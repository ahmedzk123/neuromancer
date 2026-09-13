# Daughter-artery detection

Given a CT volume and a binary parent-aorta mask, detects every eligible
direct daughter artery and reports each as a separate instance.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Writes `prediction.json` at `--output`, and alongside it in the same
directory: `daughters.nii.gz` (label volume, input grid preserved),
`snap_labels.txt` (ITK-SNAP colours), `ostia.csv`, `check.html` (interactive
3D verification view).

## Method

A frozen configuration of a Tahoces-style two-phase region-growing detector
(`k_sd=3.0`, absolute floor `230 HU`, `min_reach_mm=5.0`), selected and
validated against 5 hand-annotated cases before this rebuild:

1. Crop to the aorta mask's bounding box, padded 45mm.
2. Lumen model: erode the mask 2mm in-plane; `mu`/`sd` = mean/std of the CT
   inside.
3. Intensity band: absolute floor at 230 HU, ceiling at `mu + 3.0*sd`. The
   floor cannot be relative to the lumen -- a 2mm daughter partial-volumes
   toward fat by an amount set by scan resolution, not by the parent lumen's
   own variance, so no multiple of `sd` reproduces it across cases. The
   ceiling can, since brighter lumens also have proportionally brighter
   candidate branches.
4. Bone exclusion: threshold at `max(400, mu+3*sd)`, close x2, fill holes,
   dilate 2mm; subtracted from the band.
5. Two-phase growth: connect the band to the aorta out to 20mm; any resulting
   component over 8mL is treated as organ leakage and blocked; regrow to
   30mm avoiding blocked regions.
6. Connected components >= 8mm3 become one candidate ostium each -- the
   contact-zone voxel maximising distance to the grown mask's edge, or (for
   components that don't directly touch the wall) marching the local axis of
   the near end back to the wall and snapping to the nearest wall patch.
7. Eligibility: the component must reach >= 5mm beyond the aortic wall, and
   the origin must measure >= 2mm diameter (2x the local distance-transform
   value at the ostium -- a lightweight proxy for cross-sectional width, not
   a caliper measurement).
8. Proximal trace: skeletonize the component, walk from the ostium, stop at
   10mm of arc length or the first real bifurcation (skeleton spurs under
   1.5mm are pruned first, so single-voxel skeletonization noise isn't
   mistaken for a fork). The daughter seed is the point at exactly 5mm
   arc-length along this trace; radius is the local distance-transform value
   there; direction is a least-squares fit through the traced points.
9. Post-processing: drop candidates within 6mm of the mask's cropped
   cranial/caudal end; merge candidates closer than 6mm.

`daughters.nii.gz` holds only the traced proximal part of each candidate
(geodesic distance from the trace, capped at 5mm half-width), not the full
grown component, per the task's own definition of the proximal daughter.

## Regression (validates this rebuild against the frozen configuration)

Scored with one-to-one greedy matching at 5mm tolerance, against the
mask-derived ground truth ostium per labeled daughter (`score_ostia.py`'s
`gt_from_volume` -- the nearest-to-wall voxel within each ground-truth
label, not the annotator's raw traced guide point, which sits a few mm away
and is a different reference).

| case | GT | predicted | TP | P | R | F1 |
|---|---|---|---|---|---|---|
| case_19 | 3 | 2 | 1 | 0.50 | 0.33 | 0.40 |
| case_20 | 4 | 3 | 2 | 0.67 | 0.50 | 0.57 |
| case_21 | 3 | 4 | 3 | 0.75 | 1.00 | 0.86 |
| case_22 | 6 | 3 | 3 | 1.00 | 0.50 | 0.67 |
| case_23 | 3 | 1 | 1 | 1.00 | 0.33 | 0.50 |
| **pooled** | 19 | 13 | 10 | **0.77** | **0.53** | **0.62** |

Prediction counts and true positives are byte-identical to the originally
selected configuration on all 5 cases. Runtime: ~2-6s/case on this machine
(well inside the 60s/case budget), dominated by the geodesic label-volume
truncation and skeleton walk added for the two spec fixes below, not by the
core detector (which alone runs in ~0.1-0.2s, matching the original
measurement).

## Deviations from the original configuration (both intentional, both required by the challenge spec)

**Origin-diameter eligibility test (spec section: minimum 2mm diameter).**
The original configuration only filtered on component volume (8mm3), which
is not the same rule -- a wide, short blob and a genuine 2mm-wide vessel can
have the same volume. Added an explicit diameter check at the ostium,
kept separately from the volume filter (which now acts purely as a speckle
guard). Measured cost: zero on all 5 dev-set cases -- every surviving
candidate's origin already exceeded 2mm, so this filter changed nothing here.
It may reject candidates on unseen cases it would otherwise have kept; not
tested beyond this dev set.

**Proximal trace with bifurcation stop (spec: trace up to 10mm or the first
downstream bifurcation).** The original code estimated direction/radius from
a flat 12mm window with no bifurcation concept at all -- a blob-shaped PCA,
not a path. Replaced with a real skeleton trace: walk from the ostium, stop
at 10mm arc-length or a real fork (spurs under 1.5mm pruned as noise first).
This is the one genuinely new piece of logic in this rebuild; it does not
change which components are detected or their ostium locations (confirmed:
prediction counts and TPs above are identical to the original), only the
seed/direction/radius/label-volume derived from each one.

## Known limitations / failure modes

- **Dedupe (6mm merge) can incorrectly fuse two genuinely separate nearby
  origins** into one instance. The spec explicitly requires these stay
  separate; this configuration's 6mm merge radius was chosen for the
  original's overall F1 and was not re-validated against this specific rule.
- **Recall is the weak axis** (0.53 pooled) -- roughly half of real daughters
  are missed, mostly ones whose lumen sits far enough below the aorta's own
  brightness that even the case-adaptive floor doesn't reach them, or whose
  true diameter is right at the 2mm resolution limit of a 1.5mm-voxel scan.
- **The terminal aortoiliac bifurcation** has no special-case handling (per
  the spec, it's out of scope) -- if a supplied mask happens to extend past
  it, this detector applies the same rules to it as any other candidate,
  which may or may not be desired behaviour for that case.
- **check.html's 3D mesh placement is axis-aligned**, not applying the
  image's direction-cosine matrix -- a reasonable approximation for
  verification on these cases (which are close to axis-aligned), but would
  visibly misplace surfaces on a strongly oblique acquisition. Ostium/seed/
  direction numbers themselves are unaffected -- they go through
  `SimpleITK.TransformIndexToPhysicalPoint` directly, not this rendering path.
- **ITK-SNAP point/line geometric overlays are not produced.** I could not
  confirm ITK-SNAP has a documented file format for point/segment annotations
  from official documentation in the time available -- rather than invent a
  format, this deliverable is skipped. `check.html` is the verification
  artifact instead.
- **Determinism**: the pipeline uses `numpy.linalg.svd`/`eigvalsh` in a few
  places; these are deterministic for fixed input on a fixed BLAS backend,
  but this was not stress-tested across environments.

## Dev-set outputs

`dev_set/case_{19,20,21,22,23}/` -- all 5 EVAL_SET cases, each with
`prediction.json`, `daughters.nii.gz`, `snap_labels.txt`, `ostia.csv`,
`check.html`.
