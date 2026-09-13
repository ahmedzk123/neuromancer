# Aortic daughter-branch detection

Given a CT volume and a binary mask of the parent abdominal aorta, detect every
eligible artery arising directly from the aorta and return each as a separate
daughter instance with an ostium, a 5 mm seed, a direction and a local radius.

Classical image processing only. CPU, no GPU, no network, deterministic.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

That is the whole interface. No manual point placement, no per-case options.

### What it writes

`prediction.json` at `--output`, and — unless you pass `--json-only` — four more
files in `<output stem>_files/` beside it:

| file | what it is |
|---|---|
| `daughters.nii.gz` | Label volume on the **input grid**, one integer label per daughter (1 = `branch_001`). Origin, spacing and direction are copied from the input, so it overlays the CT in ITK-SNAP with no adjustment. |
| `snap_labels.txt` | ITK-SNAP Label Description File: load it with **Segmentation → Import Label Descriptions** so each daughter gets its own colour and name. |
| `ostia.csv` | One row per daughter, for anyone who would rather not parse JSON. |
| `check.html` | Self-contained 3D view: aorta mask, detected daughters, ostium markers, direction arrows, seeds and proximal paths. Open it in any browser. |

`prediction.json` carries the four required fields (`ostium_xyz_mm`,
`seed_xyz_mm`, `radius_mm`, `direction_xyz`) plus the full proximal centreline,
the origin-diameter estimate, the measurement method and status for each value,
and a `policy` block stating the eligibility rules that were applied. All
coordinates are physical millimetres from
`SimpleITK.TransformIndexToPhysicalPoint`; no voxel indices are reported.

## Method

Five steps, all classical.

1. **Intensity model.** Erode the supplied mask by 2 mm and take the CT inside
   it: that is this patient's contrast level. Enhancement varies enormously
   between scans — across our development cases the lumen ran from 302 to 579 HU.
2. **Band.** Keep voxels in `[230 HU, mean + 3·sd]`. The **floor is absolute and
   the ceiling is relative**, which is the one design decision worth defending:
   a daughter's apparent HU sits far below the parent's because a 2 mm vessel in
   a 1.5 mm voxel is mostly partial volume, and that offset is set by resolution,
   not by the lumen's variance. On one case the daughters measured 185 HU under
   the parent — about seven MADs — so no multiple of the spread reaches them.
   The ceiling must stay relative because one case's blood is brighter than
   another case's bone.
3. **Bone exclusion.** Threshold above `max(400, mean + 3·sd)`, close, fill and
   dilate 2 mm. Trabecular bone is a mesh of thin struts and reads as vessel to
   almost any detector; the dilation is because the strongest response is at the
   bone edge, not inside it.
4. **Two-phase growth with leak detection** (after Tahoces et al.). Grow the band
   out from the mask to 20 mm, flag any component over 8 mL as an organ leak,
   then regrow to 30 mm avoiding the flagged regions.
5. **Ostia and proximal trace.** One ostium per connected component, at the
   contact voxel furthest from the component's own edge. Candidates that never
   reach 5 mm beyond the wall are dropped, ostia within 6 mm of the mask's ends
   are dropped (a cropped end is not an origin), and ostia within 6 mm of each
   other are merged. Each survivor is then traced **geodesically** for up to
   10 mm or until the first bifurcation, and that path gives the seed, direction
   and radius.

The trace is geodesic rather than Euclidean because a Euclidean window on a
curving vessel counts its own far wall as proximal. A bifurcation is a geodesic
shell splitting into two limbs that persist for two consecutive shells; the
persistence test stops a single speckle from truncating the trace. The radius is
an area-equivalent radius on a 0.25 mm plane perpendicular to the path, not the
distance transform of a binary blob — on a 1.5 mm grid the latter is quantised
to half-voxel steps.

Every constant lives in `detector.Config` and `trace.TraceConfig`.

## Runtime

Roughly **1–3 s per case** on one core for the detector and trace; the rest is
NIfTI I/O. Well inside the 60 s target. Peak memory is a few hundred MB, set by
the cropped volume. All per-daughter work happens inside that daughter's
bounding box, which is what keeps the trace at ~0.1 s rather than ~7 s.

## Development set

```
python run_dev.py --manifest cases.csv --outdir dev --gates 0,2
```

`cases.csv` is `case_id,image,aorta_mask,ref`. `--gates` runs the whole set at
several origin-diameter thresholds so the cost of the 2 mm eligibility rule is
measured rather than assumed. Writes predictions, per-case artefacts and
`dev/scores.csv`.

To score a single case:

```
python score.py --pred prediction.json --ref annotation.json
```

**Score against the annotation JSON, not the label volume.** `score.py` accepts
both, but a label volume only yields a *derived* ostium — the labelled voxel
nearest the wall — and on subject020 that proxy sits **1.3 to 7.2 mm** from the
`ostium_xyz_mm` the annotation actually declares. The worst case is a daughter
that runs alongside the aorta, where "nearest voxel to the wall" can land
anywhere along its length. At a 5 mm matching tolerance that difference decides
true positives.

Matching is optimal one-to-one (Hungarian on the ostium distance matrix). Greedy
nearest-neighbour lets one lucky prediction claim a reference point a closer
prediction should have had, turning one hit into two mistakes.

## Known failure cases

- **Low-contrast studies.** The 230 HU floor is absolute. On a scan whose
  daughters fall below it they will not be found. This is the deliberate trade
  for surviving the bright-lumen cases; a relative floor fails those instead.
- **Daughters running parallel to the aorta.** The ostium is the component's
  contact voxel furthest from its own edge. For a vessel that hugs the aorta the
  true origin and the deepest contact can be centimetres apart.
- **One ostium per connected component.** Where the supplied mask
  under-segments, a thin sheath of lumen-intensity blood coats the aorta and can
  weld two adjacent branch bases into one component — which then reports one
  ostium instead of two. This is the main cause of missed detections when two
  origins are close together.
- **The 2 mm origin gate is near the voxel size.** At 1.5 mm isotropic a 2 mm
  origin is 1.3 voxels across, so eligibility near the threshold is noisy in
  both directions. Run `--gates 0,2` to see what it costs on your data.
- **Iliac bifurcation.** Ostia within 6 mm of the mask ends are dropped, which
  is also where the iliacs sit. That matches the core task, in which the terminal
  division is out of scope, but it means the optional iliac extension needs that
  rule relaxed.
- **Precision is the weaker half.** Recall is the easier score here; most of the
  remaining error is extra candidates, typically small bright structures next to
  the wall that satisfy the band, the 5 mm reach and the origin diameter.

## Files

| file | role |
|---|---|
| `run.py` | CLI and orchestration |
| `detector.py` | Steps 1–5: intensity model, band, bone, growth, ostia, eligibility |
| `trace.py` | Geodesic proximal trace, bifurcation stop, seed / direction / radius |
| `outputs.py` | JSON, label volume, ITK-SNAP labels, CSV, 3D view |
| `score.py` | Hungarian matching and metrics against reference annotations |
| `run_dev.py` | Batch over the development set |
