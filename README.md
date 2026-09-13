# Aortic Branch Case Visualizer

Inspect a single CT + aorta-mask pair from the aortic-branch challenge: prints a
geometry/intensity report and saves a set of diagnostic PNGs to help spot
candidate branch vessels (celiac trunk, SMA, renals) before doing anything
more automated.

## What it produces

Running the script writes the following into `--outdir`:

| File                     | Description                                                                 |
|--------------------------|-------------------------------------------------------------------------------|
| `00_report.txt`          | Geometry, intensity, and mask stats (voxel spacing, orientation, lumen HU)   |
| `01_ortho.png`           | Axial, coronal, sagittal views through the aorta centroid, mask overlaid    |
| `02_axial_montage.png`   | Evenly spaced axial slices spanning the aortic segment                     |
| `03_mip.png`             | Coronal + sagittal slab MIPs — the clearest view for spotting daughter branches |
| `04_aorta_3d.png`        | Marching-cubes surface render of the supplied aorta mask                   |
| `05_wall_shell.png`      | Mean CT intensity in a thin shell outside the aortic wall, unrolled angle-vs-z |

## Setup

Clone the repo, then create a virtual environment and install dependencies:

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is pinned (exact `==` versions via `pip freeze`) so everyone
on the team installs an identical environment. If you add or upgrade a
package, reinstall, re-run `pip freeze > requirements.txt`, and commit the
change so the rest of the team stays in sync.

## Usage

```bash
python visualize_case.py \
  --image data/subject001/orig1.nii \
  --aorta-mask data/subject001/mask1.nii \
  --outdir viz/subject001 \
  --case-id subject001
```

Add `--interactive` for a scrollable axial window (mouse wheel / arrow keys /
slider) instead of only saving PNGs. Add `--skip-3d` to skip the marching-cubes
surface render, which is the slowest step and requires `scikit-image`.


## Detecting the branches: `branch_bench.py`

`view.py` only *shows* you the case. `branch_bench.py` finds the daughter
arteries and writes the labeled `prediction.json` (`case_id` / `parent` /
`daughters`, each vessel numbered `branch_001`, `branch_002`, ...).

```bash
python branch_bench.py \
  --image "TORALIS CHALLENGE\subject001\orig1.nii" \
  --aorta-mask "TORALIS CHALLENGE\subject001\mask1.nii" \
  --outdir bench --output prediction.json --case-id subject001
```

### How it locates an ostium

Ostia are found **on the aortic surface**, not inferred from a blob grown off
it. Every boundary voxel of the supplied mask casts a cone of rays outward
along its smoothed surface normal, and a ray survives only while it stays in
contrast-filled lumen, out of bone, out of the parent aorta, and *genuinely
getting further away from the aorta*. The surviving peaks are suppressed to
one per opening, refined to sub-voxel position on the mask's `sdf = 0`
isosurface, and then traced down the daughter's own lumen with a
medialness-weighted minimal path to get the 5 mm seed, direction and calibre.

That departure requirement is the point. The dominant false positive around an
aorta is the one-voxel partial-volume rind on its own wall — bright, thin,
tube-shaped and elongated for tens of mm, so intensity and shape tests cannot
tell it from a branch. It is trivial to reject geometrically, because it never
gets more than about a voxel clear of the wall.

### Flags worth knowing

| Flag | Default | What it does |
|------|---------|--------------|
| `--method` | `wall` | `wall` (above), `m3` (older region growing, kept for comparison), `both` |
| `--max-branches` | 25 | Cap on reported daughters, best score first |
| `--lumen-floor` | 0.35 | Fuzzy-lumen membership a ray must stay above. Lower to ~0.25 to chase faint/small vessels |
| `--min-reach-mm` | 3.5 | How far clear of the aorta a ray must end. **Below ~2 the wall rind returns** |
| `--min-seed-dist-mm` | 1.5 | Clearance required at the traced 5 mm seed. Note `5*cos(angle)`: 2.0 already rejects takeoffs beyond ~66°, so don't raise it casually |
| `--min-sep-mm` | 7.0 | Minimum spacing between two ostia |
| `--lumen-hu` | auto | Override the intensity model, e.g. `300,25`, when the report warns of dense outliers (stent/calcium) |

`<outdir>/<method>/ostia.csv` carries per-branch diagnostics (`score`,
`path_mm`, `seed_clear_mm`, `tip_clear_mm`) so a detection can be judged
without re-running, and `render.html` is a standalone rotatable 3D view of the
aorta with the traced branches and ostium markers.

There is no ground truth in the dataset, so these thresholds are set from
anatomy and geometry rather than fitted to a score. If a scoring script turns
up, `--min-reach-mm`, `--lumen-floor` and `--max-branches` are the three knobs
that trade recall against precision.

## Notes

to run view.py:
- --image says "the next thing is the image path"
- --aorta-mask says "the next thing is the mask path"
- --outdir says "the next thing is where to save output"
- --case-id says "the next thing is a label for the case"

command: 

`python view.py --image "TORALIS CHALLENGE\subject001\orig1.nii" --aorta-mask "TORALIS CHALLENGE\subject001\mask1.nii" --outdir "TORALIS CHALLENGE\subject001\viz" --case-id subject001`

- Image and mask must share the same voxel grid. If they don't, resample the
  mask onto the image first — the script will raise an error otherwise.
- All physical measurements (millimeters, slice extents) go through
  `TransformIndexToPhysicalPoint`, never raw index arithmetic, since voxel
  spacing and orientation vary between scans.
- SimpleITK arrays are indexed `[z, y, x]`; keep this in mind if you extend
  the script.