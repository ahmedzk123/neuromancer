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