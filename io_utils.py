"""io_utils.py -- NIfTI reading and the daughters.nii.gz label-volume writer."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def load_case(image_path: str, mask_path: str):
    """Returns (img, ct_array[z,y,x] float32, mask_array[z,y,x] bool)."""
    img = sitk.ReadImage(image_path)
    msk = sitk.ReadImage(mask_path)
    if img.GetSize() != msk.GetSize():
        raise ValueError(f"Grid mismatch: image {img.GetSize()} vs mask {msk.GetSize()}")
    ct = sitk.GetArrayFromImage(img).astype(np.float32)
    mk = sitk.GetArrayFromImage(msk) > 0
    return img, ct, mk


def write_daughter_labels(path: str, img, full_shape, candidates):
    """One integer label per daughter (1=branch_001, 2=branch_002, ...), 0
    elsewhere, on the INPUT grid with its origin/spacing/direction preserved --
    it must overlay the CT in ITK-SNAP with no adjustment.

    Each candidate's label_mask is crop-local (from detector.Ctx); this places
    it back into the full input volume at ctx.region before writing.
    """
    out = np.zeros(full_shape, dtype=np.int16)
    for k, c in enumerate(candidates, start=1):
        region = c["ctx_region"]
        sub = out[region]
        sub[c["label_mask"]] = k
        out[region] = sub

    vol = sitk.GetImageFromArray(out)
    vol.SetOrigin(img.GetOrigin())
    vol.SetSpacing(img.GetSpacing())
    vol.SetDirection(img.GetDirection())
    sitk.WriteImage(vol, path)


def write_snap_labels(path: str, candidates):
    """ITK-SNAP Label Description File: idx R G B A vis mesh "label"."""
    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    ]
    lines = ["# ITK-SNAP Label Description File", "# IDX R G B A VIS MSH LABEL",
             '0 0 0 0 0 0 0 "Background"']
    for k, c in enumerate(candidates, start=1):
        r, g, b = palette[(k - 1) % len(palette)]
        lines.append(f'{k} {r} {g} {b} 1 1 1 "{c["instance_id"]}"')
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
