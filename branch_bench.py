#!/usr/bin/env python3
"""
branch_bench.py -- SELF-CONTAINED. Detect daughter arteries branching off the
supplied aorta mask and write a labeled prediction.json in the required
submission format.

    pip install SimpleITK numpy scipy matplotlib scikit-image plotly
    python branch_bench.py --image orig1.nii --aorta-mask mask1.nii \
        --outdir bench --output prediction.json --case-id subject001

METHODS  (--method, default "wall")
  wall   Wall scan. Ostia are found ON the aortic surface: every boundary
         voxel casts a cone of rays outward and keeps the longest unbroken
         run of contrast-filled lumen that genuinely DEPARTS the aorta. The
         surviving peaks are then traced geodesically down their own lumen
         to get the 5 mm seed, direction and calibre.
  m3     The older Tahoces-style two-phase expansion with leak detection,
         gated by a PCA shape filter. Kept for comparison -- see WHY BELOW.
  both   wall first, then any m3 detection that is not already covered.

WHY THE WALL SCAN REPLACED M3 AS THE DEFAULT
  m3 region-grows everything in the lumen HU band that is connected to the
  aorta, so its candidates include the one-voxel partial-volume rind that
  hugs the aortic wall for tens of mm. Those strands pass the PCA shape gate
  (they are long and thin) and their "ostium" is then the arbitrary point
  along the strand that happens to sit nearest the wall -- which is why the
  reported ostia did not sit on real branch openings. The wall scan cannot
  make that mistake: a candidate only counts if a ray leaves the surface and
  is still >= min_reach_mm away from the aorta at the end of the run, which a
  wall-hugging rind never is.

OUTPUTS
  <outdir>/<method>/{ostia.csv, prediction.json, render.html, overview.png}
  <output path>  -- the labeled prediction (case_id / parent / daughters),
                     each detected vessel numbered branch_001, branch_002, ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "2026-09-13.7-wall-scan"

CTA_LEVEL, CTA_WIDTH = 200.0, 700.0
MASK_CMAP = ListedColormap([(0, 0, 0, 0), (1.0, 0.25, 0.25, 0.35)])


# -----------------------------------------------------------------------------
# helpers (identical to the ones in the visualiser script)
# -----------------------------------------------------------------------------

def window(slab: np.ndarray, level: float = CTA_LEVEL, width: float = CTA_WIDTH):
    lo, hi = level - width / 2.0, level + width / 2.0
    return np.clip((slab - lo) / (hi - lo), 0.0, 1.0)


def mask_bbox(mk: np.ndarray, pad_vox=(0, 0, 0), crop: bool = True):
    """Return slices bounding the mask, padded and clipped.

    crop=False returns the full volume -- every figure then shows the whole
    field of view, which is the default behaviour of this script.
    """
    if not crop:
        return tuple(slice(0, s) for s in mk.shape)
    idx = np.where(mk > 0)
    if len(idx[0]) == 0:
        return tuple(slice(0, s) for s in mk.shape)
    out = []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - pad_vox[ax], 0)
        hi = min(int(idx[ax].max()) + pad_vox[ax] + 1, mk.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)


def _looks_gzipped(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _read_via_nibabel(path: str):
    """Last-resort NIfTI reader for files ITK refuses, converted to ITK's world.

    nibabel reports the affine in RAS; ITK works in LPS and its own NIfTI
    reader negates the first two axes on the way in. Applying the same flip
    here means a file loaded down this path lands in exactly the same world
    coordinates as one loaded by SimpleITK directly -- verified against a
    readable case, spacing/origin/direction/voxels all identical. Without the
    flip the ostia for these cases would come out mirrored in x and y, which
    is the kind of error that looks plausible in a JSON and is invisible until
    someone overlays it on the scan.
    """
    import nibabel as nib

    n = nib.load(path)
    aff = np.asarray(n.affine, float)
    flip = np.diag([-1.0, -1.0, 1.0])
    rot = flip @ aff[:3, :3]
    spacing = np.linalg.norm(rot, axis=0)
    spacing[spacing <= 0] = 1.0
    # nearest true rotation to what was written: the polar factor of the SVD.
    # It preserves handedness (det keeps its sign), so a legitimately
    # left-handed volume is not silently mirrored.
    u, _, vt = np.linalg.svd(rot / spacing)
    direction = u @ vt

    img = sitk.GetImageFromArray(
        np.ascontiguousarray(np.asanyarray(n.dataobj).T))   # [x,y,z] -> [z,y,x]
    img.SetSpacing([float(v) for v in spacing])
    img.SetOrigin([float(v) for v in flip @ aff[:3, 3]])
    img.SetDirection([float(v) for v in direction.flatten()])
    return img


def _read_nifti(path: str):
    """Read a NIfTI, working around two defects present in this dataset.

    Neither is exotic and both make SimpleITK fail outright, so without this
    the affected cases simply cannot be submitted:

      * Some volumes hold gzip content under a plain `.nii` name. ITK selects
        its reader by EXTENSION, hands the compressed bytes to the NIfTI
        parser and reports "not recognized as a NIFTI file". Decompress to a
        real temporary .nii first and the same reader is happy.
      * Some volumes have direction cosines that are off orthonormal by about
        1e-3 -- a scanner rounding its own rotation matrix. ITK rejects them
        with "only supports orthonormal direction cosines". nibabel does not
        care, so read it there and square the matrix up.
    """
    src, tmp = path, None
    if _looks_gzipped(path) and not path.lower().endswith(".gz"):
        import gzip
        import shutil
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".nii")
        os.close(fd)
        with gzip.open(path, "rb") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        src = tmp
    try:
        try:
            return sitk.ReadImage(src)
        except RuntimeError as exc:
            if "orthonormal" not in str(exc).lower():
                raise
            print(f"  NOTE: {os.path.basename(path)} has non-orthonormal "
                  f"direction cosines; reading via nibabel and squaring them "
                  f"up.", file=sys.stderr)
            return _read_via_nibabel(src)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_case(image_path: str, mask_path: str):
    """Read image + mask, verify they share a grid, return (img, mask, arrays)."""
    img = _read_nifti(image_path)
    msk = _read_nifti(mask_path)

    if img.GetSize() != msk.GetSize():
        raise ValueError(
            f"Grid mismatch: image {img.GetSize()} vs mask {msk.GetSize()}. "
            "Resample the mask onto the image before continuing."
        )
    if not np.allclose(img.GetSpacing(), msk.GetSpacing(), atol=1e-4):
        print("WARNING: spacing differs slightly between image and mask", file=sys.stderr)

    ct = sitk.GetArrayFromImage(img).astype(np.float32)          # [z, y, x]
    mk = (sitk.GetArrayFromImage(msk) > 0).astype(np.uint8)      # [z, y, x]
    return img, msk, ct, mk


def geometry_report(img, ct, mk) -> str:
    sx, sy, sz = img.GetSpacing()
    nz, ny, nx = ct.shape
    vox_mm3 = sx * sy * sz
    n_mask = int(mk.sum())

    zs = np.where(mk.any(axis=(1, 2)))[0]
    if len(zs):
        z0, z1 = int(zs[0]), int(zs[-1])
        p0 = img.TransformIndexToPhysicalPoint((int(nx // 2), int(ny // 2), z0))
        p1 = img.TransformIndexToPhysicalPoint((int(nx // 2), int(ny // 2), z1))
        craniocaudal = abs(p1[2] - p0[2])
        zrange = f"slices {z0}-{z1} ({z1 - z0 + 1} slices, {craniocaudal:.1f} mm cranio-caudal)"
    else:
        zrange = "EMPTY MASK"

    inside = ct[mk == 1]
    lines = [
        "=" * 68,
        "CASE GEOMETRY",
        "=" * 68,
        f"  size (x,y,z)        : {nx} x {ny} x {nz}",
        f"  spacing (mm)        : {sx:.3f}, {sy:.3f}, {sz:.3f}",
        f"  origin (mm)         : {tuple(round(v, 2) for v in img.GetOrigin())}",
        f"  direction           : {tuple(round(v, 3) for v in img.GetDirection())}",
        f"  axis codes          : {sitk_orientation(img)}",
        "",
        "CT INTENSITIES",
        f"  full range (HU)     : {ct.min():.0f} .. {ct.max():.0f}",
        f"  1st/99th pct        : {np.percentile(ct, 1):.0f} .. {np.percentile(ct, 99):.0f}",
        "",
        "AORTA MASK",
        f"  voxels              : {n_mask:,}   ({n_mask * vox_mm3 / 1000.0:.1f} mL)",
        f"  z-extent            : {zrange}",
    ]
    if n_mask:
        lines += [
            f"  lumen HU mean/std   : {inside.mean():.0f} / {inside.std():.0f}",
            f"  lumen HU 5/50/95    : {np.percentile(inside, 5):.0f} / "
            f"{np.percentile(inside, 50):.0f} / {np.percentile(inside, 95):.0f}",
            "",
            "  -> Use the lumen percentiles to set a case-adaptive threshold for branch",
            "     hunting; absolute HU varies a lot with contrast timing between subjects.",
        ]
    lines.append("=" * 68)
    return "\n".join(lines)


def sitk_orientation(img) -> str:
    """Approximate nibabel's aff2axcodes for a SimpleITK image."""
    d = np.array(img.GetDirection()).reshape(3, 3)
    labels = [("L", "R"), ("P", "A"), ("I", "S")]
    codes = []
    for axis in range(3):
        col = d[:, axis]
        k = int(np.argmax(np.abs(col)))
        codes.append(labels[k][1] if col[k] > 0 else labels[k][0])
    return "".join(codes)


def _surface(binary, spacing, target_vox, smooth=0.8):
    """Downsample a binary volume to ~isotropic target_vox mm and marching-cube it.

    CAREFUL with target_vox and smooth on thin structures. Downsampling to 1.5 mm
    and then blurring by sigma 0.8 (of the COARSE grid, so ~1.2 mm) drives a 2 mm
    vessel below the 0.5 isolevel and it disappears entirely -- the detection is
    still in the data, the render just deleted it. Use fine_surface() for anything
    vessel-sized; keep this one for the big aorta where smoothing is cosmetic.

    Returns (verts_xyz_mm, faces) or (None, None) if the volume is empty.
    """
    from skimage.measure import marching_cubes

    if not binary.any():
        return None, None
    sx, sy, sz = spacing
    zoom = (sz / target_vox, sy / target_vox, sx / target_vox)
    small = ndi.zoom(binary.astype(np.float32), zoom, order=1)
    if smooth:
        small = ndi.gaussian_filter(small, smooth)
    if small.max() < 0.5:
        return None, None
    verts, faces, _, _ = marching_cubes(small, level=0.5, spacing=(target_vox,) * 3)
    return verts[:, [2, 1, 0]], faces          # (z,y,x) -> (x,y,z)


def smooth_mesh(verts, faces, iters=12, lam=0.55, mu=-0.58):
    """Taubin mesh smoothing: removes the voxel staircase without shrinking.

    Marching cubes on a binary mask is blocky by construction -- every face is
    axis-aligned, so a tube looks like a flight of stairs, and anisotropic slices
    make the steps in z worse. Plain Laplacian smoothing fixes the look but
    shrinks thin tubes away. Taubin alternates a positive and a slightly larger
    negative step, which smooths while pushing the surface back out.
    """
    if verts is None or len(verts) < 4:
        return verts
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    src, dst = e[:, 0], e[:, 1]
    cnt = np.bincount(src, minlength=n).astype(np.float64)
    cnt[cnt == 0] = 1.0
    v = verts.astype(np.float64).copy()
    for it in range(iters):
        avg = np.empty_like(v)
        for k in range(3):
            avg[:, k] = np.bincount(src, weights=v[dst, k], minlength=n) / cnt
        v += (lam if it % 2 == 0 else mu) * (avg - v)
    return v.astype(np.float32)


def fine_surface(binary, spacing, cap=400_000, iso=0.40, blur_vox=0.9,
                 smooth_iters=12):
    """Isotropic, mildly blurred, low-isolevel marching cubes + Taubin smoothing.

    The three settings work together. Resampling to isotropic kills the z
    staircase. A small blur (in VOXELS of the isotropic grid, so it scales) makes
    the surface continuous. Dropping the isolevel from 0.5 to 0.40 compensates
    for the blur so thin vessels survive instead of dissolving -- blurring at
    level 0.5 is exactly what was deleting them before.
    """
    from skimage.measure import marching_cubes

    if not binary.any():
        return None, None
    base = float(min(spacing))
    sx, sy, sz = spacing
    for mult in (1.0, 1.3, 1.7, 2.2, 3.0):
        tv = base * mult
        small = ndi.zoom(binary.astype(np.float32),
                         (sz / tv, sy / tv, sx / tv), order=1)
        if blur_vox:
            small = ndi.gaussian_filter(small, blur_vox)
        if small.max() < iso:
            return None, None
        verts, faces, _, _ = marching_cubes(small, level=iso, spacing=(tv,) * 3)
        if len(faces) <= cap or mult == 3.0:
            verts = verts[:, [2, 1, 0]]
            return smooth_mesh(verts, faces, iters=smooth_iters), faces
    return None, None


def lumen_band(ct, mk, k_sd=2.5, erode_mm=2.0, spacing=(1, 1, 1)):
    """Lumen HU model, estimated ROBUSTLY.

    mean/std are wrecked by anything dense inside the mask -- a stent, calcified
    plaque, or a mask that drifts onto bone. One such case gave mean 579, sd 97,
    which is physically impossible for arterial blood and poisoned every
    threshold downstream. Median and MAD ignore those outliers.

    Returns (lo, hi, centre, spread, flag) -- flag is a warning string or "".
    """
    sx, sy, sz = spacing
    er_iter = max(int(round(erode_mm / min(sx, sy))), 1)
    core = ndi.binary_erosion(mk, iterations=er_iter)
    if core.sum() < 50:
        core = mk.astype(bool)
    v = ct[core]
    mean, sd = float(v.mean()), float(v.std())
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826          # -> sd equivalent
    spread = max(mad, 8.0)

    flag = ""
    if sd > 2.5 * spread and sd > 50:
        flag = (f"mask contains dense outliers (mean {mean:.0f}/sd {sd:.0f} vs "
                f"median {med:.0f}/MAD {spread:.0f}) - stent, calcium or "
                f"misalignment. Using the robust estimate.")
    if med > 450:
        flag += (" | median lumen > 450 HU is very high for blood; check the "
                 "mask actually covers the lumen.")
    return med - k_sd * spread, med + k_sd * spread, med, spread, flag



def link_to_aorta(pts_vox, dist_out, axis_mm, spacing,
                  max_gap_mm=15.0, step_mm=0.4, hit_mm=0.9,
                  end_mm=6.0, sub_mk=None, patch_mm=4.0):
    """March from a candidate's aorta-facing END along its LOCAL axis to the wall.

    Three things matter for where the ostium lands:
      * start from the CENTROID of the near end, not the single nearest voxel --
        one noisy voxel off the tube axis throws the ray sideways;
      * use the axis of just the near end, not of the whole component -- the SMA
        curves through 90 degrees, so its global axis points nowhere useful;
      * once the ray hits, snap to the centre of the local wall patch, which is
        what "centre of the opening" means.

    Returns (hit, gap_mm, ostium_vox or None, outward_unit_axis).
    """
    sx, sy, sz = spacing
    sp = np.array([sz, sy, sx])
    pts_mm = pts_vox * sp
    d_here = dist_out[tuple(pts_vox.T.astype(int))]
    near_i = int(np.argmin(d_here))

    # near-end cluster, and its own local axis
    dsel = np.linalg.norm(pts_mm - pts_mm[near_i], axis=1) <= end_mm
    if dsel.sum() >= 4:
        end_pts = pts_mm[dsel]
        start_mm = end_pts.mean(axis=0)
        c = end_pts - start_mm
        _, _, vv = np.linalg.svd(c, full_matrices=False)
        u = vv[0]
    else:
        start_mm = pts_mm[near_i]
        u = np.asarray(axis_mm, float)
    u = u / max(np.linalg.norm(u), 1e-9)
    p0 = start_mm / sp

    best = (False, np.inf, None, u)
    for sign in (1.0, -1.0):
        prev = None
        for t in np.arange(step_mm, max_gap_mm + step_mm, step_mm):
            p = p0 + sign * u * t / sp
            if np.any(p < 0) or np.any(p >= np.array(dist_out.shape) - 1):
                break
            d = float(ndi.map_coordinates(dist_out, p[:, None], order=1)[0])
            if prev is not None and d > prev + 0.6:
                break
            prev = d
            if d <= hit_mm:
                if t < best[1]:
                    ost = p
                    # snap to the centre of the wall patch around the hit
                    if sub_mk is not None:
                        wall = (ndi.binary_dilation(sub_mk, iterations=1)
                                & ~ndi.binary_erosion(sub_mk, iterations=1))
                        wp = np.argwhere(wall).astype(float)
                        if len(wp):
                            dd = np.linalg.norm((wp - p) * sp, axis=1)
                            loc = wp[dd <= patch_mm]
                            if len(loc) >= 3:
                                ost = loc.mean(axis=0)
                    best = (True, float(t), ost, sign * u)
                break
    return best


def branch_roi(sub_ct, sub_mk, spacing, roi_mm, mu, sd,
               hi_sd=3.0, lo_sd=4.0, bone_dilate_mm=2.0, bone_floor_hu=None):
    """Where it is worth looking for a daughter vessel.

    A shell around the aorta, intensity-bounded on BOTH sides. The upper bound is
    the important one: cortical bone sits directly against the aorta, and
    trabecular bone is a dense network of tiny struts -- which is precisely what a
    Hessian line-detector responds to. Without an upper cut, Frangi reports the
    vertebra and the real branches get buried.

    The bone exclusion is dilated by a couple of mm because the strongest Hessian
    response is at the bone EDGE, not inside it.
    """
    sx, sy, sz = spacing
    dist_out = ndi.distance_transform_edt(~sub_mk, sampling=(sz, sy, sx))
    # Never cut below plausible arterial enhancement: a bright CTA lumen can
    # reach 450 HU, so a purely relative cut would delete the vessels.
    # An intensity-only bone cut fails whenever the lumen estimate is high: the
    # cut floats up above bone and the vertebra stays in the search region.
    # Bone is also BIG and solid, while a daughter vessel is thin -- so require
    # size as well, which holds no matter how bright the scan is.
    hi_cut = (max(350.0, mu + hi_sd * sd) if bone_floor_hu is None
              else float(bone_floor_hu))
    seed = (sub_ct > hi_cut) & ~ndi.binary_dilation(sub_mk, iterations=2)
    seed = ndi.binary_closing(seed, iterations=2)
    lab_b, n_b = ndi.label(seed)
    bone = np.zeros_like(seed)
    if n_b:
        vox_mm3 = sx * sy * sz
        big = np.bincount(lab_b.ravel()) * vox_mm3 / 1000.0 > 2.0   # > 2 mL
        big[0] = False
        bone = big[lab_b]
    # Trabecular marrow sits in the lumen HU range, so fill the cortex shell.
    bone = ndi.binary_fill_holes(bone)
    it = max(int(round(bone_dilate_mm / min(sx, sy))), 1)
    bone = ndi.binary_dilation(bone, iterations=it)
    roi = ((dist_out > 0) & (dist_out <= roi_mm)
           & (sub_ct > mu - lo_sd * sd) & (~bone))
    return roi, dist_out, bone, hi_cut


def write_interactive_html(out_html, meshes, title, subtitle="", axis_note=""):
    """Self-contained rotatable 3D page (plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        if not getattr(write_interactive_html, "_warned", False):
            write_interactive_html._warned = True
            print("  NOTE: plotly not installed -- render.html will be skipped."
                  "  pip install plotly", file=sys.stderr)
        return False

    traces = []
    for m in meshes:
        if m.get("kind") == "points":
            pt = m["points"]
            traces.append(go.Scatter3d(
                x=pt[:, 0], y=pt[:, 1], z=pt[:, 2], mode="markers",
                marker=dict(size=m.get("size", 6), color=m.get("color", "#111"),
                            symbol="diamond", line=dict(width=1, color="white")),
                name=m.get("name", "ostia"), showlegend=True,
                text=m.get("labels", None),
                hovertemplate="%{text}<extra></extra>"))
            continue
        if m.get("kind") == "lines":
            pt = m["points"]
            traces.append(go.Scatter3d(
                x=pt[:, 0], y=pt[:, 1], z=pt[:, 2], mode="lines",
                line=dict(width=m.get("width", 5), color=m.get("color", "#111")),
                name=m.get("name", "links"), showlegend=True, hoverinfo="skip"))
            continue
        v, f = m["verts"], m["faces"]
        if v is None or len(v) == 0:
            continue
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=m.get("color", "#d94a4a"), opacity=m.get("opacity", 1.0),
            name=m.get("name", "surface"), showlegend=True,
            visible=m.get("visible", True),
            hovertemplate=(m.get("text", m.get("name", "")) + "<extra></extra>"),
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15,
                          roughness=0.85)))
    if not traces:
        return False

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px'>"
                        f"{subtitle}</span>", x=0.01, xanchor="left"),
        scene=dict(aspectmode="data",
                   xaxis_title=f"x (mm){axis_note}",
                   yaxis_title=f"y (mm){axis_note}",
                   zaxis_title=f"z (mm){axis_note}",
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
        legend=dict(itemsizing="constant", title="click to toggle"),
        margin=dict(l=0, r=0, t=70, b=0), template="plotly_white")
    fig.write_html(out_html, include_plotlyjs="inline", full_html=True)
    return True


def write_obj(path, verts, faces, name="surface"):
    """Plain Wavefront OBJ -- opens in 3D Slicer, MeshLab, Blender, Windows 3D
    Viewer, Preview on macOS. No dependencies, no internet, fully interactive."""
    with open(path, "w") as fh:
        fh.write(f"o {name}\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")


# -----------------------------------------------------------------------------
# shared plumbing
# -----------------------------------------------------------------------------

class Ctx:
    """Everything the methods share: cropped volumes, geometry, intensity model."""

    def __init__(self, img, ct, mk, spacing, margin_mm=45.0, roi_mm=30.0,
                 min_elong=1.8, min_length_mm=3.0):
        self.img, self.spacing = img, spacing
        # shape gate used by components(): a candidate connected component
        # must be at least this elongated (length / width) and this long
        # (mm along its own principal axis) to count as vessel-like, not a
        # compact fleck/blob. See _looks_like_vessel().
        self.min_elong = min_elong
        self.min_length_mm = min_length_mm
        sx, sy, sz = spacing
        self.sx, self.sy, self.sz = sx, sy, sz
        self.vox_mm3 = sx * sy * sz
        pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
               int(round(margin_mm / sx)))
        self.region = mask_bbox(mk, pad, crop=True)
        self.ct = ct[self.region]
        self.mk = mk[self.region].astype(bool)
        self.z0 = self.region[0].start or 0
        self.y0 = self.region[1].start or 0
        self.x0 = self.region[2].start or 0

        self.lo, self.hi, self.mu, self.sd, self.flag = lumen_band(
            ct, mk, spacing=spacing)
        self.roi, self.dist_out, self.bone, self.hi_cut = branch_roi(
            self.ct, self.mk, spacing, roi_mm, self.mu, self.sd)
        self.roi_mm = roi_mm
        self.wall = (ndi.binary_dilation(self.mk, iterations=1)
                     & ~ndi.binary_erosion(self.mk, iterations=1))
        self.sp = np.array([sz, sy, sx])
        # Direction matrix. Ignoring it puts marching-cubes surfaces in a
        # MIRRORED position relative to points that went through
        # TransformIndexToPhysicalPoint -- mesh and markers end up in different
        # parts of the scene. Column j is the world direction of voxel axis j.
        try:
            self.D = np.array(img.GetDirection(), float).reshape(3, 3)
        except Exception:
            self.D = np.eye(3)
        self.corner = None

        # aorta centreline: centroid per slice, for angles and radial geometry
        self.cl = {}
        for z in np.where(self.mk.any(axis=(1, 2)))[0]:
            cy, cx = ndi.center_of_mass(self.mk[z])
            self.cl[int(z)] = (cy, cx)

    def verts_to_world(self, v):
        """Marching-cubes vertices (mm from the crop corner, x/y/z) -> world mm."""
        if v is None:
            return None
        if self.corner is None:
            self.corner = self.to_mm((0, 0, 0))
        return self.corner + np.asarray(v, float) @ self.D.T

    def to_mm(self, vox):
        """voxel (z, y, x) in the CROP -> physical (x, y, z) mm, SUB-VOXEL.

        This used to round to the nearest integer index and call
        TransformIndexToPhysicalPoint, which threw away every fraction of a
        voxel the detector had worked out: an ostium refined to the sdf=0
        isosurface, or a centroid averaged over a wall patch, was snapped
        straight back onto the voxel lattice. On a 0.78 x 0.78 x 0.80 mm grid
        that is up to ~0.7 mm of pure quantisation error on a target whose
        whole point is sub-millimetre placement, and it made neighbouring
        candidates collapse onto identical coordinates.
        TransformContinuousIndexToPhysicalPoint keeps the fraction and applies
        the same origin/spacing/direction transform.
        """
        v = np.asarray(vox, float)
        return np.array(self.img.TransformContinuousIndexToPhysicalPoint(
            (float(v[2]) + self.x0, float(v[1]) + self.y0,
             float(v[0]) + self.z0)), float)


class Result:
    def __init__(self, name, mask, ostia_vox, dirs=None, radii=None,
                 seeds=None, runtime=0.0, note="", extras=None, paths=None):
        self.name = name
        self.mask = mask                       # bool volume of believed branches
        self.ostia_vox = list(ostia_vox)       # [(z, y, x)] in crop coords
        self.dirs = dirs or [None] * len(self.ostia_vox)
        self.radii = radii or [None] * len(self.ostia_vox)
        self.seeds = seeds or [None] * len(self.ostia_vox)  # traced ~5mm point
        # traced centreline per branch, crop voxel coords, ostium first.
        # Kept because it is the only thing that can tell two contact points
        # on one vessel apart from two vessels -- see
        # prune_duplicate_branches().
        self.paths = paths or [None] * len(self.ostia_vox)
        # per-branch diagnostics (score, traced length, seed clearance, ...)
        # written to ostia.csv so a detection can be judged without re-running
        self.extras = extras or [{} for _ in self.ostia_vox]
        self.runtime = runtime
        self.note = note

    def select(self, keep):
        """Keep only the branches at the given indices, all fields in step.

        Every filter (drop_end_caps, prune_duplicate_branches) has to subset
        the same parallel lists. Doing it by hand in each one is how a field
        gets forgotten and the diagnostics end up describing a different
        branch than the coordinates.
        """
        for attr in ("ostia_vox", "dirs", "radii", "seeds", "paths", "extras"):
            setattr(self, attr, [getattr(self, attr)[k] for k in keep])
        return self


def _looks_like_vessel(comp_mask, ctx):
    """PCA shape gate: keep tube-like components, drop compact blobs.

    A volume-only cut keeps anything that clears the voxel-count bar, blobby
    or not -- which is exactly what was showing up as "plaque": compact
    specks of leftover vesselness/OOF response or band noise near the wall
    that have enough voxels to survive but no vessel shape. A real proximal
    branch segment is elongated (one long principal axis, two short ones),
    so this fits PCA to the component's physical (mm) coordinates and checks
    both the length along the long axis and the aspect ratio against the two
    short axes. Cheap, no CT intensity needed, and it runs on every method's
    output the same way since they all funnel through components().
    """
    pts = np.argwhere(comp_mask).astype(float) * ctx.sp
    if len(pts) < 4:
        return False
    c = pts - pts.mean(axis=0)
    cov = np.cov(c.T)
    evals, evecs = np.linalg.eigh(cov)          # ascending eigenvalues
    proj = c @ evecs
    extents = proj.max(axis=0) - proj.min(axis=0)   # physical mm, per axis
    length = extents[-1]                        # longest (principal) axis
    width = max(extents[0], extents[1], 1e-6)    # widest of the two minor axes
    return length >= ctx.min_length_mm and (length / width) >= ctx.min_elong


def components(mask, ctx, min_mm3=8.0):
    lab, _ = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(min_mm3 / ctx.vox_mm3), 1)
    keep = []
    for i in np.argsort(sizes)[::-1]:
        if sizes[i] < min_vox:
            continue
        if not _looks_like_vessel(lab == i, ctx):
            continue
        keep.append(int(i))
    return lab, keep


def _ostium_on_wall(comp, wall_dist, wall_pts_mm, ctx, patch_mm=4.0):
    """Snap the ostium to the centroid of the aortic WALL patch this
    component touches, instead of a point a voxel or two inside the branch.

    The previous approach picked "the component voxel nearest the wall with
    the largest local thickness" as the ostium -- a reasonable proxy, but
    that point sits just past the true opening, biased into the branch by
    however far the vesselness/band signal happens to taper at the junction.
    The aortic wall itself ('ctx.wall', the thin shell of the mask boundary)
    is where the opening actually is. So: find the wall voxel nearest to
    this component, then average every wall voxel within patch_mm of it --
    the centroid of the local opening on the aorta's OWN surface, not the
    branch's.

    wall_dist/wall_pts_mm are precomputed once per ostia_from_mask() call and
    passed in, since recomputing a full-volume EDT per candidate component
    would be wasteful.

    Returns ostium voxel coords (z, y, x), or None if this component isn't
    actually near the wall (shouldn't happen for a grown-from-the-aorta
    candidate, but callers should fall back to the axis-extension method).
    """
    comp_pts = np.argwhere(comp)
    if len(comp_pts) == 0 or len(wall_pts_mm) == 0:
        return None
    near_i = int(np.argmin(wall_dist[tuple(comp_pts.T)]))
    if not np.isfinite(wall_dist[tuple(comp_pts[near_i])]):
        return None
    anchor_mm = comp_pts[near_i].astype(float) * ctx.sp
    dd = np.linalg.norm(wall_pts_mm - anchor_mm, axis=1)
    idx = np.where(dd <= patch_mm)[0]
    if len(idx) < 3:
        idx = np.array([int(np.argmin(dd))])
    return (wall_pts_mm[idx] / ctx.sp).mean(axis=0)


def _geodesic_trace(comp, anchor_vox, ctx, max_mm=12.0):
    """Geodesic distance (mm) from anchor_vox outward through `comp` ONLY.

    The previous direction/radius estimate took every branch voxel within a
    straight-line 12mm Euclidean ball of the ostium, and measured radius
    off an EDT computed on the WHOLE multi-branch mask -- so a curving
    vessel got a straight-line-biased direction, and a candidate sitting
    near a different branch (or leftover leak) could have its radius pulled
    by that neighbour's thickness, not its own. Confining the walk to this
    one component (cost=inf everywhere else) and following it geodesically
    fixes both: it can't see other branches, and it follows the actual
    curve, which is also what the challenge's "trace the proximal branch"
    wording asks for.

    Returns (gd, ok). gd is mm-distance from anchor_vox, np.inf outside
    comp / unreached voxels. ok is False if tracing failed (component too
    small/degenerate for the graph solver).
    """
    from skimage.graph import MCP_Geometric
    if comp.sum() < 4:
        return None, False
    try:
        cost = np.where(comp, 1.0, np.inf)
        mcp = MCP_Geometric(cost, sampling=tuple(ctx.sp))
        gd, _ = mcp.find_costs([tuple(int(v) for v in anchor_vox)])
    except Exception:
        return None, False
    return np.where(comp, gd, np.inf), True


def ostia_from_mask(mask, ctx, min_mm3=8.0, max_n=25, use_contact=True,
                    proximal_mm=10.0):
    """Shared post-step: components -> one labeled daughter each.

    Ostium: snapped to the aortic wall surface (_ostium_on_wall), not left
    wherever the branch's own near-wall voxels happened to be deepest.
    Seed/direction/radius: from a geodesic walk confined to this component
    only (_geodesic_trace), up to proximal_mm -- the seed is the traced
    point closest to 5mm out, matching the challenge's "along the daughter
    path" definition, rather than assuming a straight line.
    Falls back to the old axis-extension + Euclidean-window estimate if a
    component isn't near the wall or the geodesic solver can't run on it,
    so a malformed candidate still yields a prediction instead of vanishing.
    """
    lab, keep = components(mask, ctx, min_mm3)
    wall_dist = ndi.distance_transform_edt(~ctx.wall, sampling=tuple(ctx.sp))
    wall_pts = np.argwhere(ctx.wall)
    wall_pts_mm = wall_pts.astype(float) * ctx.sp

    ostia, dirs, radii, seeds = [], [], [], []
    for i in keep[:max_n]:
        comp = lab == i
        pts = np.argwhere(comp).astype(float)

        ost = _ostium_on_wall(comp, wall_dist, wall_pts_mm, ctx) \
            if use_contact else None
        if ost is None:
            pmm = pts * ctx.sp
            cen = pmm - pmm.mean(axis=0)
            step = max(len(cen) // 4000, 1)
            _, _, vv = np.linalg.svd(cen[::step], full_matrices=False)
            try:
                hit, _g, ost, _u = link_to_aorta(pts, ctx.dist_out, vv[0],
                                                 ctx.spacing, sub_mk=ctx.mk)
            except TypeError:      # older copy of the helper, no wall snapping
                hit, _g, ost, _u = link_to_aorta(pts, ctx.dist_out, vv[0],
                                                 ctx.spacing)
            if not hit or ost is None:
                continue
            ost = np.asarray(ost, float)

        # anchor = this component's OWN voxel nearest the chosen ostium --
        # where its lumen actually starts, used as the geodesic walk origin
        d_anchor = np.linalg.norm((pts - ost) * ctx.sp, axis=1)
        anchor = pts[int(np.argmin(d_anchor))]

        seed_vox, u, r = None, None, None
        gd, ok = _geodesic_trace(comp, anchor, ctx, max_mm=proximal_mm + 2.0)
        if ok:
            reach = np.isfinite(gd)
            if reach.sum() >= 5:
                idxs = np.argwhere(reach)
                dists = gd[reach]
                seed_i = int(np.argmin(np.abs(dists - 5.0)))
                seed_vox = idxs[seed_i].astype(float)
                win = idxs[np.abs(dists - 5.0) <= 2.5]
                if len(win) >= 3:
                    w_mm = win * ctx.sp
                    c = w_mm - w_mm.mean(axis=0)
                    _, _, vv = np.linalg.svd(c, full_matrices=False)
                    u = vv[0]
                    if np.dot(w_mm.mean(axis=0)
                             - np.asarray(ost, float) * ctx.sp, u) < 0:
                        u = -u
                    comp_edt = ndi.distance_transform_edt(comp, sampling=tuple(ctx.sp))
                    r = float(comp_edt[tuple(int(round(v)) for v in seed_vox)])

        if u is None:
            # geodesic trace failed -- old Euclidean-window fallback so this
            # candidate still produces a prediction rather than being dropped
            omm = ost * ctx.sp
            d = np.linalg.norm(pts * ctx.sp - omm, axis=1)
            sel = d <= 12.0
            if sel.sum() >= 5:
                pm = pts[sel] * ctx.sp
                c = pm - pm.mean(axis=0)
                _, _, vv = np.linalg.svd(c, full_matrices=False)
                u = vv[0]
                if np.dot(pm.mean(axis=0) - omm, u) < 0:
                    u = -u
                uu = u / max(np.linalg.norm(u), 1e-9)
                seed_vox = ost + uu * 5.0 / ctx.sp
                band = pts[sel][(d[sel] >= 3.5) & (d[sel] <= 6.5)]
                if len(band):
                    edt_full = ndi.distance_transform_edt(mask, sampling=tuple(ctx.sp))
                    r = float(ndi.map_coordinates(
                        edt_full, band.mean(axis=0)[:, None], order=1)[0])

        ostia.append(ost)
        seeds.append(seed_vox if seed_vox is not None else ost)
        dirs.append(u)
        radii.append(r)
    return ostia, dirs, radii, seeds


def drop_end_caps(res, ctx, margin_mm=6.0):
    """The flat cropped top and bottom of the mask are not branch origins."""
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return res
    zlo, zhi = zs[0] * ctx.sz, zs[-1] * ctx.sz
    keep = [k for k, o in enumerate(res.ostia_vox)
            if (o[0] * ctx.sz - zlo) >= margin_mm and (zhi - o[0] * ctx.sz) >= margin_mm]
    return res.select(keep)


def _proximal_path(path, ctx, mm):
    """The first `mm` of a traced centreline, in physical mm coordinates."""
    pts = np.asarray(path, float) * ctx.sp
    if len(pts) < 2:
        return pts
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts[cum <= mm]


def _same_daughter(res, a, b, ctx, min_sep_mm, seed_sep_mm, near_mm,
                   dir_tol, proximal_mm, overlap_frac):
    """Are detections a and b two views of one daughter artery?

    Four tests, cheapest first; any one of them means "already have this
    vessel". They are deliberately about the TRACED vessel, not the ostium,
    because the ostium is the one thing that cannot distinguish the two cases.
    """
    oa = np.asarray(res.ostia_vox[a], float) * ctx.sp
    ob = np.asarray(res.ostia_vox[b], float) * ctx.sp
    d_ost = float(np.linalg.norm(oa - ob))

    # 1. plain non-maximum suppression on the opening itself
    if d_ost < min_sep_mm:
        return True

    # 2. the two 5 mm seeds landed in the same lumen. Two walks that END UP
    #    together are the same vessel however far apart they started, and
    #    this is the test that catches the usual failure.
    sa, sb = res.seeds[a], res.seeds[b]
    d_seed = None
    if sa is not None and sb is not None:
        d_seed = float(np.linalg.norm(
            (np.asarray(sa, float) - np.asarray(sb, float)) * ctx.sp))
        if d_seed < seed_sep_mm:
            return True

    # 3. the PROXIMAL centrelines coincide: half of one runs within a calibre
    #    of the other. Proximal only, on purpose -- two genuinely different
    #    branches can drain into one distal network, and comparing the full
    #    paths would merge them wrongly.
    pa, pb = res.paths[a], res.paths[b]
    if pa is not None and pb is not None:
        A = _proximal_path(pa, ctx, proximal_mm)
        B = _proximal_path(pb, ctx, proximal_mm)
        if len(A) and len(B):
            tol = max(res.radii[a] or 1.0, res.radii[b] or 1.0, 1.0) + 1.0
            dm = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            if max(float((dm.min(axis=1) <= tol).mean()),
                   float((dm.min(axis=0) <= tol).mean())) >= overlap_frac:
                return True

    # 4. nearby openings pointing the same way whose traces CONVERGE. The
    #    convergence clause is what makes this safe: two genuinely different
    #    branches running parallel keep their separation, so their seeds stay
    #    about as far apart as their ostia, while two contact points on one
    #    vessel get closer together as both walks drop into the same lumen.
    #    Without it, a left and a right intercostal at one vertebral level
    #    (openings ~13 mm apart on opposite walls) would be at risk -- though
    #    those also point in opposite directions, which is why the dot
    #    product is signed and not an absolute value.
    da, db = res.dirs[a], res.dirs[b]
    if (d_ost < near_mm and da is not None and db is not None
            and float(np.dot(da, db)) >= dir_tol
            and d_seed is not None and d_seed < d_ost):
        return True
    return False


def prune_duplicate_branches(res, ctx, min_sep_mm=7.0, seed_sep_mm=5.0,
                             near_mm=14.0, dir_tol=0.60, proximal_mm=12.0,
                             overlap_frac=0.5, max_n=6):
    """Collapse detections that are really the same daughter, then cap the count.

    The wall scan suppresses non-maxima by OSTIUM DISTANCE only, and that is
    not enough. A real opening is a patch of wall several mm across, and a
    vessel that runs tangentially before turning away presents several
    surface peaks further apart than min_sep_mm that all trace into the same
    lumen. On subject001 that produced three separate "branches" whose ostia
    were 7-9 mm apart, whose radii all agreed to within 0.5 mm, whose
    directions agreed to dot = 0.86, and whose 5 mm seeds were CLOSER
    together than their ostia -- the signature of one vessel found three
    times. See _same_daughter() for the tests that catch it.

    Survivors are taken best-score-first, so whichever member of a merged
    group had the strongest ray evidence is the one that is reported, and the
    final list stays in descending confidence order.

    max_n is a hard upper bound applied AFTER merging, so it spends its budget
    on distinct vessels instead of on several copies of the loudest one.
    """
    n0 = len(res.ostia_vox)
    order = sorted(range(n0),
                   key=lambda k: -float(res.extras[k].get("score") or 0.0))
    keep = []
    for k in order:
        if any(_same_daughter(res, k, j, ctx, min_sep_mm, seed_sep_mm,
                              near_mm, dir_tol, proximal_mm, overlap_frac)
               for j in keep):
            continue
        keep.append(k)
        if max_n and len(keep) >= max_n:
            break
    res.select(keep)
    merged = n0 - len(keep)
    if merged:
        res.note = (res.note + f"; {merged} duplicate/surplus dropped"
                    ).lstrip("; ")
    return res


# -----------------------------------------------------------------------------
# wall scan: find the ostia ON the aortic surface, then trace outward
# -----------------------------------------------------------------------------

def lumen_floor_hu(mu):
    """HU at which soft_lumen() membership crosses 0.5. Its own function so
    the diagnostic report quotes the number the detector actually uses rather
    than a copy of the formula that can drift out of step with it."""
    return min(max(120.0, 0.5 * (mu + 50.0)), 250.0)


def soft_lumen(ct, mu, sd):
    """Fuzzy "this voxel is contrast-filled arterial lumen" map, in [0, 1].

    A hard HU band is the wrong tool for daughter arteries. A 2 mm intercostal
    on a 0.78 mm grid is three voxels across, so partial volume drags its
    centre HU well below the aortic lumen -- a band centred on the aorta's own
    median throws those vessels away, and widening the band far enough to keep
    them starts swallowing enhanced parenchyma.

    So the lower edge is placed on PHYSICS rather than on a multiple of the
    lumen spread: a voxel that is half lumen and half soft tissue lands at
    (mu + 50)/2, so that midpoint is exactly where membership should cross
    0.5. Above it a voxel is more lumen than tissue; below it, less. The upper
    edge exists to drop calcium, stents and bone, which read far brighter than
    any opacified vessel.

    The floor is then CAPPED at 250 HU. Even the robust median is only as good
    as the mask: three cases here report a lumen of 520-560 HU, which no blood
    reaches, because the segmentation includes a stent or dense calcium. The
    midpoint rule then puts the floor near 290 HU and quietly deletes every
    daughter dimmer than that -- and small arteries routinely sit at 200-300
    HU once partial volume has had its way. Capping only ever makes the
    detector more permissive, and it only binds when the estimate is already
    implausible (mu > 450), so well-behaved cases are untouched.
    """
    floor_hu = lumen_floor_hu(mu)
    soft = max(0.25 * (mu - floor_hu), 2.0 * sd, 10.0)
    ceil_hu = mu + max(150.0, 6.0 * sd)
    rise = np.clip((ct - floor_hu) / soft, 0.0, 1.0)
    fall = np.clip((ceil_hu + soft - ct) / soft, 0.0, 1.0)
    return (rise * fall).astype(np.float32)


def _neighbour_offsets(sp, radius_mm):
    """Integer voxel offsets whose centres lie within radius_mm PHYSICALLY.

    A fixed cube of offsets is the wrong neighbourhood on an anisotropic grid:
    +-1 voxel is 0.78 mm in plane and 0.80 mm through plane here, but 3 mm
    through plane on a coarser series, so the same cube means very different
    things in the two directions.
    """
    sp = np.asarray(sp, float)
    lim = np.ceil(radius_mm / sp).astype(int)
    zz, yy, xx = np.mgrid[-lim[0]:lim[0] + 1, -lim[1]:lim[1] + 1,
                          -lim[2]:lim[2] + 1]
    d2 = (zz * sp[0]) ** 2 + (yy * sp[1]) ** 2 + (xx * sp[2]) ** 2
    keep = d2 <= radius_mm ** 2 + 1e-9
    return np.stack([zz[keep], yy[keep], xx[keep]], axis=1).astype(int)


def _ridge_recentre(path, edt, sp, radius_mm=1.6):
    """Nudge a traced path onto the local EDT ridge, i.e. the lumen centre.

    Even with a medialness-weighted cost the minimal path clips the inside of
    every bend, so the radius sampled on it reads low and the tangent is
    slightly off. Each point is replaced by the EDT-weighted centroid of its
    physical neighbourhood, using only the upper half of the local EDT range
    so wall voxels get no vote. That is a smooth correction -- snapping to the
    single best voxel instead makes the path zigzag between equally good
    neighbours and ruins the tangent.

    path[0] is left alone: it is the ostium, which belongs ON the aortic wall
    and must not be pulled into the middle of the daughter.
    """
    off = _neighbour_offsets(sp, radius_mm)
    shp = np.array(edt.shape)
    out = np.array(path, float, copy=True)
    for k in range(1, len(path)):
        q = np.rint(path[k]).astype(int)[None, :] + off
        q = q[np.all((q >= 0) & (q < shp), axis=1)]
        if len(q) == 0:
            continue
        vals = edt[tuple(q.T)]
        w = np.maximum(vals - 0.5 * float(vals.max()), 0.0)
        if w.sum() > 1e-9:
            out[k] = (q * w[:, None]).sum(axis=0) / w.sum()
    return out


def _surface_and_normals(ctx, smooth_mm=1.2):
    """Aortic boundary layer + a smooth OUTWARD normal at every boundary voxel.

    The normal has to come from the signed distance field, not from the voxel
    geometry. A binary mask's own surface is a staircase, so its face normals
    only ever point along the six axis directions; a ray cast along one of
    those leaves at up to 45 degrees from the true surface normal and misses
    the branch. grad(sdf) is continuous and correct between voxels, and a mild
    blur -- specified in MM and converted to per-axis sigma, so it behaves the
    same on anisotropic grids -- removes the residual staircase without moving
    the sdf=0 isosurface.

    Returns (surf, normals, sdf): surf is the inner boundary layer (so every
    point is inside the mask), normals is (z,y,x,3) in mm-space, sdf is
    negative inside / positive outside, in mm.
    """
    dist_in = ndi.distance_transform_edt(ctx.mk, sampling=tuple(ctx.sp))
    sdf = (ctx.dist_out - dist_in).astype(np.float32)
    sig = [smooth_mm / float(s) for s in ctx.sp]
    grads = np.gradient(ndi.gaussian_filter(sdf, sigma=sig), *[float(s) for s in ctx.sp])
    nrm = np.stack(grads, axis=-1).astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    surf = ctx.mk & ~ndi.binary_erosion(ctx.mk, iterations=1)
    return surf, nrm, sdf


def _drop_cap_surface(surf, nrm, ctx, margin_mm=6.0, axial_dot=0.7):
    """Remove the flat truncated ENDS of the mask from the scanned surface.

    The supplied segmentation stops somewhere -- at the edge of the field of
    view, or wherever whoever drew it stopped. At that cut face the real aorta
    CONTINUES outside the mask, so a ray leaving the cap runs straight down
    the parent's own lumen: perfect contrast, and gaining distance from the
    mask the whole way, so it outscores every genuine branch.

    Rejecting every candidate within margin_mm of the mask's z extent (the old
    drop_end_caps rule, still used for --method m3) is too blunt. On a 32-slice
    segment that is a quarter of the available aorta, and it throws away real
    branches that happen to arise near the end -- on this dataset it was the
    reason two short-segment cases reported no daughters at all despite the
    ray scan finding them.

    The cap is specifically the part of that end whose normal points ALONG the
    vessel. Requiring BOTH conditions puts each error where it belongs: side
    wall near the end is kept, flat end face is not.
    """
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return surf
    zlo, zhi = zs[0] * ctx.sz, zs[-1] * ctx.sz
    z_mm = np.arange(ctx.mk.shape[0], dtype=float)[:, None, None] * ctx.sz
    near_end = ((z_mm - zlo) < margin_mm) | ((zhi - z_mm) < margin_mm)
    # nrm is a unit vector in mm-space, so |n_z| IS the cosine to the z axis
    axial = np.abs(nrm[..., 0]) >= axial_dot
    return surf & ~(near_end & axial)


# (angle from the normal in degrees, how many directions at that angle)
_CONE = ((0.0, 1), (28.0, 6), (50.0, 8), (70.0, 10))


def _ray_cone(n, cone=_CONE):
    """Unit directions (m, ND, 3) in a cone about each outward normal n (m, 3).

    A daughter artery does not leave along the wall normal. Renals are close
    to it, but the celiac trunk and SMA peel off the anterior aorta at 40-70
    degrees, so a single normal ray finds only some of the anatomy. The cone
    is built per point from an orthonormal tangent basis, which keeps the
    whole thing one vectorised expression over all surface points at once --
    a per-point loop over directions is far too slow at ~8000 surface voxels.
    """
    a = np.zeros_like(n)
    a[:, 0] = 1.0
    a[np.abs(n[:, 0]) > 0.9] = (0.0, 1.0, 0.0)
    t1 = np.cross(n, a)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-6)
    t2 = np.cross(n, t1)
    dirs = []
    for ang, k in cone:
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for j in range(k):
            # stagger successive rings so the directions do not line up in
            # meridians and leave unsampled wedges between them
            b = 2.0 * np.pi * j / k + 0.5 * np.pi * ang / 90.0
            dirs.append(ca * n + sa * (np.cos(b) * t1 + np.sin(b) * t2))
    return np.stack(dirs, axis=1).astype(np.float32)


def scan_wall_for_branches(ctx, L, lumen_floor=0.35, step_mm=0.8, max_mm=16.0,
                           depart=0.28, min_len_mm=5.0, min_reach_mm=3.5,
                           exit_mm=2.0, block=1500):
    """Score every aortic surface voxel by "does a vessel leave from here".

    For each surface voxel and each direction in its cone, walk outward in
    step_mm increments and keep the longest UNBROKEN run in which the ray is

      * inside the fuzzy lumen           (L >= lumen_floor)
      * not in bone                      (the vertebra abuts the aorta, and
                                          trabecular struts read as lumen)
      * actually getting away from the aorta
                                         (dist_out >= depart * t - 0.8)
      * out of the parent aorta by exit_mm
                                         (the ray starts ON the wall, so it
                                          should leave almost at once)

    The DEPARTURE condition is the whole point, and the thing the old pipeline
    lacked. The dominant false positive around an aorta is the one-voxel
    partial-volume rind on its own wall: bright, thin, tube-shaped, elongated
    for tens of mm, and therefore indistinguishable from a branch by
    intensity or by PCA shape. It is trivially distinguishable by geometry --
    it never gets more than about a voxel away from the wall. `depart` is set
    just under cos(70 deg) = 0.34 so that even the most oblique ray in the
    cone is still required to make real outward progress, and the -0.8 mm
    slack lets a ray cross the junction (where it is still inside the mask,
    dist_out = 0) without being killed on its first step.

    The EXIT condition looks pedantic and is not. Rays start at a boundary
    voxel centre, which is INSIDE the mask, where the aorta's own lumen scores
    a perfect 1.0 -- so an oblique ray is free to cut a chord through the
    parent vessel and come out somewhere else on the wall. It then finds a
    real daughter over there and reports its ostium back at the entry point,
    several mm from the opening it actually found. Forcing the ray out of the
    mask within exit_mm (about one voxel of travel even at the most oblique
    angle in the cone) keeps each ray's evidence local to its own foot point.

    Returns (P, score, dirs, length, reach, sdf, nrm); score is 0 wherever no
    direction produced an acceptable run.
    """
    surf, nrm, sdf = _surface_and_normals(ctx)
    surf = _drop_cap_surface(surf, nrm, ctx)
    P = np.argwhere(surf).astype(np.float32)
    if len(P) == 0:
        z = np.zeros(0, np.float32)
        return P, z, np.zeros((0, 3), np.float32), z, z, sdf, nrm

    n = nrm[surf]                                       # (M, 3), mm-space
    ts = (np.arange(1, int(np.ceil(max_mm / step_mm)) + 1, dtype=np.float32)
          * step_mm)
    K = len(ts)
    bone_f = ctx.bone.astype(np.float32)
    mk_f = ctx.mk.astype(np.float32)
    M = len(P)
    score = np.zeros(M, np.float32)
    bdir = np.zeros((M, 3), np.float32)
    blen = np.zeros(M, np.float32)
    breach = np.zeros(M, np.float32)

    # blocked so the sample buffer stays a few tens of MB no matter how big
    # the mask surface is
    for s0 in range(0, M, block):
        s1 = min(s0 + block, M)
        D = _ray_cone(n[s0:s1])                         # (m, ND, 3)
        m, ND = D.shape[0], D.shape[1]
        # voxel-space step per mm is D / sp, so sample at P + t * D / sp
        c = (P[s0:s1, None, None, :]
             + ts[None, None, :, None] * (D / ctx.sp)[:, :, None, :])
        c = np.ascontiguousarray(c.reshape(-1, 3).T)
        Lv = ndi.map_coordinates(L, c, order=1, mode="constant",
                                 cval=0.0).reshape(m, ND, K)
        Dv = ndi.map_coordinates(ctx.dist_out, c, order=1, mode="constant",
                                 cval=0.0).reshape(m, ND, K)
        # cval=1 -> off-volume counts as blocked, so rays cannot score by
        # running out of the field of view
        Bv = ndi.map_coordinates(bone_f, c, order=1, mode="constant",
                                 cval=1.0).reshape(m, ND, K)
        Iv = ndi.map_coordinates(mk_f, c, order=1, mode="constant",
                                 cval=0.0).reshape(m, ND, K)
        del c
        alive = ((Lv >= lumen_floor) & (Bv < 0.5)
                 & (Dv >= depart * ts[None, None, :] - 0.8)
                 & ((Iv < 0.5) | (ts[None, None, :] <= exit_mm)))
        run = np.cumprod(alive, axis=2).astype(np.float32)
        nrun = run.sum(axis=2)
        length = nrun * step_mm
        reach = (Dv * run).max(axis=2)
        meanL = (Lv * run).sum(axis=2) / np.maximum(nrun, 1.0)
        ok = (length >= min_len_mm) & (reach >= min_reach_mm)
        sc = np.where(ok, length + 2.0 * reach + 4.0 * meanL, 0.0)
        j = sc.argmax(axis=1)
        r = np.arange(m)
        score[s0:s1] = sc[r, j]
        bdir[s0:s1] = D[r, j]
        blen[s0:s1] = length[r, j]
        breach[s0:s1] = reach[r, j]
    return P, score, bdir, blen, breach, sdf, nrm


def _nms_surface(P, score, sp, min_sep_mm=7.0, max_n=60):
    """Greedy non-maximum suppression over the scored surface points.

    A real ostium lights up a patch of wall, not one voxel, so without this
    the top of the ranking is one opening repeated dozens of times.
    """
    order = np.argsort(score)[::-1]
    order = order[score[order] > 0]
    picked, pts = [], []
    for i in order:
        p = P[i] * sp
        if any(np.linalg.norm(p - q) < min_sep_mm for q in pts):
            continue
        picked.append(int(i))
        pts.append(p)
        if len(picked) >= max_n:
            break
    return picked


def _refine_ostium(P, score, dirs, idx, ctx, sdf, nrm, patch_mm=2.5):
    """Sub-voxel ostium: score-weighted patch centroid, projected onto sdf=0.

    Two steps, each fixing a different error. First, the argmax surface voxel
    is one sample of a noisy score field, so average the agreeing neighbours
    (same opening, similar outward direction) weighted by score -- that is the
    centre of the opening rather than its brightest voxel. Second, the surface
    voxel centre is not the surface: it is the last lattice point INSIDE the
    mask, so it sits up to half a voxel proud of the true boundary. A few
    Newton steps along the normal put the point on the sdf=0 isosurface, which
    is the actual aortic wall. Stepping along the normal rather than along the
    ray makes the step exact, since |grad(sdf)| = 1 there.
    """
    p0 = P[idx].astype(float)
    u0 = dirs[idx].astype(float)
    d = np.linalg.norm((P - P[idx]) * ctx.sp, axis=1)
    agree = dirs @ dirs[idx]
    sel = np.where((d <= patch_mm) & (score > 0) & (agree >= 0.5))[0]
    if len(sel) >= 3:
        w = score[sel].astype(float)
        ost = (P[sel].astype(float) * w[:, None]).sum(axis=0) / w.sum()
        u = (dirs[sel].astype(float) * w[:, None]).sum(axis=0)
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-9 else u0
    else:
        ost, u = p0, u0

    hi = np.array(sdf.shape, float) - 1.001
    for _ in range(8):
        ost = np.clip(ost, 0.0, hi)
        s = float(ndi.map_coordinates(sdf, ost[:, None], order=1)[0])
        if abs(s) < 0.05:
            break
        nv = np.array([float(ndi.map_coordinates(nrm[..., k], ost[:, None],
                                                 order=1)[0]) for k in range(3)])
        nn = np.linalg.norm(nv)
        if nn < 1e-6:
            break
        ost = ost - (s * (nv / nn)) / ctx.sp
    return np.clip(ost, 0.0, hi), u, (nrm[tuple(np.rint(np.clip(
        p0, 0, hi)).astype(int))]).astype(float)


def trace_branch(ctx, ost_vox, u0, L, lumen_floor=0.35, reach_mm=26.0,
                 proximal_mm=5.0, max_path_mm=30.0):
    """Follow the daughter's OWN lumen out from the ostium, geodesically.

    The straight ray that found the ostium is a detector, not a centreline --
    proximal branches curve, so the point 5 mm along the ray is not the point
    5 mm along the vessel. This walks the real thing: a minimal-cost path
    through the fuzzy lumen outside the aorta, seeded just past the wall and
    traced back from the tip. The tip is chosen by `dist_out - 0.12 * cost`,
    i.e. the point that gets furthest from the aorta for the least travel,
    which follows the branch instead of wandering into whatever the lumen band
    happens to be connected to.

    The cost is MEDIALNESS-WEIGHTED, not unit. A unit cost makes the solver
    find the shortest way through the vessel, which clips every corner and
    runs along the lumen wall -- the giveaway was that almost every reported
    radius came back as exactly one voxel, because the sampled point was
    always adjacent to background. Dividing by (distance-to-wall x brightness)
    makes the cheap route the middle of the vessel, so the path is a
    centreline and the radius measured on it is the real calibre. Arc length
    is then measured geometrically from the returned points, since the cost
    field is no longer in millimetres.

    Everything runs in a LOCAL box around the ostium. A whole-volume geodesic
    would be slow and, worse, would let one leak on the far side of the aorta
    influence every candidate.

    Returns a dict, or None if this candidate cannot be traced.
    """
    from skimage.graph import MCP_Geometric

    shape = np.array(ctx.ct.shape)
    half = np.ceil((reach_mm + 8.0) / ctx.sp).astype(int)
    c = np.rint(ost_vox).astype(int)
    lo = np.maximum(c - half, 0)
    hi = np.minimum(c + half + 1, shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    Lb, db = L[sl], ctx.dist_out[sl]
    grow = (Lb >= lumen_floor) & ~ctx.mk[sl] & ~ctx.bone[sl] & (db <= reach_mm)
    if grow.sum() < 8:
        return None

    o_loc = np.asarray(ost_vox, float) - lo
    start = o_loc + (np.asarray(u0, float) / ctx.sp) * 1.4
    gp = np.argwhere(grow).astype(float)
    dd = np.linalg.norm((gp - start) * ctx.sp, axis=1)
    if not len(dd) or dd.min() > 3.0:
        return None
    s_vox = tuple(int(v) for v in gp[int(np.argmin(dd))])

    edt = ndi.distance_transform_edt(grow, sampling=tuple(ctx.sp))
    med = np.clip(edt / 1.6, 0.0, 1.0) * np.clip(Lb, 0.0, 1.0)
    cost = np.where(grow, 1.0 / (0.18 + 0.82 * med), np.inf)   # in [1.0, 5.6]
    try:
        mcp = MCP_Geometric(cost, sampling=tuple(ctx.sp))
        gd, _ = mcp.find_costs([s_vox])
    except Exception:
        return None
    gd = np.where(grow, gd, np.inf)
    # cost >= 1 everywhere, so a budget of 4x the path limit is generous but
    # still bounds how far the tip search can look
    ok = np.isfinite(gd) & (gd <= 4.0 * max_path_mm)
    if ok.sum() < 6:
        return None

    obj = np.where(ok, db - 0.12 * gd, -np.inf)
    tip = np.unravel_index(int(np.argmax(obj)), obj.shape)
    try:
        path = np.asarray(mcp.traceback(tip), float)
    except Exception:
        return None
    if len(path) < 2:
        return None

    # prepend the refined ostium so arc length is measured from the wall
    path = np.vstack([o_loc[None, :], path])
    path = _ridge_recentre(path, edt, ctx.sp)
    pmm = path * ctx.sp
    seg = np.linalg.norm(np.diff(pmm, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] < 1e-6:
        return None
    if cum[-1] > max_path_mm:                     # truncate to the proximal bit
        keep = int(np.searchsorted(cum, max_path_mm)) + 1
        path, pmm, cum = path[:keep], pmm[:keep], cum[:keep]
        seg = seg[:keep - 1]

    j = int(np.argmin(np.abs(cum - proximal_mm)))
    seed = path[j]

    # direction at the ostium: fit the proximal segment, not the whole path
    w = path[cum <= max(6.0, float(cum[j]))]
    if len(w) >= 3:
        wmm = w * ctx.sp
        cc = wmm - wmm.mean(axis=0)
        _, _, vv = np.linalg.svd(cc, full_matrices=False)
        u = vv[0]
        if np.dot(u, wmm[-1] - wmm[0]) < 0:
            u = -u
    else:
        u = np.asarray(u0, float)
    u = u / max(np.linalg.norm(u), 1e-9)

    r = float(ndi.map_coordinates(edt, seed[:, None], order=1)[0])
    # how far from the aorta the 5 mm seed actually sits -- the cheapest and
    # sharpest check that this is a branch leaving the wall rather than a path
    # sliding ALONG it (see the min_seed_dist_mm gate in wall_scan)
    seed_dist = float(ndi.map_coordinates(db, seed[:, None], order=1)[0])
    # Clearance at the END OF THE REPORTED PATH, not at the tip the geodesic
    # originally found. Those differ whenever the path was truncated above,
    # and reading db[tip] there described a point that is no longer part of
    # the branch being returned -- which made the min_end_dist_mm gate test
    # one place and the tip_clear_mm diagnostic report another.
    end_dist = float(ndi.map_coordinates(db, path[-1][:, None], order=1)[0])

    # corridor for rendering: the path, thickened to the local calibre but
    # clipped back to lumen so it cannot bleed into surrounding tissue
    pi = np.rint(path).astype(int)
    pi = pi[np.all((pi >= 0) & (pi < np.array(grow.shape)), axis=1)]
    cor = np.zeros_like(grow)
    if len(pi):
        cor[tuple(pi.T)] = True
        it = max(1, int(round(max(r, 0.8) / float(min(ctx.sp)))))
        cor = ndi.binary_dilation(cor, iterations=it) & grow
        cor[tuple(pi.T)] = True

    return dict(seed=seed + lo, u=u, radius=r, path=path + lo,
                length_mm=float(cum[-1]), end_dist_mm=end_dist,
                seed_dist_mm=seed_dist, slice=sl, corridor=cor)


def wall_scan(ctx, max_n=25, lumen_floor=0.35, min_sep_mm=7.0,
              max_ray_mm=16.0, min_len_mm=5.0, min_reach_mm=3.5,
              min_path_mm=6.0, min_end_dist_mm=4.0, min_seed_dist_mm=1.5,
              radius_mm=(0.4, 9.0), min_outward=0.10):
    """Primary detector: ostia on the wall, then one traced daughter each."""
    t0 = time.time()
    L = soft_lumen(ctx.ct, ctx.mu, ctx.sd)
    P, score, dirs, _len, _reach, sdf, nrm = scan_wall_for_branches(
        ctx, L, lumen_floor=lumen_floor, max_mm=max_ray_mm,
        min_len_mm=min_len_mm, min_reach_mm=min_reach_mm)
    empty = np.zeros_like(ctx.mk)
    if not len(P) or not (score > 0).any():
        return Result("Wall scan", empty, [], runtime=time.time() - t0,
                      note="no wall candidate survived the ray test")

    n_peaks = int((score > 0).sum())
    picked = _nms_surface(P, score, ctx.sp, min_sep_mm, max_n=max_n * 3)

    items, rejected = [], 0
    for i in picked:
        ost, u0, nvec = _refine_ostium(P, score, dirs, i, ctx, sdf, nrm)
        tr = trace_branch(ctx, ost, u0, L, lumen_floor=lumen_floor)
        if tr is None:
            rejected += 1
            continue
        # final gates on the TRACED vessel, which knows things the ray did not.
        # min_seed_dist_mm is the important one: it says the point 5 mm along
        # the daughter must itself be clear of the aorta. A path that arises
        # and then slides caudally along the aortic wall satisfies every
        # intensity and shape test but fails this outright, and that was the
        # last family of false positives left after the ray departure test.
        if (tr["length_mm"] < min_path_mm
                or tr["end_dist_mm"] < min_end_dist_mm
                or tr["seed_dist_mm"] < min_seed_dist_mm
                or not (radius_mm[0] <= tr["radius"] <= radius_mm[1])
                or float(np.dot(tr["u"], nvec)) < min_outward):
            rejected += 1
            continue
        items.append((float(score[i]), ost, tr))

    # dedupe on the REFINED ostia (the refinement moves points, so two peaks
    # can converge on one opening after it), keeping the better score
    items.sort(key=lambda t: t[0], reverse=True)
    kept = []
    for sc, ost, tr in items:
        p = np.asarray(ost, float) * ctx.sp
        if any(np.linalg.norm(p - np.asarray(q, float) * ctx.sp) < min_sep_mm
               for _, q, _ in kept):
            continue
        kept.append((sc, ost, tr))
        if len(kept) >= max_n:
            break

    mask = empty
    for _sc, _ost, tr in kept:
        mask[tr["slice"]] |= tr["corridor"]
    return Result(
        "Wall scan", mask, [o for _, o, _ in kept],
        dirs=[t["u"] for _, _, t in kept],
        radii=[t["radius"] for _, _, t in kept],
        seeds=[t["seed"] for _, _, t in kept],
        paths=[t["path"] for _, _, t in kept],
        extras=[dict(score=round(sc, 2),
                     path_mm=round(t["length_mm"], 2),
                     seed_clear_mm=round(t["seed_dist_mm"], 2),
                     tip_clear_mm=round(t["end_dist_mm"], 2))
                for sc, _, t in kept],
        runtime=time.time() - t0,
        note=f"{n_peaks} scored wall voxels -> {len(picked)} peaks -> "
             f"{len(kept)} traced ({rejected} rejected)")


def merge_results(primary, fallback, ctx, merge_mm=6.0):
    """Keep every primary detection, then add non-overlapping fallback ones."""
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    ostia = list(primary.ostia_vox)
    dirs, radii, seeds = (list(primary.dirs), list(primary.radii),
                          list(primary.seeds))
    paths, extras = list(primary.paths), list(primary.extras)
    for o, d, r, s, pth, x in zip(fallback.ostia_vox, fallback.dirs,
                                  fallback.radii, fallback.seeds,
                                  fallback.paths, fallback.extras):
        p = np.asarray(o, float) * ctx.sp
        if any(np.linalg.norm(p - np.asarray(q, float) * ctx.sp) < merge_mm
               for q in ostia):
            continue
        ostia.append(o); dirs.append(d); radii.append(r); seeds.append(s)
        paths.append(pth)
        extras.append(dict(x, source=fallback.name))
    n_add = len(ostia) - len(primary.ostia_vox)
    return Result(f"{primary.name} + fallback", primary.mask | fallback.mask,
                  ostia, dirs=dirs, radii=radii, seeds=seeds, paths=paths,
                  extras=extras,
                  runtime=primary.runtime + fallback.runtime,
                  note=f"{len(primary.ostia_vox)} {primary.name.lower()} + "
                       f"{n_add} {fallback.name.lower()}")


# -----------------------------------------------------------------------------
# M3  Tahoces-style two-phase expansion with leak detection
# -----------------------------------------------------------------------------

def m3_expansion(ctx, phase1_mm=20.0, phase2_mm=30.0, leak_mL=8.0):
    t0 = time.time()
    band = (ctx.ct >= ctx.lo) & (ctx.ct <= ctx.hi) & ~ctx.bone

    def grow(limit_mm, blocked):
        allowed = band & (ctx.dist_out <= limit_mm) & ~blocked
        lab, _ = ndi.label(allowed | ctx.mk)
        touch = set(int(v) for v in np.unique(lab[ctx.mk]) if v > 0)
        return np.isin(lab, list(touch)) & ~ctx.mk

    # phase 1: grow to 20 mm, flag components that blow up (organ leakage)
    g1 = grow(phase1_mm, np.zeros_like(band))
    lab1, n1 = ndi.label(g1)
    blocked = np.zeros_like(band)
    n_leak = 0
    if n1:
        sizes = np.bincount(lab1.ravel()) * ctx.vox_mm3 / 1000.0
        leaks = [i for i in range(1, n1 + 1) if sizes[i] > leak_mL]
        n_leak = len(leaks)
        if leaks:
            blocked = np.isin(lab1, leaks)

    # phase 2: grow further, avoiding the flagged regions -- and re-check, since
    # a component can stay small at 20 mm and then blow into an organ at 30 mm.
    g2 = grow(phase2_mm, blocked)
    lab2, n2 = ndi.label(g2)
    if n2:
        sizes2 = np.bincount(lab2.ravel()) * ctx.vox_mm3 / 1000.0
        leaks2 = [i for i in range(1, n2 + 1) if sizes2[i] > leak_mL]
        if leaks2:
            n_leak += len(leaks2)
            g2 = g2 & ~np.isin(lab2, leaks2)
    o, d, r, s = ostia_from_mask(g2, ctx)
    return Result("M3 expansion", g2, o, dirs=d, radii=r, seeds=s,
                  runtime=time.time() - t0, note=f"{n_leak} leak(s) blocked")


def build_prediction(res, ctx, case_id):
    """Label every detected daughter and build the required prediction dict.

    Each entry in res.ostia_vox becomes one branch_NNN instance (sequential,
    1-indexed) with parent_instance_id "aorta" -- this is the one place that
    assigns instance IDs, so save_method() and the top-level --output file
    both call it instead of building the dict twice and risking the two
    falling out of sync.

    seed_xyz_mm comes from res.seeds -- the actual traced point closest to
    5mm along the branch's own geodesic path (see _geodesic_trace) -- rather
    than assuming a straight line from the ostium along `direction_xyz`,
    which only matches reality when the vessel doesn't curve.
    """
    daughters = []
    for k, (o, dr, rad, sd) in enumerate(
            zip(res.ostia_vox, res.dirs, res.radii, res.seeds), 1):
        p = ctx.to_mm(o)
        v = ([float(dr[2]), float(dr[1]), float(dr[0])] if dr is not None
             else [0.0, 0.0, 1.0])
        v = list(np.round(np.array(v) / max(np.linalg.norm(v), 1e-9), 4))
        seed_mm = ctx.to_mm(sd) if sd is not None else (p + np.array(v) * 5.0)
        daughters.append({
            "instance_id": f"branch_{k:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(float(q), 3) for q in p],
            "seed_xyz_mm": [round(float(q), 3) for q in seed_mm],
            "radius_mm": round(float(rad), 3) if rad else None,
            "direction_xyz": v,
        })
    return {"case_id": case_id, "parent": {"instance_id": "aorta"},
            "daughters": daughters}


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------
# per-method output: one folder each
# -----------------------------------------------------------------------------

def save_method(res, ctx, outdir, case_id):
    """Everything for one method lands in outdir/<method>/ ."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = os.path.join(outdir, res.name.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)

    # ---- ostia.csv (seed point + detector diagnostics, for spot-checking) --
    diag_cols = ["score", "path_mm", "seed_clear_mm", "tip_clear_mm", "source"]
    with open(os.path.join(d, "ostia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "seed_x_mm", "seed_y_mm", "seed_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"] + diag_cols)
        for k, (o, dr, rad, sd, ex) in enumerate(
                zip(res.ostia_vox, res.dirs, res.radii, res.seeds,
                    res.extras), 1):
            p = ctx.to_mm(o)
            sp_mm = ctx.to_mm(sd) if sd is not None else p
            dd = ([round(float(dr[2]), 4), round(float(dr[1]), 4),
                   round(float(dr[0]), 4)] if dr is not None else ["", "", ""])
            w.writerow([f"branch_{k:03d}", round(p[0], 2), round(p[1], 2),
                        round(p[2], 2), round(sp_mm[0], 2), round(sp_mm[1], 2),
                        round(sp_mm[2], 2), *dd, round(rad, 3) if rad else ""]
                       + [ex.get(c, "") for c in diag_cols])

    # ---- prediction.json (per-method copy, same labeling as --output) ----
    pred = build_prediction(res, ctx, case_id)
    with open(os.path.join(d, "prediction.json"), "w") as fh:
        json.dump(pred, fh, indent=2)

    # ---- render.html : standalone, opens in its own tab ----
    meshes = []
    av, af = fine_surface(ctx.mk, ctx.spacing, cap=120_000)
    if av is not None:
        meshes.append(dict(verts=ctx.verts_to_world(av), faces=af,
                           name="aorta", color="#d94a4a", opacity=0.30,
                           text="parent aorta"))
    bv, bf = fine_surface(res.mask & ~ctx.mk, ctx.spacing, cap=90_000)
    if bv is not None:
        meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                           name="detected branches", color="#f2c14e",
                           opacity=0.95, text=f"{res.name} branch voxels"))
    if res.ostia_vox:
        P = np.array([ctx.to_mm(o) for o in res.ostia_vox])
        meshes.append(dict(kind="points", points=P, name="ostia",
                           color="#111111", size=8,
                           labels=[f"branch_{k:03d}" for k in
                                   range(1, len(res.ostia_vox) + 1)]))
    write_interactive_html(
        os.path.join(d, "render.html"), meshes,
        title=f"{case_id} — {res.name}",
        subtitle=f"{len(res.ostia_vox)} ostia · {res.runtime:.1f}s · {res.note}")

    # ---- overview.png : 3D, the busiest axial slice, ostia on the unrolled wall
    fig = plt.figure(figsize=(16, 5.2), facecolor="white")

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    allv = []
    for binary, col, alpha in ((ctx.mk, "#d94a4a", 0.22),
                               (res.mask & ~ctx.mk, "#f2c14e", 0.95)):
        v, f = fine_surface(binary, ctx.spacing, cap=18_000)
        if v is None:
            continue
        c = Poly3DCollection(v[f], alpha=alpha)
        c.set_facecolor(col)
        c.set_edgecolor("none")
        ax.add_collection3d(c)
        allv.append(v)
    if res.ostia_vox:
        P = np.array([np.asarray(o, float) * ctx.sp for o in res.ostia_vox])
        ax.scatter(P[:, 2], P[:, 1], P[:, 0], s=55, c="#111111", marker="D",
                   depthshade=False)
    if allv:
        st = np.vstack(allv)
        ax.set_xlim(st[:, 0].min(), st[:, 0].max())
        ax.set_ylim(st[:, 1].min(), st[:, 1].max())
        ax.set_zlim(st[:, 2].min(), st[:, 2].max())
        try:
            ax.set_box_aspect([np.ptp(st[:, i]) for i in range(3)])
        except Exception:
            pass
    ax.view_init(elev=12, azim=-75)
    ax.set_title("branches (gold) on the aorta", fontsize=10)
    ax.set_axis_off()

    # Show the axial slice carrying the most ostia. The window is +-3 MM, not
    # +-1 slice, so the panel does not silently get less useful on a thick
    # series; the 1/rank weight breaks ties toward the better-scored branch
    # instead of landing on whichever weak detection came first.
    half = max(int(round(3.0 / ctx.sz)), 1)
    zc = np.zeros(ctx.ct.shape[0])
    for rank, o in enumerate(res.ostia_vox):
        z = int(round(o[0]))
        if 0 <= z < len(zc):
            zc[max(z - half, 0):z + half + 1] += 1.0 + 1.0 / (rank + 1)
    z_show = int(np.argmax(zc)) if zc.max() else ctx.ct.shape[0] // 2
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(window(ctx.ct[z_show]), cmap="gray", vmin=0, vmax=1,
              aspect=ctx.sy / ctx.sx)
    ov = np.zeros(ctx.ct[z_show].shape + (4,), np.float32)
    ov[res.mask[z_show] & ~ctx.mk[z_show]] = (1.0, 0.78, 0.25, 0.75)
    ov[ctx.mk[z_show]] = (0.85, 0.29, 0.29, 0.45)
    ax.imshow(ov, aspect=ctx.sy / ctx.sx, interpolation="nearest")
    for o in res.ostia_vox:
        if abs(o[0] - z_show) <= 2:
            ax.plot(o[2], o[1], "D", ms=8, mfc="#00e5ff", mec="white", mew=1)
    ax.set_title(f"axial slice {z_show}", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    if res.ostia_vox:
        A, Z = [], []
        for o in res.ostia_vox:
            z = int(round(o[0]))
            if z not in ctx.cl:
                z = min(ctx.cl, key=lambda q: abs(q - z))
            cy, cx = ctx.cl[z]
            A.append(np.degrees(np.arctan2((o[1] - cy) * ctx.sy,
                                           (o[2] - cx) * ctx.sx)))
            Z.append(ctx.to_mm(o)[2])
        ax.scatter(A, Z, s=110, c="#f2c14e", edgecolors="#111", zorder=3)
        for a, z, k in zip(A, Z, range(1, len(A) + 1)):
            ax.annotate(f"{k:02d}", (a, z), fontsize=8,
                        xytext=(6, 4), textcoords="offset points")
    ax.set_xlim(-185, 185)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("angle around the aorta (deg)")
    ax.set_ylabel("z (mm)")
    ax.grid(alpha=0.3)
    ax.set_title("ostia on the unrolled wall", fontsize=10)

    fig.suptitle(f"{case_id} — {res.name}   ·   {len(res.ostia_vox)} ostia   ·   "
                 f"{res.runtime:.1f}s   ·   {res.note}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "overview.png"), dpi=130, facecolor="white")
    plt.close(fig)
    return d


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--outdir", default="bench")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--roi-mm", type=float, default=30.0)
    ap.add_argument("--lumen-hu", default=None,
                    help="override the automatic lumen model, e.g. 300,25 "
                         "(centre,spread in HU). Use when the report warns that "
                         "the mask contains dense outliers.")
    ap.add_argument("--output", default="prediction.json",
                    help="path to write the labeled prediction JSON "
                         "(case_id / parent / daughters, matching the "
                         "required submission format)")
    ap.add_argument("--min-elong", type=float, default=1.8,
                    help="shape gate: min length/width ratio for a candidate "
                         "component to be kept as vessel-like (higher = "
                         "stricter, fewer blobby false positives; try 2.5-3 "
                         "if you're still seeing compact false branches)")
    ap.add_argument("--min-branch-len-mm", type=float, default=3.0,
                    help="shape gate: min physical length (mm) a candidate "
                         "component must span along its own principal axis "
                         "(--method m3 only)")
    ap.add_argument("--method", choices=("wall", "m3", "both"), default="wall",
                    help="wall = ostia found on the aortic surface by ray "
                         "casting (default); m3 = the older region-growing "
                         "method; both = wall plus any m3 detection the wall "
                         "scan missed")
    ap.add_argument("--max-branches", type=int, default=6,
                    help="hard upper bound on reported daughters, best score "
                         "first. Applied AFTER duplicate merging, so the "
                         "budget goes on distinct vessels rather than on "
                         "several copies of the loudest one")
    ap.add_argument("--dup-seed-mm", type=float, default=5.0,
                    help="merge two detections whose traced 5 mm seeds land "
                         "within this distance: two walks that ended up in "
                         "the same lumen are the same vessel no matter how "
                         "far apart their wall contact points were")
    ap.add_argument("--dup-near-mm", type=float, default=14.0,
                    help="range over which two openings pointing the same "
                         "way are merged, provided their traces also "
                         "CONVERGE (seeds closer together than ostia). Raise "
                         "to merge harder; lower if distinct neighbouring "
                         "branches are being collapsed")
    ap.add_argument("--lumen-floor", type=float, default=0.35,
                    help="wall scan: fuzzy-lumen membership a ray must stay "
                         "above (0-1). Lower it to ~0.25 to chase faint or "
                         "very small vessels, raise it to ~0.5 to be strict")
    ap.add_argument("--min-reach-mm", type=float, default=3.5,
                    help="wall scan: how far from the aorta a ray must still "
                         "be at the end of its run. This is the gate that "
                         "rejects the partial-volume rind on the aortic wall; "
                         "below ~2 mm that rind starts coming back")
    ap.add_argument("--min-sep-mm", type=float, default=7.0,
                    help="wall scan: minimum spacing between two ostia")
    ap.add_argument("--min-seed-dist-mm", type=float, default=1.5,
                    help="wall scan: how far clear of the aorta the traced "
                         "5 mm seed must sit. Note the geometry: a daughter "
                         "leaving at angle A from the wall normal only gains "
                         "5*cos(A) mm of clearance by 5 mm, so 2.0 already "
                         "rejects takeoffs beyond ~66 deg (celiac, SMA). "
                         "Raise it only if paths sliding along the wall "
                         "reappear")
    args = ap.parse_args()

    case_id = args.case_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.image))) or "case"
    os.makedirs(args.outdir, exist_ok=True)

    print(f"  branch_bench {__version__}")
    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    ctx = Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm,
              min_elong=args.min_elong, min_length_mm=args.min_branch_len_mm)
    if args.lumen_hu:
        c, sp_ = (float(q) for q in args.lumen_hu.split(","))
        ctx.mu, ctx.sd = c, sp_
        ctx.lo, ctx.hi = c - 2.5 * sp_, c + 2.5 * sp_
        ctx.roi, ctx.dist_out, ctx.bone, ctx.hi_cut = branch_roi(
            ctx.ct, ctx.mk, ctx.spacing, args.roi_mm, c, sp_)
        ctx.flag = f"lumen model overridden to {c:.0f} +/- {sp_:.0f} HU"
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    diag = [
        f"lumen (robust)   : {ctx.mu:.0f} HU, spread {ctx.sd:.0f}",
        f"band             : {ctx.lo:.0f} - {ctx.hi:.0f} HU",
        f"bone cut         : > {ctx.hi_cut:.0f} HU and > 2 mL "
        f"({ctx.bone.sum() * ctx.vox_mm3 / 1000:.0f} mL excluded)",
        f"search region    : {ctx.roi.sum() * ctx.vox_mm3 / 1000:.0f} mL",
        f"aorta mask       : {ctx.mk.sum() * ctx.vox_mm3 / 1000:.1f} mL, "
        f"{len(zs)} slices, {len(zs) * ctx.sz:.0f} mm long",
        f"crop             : {ctx.ct.shape}",
        f"method           : {args.method}",
    ]
    if args.method in ("wall", "both"):
        fl = lumen_floor_hu(ctx.mu)
        diag.append(f"wall scan        : lumen floor {args.lumen_floor:.2f} "
                    f"(~{fl:.0f} HU), min reach {args.min_reach_mm:.1f} mm, "
                    f"min seed clearance {args.min_seed_dist_mm:.1f} mm, "
                    f"min separation {args.min_sep_mm:.1f} mm")
    if args.method in ("m3", "both"):
        diag.append(f"shape gate       : length >= {ctx.min_length_mm:.1f} mm, "
                    f"elongation >= {ctx.min_elong:.1f} "
                    f"(--min-elong / --min-branch-len-mm)")
    if ctx.flag:
        diag.append("WARNING          : " + ctx.flag)
    for line in diag:
        print("  " + line)
    print()
    with open(os.path.join(args.outdir, "case_report.txt"), "w") as fh:
        fh.write(f"{case_id}\n" + "\n".join(diag) + "\n")

    t0 = time.time()
    primary = None
    if args.method in ("wall", "both"):
        primary = wall_scan(ctx, max_n=args.max_branches,
                            lumen_floor=args.lumen_floor,
                            min_sep_mm=args.min_sep_mm,
                            min_reach_mm=args.min_reach_mm,
                            min_seed_dist_mm=args.min_seed_dist_mm)
    # The flat cropped ends of the mask are not branch origins: the aorta
    # continues outside the mask there, so anything growing off the cap is the
    # parent vessel. The wall scan already excludes that surface at the source
    # (_drop_cap_surface), and does it by normal direction rather than by a
    # blanket z-margin, so only m3 needs the blunt version -- applying it to
    # the wall result as well would re-delete genuine branches that arise near
    # the end of a short segment.
    fallback = (drop_end_caps(m3_expansion(ctx), ctx)
                if args.method in ("m3", "both") else None)
    res = merge_results(primary, fallback, ctx, merge_mm=args.min_sep_mm)
    res = prune_duplicate_branches(
        res, ctx, min_sep_mm=args.min_sep_mm, seed_sep_mm=args.dup_seed_mm,
        near_mm=args.dup_near_mm, max_n=args.max_branches)
    d = save_method(res, ctx, args.outdir, case_id)
    print(f"  {res.name:<18} {len(res.ostia_vox):>3} ostia  {res.runtime:>6.1f}s  "
          f"-> {os.path.basename(d)}/   (total {time.time() - t0:.1f}s)")

    pred = build_prediction(res, ctx, case_id)
    with open(args.output, "w") as fh:
        json.dump(pred, fh, indent=2)
    labels = ", ".join(dd["instance_id"] for dd in pred["daughters"]) or "none"
    print(f"\n  {len(pred['daughters'])} daughter(s) labeled -> {args.output}"
          f"\n    {labels}")


if __name__ == "__main__":
    main()