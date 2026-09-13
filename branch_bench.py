#!/usr/bin/env python3
"""
branch_bench.py -- SELF-CONTAINED. Run 8 independent branch-detection methods on
one CT + aorta-mask pair and compare what they find. No other file needed.

    pip install SimpleITK numpy scipy matplotlib scikit-image plotly
    python branch_bench.py --image orig1.nii --aorta-mask mask1.nii \
        --outdir bench --case-id subject001

THE METHODS
  M1 frangi        Hessian vesselness + geometric linking back to the wall.
  M2 band_connect  Naive control: intensity band connected to the aorta.
  M3 expansion     Tahoces-style two-phase growing with leak detection.
  M4 wall_flux     Unrolled shell around the wall; peaks are openings.
  M5 escape        Fast-marching front with a soft intensity cost.
  M6 oof           Optimally Oriented Flux (sphere-sampled).
  M7 bulge         Mask geometry ONLY -- never reads the CT.
  M8 consensus     Votes over M1-M7. Your label-free validation.

OUTPUTS
  20_methods_3d.png  21_methods_axial.png  22_methods_agreement.png
  23_methods_ostia_map.png  methods_ostia.csv  methods_summary.csv
  pred_m1.json ... pred_m8.json  methods_interactive.html
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

__version__ = "2026-09-14.8"

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


def load_case(image_path: str, mask_path: str):
    """Read image + mask, verify they share a grid, return (img, mask, arrays)."""
    img = sitk.ReadImage(image_path)
    msk = sitk.ReadImage(mask_path)

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


def frangi_3d(vol, spacing, sigmas_mm, roi=None, alpha=0.5, beta=0.5, c=None):
    """Multi-scale 3D Frangi vesselness for BRIGHT tubes on a dark background.

    Derivatives are taken in physical units, so anisotropic slice thickness is
    handled and the sigmas are genuine millimetres. `roi` restricts the expensive
    eigendecomposition to voxels worth evaluating.

    Returns (vesselness, best_sigma_mm).
    """
    sx, sy, sz = spacing
    sp = (sz, sy, sx)
    if roi is None:
        roi = np.ones(vol.shape, bool)
    idx = np.where(roi)
    n = len(idx[0])
    if n == 0:
        return np.zeros(vol.shape, np.float32), np.zeros(vol.shape, np.float32)

    best = np.zeros(n, np.float32)
    best_sigma = np.zeros(n, np.float32)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    for s_mm in sigmas_mm:
        sig = tuple(max(s_mm / q, 0.35) for q in sp)
        H = np.empty((n, 3, 3), np.float32)
        for (i, j) in pairs:
            order = [0, 0, 0]
            order[i] += 1
            order[j] += 1
            d = ndi.gaussian_filter(vol, sig, order=order, mode="nearest")
            d = d / (sp[i] * sp[j]) * (s_mm ** 2)
            v = d[idx]
            H[:, i, j] = v
            H[:, j, i] = v

        ev = np.linalg.eigvalsh(H)
        del H
        order2 = np.argsort(np.abs(ev), axis=1)
        ev = np.take_along_axis(ev, order2, axis=1)
        l1, l2, l3 = ev[:, 0], ev[:, 1], ev[:, 2]
        del ev

        a2, a3 = np.abs(l2), np.abs(l3)
        with np.errstate(divide="ignore", invalid="ignore"):
            RA = np.where(a3 > 0, a2 / a3, 0.0)
            RB = np.where(a2 * a3 > 0, np.abs(l1) / np.sqrt(a2 * a3), 0.0)
        S = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        # c from a high percentile, not the max: one very bright object (bone, a
        # surgical clip) would otherwise define it and suppress every vessel.
        cc = c if c is not None else 0.5 * float(
            np.percentile(S, 99) if S.size else 1.0)
        cc = max(cc, 1e-6)

        V = ((1.0 - np.exp(-(RA ** 2) / (2 * alpha ** 2)))
             * np.exp(-(RB ** 2) / (2 * beta ** 2))
             * (1.0 - np.exp(-(S ** 2) / (2 * cc ** 2))))
        V[(l2 > 0) | (l3 > 0)] = 0.0
        V = np.nan_to_num(V, nan=0.0)

        upd = V > best
        best[upd] = V[upd]
        best_sigma[upd] = s_mm

    out = np.zeros(vol.shape, np.float32)
    out[idx] = best
    scale = np.zeros(vol.shape, np.float32)
    scale[idx] = best_sigma
    return out, scale


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
# per-subject calibration -- measure, do not guess
# -----------------------------------------------------------------------------

class Calib:
    """Numbers measured from THIS subject, all in physical units.

    The cohort is not homogeneous: observed voxel sizes differ by 2x and lumen
    brightness by 230 HU between subjects. Anything expressed in voxels, or any
    fixed HU threshold, silently means something different per case -- so every
    value here is derived from the case itself and stated in mm / mm^3 / HU.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def lines(self):
        return [
            f"voxel            : {self.sx:.2f} x {self.sy:.2f} x {self.sz:.2f} mm"
            f"  ({self.vox_mm3:.3f} mm3)",
            f"lumen            : {self.centre:.0f} HU, spread {self.spread:.0f}"
            f"   (mean {self.mean:.0f} +/- {self.sd:.0f})",
            f"rim offset       : {self.rim_mm:.2f} mm"
            f"   ({self.rim_note})",
            f"band             : {self.lo:.0f} - {self.hi:.0f} HU",
            f"bone cut         : > {self.bone_hu:.0f} HU and > 2 mL",
            f"min branch       : {self.min_branch_mm3:.0f} mm3",
            f"eligible         : must reach {self.eligible_mm:.0f} mm beyond "
            f"the wall",
            f"leak cap         : {self.leak_mL:.1f} mL "
            f"(aorta is {self.aorta_mL:.1f} mL)",
            f"despeckle        : components below min branch size are dropped "
            f"(no morphological opening)",
        ]


def calibrate(ct, mk, spacing, override_lumen=None, override_rim=None,
              override_band=None, min_branch_mm3=15.0):
    """Measure the parameters this case needs."""
    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    mkb = mk.astype(bool)

    # ---- lumen: median + MAD, so a stent or calcium cannot move it ----
    er = max(int(round(2.0 / min(sx, sy))), 1)
    core = ndi.binary_erosion(mkb, iterations=er)
    if core.sum() < 50:
        core = mkb
    v = ct[core]
    mean, sd = float(v.mean()), float(v.std())
    centre = float(np.median(v))
    spread = max(float(np.median(np.abs(v - centre))) * 1.4826, 8.0)
    if override_lumen:
        centre, spread = override_lumen

    # ---- rim: how far does the supplied mask UNDER-segment the lumen? ----
    # Measured by the FRACTION of each outward shell that is still blood-like,
    # not by the shell's median. Under-segmentation is usually patchy -- tight in
    # places, a voxel short in others -- so the median of a shell can sit in fat
    # while a third of it is still lumen. That third is enough to weld every
    # branch into one sheath wrapping the whole aorta.
    dist = ndi.distance_transform_edt(~mkb, sampling=(sz, sy, sx))
    step = max(min(sx, sy, sz) * 0.6, 0.25)
    lo_probe = centre - max(3.0 * spread, 120.0)
    rim_mm, prof = 0.0, []
    for d in np.arange(step, 6.0 + step, step):
        sel = (dist > d - step) & (dist <= d)
        if sel.sum() < 40:
            continue
        frac = float(((ct[sel] >= lo_probe) & (ct[sel] <= centre + 3 * spread)
                      ).mean())
        prof.append((round(float(d), 2), round(frac, 2)))
        if frac >= 0.30:
            rim_mm = float(d)
        else:
            break
    # Always strip at least one in-plane voxel. A branch runs 5-10 mm outward, so
    # losing its first voxel costs nothing, while NOT stripping lets a single
    # partially under-segmented ring merge everything into one component.
    rim_mm = float(np.clip(max(rim_mm, min(sx, sy)), 0.0, 4.0))
    rim_note = f"blood-like fraction per shell {prof[:5]}"
    if override_rim is not None:
        rim_mm, rim_note = float(override_rim), "overridden"

    # ---- band: wide BELOW the lumen, tight above ----
    # A 1-2 voxel branch is mostly partial volume, so its mean sits well under
    # the parent's -- measured at 384-503 HU against a 532 HU aorta on one case.
    # A symmetric band around the lumen throws those away.
    lo = max(120.0, centre - max(4.0 * spread, 150.0))
    hi = centre + max(2.5 * spread, 60.0)
    if override_band:
        lo, hi = override_band
    bone_hu = max(centre + 3.0 * spread, 350.0)

    aorta_mL = float(mkb.sum()) * vox_mm3 / 1000.0
    eligible_mm = 5.0
    classic = False
    leak_mL = float(np.clip(0.10 * aorta_mL, 1.0, 4.0))

    return Calib(leak_mL=leak_mL, aorta_mL=aorta_mL, classic=classic,
                 eligible_mm=eligible_mm, sx=sx, sy=sy, sz=sz, vox_mm3=vox_mm3, centre=centre,
                 spread=spread, mean=mean, sd=sd, rim_mm=rim_mm,
                 rim_note=rim_note, lo=lo, hi=hi, bone_hu=bone_hu,
                 profile=prof, min_branch_mm3=min_branch_mm3)


# -----------------------------------------------------------------------------
# shared plumbing
# -----------------------------------------------------------------------------

class Ctx:
    def __init__(self, img, ct, mk, spacing, cal, margin_mm=45.0, roi_mm=30.0):
        self.img, self.spacing, self.cal = img, spacing, cal
        sx, sy, sz = spacing
        self.sx, self.sy, self.sz = sx, sy, sz
        self.vox_mm3 = cal.vox_mm3
        pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
               int(round(margin_mm / sx)))
        self.region = mask_bbox(mk, pad, crop=True)
        self.ct = ct[self.region]
        self.mk = mk[self.region].astype(bool)
        self.z0 = self.region[0].start or 0
        self.y0 = self.region[1].start or 0
        self.x0 = self.region[2].start or 0
        self.sp = np.array([sz, sy, sx])
        self.roi_mm = roi_mm
        self.lo, self.hi = cal.lo, cal.hi
        self.mu, self.sd = cal.centre, cal.spread

        try:
            self.D = np.array(img.GetDirection(), float).reshape(3, 3)
        except Exception:
            self.D = np.eye(3)
        self.corner = None

        self.dist_out = ndi.distance_transform_edt(~self.mk,
                                                   sampling=(sz, sy, sx))
        # bone: bright AND bulky. Intensity alone fails when the lumen is bright.
        seed = ((self.ct > cal.bone_hu)
                & ~ndi.binary_dilation(self.mk, iterations=2))
        seed = ndi.binary_closing(seed, iterations=2)
        lab_b, n_b = ndi.label(seed)
        self.bone = np.zeros_like(seed)
        if n_b:
            big = np.bincount(lab_b.ravel()) * self.vox_mm3 / 1000.0 > 2.0
            big[0] = False
            self.bone = big[lab_b]
        self.bone = ndi.binary_dilation(ndi.binary_fill_holes(self.bone),
                                        iterations=max(int(round(
                                            2.0 / min(sx, sy))), 1))

        self.band = ((self.ct >= cal.lo) & (self.ct <= cal.hi) & ~self.bone)
        # the search region starts BEYOND the rim
        self.roi = (self.band & (self.dist_out > cal.rim_mm)
                    & (self.dist_out <= roi_mm))
        self.wall = (ndi.binary_dilation(self.mk, iterations=1)
                     & ~ndi.binary_erosion(self.mk, iterations=1))
        self.wall_pts = np.argwhere(self.wall).astype(float)
        self.hi_cut = cal.bone_hu

        self.cl = {}
        for z in np.where(self.mk.any(axis=(1, 2)))[0]:
            cy, cx = ndi.center_of_mass(self.mk[z])
            self.cl[int(z)] = (cy, cx)

    def verts_to_world(self, v):
        if v is None:
            return None
        if self.corner is None:
            self.corner = self.to_mm((0, 0, 0))
        return self.corner + np.asarray(v, float) @ self.D.T

    def to_mm(self, vox):
        v = np.asarray(vox, float)
        return np.array(self.img.TransformIndexToPhysicalPoint(
            (int(round(v[2])) + self.x0, int(round(v[1])) + self.y0,
             int(round(v[0])) + self.z0)), float)

    def strip_rim(self, mask):
        """Remove the shell of under-segmented lumen hugging the aorta."""
        return mask & (self.dist_out > self.cal.rim_mm)

    def despeckle(self, mask, min_mm3=None):
        """Drop small components INSTEAD of morphological opening.

        Opening erodes then dilates, which annihilates anything thinner than the
        structuring element. Reference branches measured 14-34 voxels, about two
        voxels across, so opening was deleting the very things being looked for.
        """
        mm3 = self.cal.min_branch_mm3 if min_mm3 is None else min_mm3
        lab, n = ndi.label(mask)
        if n == 0:
            return mask
        keep = np.bincount(lab.ravel()) * self.vox_mm3 >= mm3
        keep[0] = False
        return keep[lab]

    def snap_to_wall(self, pt_vox):
        """Nearest point on the aortic wall -- the ostium lives on the wall."""
        if not len(self.wall_pts):
            return np.asarray(pt_vox, float)
        d = np.linalg.norm((self.wall_pts - np.asarray(pt_vox, float)) * self.sp,
                           axis=1)
        return self.wall_pts[int(np.argmin(d))]


class Result:
    def __init__(self, name, mask, ostia_vox, dirs=None, radii=None,
                 runtime=0.0, note=""):
        self.name = name
        self.mask = mask
        self.ostia_vox = list(ostia_vox)
        self.dirs = dirs or [None] * len(self.ostia_vox)
        self.radii = radii or [None] * len(self.ostia_vox)
        self.runtime = runtime
        self.note = note


def ostia_from_mask(mask, ctx, max_n=25):
    """Components -> one ostium each, with direction and radius."""
    lab, n = ndi.label(mask)
    if n == 0:
        return [], [], []
    sizes = np.bincount(lab.ravel()) * ctx.vox_mm3
    sizes[0] = 0
    keep = [int(i) for i in np.argsort(sizes)[::-1]
            if sizes[i] >= ctx.cal.min_branch_mm3][:max_n]
    edt = ndi.distance_transform_edt(mask, sampling=tuple(ctx.sp))
    near = ndi.binary_dilation(ctx.mk, iterations=max(
        int(round((ctx.cal.rim_mm + 2.0) / min(ctx.sx, ctx.sy))), 2))

    ostia, dirs, radii = [], [], []
    for i in keep:
        comp = lab == i
        pts = np.argwhere(comp).astype(float)
        contact = comp & near
        if contact.any():
            cp = np.argwhere(contact)
            deepest = cp[int(np.argmax(edt[tuple(cp.T)]))]
            ost = ctx.snap_to_wall(deepest)      # the opening is ON the wall
        else:
            pmm = pts * ctx.sp
            cen = pmm - pmm.mean(axis=0)
            st = max(len(cen) // 4000, 1)
            _, _, vv = np.linalg.svd(cen[::st], full_matrices=False)
            hit, _g, ost, _u = link_to_aorta(pts, ctx.dist_out, vv[0],
                                             ctx.spacing, sub_mk=ctx.mk)
            if not hit or ost is None:
                continue
            ost = np.asarray(ost, float)

        omm = ost * ctx.sp
        d = np.linalg.norm(pts * ctx.sp - omm, axis=1)
        sel = d <= 12.0
        if sel.sum() >= 4:
            pm = pts[sel] * ctx.sp
            c = pm - pm.mean(axis=0)
            _, _, vv = np.linalg.svd(c, full_matrices=False)
            u = vv[0]
            if np.dot(pm.mean(axis=0) - omm, u) < 0:
                u = -u
            band = pts[sel][(d[sel] >= 3.5) & (d[sel] <= 6.5)]
            r = (float(ndi.map_coordinates(edt, band.mean(axis=0)[:, None],
                                           order=1)[0]) if len(band) else None)
        else:
            u, r = None, None
        ostia.append(ost)
        dirs.append(u)
        radii.append(r)
    return ostia, dirs, radii


def drop_end_caps(res, ctx, margin_mm=6.0):
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return res
    zlo, zhi = zs[0] * ctx.sz, zs[-1] * ctx.sz
    keep = [k for k, o in enumerate(res.ostia_vox)
            if (o[0] * ctx.sz - zlo) >= margin_mm
            and (zhi - o[0] * ctx.sz) >= margin_mm]
    res.ostia_vox = [res.ostia_vox[k] for k in keep]
    res.dirs = [res.dirs[k] for k in keep]
    res.radii = [res.radii[k] for k in keep]
    return res


def dedupe(res, ctx, merge_mm=6.0):
    kept = []
    for k, o in enumerate(res.ostia_vox):
        p = np.asarray(o, float) * ctx.sp
        if any(np.linalg.norm(p - np.asarray(q, float) * ctx.sp) < merge_mm
               for q, _ in kept):
            continue
        kept.append((o, k))
    res.ostia_vox = [o for o, _ in kept]
    res.dirs = [res.dirs[k] for _, k in kept]
    res.radii = [res.radii[k] for _, k in kept]
    return res


def ball(r_mm, spacing):
    sx, sy, sz = spacing
    rz, ry, rx = (max(int(np.ceil(r_mm / q)), 1) for q in (sz, sy, sx))
    zz, yy, xx = np.ogrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    return ((zz * sz) ** 2 + (yy * sy) ** 2 + (xx * sx) ** 2) <= r_mm ** 2


def remove_sheath(grown, ctx, core_mm=4.0, reach_mm=None):
    """Delete the wall-hugging sheath while KEEPING every branch junction.

    The sheath and a branch base are the same thickness, so no erosion or opening
    can separate them -- both are thin. What actually differs is that a branch
    has substance continuing outward and a sheath patch does not.

    So: take the shafts (everything far enough from the wall that the sheath
    cannot reach), then grow them back inward THROUGH the grown mask only. A
    branch base is a few mm of geodesic distance from its own shaft, so it is
    recovered together with the junction. Sheath that leads nowhere is never
    reached and disappears. Nothing is cut; the branches are simply never
    disconnected from the wall in the first place.
    """
    reach_mm = (core_mm + 4.0) if reach_mm is None else reach_mm
    core = grown & (ctx.dist_out >= core_mm)
    if not core.any():
        return np.zeros_like(grown)
    st = ball(min(ctx.sp) * 1.01, ctx.spacing)
    keep = core
    for _ in range(max(int(np.ceil(reach_mm / min(ctx.sp))), 1)):
        grown_once = ndi.binary_dilation(keep, structure=st) & grown
        if (grown_once == keep).all():
            break
        keep = grown_once
    return keep


def ostia_centerline(lab, ctx, max_out_mm=12.0, eligible_mm=None):
    """Ostium, direction, seed and radius from a shell-centroid centreline.

    Now that each branch stays attached to the aorta, the junction can be used
    instead of worked around. Step outward in thin shells; in each shell take the
    lumen-weighted centroid of the branch. Those centroids form a centreline that
    begins AT the wall, so the ostium is simply where it starts -- the centre of
    the opening, by construction, rather than a point inferred by extending a
    line backwards.
    """
    eligible_mm = ctx.cal.eligible_mm if eligible_mm is None else eligible_mm
    step = float(min(ctx.sp))
    ostia, dirs, radii = [], [], []
    for i in [int(v) for v in np.unique(lab) if v > 0]:
        comp = lab == i
        if comp.sum() * ctx.vox_mm3 < ctx.cal.min_branch_mm3:
            continue
        reach = float(ctx.dist_out[comp].max())
        if reach < eligible_mm:
            continue                       # never leaves the wall -> not eligible
        edt = ndi.distance_transform_edt(comp, sampling=tuple(ctx.sp))

        line = []                          # (distance from wall, centroid voxel)
        for d in np.arange(0.0, min(max_out_mm, reach) + step, step):
            sel = comp & (ctx.dist_out >= d) & (ctx.dist_out < d + step)
            pts = np.argwhere(sel)
            if len(pts) == 0:
                continue
            w = edt[tuple(pts.T)] + 1e-6   # bias toward the lumen centre
            line.append((float(d), (pts * w[:, None]).sum(0) / w.sum()))
        if len(line) < 3:
            continue

        ost = ctx.snap_to_wall(line[0][1])
        prox = np.array([c for d, c in line if 1.0 <= d <= 10.0], float)
        if len(prox) >= 2:
            pm = prox * ctx.sp
            u = pm[-1] - pm[0]
            n = np.linalg.norm(u)
            u = u / n if n > 1e-6 else None
        else:
            u = None
        seed = min(line, key=lambda t: abs(t[0] - 5.0))[1]
        r = float(ndi.map_coordinates(edt, np.asarray(seed, float)[:, None],
                                      order=1)[0])
        ostia.append(ost)
        dirs.append(u)
        radii.append(r)
    return ostia, dirs, radii


def split_outward(grown, ctx, seed_margin_mm=1.0):
    """One label per OUTWARD structure, each KEEPING its junction with the wall.

    The mask under-segments slightly, so a shell of lumen-intensity blood hugs
    the aorta and welds every branch into one component. Cutting that shell out
    fixes the merging but amputates every branch at its base -- the junction is
    exactly where the ostium is, so that trade is a bad one.

    Instead: take seeds from the part of each structure that reaches BEYOND the
    shell, then flood them back through the whole grown region. Branches come out
    separated from each other and still attached to the wall. Sheath that leads
    nowhere gets no seed and is dropped.
    """
    from skimage.segmentation import watershed

    seed_zone = grown & (ctx.dist_out > ctx.cal.rim_mm + seed_margin_mm)
    markers, n = ndi.label(seed_zone)
    if n == 0:
        return np.zeros(grown.shape, np.int32), 0
    sizes = np.bincount(markers.ravel()) * ctx.vox_mm3
    keep = sizes >= ctx.cal.min_branch_mm3
    keep[0] = False
    if not keep.any():
        return np.zeros(grown.shape, np.int32), 0
    remap = np.zeros(len(sizes), np.int32)
    remap[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1)
    markers = remap[markers]
    # flat elevation -> geodesic nearest-seed assignment inside `grown`
    lab = watershed(np.zeros(grown.shape, np.uint8), markers, mask=grown)
    return lab.astype(np.int32), int(markers.max())


def ostia_from_labels(lab, ctx, max_n=25, eligible_mm=None):
    """One ostium per label, taken at its contact with the aortic wall.

    Structures that never get far from the wall are rejected. That is the task's
    own eligibility rule -- a daughter must be followable at least 5 mm beyond
    the aortic wall -- and it is exactly what separates a real branch from the
    shell of under-segmented lumen that coats the aorta and can otherwise pick up
    a seed of its own.
    """
    eligible_mm = ctx.cal.eligible_mm if eligible_mm is None else eligible_mm
    ids = [int(i) for i in np.unique(lab) if i > 0]
    sizes = {i: float((lab == i).sum()) * ctx.vox_mm3 for i in ids}
    ids = sorted(ids, key=lambda i: -sizes[i])[:max_n]
    near = ndi.binary_dilation(ctx.mk, iterations=2)

    ostia, dirs, radii = [], [], []
    for i in ids:
        comp = lab == i
        if sizes[i] < ctx.cal.min_branch_mm3:
            continue
        if float(ctx.dist_out[comp].max()) < eligible_mm:
            continue                     # hugs the wall -> not an eligible daughter
        edt = ndi.distance_transform_edt(comp, sampling=tuple(ctx.sp))
        pts = np.argwhere(comp).astype(float)
        contact = comp & near
        if contact.any():
            cp = np.argwhere(contact)
            ost = ctx.snap_to_wall(cp[int(np.argmax(edt[tuple(cp.T)]))])
        else:
            pmm = pts * ctx.sp
            cen = pmm - pmm.mean(axis=0)
            st = max(len(cen) // 4000, 1)
            _, _, vv = np.linalg.svd(cen[::st], full_matrices=False)
            hit, _g, ost, _u = link_to_aorta(pts, ctx.dist_out, vv[0],
                                             ctx.spacing, sub_mk=ctx.mk)
            if not hit or ost is None:
                continue
            ost = np.asarray(ost, float)

        omm = ost * ctx.sp
        d = np.linalg.norm(pts * ctx.sp - omm, axis=1)
        # direction and radius from the proximal segment, ignoring the junction
        sel = (d >= 2.0) & (d <= 12.0)
        if sel.sum() >= 4:
            pm = pts[sel] * ctx.sp
            c = pm - pm.mean(axis=0)
            _, _, vv = np.linalg.svd(c, full_matrices=False)
            u = vv[0]
            if np.dot(pm.mean(axis=0) - omm, u) < 0:
                u = -u
            band = pts[sel][(d[sel] >= 3.5) & (d[sel] <= 6.5)]
            r = (float(ndi.map_coordinates(edt, band.mean(axis=0)[:, None],
                                           order=1)[0]) if len(band) else None)
        else:
            u, r = None, None
        ostia.append(ost)
        dirs.append(u)
        radii.append(r)
    return ostia, dirs, radii


def grow_from_aorta(ctx, limit_mm, blocked=None):
    """Everything band-like and connected to the aorta, out to limit_mm.

    Growth runs THROUGH the rim so connectivity is preserved; the rim is only
    stripped afterwards, when components are formed.
    """
    allowed = ctx.band & (ctx.dist_out <= limit_mm)
    if blocked is not None:
        allowed = allowed & ~blocked
    lab, _ = ndi.label(allowed | ctx.mk)
    touch = set(int(v) for v in np.unique(lab[ctx.mk]) if v > 0)
    return np.isin(lab, list(touch)) & ~ctx.mk


# -----------------------------------------------------------------------------
# the methods
# -----------------------------------------------------------------------------

def m1_frangi(ctx, sigmas=None, frac=0.30):
    t0 = time.time()
    if sigmas is None:                       # scales in mm, not voxels
        sigmas = (0.7, 1.0, 1.5, 2.0, 3.0)
    vess, _ = frangi_3d(ctx.ct, ctx.spacing, sigmas, roi=ctx.roi)
    thr = frac * float(np.percentile(vess[ctx.roi], 99.5)) if ctx.roi.any() else 0
    cand = ctx.despeckle((vess >= thr) & ctx.roi)
    o, d, r = ostia_from_mask(cand, ctx)
    return Result("M1 frangi", cand, o, d, r, time.time() - t0, f"thr={thr:.3f}")


def m2_band_connect(ctx):
    t0 = time.time()
    lab, n = split_outward(grow_from_aorta(ctx, ctx.roi_mm), ctx)
    o, d, r = ostia_from_labels(lab, ctx)
    return Result("M2 band_connect", lab > 0, o, d, r, time.time() - t0,
                  f"band {ctx.lo:.0f}-{ctx.hi:.0f} HU, {n} structures")


def m3_classic(ctx, phase1_mm=20.0, phase2_mm=30.0, leak_mL=8.0):
    """The ORIGINAL M3, unchanged: grow, block oversized components, regrow.

    Components are kept whole, so each branch keeps the voxels where it meets the
    aorta and the junctions look like real junctions. The cost is a shell of
    under-segmented lumen coating the aorta, which is cosmetic; trying to remove
    it is what broke the junctions.
    """
    t0 = time.time()
    band = (ctx.ct >= ctx.lo) & (ctx.ct <= ctx.hi) & ~ctx.bone

    def grow(limit_mm, blocked):
        allowed = band & (ctx.dist_out <= limit_mm) & ~blocked
        lab, _ = ndi.label(allowed | ctx.mk)
        touch = set(int(v) for v in np.unique(lab[ctx.mk]) if v > 0)
        return np.isin(lab, list(touch)) & ~ctx.mk

    g1 = grow(phase1_mm, np.zeros_like(band))
    lab1, n1 = ndi.label(g1)
    blocked = np.zeros_like(band)
    n_leak = 0
    if n1:
        mL = np.bincount(lab1.ravel()) * ctx.vox_mm3 / 1000.0
        leaks = [i for i in range(1, n1 + 1) if mL[i] > leak_mL]
        n_leak = len(leaks)
        if leaks:
            blocked = np.isin(lab1, leaks)

    g2 = ndi.binary_opening(grow(phase2_mm, blocked), iterations=1)
    lab2, _ = ndi.label(g2)
    o, d, r = ostia_from_labels(lab2, ctx, eligible_mm=0.0)
    return Result("M3 expansion", g2, o, d, r, time.time() - t0,
                  f"classic | {n_leak} leak(s) blocked, cap {leak_mL} mL")


def m3_expansion(ctx, phase_mm=30.0, leak_mL=None, core_mm=4.0):
    """Original growth, kept whole; sheath removed after; leaks judged last.

    Order matters. Before the sheath is removed, every branch and the wall
    coating are ONE component, so any per-component size rule sees a single
    oversized blob and blocks the lot -- which is how the celiac and renals were
    being deleted. Leak handling therefore runs last, once components are real
    branches, and a leak is truncated rather than dropped.
    """
    if ctx.cal.classic:
        return m3_classic(ctx)
    t0 = time.time()
    leak_mL = ctx.cal.leak_mL if leak_mL is None else leak_mL
    band = (ctx.ct >= ctx.lo) & (ctx.ct <= ctx.hi) & ~ctx.bone

    allowed = band & (ctx.dist_out <= phase_mm)
    lb, _ = ndi.label(allowed | ctx.mk)
    touch = set(int(v) for v in np.unique(lb[ctx.mk]) if v > 0)
    grown = np.isin(lb, list(touch)) & ~ctx.mk

    cleaned = remove_sheath(grown, ctx, core_mm=core_mm)
    lab, n = ndi.label(cleaned)

    out = np.zeros(lab.shape, np.int32)
    n_trunc = n_drop = 0
    for i in range(1, n + 1):
        comp = lab == i
        if not comp.any():
            continue
        if comp.sum() * ctx.vox_mm3 / 1000.0 <= leak_mL:
            out[comp] = i
            continue
        for lim in (10.0, 6.0, 3.5):        # a leak means the growth ran too far
            t = comp & (ctx.dist_out <= lim)
            if t.any() and t.sum() * ctx.vox_mm3 / 1000.0 <= leak_mL:
                out[t] = i
                n_trunc += 1
                break
        else:
            n_drop += 1

    o, d, r = ostia_centerline(out, ctx)
    removed = (grown.sum() - cleaned.sum()) * ctx.vox_mm3 / 1000.0
    return Result("M3 expansion", out > 0, o, d, r, time.time() - t0,
                  f"{n} structures, {removed:.2f} mL sheath removed, "
                  f"{n_trunc} truncated, {n_drop} dropped")


def m4_wall_flux(ctx, n_ang=120, min_sep_mm=6.0):
    t0 = time.time()
    shell = (ctx.cal.rim_mm + 0.5, ctx.cal.rim_mm + 6.0)
    zs = sorted(ctx.cl)
    prof = np.full((len(zs), n_ang), np.nan, np.float32)
    for row, z in enumerate(zs):
        sh = ((ctx.dist_out[z] >= shell[0]) & (ctx.dist_out[z] <= shell[1])
              & ~ctx.bone[z])
        ys, xs = np.where(sh)
        if not len(ys):
            continue
        cy, cx = ctx.cl[z]
        th = np.arctan2((ys - cy) * ctx.sy, (xs - cx) * ctx.sx)
        b = np.clip(((th + np.pi) / (2 * np.pi) * n_ang).astype(int), 0, n_ang - 1)
        v = ctx.ct[z][ys, xs]
        sums = np.bincount(b, weights=v, minlength=n_ang)
        cnts = np.bincount(b, minlength=n_ang)
        with np.errstate(invalid="ignore"):
            prof[row] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)

    sm = ndi.gaussian_filter(np.nan_to_num(prof, nan=ctx.lo - 100),
                             (1.2, 1.2), mode="wrap")
    thr = ctx.lo
    peaks = (sm == ndi.maximum_filter(sm, size=(7, 9), mode="wrap")) & (sm > thr)

    ostia, mask = [], np.zeros_like(ctx.mk)
    for row, col in np.argwhere(peaks):
        z = zs[row]
        ang = (col + 0.5) / n_ang * 2 * np.pi - np.pi
        cy, cx = ctx.cl[z]
        for rad in np.arange(0.5, 40.0, 0.5):
            yy = cy + rad * np.sin(ang) / ctx.sy
            xx = cx + rad * np.cos(ang) / ctx.sx
            if not (0 <= yy < ctx.mk.shape[1] and 0 <= xx < ctx.mk.shape[2]):
                break
            if not ctx.mk[z, int(yy), int(xx)]:
                ostia.append(np.array([z, yy, xx], float))
                mask[z, int(yy), int(xx)] = True
                break
    res = Result("M4 wall_flux", ndi.binary_dilation(mask, iterations=2),
                 ostia, None, None, time.time() - t0, f"thr {thr:.0f} HU")
    return dedupe(res, ctx, min_sep_mm)


def m5_escape(ctx, max_cost=8.0):
    t0 = time.time()
    try:
        from skimage.graph import MCP_Geometric
    except ImportError:
        return Result("M5 escape", np.zeros_like(ctx.mk), [], None, None, 0.0,
                      "skimage.graph unavailable")
    z = np.clip((ctx.ct - ctx.lo) / max(ctx.hi - ctx.lo, 1.0), 0, 1)
    cost = (1.0 / (0.08 + z)).astype(np.float64)
    cost[ctx.bone | (ctx.dist_out > ctx.roi_mm)] = np.inf
    cost[ctx.mk] = 0.05

    mcp = MCP_Geometric(cost, sampling=tuple(ctx.sp))
    starts = np.argwhere(ctx.wall & ctx.mk)
    if not len(starts):
        return Result("M5 escape", np.zeros_like(ctx.mk), [], None, None,
                      time.time() - t0, "no wall")
    st = max(len(starts) // 1500, 1)
    try:
        gd, _ = mcp.find_costs(starts[::st].tolist(),
                               max_cumulative_cost=max_cost)
    except TypeError:
        gd, _ = mcp.find_costs(starts[::st].tolist())
    gd = np.nan_to_num(gd, nan=1e6, posinf=1e6)
    lab, n = split_outward((gd < max_cost) & ~ctx.mk, ctx)
    o, d, r = ostia_from_labels(lab, ctx)
    return Result("M5 escape", lab > 0, o, d, r, time.time() - t0,
                  f"cost<{max_cost}, {n} structures")


def _sphere_dirs(n=42):
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)


def m6_oof(ctx, radii_mm=(1.0, 1.5, 2.5), n_dir=42, frac=0.30):
    t0 = time.time()
    idx = np.argwhere(ctx.roi)
    if not len(idx):
        return Result("M6 oof", np.zeros_like(ctx.mk), [], None, None, 0.0,
                      "empty search region")
    gz, gy, gx = np.gradient(ctx.ct.astype(np.float32), ctx.sz, ctx.sy, ctx.sx)
    dirs = _sphere_dirs(n_dir)
    best = np.zeros(len(idx), np.float32)
    for r_mm in radii_mm:
        Q = np.zeros((len(idx), 3, 3), np.float32)
        for u in dirs:
            co = np.clip(idx + (u * r_mm / ctx.sp), 0,
                         np.array(ctx.ct.shape) - 1).T
            g = np.stack([ndi.map_coordinates(a, co, order=1)
                          for a in (gz, gy, gx)], axis=1)
            Q += (g * u).sum(axis=1)[:, None, None] * np.outer(u, u)[None]
        Q /= n_dir
        ev = np.linalg.eigvalsh(Q)
        ev = np.take_along_axis(ev, np.argsort(np.abs(ev), axis=1), axis=1)
        best = np.maximum(best, np.clip(-(ev[:, 1] + ev[:, 2]) / 2.0, 0, None)
                          * (r_mm ** 0.5))
    out = np.zeros(ctx.ct.shape, np.float32)
    out[ctx.roi] = best
    thr = frac * float(np.percentile(best, 99.5)) if best.size else 0.0
    cand = ctx.despeckle((out >= thr) & ctx.roi)
    o, d, r = ostia_from_mask(cand, ctx)
    return Result("M6 oof", cand, o, d, r, time.time() - t0, f"thr={thr:.3f}")


def m7_bulge(ctx, n_ang=90, k_sd=2.0, min_sep_mm=6.0):
    t0 = time.time()
    zs = sorted(ctx.cl)
    rad = np.full((len(zs), n_ang), np.nan, np.float32)
    for row, z in enumerate(zs):
        cy, cx = ctx.cl[z]
        ys, xs = np.where(ctx.mk[z])
        if len(ys) < 8:
            continue
        dy, dx = (ys - cy) * ctx.sy, (xs - cx) * ctx.sx
        th = np.arctan2(dy, dx)
        rr = np.hypot(dy, dx)
        b = np.clip(((th + np.pi) / (2 * np.pi) * n_ang).astype(int), 0, n_ang - 1)
        for k in range(n_ang):
            m = b == k
            if m.any():
                rad[row, k] = rr[m].max()
    filled = np.nan_to_num(rad, nan=np.nanmedian(rad)
                           if np.isfinite(rad).any() else 0)
    dev = filled - ndi.gaussian_filter(filled, (3.0, 4.0), mode="wrap")
    sdv = float(np.nanstd(dev)) or 1.0
    peaks = ((dev == ndi.maximum_filter(dev, size=(5, 7), mode="wrap"))
             & (dev > k_sd * sdv))
    ostia, mask = [], np.zeros_like(ctx.mk)
    for row, col in np.argwhere(peaks):
        z = zs[row]
        ang = (col + 0.5) / n_ang * 2 * np.pi - np.pi
        cy, cx = ctx.cl[z]
        rr = filled[row, col]
        yy, xx = cy + rr * np.sin(ang) / ctx.sy, cx + rr * np.cos(ang) / ctx.sx
        if 0 <= yy < ctx.mk.shape[1] and 0 <= xx < ctx.mk.shape[2]:
            ostia.append(np.array([z, yy, xx], float))
            mask[z, int(yy), int(xx)] = True
    res = Result("M7 bulge", ndi.binary_dilation(mask, iterations=2), ostia,
                 None, None, time.time() - t0,
                 f"dev > {k_sd} sd ({k_sd * sdv:.2f} mm)")
    return dedupe(res, ctx, min_sep_mm)


def m8_consensus(results, ctx, tol_mm=6.0, min_votes=3):
    t0 = time.time()
    pts, who = [], []
    for r in results:
        for o in r.ostia_vox:
            pts.append(np.asarray(o, float) * ctx.sp)
            who.append(r.name)
    if not pts:
        return Result("M8 consensus", np.zeros_like(ctx.mk), [], None, None,
                      0.0, "nothing to vote on"), []
    pts = np.array(pts)
    used = np.zeros(len(pts), bool)
    ostia, votes = [], []
    order = np.argsort([-sum(np.linalg.norm(pts - p, axis=1) < tol_mm)
                        for p in pts])
    for i in order:
        if used[i]:
            continue
        m = (~used) & (np.linalg.norm(pts - pts[i], axis=1) < tol_mm)
        names = {who[j] for j in np.flatnonzero(m)}
        used |= m
        if len(names) >= min_votes:
            ostia.append(pts[m].mean(axis=0) / ctx.sp)
            votes.append(sorted(names))
    mask = np.zeros_like(ctx.mk)
    for o in ostia:
        zz, yy, xx = [int(round(q)) for q in o]
        if (0 <= zz < mask.shape[0] and 0 <= yy < mask.shape[1]
                and 0 <= xx < mask.shape[2]):
            mask[zz, yy, xx] = True
    return Result("M8 consensus", ndi.binary_dilation(mask, iterations=2),
                  ostia, None, None, time.time() - t0,
                  f">={min_votes} methods agree"), votes


# -----------------------------------------------------------------------------
# ground-truth scoring
# -----------------------------------------------------------------------------

def load_labels(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int32)


def gt_ostia(gt_lab, ctx, min_label=2):
    """Reference ostia: label 1 is the parent, every label >= 2 is a daughter."""
    near = ndi.binary_dilation(ctx.mk, iterations=3)
    out = []
    for L in sorted(int(v) for v in np.unique(gt_lab) if v >= min_label):
        comp = gt_lab == L
        if not comp.any():
            continue
        edt = ndi.distance_transform_edt(comp, sampling=tuple(ctx.sp))
        contact = comp & near
        pts = np.argwhere(contact if contact.any() else comp)
        deep = pts[int(np.argmax(edt[tuple(pts.T)]))].astype(float)
        out.append((L, ctx.snap_to_wall(deep), float(comp.sum()) * ctx.vox_mm3))
    return out


def score(results, ctx, refs, match_mm, outdir):
    R = [np.asarray(o, float) * ctx.sp for _, o, _ in refs]
    rows = []
    for res in results:
        P = [np.asarray(o, float) * ctx.sp for o in res.ostia_vox]
        cand = sorted(((float(np.linalg.norm(p - r)), i, j)
                       for i, p in enumerate(P) for j, r in enumerate(R)),
                      key=lambda t: t[0])
        tookP, tookR, hits = set(), set(), []
        for dist, i, j in cand:
            if dist > match_mm or i in tookP or j in tookR:
                continue
            tookP.add(i)
            tookR.add(j)
            hits.append(dist)
        tp, fp, fn = len(hits), len(P) - len(hits), len(R) - len(hits)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append(dict(method=res.name, tp=tp, fp=fp, fn=fn,
                         precision=round(prec, 3), recall=round(rec, 3),
                         f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                         mean_ostium_err_mm=round(float(np.mean(hits)), 2)
                         if hits else ""))
    with open(os.path.join(outdir, "scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  SCORED against {len(R)} reference branches "
          f"(match radius {match_mm:.0f} mm)")
    print(f"  {'method':<18}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}"
          f"{'F1':>7}{'err mm':>8}")
    for r in sorted(rows, key=lambda q: -q["f1"]):
        print(f"  {r['method']:<18}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}"
              f"{r['precision']:>7.2f}{r['recall']:>7.2f}{r['f1']:>7.2f}"
              f"{str(r['mean_ostium_err_mm']):>8}")
    return rows


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------
# per-method output: one folder each
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# per-method output: one folder each
# -----------------------------------------------------------------------------

def save_method(res, ctx, outdir, case_id):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = os.path.join(outdir, res.name.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "ostia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"])
        for k, (o, dr, rad) in enumerate(zip(res.ostia_vox, res.dirs,
                                             res.radii), 1):
            p_ = ctx.to_mm(o)
            dd = ([round(float(dr[2]), 4), round(float(dr[1]), 4),
                   round(float(dr[0]), 4)] if dr is not None else ["", "", ""])
            w.writerow([f"branch_{k:03d}", round(p_[0], 2), round(p_[1], 2),
                        round(p_[2], 2), *dd, round(rad, 3) if rad else ""])

    ds = []
    for k, (o, dr, rad) in enumerate(zip(res.ostia_vox, res.dirs, res.radii), 1):
        p_ = ctx.to_mm(o)
        v = ([float(dr[2]), float(dr[1]), float(dr[0])] if dr is not None
             else [0.0, 0.0, 1.0])
        v = list(np.round(np.array(v) / max(np.linalg.norm(v), 1e-9), 4))
        ds.append({"instance_id": f"branch_{k:03d}",
                   "parent_instance_id": "aorta",
                   "ostium_xyz_mm": [round(float(q), 3) for q in p_],
                   "seed_xyz_mm": [round(float(q), 3)
                                   for q in (p_ + np.array(v) * 5.0)],
                   "radius_mm": round(float(rad), 3) if rad else None,
                   "direction_xyz": v})
    with open(os.path.join(d, "prediction.json"), "w") as fh:
        json.dump({"case_id": case_id, "parent": {"instance_id": "aorta"},
                   "daughters": ds}, fh, indent=2)

    meshes = []
    av, af = fine_surface(ctx.mk, ctx.spacing, cap=120_000)
    if av is not None:
        meshes.append(dict(verts=ctx.verts_to_world(av), faces=af, name="aorta",
                           color="#d94a4a", opacity=0.30, text="parent aorta"))
    bv, bf = fine_surface(res.mask & ~ctx.mk, ctx.spacing, cap=90_000)
    if bv is not None:
        meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                           name="detected branches", color="#f2c14e",
                           opacity=0.95, text=f"{res.name} branch voxels"))
    if res.ostia_vox:
        P = np.array([ctx.to_mm(o) for o in res.ostia_vox])
        meshes.append(dict(kind="points", points=P, name="ostia",
                           color="#111111", size=8,
                           labels=[f"branch_{k:03d}"
                                   for k in range(1, len(res.ostia_vox) + 1)]))
    write_interactive_html(
        os.path.join(d, "render.html"), meshes,
        title=f"{case_id} - {res.name}",
        subtitle=f"{len(res.ostia_vox)} ostia | {res.runtime:.1f}s | {res.note}")

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

    zc = np.zeros(ctx.ct.shape[0])
    for o in res.ostia_vox:
        z = int(round(o[0]))
        if 0 <= z < len(zc):
            zc[max(z - 1, 0):z + 2] += 1
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
            ax.annotate(f"{k:02d}", (a, z), fontsize=8, xytext=(6, 4),
                        textcoords="offset points")
    ax.set_xlim(-185, 185)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("angle around the aorta (deg)")
    ax.set_ylabel("z (mm)")
    ax.grid(alpha=0.3)
    ax.set_title("ostia on the unrolled wall", fontsize=10)

    fig.suptitle(f"{case_id} - {res.name}   |   {len(res.ostia_vox)} ostia   |   "
                 f"{res.runtime:.1f}s   |   {res.note}", fontsize=12)
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
    ap.add_argument("--mask-label", type=int, default=None,
                    help="use only this label of the mask file as the parent")
    ap.add_argument("--gt", default=None,
                    help="multi-label ground truth (label 1 = aorta, >=2 = "
                         "reference branches). Enables scoring.")
    ap.add_argument("--gt-min-label", type=int, default=2)
    ap.add_argument("--match-mm", type=float, default=10.0,
                    help="a prediction counts as a hit within this distance")
    ap.add_argument("--min-branch-mm3", type=float, default=15.0,
                    help="smallest component treated as a branch")
    ap.add_argument("--classic", action="store_true",
                    help="restore the ORIGINAL behaviour: whole components with "
                         "their wall junctions intact, symmetric HU band, 20/30 "
                         "mm growth, 8 mL leak cap, no rim or eligibility "
                         "filtering. Use this if the newer logic is worse.")
    ap.add_argument("--core-mm", type=float, default=4.0,
                    help="distance from the wall at which a structure counts as "
                         "a branch shaft; shafts are grown back inward to "
                         "recover their junctions (M3)")
    ap.add_argument("--eligible-mm", type=float, default=None,
                    help="a daughter must be followable this far beyond the "
                         "wall (the task's own rule; default 5)")
    ap.add_argument("--leak-ml", type=float, default=None,
                    help="a grown component larger than this is treated as a "
                         "leak into an organ and dropped")
    ap.add_argument("--rim-mm", type=float, default=None,
                    help="override the measured rim offset")
    ap.add_argument("--band", default=None,
                    help="override the measured HU band, e.g. 150,420")
    ap.add_argument("--lumen-hu", default=None,
                    help="override the automatic lumen model, e.g. 300,25 "
                         "(centre,spread in HU). Use when the report warns that "
                         "the mask contains dense outliers.")
    ap.add_argument("--tol-mm", type=float, default=6.0)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--skip", default="", help="e.g. M5,M6")
    args = ap.parse_args()

    case_id = args.case_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.image))) or "case"
    os.makedirs(args.outdir, exist_ok=True)

    print(f"  branch_bench {__version__}")
    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    if args.mask_label is not None:
        mk = (load_labels(args.aorta_mask) == args.mask_label).astype(np.uint8)
        print(f"  parent = label {args.mask_label} only "
              f"({int(mk.sum())} voxels)")
    elif len(np.unique(load_labels(args.aorta_mask))) > 2:
        print("  WARNING: the mask has several labels and they are being "
              "MERGED. If labels >= 2 are branches, pass --mask-label 1, "
              "otherwise they cannot be found.", file=sys.stderr)

    cal = calibrate(
        ct, mk, img.GetSpacing(),
        override_lumen=(tuple(float(q) for q in args.lumen_hu.split(","))
                        if args.lumen_hu else None),
        override_rim=args.rim_mm,
        override_band=(tuple(float(q) for q in args.band.split(","))
                       if args.band else None),
        min_branch_mm3=args.min_branch_mm3)
    if args.classic:
        # exactly the settings the original run used, before any later change
        cal.classic = True
        cal.lo = cal.centre - 2.5 * cal.spread
        cal.hi = cal.centre + 2.5 * cal.spread
        cal.bone_hu = max(400.0, cal.centre + 3.0 * cal.spread)
        cal.rim_mm = 0.0
        cal.rim_note = "classic: no rim handling"
        cal.min_branch_mm3 = 8.0
        cal.eligible_mm = 0.0
        cal.leak_mL = 8.0
    # explicit overrides win over --classic, so a hand-set band still applies
    if args.band:
        cal.lo, cal.hi = (float(q) for q in args.band.split(","))
    if args.lumen_hu:
        c_, s_ = (float(q) for q in args.lumen_hu.split(","))
        cal.centre, cal.spread = c_, s_
        if not args.band:
            cal.lo, cal.hi = c_ - 2.5 * s_, c_ + 2.5 * s_
        cal.bone_hu = max(cal.hi + 40.0, 350.0)
    if args.leak_ml:
        cal.leak_mL = args.leak_ml
    if args.eligible_mm:
        cal.eligible_mm = args.eligible_mm
    for line in cal.lines():
        print("  " + line)
    with open(os.path.join(args.outdir, "calibration.txt"), "w") as fh:
        fh.write(f"{case_id}\n" + "\n".join(cal.lines()) + "\n")

    ctx = Ctx(img, ct, mk, img.GetSpacing(), cal, roi_mm=args.roi_mm)
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    print(f"  aorta mask       : {ctx.mk.sum() * ctx.vox_mm3 / 1000:.1f} mL, "
          f"{len(zs)} slices, {len(zs) * ctx.sz:.0f} mm long")
    print(f"  search region    : "
          f"{ctx.roi.sum() * ctx.vox_mm3 / 1000:.1f} mL   "
          f"bone excluded {ctx.bone.sum() * ctx.vox_mm3 / 1000:.0f} mL\n")

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    plan = [("M1", m1_frangi), ("M2", m2_band_connect), ("M3", m3_expansion),
            ("M4", m4_wall_flux), ("M5", m5_escape), ("M6", m6_oof),
            ("M7", m7_bulge)]

    results = []
    for mid, fn in plan:
        if mid in skip:
            continue
        try:
            r = dedupe(drop_end_caps(
                (fn(ctx, core_mm=args.core_mm) if fn is m3_expansion
                 else fn(ctx)), ctx), ctx)
            results.append(r)
        except Exception as exc:
            print(f"  {mid} FAILED: {exc}", file=sys.stderr)

    cons, votes = m8_consensus(results, ctx, args.tol_mm, args.min_votes)
    results.append(cons)

    # agreement, for the one summary table
    n = len(results)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(results):
        A = [np.asarray(o, float) * ctx.sp for o in a.ostia_vox]
        for j, b in enumerate(results):
            B = [np.asarray(o, float) * ctx.sp for o in b.ostia_vox]
            if A and B:
                M[i, j] = 100.0 * sum(
                    1 for p in A
                    if min(np.linalg.norm(p - q) for q in B) <= args.tol_mm) / len(A)

    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_ostia", "runtime_s", "note", "mean_agreement_pct"])
        for i, r in enumerate(results):
            row = M[i][np.arange(n) != i]
            w.writerow([r.name, len(r.ostia_vox), round(r.runtime, 2), r.note,
                        round(float(np.nanmean(row)), 1)
                        if np.isfinite(row).any() else ""])

    if args.gt:
        refs = gt_ostia(load_labels(args.gt)[ctx.region], ctx, args.gt_min_label)
        print(f"\n  ground truth: {len(refs)} branches "
              f"({', '.join(f'L{L}:{v:.0f}mm3' for L, _, v in refs)})")
        if refs:
            score(results, ctx, refs, args.match_mm, args.outdir)

    for r in results:
        d = save_method(r, ctx, args.outdir, case_id)
        print(f"  {r.name:<18} {len(r.ostia_vox):>3} ostia  {r.runtime:>6.1f}s  "
              f"-> {os.path.basename(d)}/")

    print(f"\n  consensus ostia (>= {args.min_votes} methods):")
    for o, v in zip(cons.ostia_vox, votes):
        p = ctx.to_mm(o)
        print(f"    ({p[0]:8.1f},{p[1]:8.1f},{p[2]:9.1f}) mm  <- {', '.join(v)}")
    print(f"\n  {args.outdir}/summary.csv  +  one folder per method "
          f"(ostia.csv, prediction.json, render.html, overview.png)")


if __name__ == "__main__":
    main()