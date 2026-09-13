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

__version__ = "2026-09-13.2"

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
    """Learn the lumen HU band from inside the (eroded) aorta mask."""
    sx, sy, sz = spacing
    er_iter = max(int(round(erode_mm / min(sx, sy))), 1)
    core = ndi.binary_erosion(mk, iterations=er_iter)
    if core.sum() < 50:
        core = mk.astype(bool)
    lumen = ct[core]
    mu, sd = float(lumen.mean()), float(lumen.std())
    return mu - k_sd * sd, mu + k_sd * sd, mu, sd


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
    hi_cut = (max(400.0, mu + hi_sd * sd) if bone_floor_hu is None
              else float(bone_floor_hu))
    # Seed on cortex, then FILL: trabecular marrow sits at 200-400 HU, right in
    # the lumen range, so thresholding alone leaves the inside of the vertebra
    # in the ROI -- and trabeculae are a mesh of struts that Frangi scores highly.
    bone = ndi.binary_closing(sub_ct > hi_cut, iterations=2)
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
        print("  plotly not installed -- skipping the interactive HTML "
              "(pip install plotly)", file=sys.stderr)
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

    def __init__(self, img, ct, mk, spacing, margin_mm=45.0, roi_mm=30.0):
        self.img, self.spacing = img, spacing
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

        self.lo, self.hi, self.mu, self.sd = lumen_band(ct, mk, spacing=spacing)
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
        """voxel (z, y, x) in the CROP -> physical (x, y, z) mm."""
        v = np.asarray(vox, float)
        return np.array(self.img.TransformIndexToPhysicalPoint(
            (int(round(v[2])) + self.x0, int(round(v[1])) + self.y0,
             int(round(v[0])) + self.z0)), float)


class Result:
    def __init__(self, name, mask, ostia_vox, dirs=None, radii=None,
                 runtime=0.0, note=""):
        self.name = name
        self.mask = mask                       # bool volume of believed branches
        self.ostia_vox = list(ostia_vox)       # [(z, y, x)] in crop coords
        self.dirs = dirs or [None] * len(self.ostia_vox)
        self.radii = radii or [None] * len(self.ostia_vox)
        self.runtime = runtime
        self.note = note


def components(mask, ctx, min_mm3=8.0):
    lab, _ = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(min_mm3 / ctx.vox_mm3), 1)
    return lab, [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]


def ostia_from_mask(mask, ctx, min_mm3=8.0, max_n=25, use_contact=True):
    """Shared post-step: components -> one ostium each.

    If the component already touches the aorta, take the contact zone and pick
    the voxel that maximises distance to the edge of the ramification -- the
    centre of the opening, following Tahoces. Otherwise fall back to extending
    the component's local axis back to the wall.
    """
    lab, keep = components(mask, ctx, min_mm3)
    edt = ndi.distance_transform_edt(mask, sampling=tuple(ctx.sp)) if mask.any() \
        else np.zeros_like(mask, float)
    near_wall = ndi.binary_dilation(ctx.mk, iterations=2)

    ostia, dirs, radii = [], [], []
    for i in keep[:max_n]:
        comp = lab == i
        pts = np.argwhere(comp).astype(float)
        contact = comp & near_wall
        if use_contact and contact.any():
            cpts = np.argwhere(contact)
            best = cpts[int(np.argmax(edt[tuple(cpts.T)]))]
            ost = best.astype(float)
        else:
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

        # outward direction + radius from the proximal 12 mm
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
    """The flat cropped top and bottom of the mask are not branch origins."""
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return res
    zlo, zhi = zs[0] * ctx.sz, zs[-1] * ctx.sz
    keep = [k for k, o in enumerate(res.ostia_vox)
            if (o[0] * ctx.sz - zlo) >= margin_mm and (zhi - o[0] * ctx.sz) >= margin_mm]
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


# -----------------------------------------------------------------------------
# M1  Frangi vesselness
# -----------------------------------------------------------------------------

def m1_frangi(ctx, sigmas=(0.7, 1.0, 1.5, 2.0, 3.0), frac=0.30):
    t0 = time.time()
    vess, _ = frangi_3d(ctx.ct, ctx.spacing, sigmas, roi=ctx.roi)
    thr = frac * float(np.percentile(vess[ctx.roi], 99.5)) if ctx.roi.any() else 0
    cand = ndi.binary_opening((vess >= thr) & ctx.roi, iterations=1)
    o, d, r = ostia_from_mask(cand, ctx, use_contact=False)
    return Result("M1 frangi", cand, o, d, r, time.time() - t0,
                  f"thr={thr:.3f}")


# -----------------------------------------------------------------------------
# M2  naive intensity band + connectivity  (the control)
# -----------------------------------------------------------------------------

def m2_band_connect(ctx):
    t0 = time.time()
    band = (ctx.ct >= ctx.lo) & (ctx.ct <= ctx.hi) & ~ctx.bone
    band = ndi.binary_opening(band, iterations=1)
    lab, _ = ndi.label(band | ctx.mk)
    touch = set(int(v) for v in np.unique(lab[ctx.mk]) if v > 0)
    grown = np.isin(lab, list(touch)) & ~ctx.mk & (ctx.dist_out <= ctx.roi_mm)
    o, d, r = ostia_from_mask(grown, ctx)
    return Result("M2 band_connect", grown, o, d, r, time.time() - t0,
                  f"band {ctx.lo:.0f}-{ctx.hi:.0f} HU")


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

    # phase 2: grow further, avoiding the flagged regions
    g2 = grow(phase2_mm, blocked)
    o, d, r = ostia_from_mask(g2, ctx)
    return Result("M3 expansion", g2, o, d, r, time.time() - t0,
                  f"{n_leak} leak(s) blocked")


# -----------------------------------------------------------------------------
# M4  wall flux -- unrolled shell, peaks are openings
# -----------------------------------------------------------------------------

def m4_wall_flux(ctx, shell=(1.5, 6.0), n_ang=120, min_sep_mm=6.0):
    t0 = time.time()
    zs = sorted(ctx.cl)
    prof = np.full((len(zs), n_ang), np.nan, np.float32)
    for row, z in enumerate(zs):
        shellz = (ctx.dist_out[z] >= shell[0]) & (ctx.dist_out[z] <= shell[1]) \
            & ~ctx.bone[z]
        ys, xs = np.where(shellz)
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

    sm = ndi.gaussian_filter(np.nan_to_num(prof, nan=ctx.mu - 4 * ctx.sd),
                             (1.2, 1.2), mode="wrap")
    thr = ctx.mu - 1.5 * ctx.sd
    peaks = (sm == ndi.maximum_filter(sm, size=(7, 9), mode="wrap")) & (sm > thr)

    ostia, mask = [], np.zeros_like(ctx.mk)
    for row, col in np.argwhere(peaks):
        z = zs[row]
        ang = (col + 0.5) / n_ang * 2 * np.pi - np.pi
        cy, cx = ctx.cl[z]
        # walk out from the centre until we leave the mask -> wall point
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


# -----------------------------------------------------------------------------
# M5  fast-marching escape with a soft intensity speed function
# -----------------------------------------------------------------------------

def m5_escape(ctx, max_cost=18.0):
    t0 = time.time()
    try:
        from skimage.graph import MCP_Geometric
    except ImportError:
        return Result("M5 escape", np.zeros_like(ctx.mk), [], None, None, 0.0,
                      "skimage.graph unavailable")
    # soft speed: 1 inside the lumen band, rising cost as intensity falls away.
    # Unlike a hard band this survives partial-volume dropout at an ostium.
    z = np.clip((ctx.ct - (ctx.mu - 3 * ctx.sd)) / (3 * ctx.sd), 0, 1)
    cost = (1.0 / (0.08 + z)).astype(np.float64)
    # np.inf is impassable; a large finite cost still gets explored, which makes
    # the front crawl over the whole crop and takes minutes instead of seconds.
    blocked = ctx.bone | (ctx.dist_out > ctx.roi_mm)
    cost[blocked] = np.inf
    cost[ctx.mk] = 0.05                       # free travel inside the lumen

    mcp = MCP_Geometric(cost, sampling=tuple(ctx.sp))
    starts = np.argwhere(ctx.wall & ctx.mk)
    if not len(starts):
        return Result("M5 escape", np.zeros_like(ctx.mk), [], None, None,
                      time.time() - t0, "no wall")
    step = max(len(starts) // 1500, 1)
    try:
        gd, _ = mcp.find_costs(starts[::step].tolist(),
                               max_cumulative_cost=max_cost)
    except TypeError:                          # newer skimage dropped the arg
        gd, _ = mcp.find_costs(starts[::step].tolist())
    gd = np.nan_to_num(gd, nan=1e6, posinf=1e6)

    reach = (gd < max_cost) & ~ctx.mk & (ctx.dist_out > 0)
    reach = ndi.binary_opening(reach, iterations=1)
    o, d, r = ostia_from_mask(reach, ctx)
    return Result("M5 escape", reach, o, d, r, time.time() - t0,
                  f"cost<{max_cost}")


# -----------------------------------------------------------------------------
# M6  Optimally Oriented Flux (sphere-sampled)
# -----------------------------------------------------------------------------

def _sphere_dirs(n=42):
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (1 + 5 ** 0.5)
    th = gold * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)          # (n, 3) as (z, y, x)


def m6_oof(ctx, radii_mm=(1.0, 1.5, 2.5), n_dir=42, frac=0.30):
    """Flux of the image gradient through a sphere of radius r.

    Unlike a Hessian, OOF integrates over a sphere SURFACE, so a bright object
    sitting next to the vessel contributes far less -- the property that makes it
    the sensible choice when the aorta is pressed against the vertebral body.
    """
    t0 = time.time()
    idx = np.argwhere(ctx.roi)
    if not len(idx):
        return Result("M6 oof", np.zeros_like(ctx.mk), [], None, None, 0.0, "no roi")
    gz, gy, gx = np.gradient(ctx.ct.astype(np.float32), ctx.sz, ctx.sy, ctx.sx)
    dirs = _sphere_dirs(n_dir)
    best = np.zeros(len(idx), np.float32)

    for r_mm in radii_mm:
        Q = np.zeros((len(idx), 3, 3), np.float32)
        for u in dirs:
            off = idx + (u * r_mm / ctx.sp)
            co = np.clip(off, 0, np.array(ctx.ct.shape) - 1).T
            g = np.stack([ndi.map_coordinates(a, co, order=1)
                          for a in (gz, gy, gx)], axis=1)
            flux = (g * u).sum(axis=1)               # inward-positive for bright
            Q += flux[:, None, None] * np.outer(u, u)[None, :, :]
        Q /= n_dir
        ev = np.linalg.eigvalsh(Q)
        order = np.argsort(np.abs(ev), axis=1)
        ev = np.take_along_axis(ev, order, axis=1)
        # bright tube: the two largest-magnitude eigenvalues strongly negative
        resp = np.clip(-(ev[:, 1] + ev[:, 2]) / 2.0, 0, None) * (r_mm ** 0.5)
        best = np.maximum(best, resp)

    out = np.zeros(ctx.ct.shape, np.float32)
    out[ctx.roi] = best
    thr = frac * float(np.percentile(best, 99.5)) if best.size else 0.0
    cand = ndi.binary_opening((out >= thr) & ctx.roi, iterations=1)
    o, d, r = ostia_from_mask(cand, ctx, use_contact=False)
    return Result("M6 oof", cand, o, d, r, time.time() - t0, f"thr={thr:.3f}")


# -----------------------------------------------------------------------------
# M7  mask geometry only -- outward bulges on a fitted tube
# -----------------------------------------------------------------------------

def m7_bulge(ctx, n_ang=90, k_sd=2.0, min_sep_mm=6.0):
    """Never looks at the CT. If the supplied mask bulges at the origins, this
    finds them for free -- and it is completely independent of every intensity
    based method, which makes it valuable for the consensus vote."""
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

    filled = np.nan_to_num(rad, nan=np.nanmedian(rad) if np.isfinite(rad).any() else 0)
    smooth = ndi.gaussian_filter(filled, (3.0, 4.0), mode="wrap")
    dev = filled - smooth
    s = float(np.nanstd(dev)) or 1.0
    peaks = ((dev == ndi.maximum_filter(dev, size=(5, 7), mode="wrap"))
             & (dev > k_sd * s))

    ostia, mask = [], np.zeros_like(ctx.mk)
    for row, col in np.argwhere(peaks):
        z = zs[row]
        ang = (col + 0.5) / n_ang * 2 * np.pi - np.pi
        cy, cx = ctx.cl[z]
        rr = filled[row, col]
        yy = cy + rr * np.sin(ang) / ctx.sy
        xx = cx + rr * np.cos(ang) / ctx.sx
        if 0 <= yy < ctx.mk.shape[1] and 0 <= xx < ctx.mk.shape[2]:
            ostia.append(np.array([z, yy, xx], float))
            mask[z, int(yy), int(xx)] = True
    res = Result("M7 bulge", ndi.binary_dilation(mask, iterations=2), ostia,
                 None, None, time.time() - t0, f"dev > {k_sd} sd ({k_sd * s:.2f} mm)")
    return dedupe(res, ctx, min_sep_mm)


# -----------------------------------------------------------------------------
# M8  consensus
# -----------------------------------------------------------------------------

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
        zz, yy, xx = [int(round(v)) for v in o]
        if 0 <= zz < mask.shape[0] and 0 <= yy < mask.shape[1] and 0 <= xx < mask.shape[2]:
            mask[zz, yy, xx] = True
    return Result("M8 consensus", ndi.binary_dilation(mask, iterations=2),
                  ostia, None, None, time.time() - t0,
                  f">={min_votes} methods agree"), votes


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------

def fig_3d(results, ctx, path, case_id, cap=18_000):
    # matplotlib's 3D renderer is soft-sorted and gets very slow past ~25k
    # polygons per axes. Eight panels at 100k each simply never finishes, so the
    # static figure uses a coarse mesh; the plotly HTML gets the detailed one.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    av, af = fine_surface(ctx.mk, ctx.spacing, cap=cap)
    n = len(results)
    cols, rows = 4, int(np.ceil(n / 4))
    fig = plt.figure(figsize=(4.2 * cols, 5.0 * rows), facecolor="white")
    for k, res in enumerate(results):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        allv = []
        if av is not None:
            c = Poly3DCollection(av[af], alpha=0.22)
            c.set_facecolor("#d94a4a")
            c.set_edgecolor("none")
            ax.add_collection3d(c)
            allv.append(av)
        bv, bf = fine_surface(res.mask & ~ctx.mk, ctx.spacing, cap=cap)
        if bv is not None:
            c = Poly3DCollection(bv[bf], alpha=0.95)
            c.set_facecolor("#f2c14e")
            c.set_edgecolor("none")
            ax.add_collection3d(c)
            allv.append(bv)
        if res.ostia_vox:
            P = np.array([np.asarray(o, float) * ctx.sp for o in res.ostia_vox])
            ax.scatter(P[:, 2], P[:, 1], P[:, 0], s=42, c="#111111",
                       marker="D", depthshade=False)
        if allv:
            s = np.vstack(allv)
            ax.set_xlim(s[:, 0].min(), s[:, 0].max())
            ax.set_ylim(s[:, 1].min(), s[:, 1].max())
            ax.set_zlim(s[:, 2].min(), s[:, 2].max())
            try:
                ax.set_box_aspect([np.ptp(s[:, i]) for i in range(3)])
            except Exception:
                pass
        ax.view_init(elev=12, azim=-75)
        ax.set_title(f"{res.name}\n{len(res.ostia_vox)} ostia · "
                     f"{res.runtime:.1f}s · {res.note}", fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"{case_id} — eight independent branch detectors "
                 f"(gold = believed branch voxels, black diamonds = ostia)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def fig_axial(results, ctx, path, case_id):
    score = np.zeros(ctx.ct.shape[0])
    for r in results:
        for o in r.ostia_vox:
            z = int(round(o[0]))
            if 0 <= z < len(score):
                score[max(z - 1, 0):z + 2] += 1
    z_show = int(np.argmax(score)) if score.max() else ctx.ct.shape[0] // 2
    n = len(results)
    cols, rows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(3.7 * cols, 3.9 * rows),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    for k, res in enumerate(results):
        ax = axes[k]
        ax.imshow(window(ctx.ct[z_show]), cmap="gray", vmin=0, vmax=1,
                  aspect=ctx.sy / ctx.sx)
        ov = np.zeros(ctx.ct[z_show].shape + (4,), np.float32)
        ov[res.mask[z_show] & ~ctx.mk[z_show]] = (1.0, 0.78, 0.25, 0.75)
        ov[ctx.mk[z_show]] = (0.85, 0.29, 0.29, 0.45)
        ax.imshow(ov, aspect=ctx.sy / ctx.sx, interpolation="nearest")
        for o in res.ostia_vox:
            if abs(o[0] - z_show) <= 2:
                ax.plot(o[2], o[1], "D", ms=7, mfc="#00e5ff", mec="white", mew=1)
        ax.set_title(f"{res.name}  ({len(res.ostia_vox)})", color="white",
                     fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{case_id} — axial slice {z_show}, all methods", color="white",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=125, facecolor="black")
    plt.close(fig)


def fig_agreement(results, ctx, path, case_id, tol_mm=6.0):
    n = len(results)
    M = np.zeros((n, n))
    for i, a in enumerate(results):
        A = [np.asarray(o, float) * ctx.sp for o in a.ostia_vox]
        for j, b in enumerate(results):
            B = [np.asarray(o, float) * ctx.sp for o in b.ostia_vox]
            if not A or not B:
                M[i, j] = np.nan
                continue
            M[i, j] = 100.0 * sum(
                1 for p in A if min(np.linalg.norm(p - q) for q in B) <= tol_mm
            ) / len(A)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    im = axes[0].imshow(M, cmap="viridis", vmin=0, vmax=100)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    names = [r.name.split()[0] for r in results]
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0].set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                axes[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                             color="white" if M[i, j] < 60 else "black",
                             fontsize=8)
    axes[0].set_title("agreement: % of ROW's ostia found by COLUMN\n"
                      f"(within {tol_mm:.0f} mm)", fontsize=11)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].barh(names[::-1], [len(r.ostia_vox) for r in results][::-1],
                 color="#4f86c6")
    axes[1].set_title("ostia found", fontsize=11)
    axes[1].grid(alpha=0.25, axis="x")

    axes[2].barh(names[::-1], [r.runtime for r in results][::-1],
                 color="#e07a5f")
    axes[2].set_title("runtime (s)", fontsize=11)
    axes[2].grid(alpha=0.25, axis="x")

    fig.suptitle(f"{case_id} — method agreement. With no ground truth, an ostium "
                 f"several independent methods find is your best evidence.",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return M


def fig_ostia_map(results, ctx, path, case_id):
    """Every method's ostia on the unrolled aortic wall: angle vs slice."""
    fig, ax = plt.subplots(figsize=(13, 8))
    markers = "osD^vP*X"
    for k, res in enumerate(results):
        if not res.ostia_vox:
            continue
        A, Z = [], []
        for o in res.ostia_vox:
            z = int(round(o[0]))
            if z not in ctx.cl:
                z = min(ctx.cl, key=lambda q: abs(q - z))
            cy, cx = ctx.cl[z]
            A.append(np.degrees(np.arctan2((o[1] - cy) * ctx.sy,
                                           (o[2] - cx) * ctx.sx)))
            Z.append(o[0] * ctx.sz)
        ax.scatter(A, Z, s=90, marker=markers[k % len(markers)], alpha=0.8,
                   label=f"{res.name} ({len(A)})")
    ax.set_xlabel("angle around the aorta (deg)   0 = +x, ±180 = -x")
    ax.set_ylabel("position along z (mm)")
    ax.invert_yaxis()
    ax.set_xlim(-185, 185)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"{case_id} — where each method puts its ostia.\n"
                 f"Vertical clusters = several methods agreeing on one origin.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--outdir", default="bench")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--roi-mm", type=float, default=30.0)
    ap.add_argument("--tol-mm", type=float, default=6.0,
                    help="distance within which two methods count as agreeing")
    ap.add_argument("--min-votes", type=int, default=3,
                    help="methods that must agree for a consensus ostium")
    ap.add_argument("--skip", default="", help="comma-separated method ids to "
                                               "skip, e.g. M6,M5")
    args = ap.parse_args()

    case_id = args.case_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.image))) or "case"
    os.makedirs(args.outdir, exist_ok=True)

    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    ctx = Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm)
    print(f"  branch_bench {__version__}")
    print(f"  lumen {ctx.mu:.0f} +/- {ctx.sd:.0f} HU   band {ctx.lo:.0f}-{ctx.hi:.0f}"
          f"   bone > {ctx.hi_cut:.0f}   crop {ctx.ct.shape}")

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    plan = [("M1", m1_frangi), ("M2", m2_band_connect), ("M3", m3_expansion),
            ("M4", m4_wall_flux), ("M5", m5_escape), ("M6", m6_oof),
            ("M7", m7_bulge)]

    results = []
    for mid, fn in plan:
        if mid in skip:
            print(f"  {mid} skipped")
            continue
        try:
            r = fn(ctx)
            r = dedupe(drop_end_caps(r, ctx), ctx)
            results.append(r)
            print(f"  {r.name:<18} {len(r.ostia_vox):>3} ostia  "
                  f"{r.runtime:>6.1f}s  {r.note}")
        except Exception as exc:
            print(f"  {mid} FAILED: {exc}", file=sys.stderr)

    cons, votes = m8_consensus(results, ctx, args.tol_mm, args.min_votes)
    results.append(cons)
    print(f"  {cons.name:<18} {len(cons.ostia_vox):>3} ostia  {cons.note}")
    for o, v in zip(cons.ostia_vox, votes):
        p = ctx.to_mm(o)
        print(f"      ({p[0]:8.1f},{p[1]:8.1f},{p[2]:9.1f}) mm  <- {', '.join(v)}")

    # ---- tables ----
    with open(os.path.join(args.outdir, "methods_ostia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"])
        for r in results:
            for o, d, rad in zip(r.ostia_vox, r.dirs, r.radii):
                p = ctx.to_mm(o)
                dd = ([round(float(d[2]), 4), round(float(d[1]), 4),
                       round(float(d[0]), 4)] if d is not None else ["", "", ""])
                w.writerow([r.name, round(p[0], 2), round(p[1], 2), round(p[2], 2),
                            *dd, round(rad, 3) if rad else ""])

    M = fig_agreement(results, ctx, os.path.join(args.outdir,
                                                 "22_methods_agreement.png"),
                      case_id, args.tol_mm)
    with open(os.path.join(args.outdir, "methods_summary.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_ostia", "runtime_s", "note",
                    "mean_agreement_pct"])
        for i, r in enumerate(results):
            row = M[i][np.arange(len(results)) != i]
            w.writerow([r.name, len(r.ostia_vox), round(r.runtime, 2), r.note,
                        round(float(np.nanmean(row)), 1) if np.isfinite(row).any()
                        else ""])

    # ---- challenge JSON per method ----
    for r in results:
        ds = []
        for k, (o, d, rad) in enumerate(zip(r.ostia_vox, r.dirs, r.radii), 1):
            p = ctx.to_mm(o)
            vec = ([float(d[2]), float(d[1]), float(d[0])] if d is not None
                   else [0.0, 0.0, 1.0])
            vec = list(np.round(np.array(vec) / max(np.linalg.norm(vec), 1e-9), 4))
            ds.append({"instance_id": f"branch_{k:03d}",
                       "parent_instance_id": "aorta",
                       "ostium_xyz_mm": [round(float(v), 3) for v in p],
                       "seed_xyz_mm": [round(float(v), 3)
                                       for v in (p + np.array(vec) * 5.0)],
                       "radius_mm": round(float(rad), 3) if rad else None,
                       "direction_xyz": vec})
        tag = r.name.split()[0].lower()
        with open(os.path.join(args.outdir, f"pred_{tag}.json"), "w") as fh:
            json.dump({"case_id": case_id, "parent": {"instance_id": "aorta"},
                       "daughters": ds}, fh, indent=2)

    # ---- figures ----
    for name, fn in [("20_methods_3d.png", fig_3d),
                     ("21_methods_axial.png", fig_axial),
                     ("23_methods_ostia_map.png", fig_ostia_map)]:
        try:
            fn(results, ctx, os.path.join(args.outdir, name), case_id)
            print(f"  wrote {os.path.join(args.outdir, name)}")
        except Exception as exc:
            print(f"  FAILED {name}: {exc}", file=sys.stderr)

    # ---- interactive ----
    meshes = []
    av, af = fine_surface(ctx.mk, ctx.spacing, cap=120_000)
    if av is not None:
        meshes.append(dict(verts=ctx.verts_to_world(av), faces=af, name="aorta",
                           color="#d94a4a", opacity=0.3, text="parent aorta"))
    palette = ["#f2c14e", "#2f9e9e", "#7b6cd9", "#e07a5f", "#3fa34d",
               "#c85b9b", "#4f86c6", "#111111"]
    for k, r in enumerate(results):
        bv, bf = fine_surface(r.mask & ~ctx.mk, ctx.spacing, cap=90_000)
        if bv is None:
            continue
        meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                           name=f"{r.name} ({len(r.ostia_vox)} ostia)",
                           color=palette[k % len(palette)], opacity=0.9,
                           visible=(r.name.startswith("M8")),
                           text=f"<b>{r.name}</b><br>{len(r.ostia_vox)} ostia<br>"
                                f"{r.runtime:.1f}s<br>{r.note}"))
        if r.ostia_vox:
            P = np.array([ctx.to_mm(o) for o in r.ostia_vox])
            meshes.append(dict(kind="points", points=P,
                               name=f"{r.name} ostia", color=palette[k % len(palette)],
                               size=7,
                               labels=[f"{r.name} ostium" for _ in r.ostia_vox]))
    write_interactive_html(
        os.path.join(args.outdir, "methods_interactive.html"), meshes,
        title=f"{case_id} — eight branch detectors",
        subtitle="Consensus is shown first. Click legend entries to bring in the "
                 "individual methods and see where they disagree.",
        axis_note="")

    print(f"\n  done -> {args.outdir}/")


if __name__ == "__main__":
    main()