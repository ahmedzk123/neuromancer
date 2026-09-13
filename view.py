#!/usr/bin/env python3
"""
visualize_case.py -- Inspect one CT + aorta-mask pair from the aortic-branch challenge.

Usage
-----
    python visualize_case.py --image orig1.nii --aorta-mask mask1.nii --outdir viz/

    # add an interactive scroll-through-slices window:
    python visualize_case.py --image orig1.nii --aorta-mask mask1.nii --interactive

What it produces (PNGs in --outdir, plus a printed header report):

  00_report.txt        geometry / intensity / mask stats
  01_ortho.png         axial, coronal, sagittal through the aorta centroid, mask overlaid
  02_axial_montage.png evenly spaced axial slices spanning the aortic segment
  03_mip.png           coronal + sagittal maximum-intensity projections -- the single
                       best view for *seeing* the daughter branches
  06_intensity_render.png  is the lumen HU band alone enough to isolate the arteries?

All figures show the FULL field of view by default. Pass --crop to restrict them
to a margin around the aorta (tighter, but you lose surrounding context).
  04_aorta_3d.png      marching-cubes surface of the supplied aorta mask
  05_wall_shell.png    mean CT intensity in a thin shell just outside the aortic wall,
                       unrolled as angle-vs-z -- branch ostia show up as bright blobs

Dependencies
------------
    pip install SimpleITK numpy matplotlib scikit-image scipy

Coordinate note: SimpleITK arrays are indexed [z, y, x]; physical points are
(x, y, z) via TransformIndexToPhysicalPoint. Every millimetre figure printed here
goes through that transform, never raw index arithmetic.
"""

from __future__ import annotations

import argparse
import re
import os
import sys

import numpy as np
import SimpleITK as sitk
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

# -----------------------------------------------------------------------------
# display constants
# -----------------------------------------------------------------------------

# Abdominal CT-angiography window. Contrast-filled aorta sits ~250-400 HU.
__version__ = "2026-09-13.1"

CTA_LEVEL, CTA_WIDTH = 200.0, 700.0


class _PhysMap:
    """Stand-in for a SimpleITK image after resampling: same physical origin,
    new spacing. Axis-aligned volumes only, which abdominal CT effectively is."""

    def __init__(self, origin, spacing, direction=None):
        self._o, self._s = tuple(origin), tuple(spacing)
        self._d = tuple(direction) if direction else (1., 0., 0.,
                                                      0., 1., 0.,
                                                      0., 0., 1.)

    def TransformIndexToPhysicalPoint(self, idx):
        return tuple(self._o[k] + idx[k] * self._s[k] for k in range(3))

    def GetOrigin(self):
        return self._o

    def GetSpacing(self):
        return self._s

    def GetDirection(self):
        return self._d


def resample_isotropic(ct, mk, spacing, target_mm):
    """Resample to isotropic voxels.

    Slice thickness is usually 2-4x the in-plane spacing, so the volume carries
    genuinely less information along z. The Hessian is computed in physical units
    either way, but the DATA is blurrier in z -- a cranio-caudal vessel is
    measured more crudely than an in-plane one. Resampling removes that
    asymmetry, at the cost of a bigger array and more runtime.
    """
    sx, sy, sz = spacing
    zoom = (sz / target_mm, sy / target_mm, sx / target_mm)
    ct2 = ndi.zoom(ct, zoom, order=1)
    mk2 = (ndi.zoom(mk.astype(np.float32), zoom, order=1) > 0.5).astype(np.uint8)
    return ct2, mk2, (target_mm, target_mm, target_mm)

MASK_CMAP = ListedColormap([(0, 0, 0, 0), (1.0, 0.25, 0.25, 0.35)])


# -----------------------------------------------------------------------------
# IO / geometry
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------

def fig_ortho(ct, mk, spacing, out_png, case_id):
    """Axial / coronal / sagittal through the aorta centre of mass."""
    sx, sy, sz = spacing
    cz, cy, cx = [int(round(c)) for c in ndi.center_of_mass(mk)] if mk.any() else \
                 [s // 2 for s in ct.shape]

    panels = [
        ("Axial",    ct[cz], mk[cz],          sy / sx, f"z = {cz}"),
        ("Coronal",  ct[:, cy], mk[:, cy],    sz / sx, f"y = {cy}"),
        ("Sagittal", ct[:, :, cx], mk[:, :, cx], sz / sy, f"x = {cx}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="black")
    for ax, (name, im, ms, aspect, sub) in zip(axes, panels):
        ax.imshow(window(im), cmap="gray", vmin=0, vmax=1,
                  aspect=aspect, origin="lower" if name != "Axial" else "upper")
        ax.imshow(ms, cmap=MASK_CMAP, vmin=0, vmax=1,
                  aspect=aspect, origin="lower" if name != "Axial" else "upper",
                  interpolation="nearest")
        ax.contour(ms, levels=[0.5], colors=["#ff4d4d"], linewidths=0.9,
                   origin="lower" if name != "Axial" else "upper")
        ax.set_title(f"{name}  ({sub})", color="white", fontsize=11)
        ax.axis("off")

    fig.suptitle(f"{case_id} — orthogonal views through aorta centroid  "
                 f"(red = supplied parent mask)", color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="black")
    plt.close(fig)


def fig_axial_montage(ct, mk, spacing, out_png, case_id, n_panels=24,
                      margin_mm=35.0, crop=False):
    """Evenly spaced axial slices spanning the aortic segment."""
    sx, sy, sz = spacing
    zs = np.where(mk.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return
    z_idx = np.linspace(zs[0], zs[-1], min(n_panels, len(zs))).round().astype(int)

    pad = (0, int(round(margin_mm / sy)), int(round(margin_mm / sx)))
    _, sly, slx = mask_bbox(mk, pad, crop=crop)

    cols = 6
    rows = int(np.ceil(len(z_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()

    for ax, z in zip(axes, z_idx):
        ax.imshow(window(ct[z, sly, slx]), cmap="gray", vmin=0, vmax=1,
                  aspect=sy / sx)
        ax.contour(mk[z, sly, slx], levels=[0.5], colors=["#ff4d4d"],
                   linewidths=1.0)
        ax.set_title(f"z={z}", color="#bbbbbb", fontsize=8, pad=2)
        ax.axis("off")
    for ax in axes[len(z_idx):]:
        ax.axis("off")

    fig.suptitle(f"{case_id} — axial sweep through the aortic segment "
                 f"(bright stubs leaving the red outline are candidate daughters)",
                 color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="black")
    plt.close(fig)


def fig_mip(ct, mk, spacing, out_png, case_id, slab_mm=45.0, crop=False):
    """Coronal + sagittal maximum-intensity projections of a slab around the aorta.

    This is the view where branch anatomy becomes obvious: the celiac trunk and SMA
    project anteriorly on the sagittal MIP, the renals laterally on the coronal.
    """
    sx, sy, sz = spacing
    if not mk.any():
        return
    pad = (0, int(round(slab_mm / sy)), int(round(slab_mm / sx)))
    slz, sly, slx = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[slz, sly, slx], mk[slz, sly, slx]

    cor = sub_ct.max(axis=1)          # project along y -> [z, x]
    sag = sub_ct.max(axis=2)          # project along x -> [z, y]
    cor_m = sub_mk.max(axis=1)
    sag_m = sub_mk.max(axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), facecolor="black")
    for ax, (im, ms, name, asp) in zip(
        axes,
        [(cor, cor_m, "Coronal MIP", sz / sx), (sag, sag_m, "Sagittal MIP", sz / sy)],
    ):
        ax.imshow(window(im, level=250, width=600), cmap="gray", vmin=0, vmax=1,
                  aspect=asp, origin="lower")
        ax.contour(ms, levels=[0.5], colors=["#ff4d4d"], linewidths=1.0,
                   origin="lower")
        ax.set_title(name, color="white", fontsize=12)
        ax.axis("off")

    scope = (f"slab MIP ±{slab_mm:.0f} mm around the aorta" if crop
             else "full field-of-view MIP")
    fig.suptitle(f"{case_id} — {scope}  (red = parent mask outline)",
                 color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="black")
    plt.close(fig)


def fig_aorta_3d(mk, spacing, out_png, case_id, target_vox=1.5, out_html=None):
    """Marching-cubes surface of the supplied mask, three viewing angles."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("skimage not installed -- skipping the 3D surface", file=sys.stderr)
        return
    if not mk.any():
        return

    sx, sy, sz = spacing
    # Downsample to roughly isotropic ~target_vox mm so marching cubes stays fast.
    zoom = (sz / target_vox, sy / target_vox, sx / target_vox)
    small = ndi.zoom(mk.astype(np.float32), zoom, order=1)
    small = ndi.gaussian_filter(small, 0.8)
    if small.max() < 0.5:
        return

    verts, faces, _, _ = marching_cubes(small, level=0.5,
                                        spacing=(target_vox,) * 3)
    # verts are (z, y, x) mm -> reorder to (x, y, z) for plotting
    verts = verts[:, [2, 1, 0]]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    views = [(20, -60, "oblique"), (0, -90, "anterior"), (0, 0, "left lateral")]
    fig = plt.figure(figsize=(15, 6), facecolor="white")
    for i, (elev, azim, name) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        coll = Poly3DCollection(verts[faces], alpha=0.9)
        coll.set_facecolor("#d94a4a")
        coll.set_edgecolor("none")
        ax.add_collection3d(coll)
        for setter, vals in (
            (ax.set_xlim, verts[:, 0]), (ax.set_ylim, verts[:, 1]),
            (ax.set_zlim, verts[:, 2]),
        ):
            setter(vals.min(), vals.max())
        try:
            ax.set_box_aspect([np.ptp(verts[:, k]) for k in range(3)])
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=11)
        ax.set_axis_off()

    fig.suptitle(f"{case_id} — supplied parent aorta mask, surface render "
                 f"(bumps on the wall are branch ostia)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)

    if out_html:
        ok = write_interactive_html(
            out_html,
            [dict(verts=verts, faces=faces, name="aorta (supplied mask)",
                  color="#d94a4a", opacity=1.0,
                  text="parent aorta<br>supplied mask")],
            title=f"{case_id} — supplied aorta mask",
            subtitle="drag to rotate, scroll to zoom. Look along the wall for "
                     "small outward bumps: those are branch origins the mask "
                     "may already be hinting at.",
            axis_note="  [relative]")
        if ok:
            print(f"  wrote {out_html}")


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


def write_obj(path, verts, faces, name="surface"):
    """Plain Wavefront OBJ -- opens in 3D Slicer, MeshLab, Blender, Windows 3D
    Viewer, Preview on macOS. No dependencies, no internet, fully interactive."""
    with open(path, "w") as fh:
        fh.write(f"o {name}\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")


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


def fig_intensity_render(ct, mk, spacing, out_png, out_txt, case_id,
                         k_sd=2.5, erode_mm=2.0, margin_mm=60.0,
                         crop=False, max_components=15,
                         min_component_mm3=30.0):
    """EXPERIMENT: is intensity alone enough to isolate the arterial tree?"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    lo, hi, mu, sd = lumen_band(ct, mk, k_sd, erode_mm, spacing)
    pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
           int(round(margin_mm / sx)))
    region = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)

    band = ndi.binary_opening((sub_ct >= lo) & (sub_ct <= hi), iterations=1)
    lab, n_lab = ndi.label(band)
    if n_lab == 0:
        return
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    touch = set(int(t) for t in
                np.unique(lab[ndi.binary_dilation(sub_mk, iterations=1)]) if t > 0)
    connected = np.isin(lab, list(touch)) if touch else np.zeros_like(band)

    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox][:40]

    lines = [
        "=" * 74,
        f"INTENSITY-ONLY EXPERIMENT — {case_id}",
        "=" * 74,
        f"  lumen HU (eroded mask)  : mean {mu:.0f}, sd {sd:.0f}",
        f"  band used (mean +/- {k_sd} sd) : {lo:.0f} .. {hi:.0f} HU",
        f"  aorta mask volume       : {sub_mk.sum() * vox_mm3 / 1000:.1f} mL",
        f"  in-band volume          : {band.sum() * vox_mm3 / 1000:.1f} mL "
        f"({band.sum() / max(sub_mk.sum(), 1):.1f}x the aorta)",
        f"  ... touching the aorta  : {connected.sum() * vox_mm3 / 1000:.1f} mL",
        "",
        f"    {'rank':>4}  {'volume mL':>10}  {'mean HU':>8}  {'touches aorta':>13}",
    ]
    for rank, i in enumerate(keep[:15], start=1):
        comp = lab == i
        lines.append(f"    {rank:>4}  {sizes[i] * vox_mm3 / 1000:>10.2f}  "
                     f"{sub_ct[comp].mean():>8.0f}  "
                     f"{('yes' if i in touch else 'no'):>13}")
    lines += [
        "",
        "  If the in-band volume is many times the aorta volume, raw intensity is",
        "  NOT separating arteries from bone, kidneys, bowel or veins. If the",
        "  'touching the aorta' figure is close to aorta + a little, connectivity",
        "  is doing the real work.",
        "=" * 74,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    big = np.isin(lab, keep[:max_components]) if keep else band
    panels = [
        ("A — supplied aorta mask", [(sub_mk, "#d94a4a", 0.95)]),
        (f"B — HU band only ({lo:.0f}..{hi:.0f})",
         [(big & ~sub_mk, "#2f9e9e", 0.55), (sub_mk, "#d94a4a", 0.95)]),
        ("C — HU band, connected to aorta",
         [(connected & ~sub_mk, "#2f9e9e", 0.75), (sub_mk, "#d94a4a", 0.95)]),
    ]
    fig = plt.figure(figsize=(17, 6.5), facecolor="white")
    for col, (title, layers) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        allv = []
        for binary, color, alpha in layers:
            v, f = fine_surface(binary, spacing, cap=120_000)
            if v is None:
                continue
            coll = Poly3DCollection(v[f], alpha=alpha)
            coll.set_facecolor(color)
            coll.set_edgecolor("none")
            ax.add_collection3d(coll)
            allv.append(v)
        if not allv:
            ax.set_axis_off()
            continue
        allv = np.vstack(allv)
        ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
        ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
        ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
        try:
            ax.set_box_aspect([np.ptp(allv[:, k]) for k in range(3)])
        except Exception:
            pass
        ax.view_init(elev=12, azim=-75)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    fig.suptitle(f"{case_id} — can intensity alone find the branches?   "
                 f"red = supplied aorta,  teal = everything else in the band",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


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


def fig_kernel_stride(ct, mk, spacing, out_png, out_txt, case_id,
                      k_sd=2.5, ksize=3, strides=(1, 2), crop=False,
                      margin_mm=60.0, max_components=15, min_component_mm3=30.0):
    """EXPERIMENT: threshold on a k x k in-plane block MEAN instead of voxels."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    lo, hi, mu, sd = lumen_band(ct, mk, k_sd, spacing=spacing)
    pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
           int(round(margin_mm / sx)))
    region = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)

    per_voxel = (sub_ct >= lo) & (sub_ct <= hi)
    block_mean = ndi.uniform_filter(sub_ct, size=(1, ksize, ksize), mode="nearest")
    mean_ok = (block_mean >= lo) & (block_mean <= hi)

    r = ksize // 2
    footprint = np.ones((1, ksize, ksize), bool)
    results = [("per-voxel (stride 1, 1x1)", per_voxel)]
    for st in strides:
        centers = np.zeros_like(mean_ok)
        centers[:, r:-r or None:st, r:-r or None:st] = \
            mean_ok[:, r:-r or None:st, r:-r or None:st]
        results.append((f"{ksize}x{ksize} block mean, stride {st}",
                        ndi.binary_dilation(centers, structure=footprint)))

    lines = [
        "=" * 78,
        f"BLOCK-MEAN KERNEL EXPERIMENT — {case_id}",
        "=" * 78,
        f"  lumen HU band           : {lo:.0f} .. {hi:.0f}  (mean {mu:.0f}, sd {sd:.0f})",
        f"  kernel                  : {ksize}x{ksize} in-plane "
        f"({ksize * sx:.1f} x {ksize * sy:.1f} mm at this spacing)",
        f"  aorta mask volume       : {sub_mk.sum() * vox_mm3 / 1000:.1f} mL",
        "",
        f"  {'variant':<32} {'volume mL':>10} {'components':>11} {'in aorta %':>11}",
    ]
    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    for name, vol in results:
        lab, _ = ndi.label(vol)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        n_comp = int((sizes >= min_vox).sum())
        recall = 100.0 * (vol & sub_mk).sum() / max(sub_mk.sum(), 1)
        lines.append(f"  {name:<32} {vol.sum() * vox_mm3 / 1000:>10.1f} "
                     f"{n_comp:>11d} {recall:>10.1f}%")
    lines += [
        "",
        "  Block averaging kills isolated noise but also dilutes any vessel",
        "  narrower than the kernel, and it ACCEPTS heterogeneous bright tissue",
        "  (trabecular bone, opacified bowel) whose mean lands in the band.",
        "=" * 78,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    dist_out = ndi.distance_transform_edt(~sub_mk, sampling=(sz, sy, sx))
    near = (dist_out > 0) & (dist_out < 20.0)
    score = (per_voxel & near).sum(axis=(1, 2))
    z_show = int(np.argmax(score)) if score.max() > 0 else sub_ct.shape[0] // 2

    n = len(results)
    fig = plt.figure(figsize=(5.6 * n, 10.5), facecolor="white")
    for col, (name, vol) in enumerate(results):
        lab, _ = ndi.label(vol)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]
        big = np.isin(lab, keep[:max_components]) if keep else vol

        ax = fig.add_subplot(2, n, col + 1, projection="3d")
        allv = []
        for binary, color, alpha in ((big & ~sub_mk, "#2f9e9e", 0.6),
                                     (sub_mk, "#d94a4a", 0.95)):
            v, f = fine_surface(binary, spacing, cap=120_000)
            if v is None:
                continue
            coll = Poly3DCollection(v[f], alpha=alpha)
            coll.set_facecolor(color)
            coll.set_edgecolor("none")
            ax.add_collection3d(coll)
            allv.append(v)
        if allv:
            allv = np.vstack(allv)
            ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
            ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
            ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
            try:
                ax.set_box_aspect([np.ptp(allv[:, k]) for k in range(3)])
            except Exception:
                pass
        ax.view_init(elev=12, azim=-75)
        ax.set_title(name, fontsize=11)
        ax.set_axis_off()

        ax2 = fig.add_subplot(2, n, n + col + 1)
        ax2.imshow(window(sub_ct[z_show]), cmap="gray", vmin=0, vmax=1,
                   aspect=sy / sx)
        overlay = np.zeros(sub_ct[z_show].shape + (4,), np.float32)
        overlay[vol[z_show]] = (0.18, 0.62, 0.62, 0.55)
        overlay[sub_mk[z_show]] = (0.85, 0.29, 0.29, 0.55)
        ax2.imshow(overlay, aspect=sy / sx, interpolation="nearest")
        ax2.set_title(f"axial slice {z_show}", fontsize=10)
        ax2.axis("off")

    fig.suptitle(
        f"{case_id} — block-mean thresholding, kernel {ksize}x{ksize}, "
        f"band {lo:.0f}-{hi:.0f} HU\nred = supplied aorta, "
        f"teal = accepted as lumen-like", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


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


def grow_bridge(sub_ct, sub_mk, comp, ostium_vox, spacing, lo, hi,
                corridor_mm=3.0, max_gap_mm=15.0):
    """Segment the ACTUAL voxels between a candidate and the aortic wall."""
    sx, sy, sz = spacing
    sp = np.array([sz, sy, sx])
    if ostium_vox is None:
        return np.zeros_like(comp)

    pts = np.argwhere(comp).astype(float)
    d = np.linalg.norm((pts - np.asarray(ostium_vox, float)) * sp, axis=1)
    start = pts[int(np.argmin(d))]
    end = np.asarray(ostium_vox, float)

    n = max(int(np.linalg.norm((end - start) * sp) / 0.3), 2)
    path = start[None, :] + (end - start)[None, :] * np.linspace(0, 1, n)[:, None]

    corridor = np.zeros(comp.shape, bool)
    idx = np.round(path).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(comp.shape)), axis=1)
    idx = idx[ok]
    if len(idx) == 0:
        return corridor
    corridor[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    it = max(int(round(corridor_mm / min(sx, sy))), 1)
    corridor = ndi.binary_dilation(corridor, iterations=it)

    bridge = corridor & (sub_ct >= lo) & (sub_ct <= hi) & ~sub_mk
    seed = comp | ndi.binary_dilation(sub_mk, iterations=1)
    lab_b, _ = ndi.label(bridge | seed)
    keep_ids = set(np.unique(lab_b[comp])) - {0}
    return np.isin(lab_b, list(keep_ids)) & bridge


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


def fig_frangi(ct, mk, spacing, out_png, out_txt, out_csv, case_id,
               sigmas_mm=(0.7, 1.0, 1.5, 2.0, 3.0), roi_mm=30.0,
               vess_pct=None, vess_frac=0.30, crop=True, margin_mm=40.0,
               min_component_mm3=8.0, max_components=20,
               out_html=None, img=None, bone_floor_hu=None,
               alpha=0.5, beta=0.5, max_gap_mm=15.0, proximal_mm=12.0,
               out_json=None, hu_tol=45.0, merge_mm=6.0,
               end_margin_mm=5.0):
    """Frangi vesselness around the aorta, then per-candidate features to CSV.

    The CSV is the handoff to a classifier: one row per connected candidate, with
    the geometric and intensity features a kNN (or anything else) would use.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import csv as _csv

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
           int(round(margin_mm / sx)))
    region = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)

    lo, hi, mu, sd = lumen_band(ct, mk, spacing=spacing)

    roi, dist_out, bone, hi_cut = branch_roi(
        sub_ct, sub_mk, spacing, roi_mm, mu, sd, bone_floor_hu=bone_floor_hu)

    t0 = __import__("time").time()
    vess, scale = frangi_3d(sub_ct, spacing, sigmas_mm, roi=roi,
                            alpha=alpha, beta=beta)
    elapsed = __import__("time").time() - t0

    # A percentile of the ROI is unstable: once bone is excluded the ROI is
    # mostly vessel, so "top 1%" throws the vessels away. Anchor to the strength
    # of the response instead of to ROI composition.
    thr = (vess_frac * float(np.percentile(vess[roi], 99.5))
           if roi.any() else 0.0) if vess_pct is None else \
        (float(np.percentile(vess[roi], vess_pct)) if roi.any() else 0.0)
    cand = (vess >= thr) & roi
    cand = ndi.binary_opening(cand, iterations=1)

    lab, _ = ndi.label(cand)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]

    wall = ndi.binary_dilation(sub_mk, iterations=2) & ~sub_mk
    com_aorta = np.array(ndi.center_of_mass(sub_mk))          # (z, y, x)

    # principal axis of the parent aorta -- a companion vein runs along it,
    # a daughter leaves across it.
    _ap = np.argwhere(sub_mk).astype(float) * np.array([sz, sy, sx])
    _ap -= _ap.mean(axis=0)
    _, _, _av = np.linalg.svd(_ap[::max(len(_ap) // 20000, 1)],
                              full_matrices=False)
    aorta_axis = _av[0] / max(np.linalg.norm(_av[0]), 1e-9)

    # ---- per-candidate features -> CSV (input for your classifier) ----
    z0, y0, x0 = (region[0].start or 0, region[1].start or 0, region[2].start or 0)
    rows = []
    for rank, i in enumerate(keep[:200], start=1):
        comp = lab == i
        pts = np.argwhere(comp).astype(np.float64)            # (z, y, x) voxels
        com = pts.mean(axis=0)
        pts_mm = pts * np.array([sz, sy, sx])
        centred = pts_mm - pts_mm.mean(axis=0)
        # principal axis + how tube-like the component is
        if len(pts) >= 3:
            _, svals, vecs = np.linalg.svd(centred, full_matrices=False)
            axis = vecs[0]
            elong = float(svals[0] / max(svals[1], 1e-6))
        else:
            axis, elong = np.array([0.0, 0.0, 1.0]), 1.0
        # how long the component actually is, along its own axis
        _proj = centred @ (axis / max(np.linalg.norm(axis), 1e-9))
        length_mm = float(_proj.max() - _proj.min())
        parallel = float(abs(np.dot(axis / max(np.linalg.norm(axis), 1e-9),
                                    aorta_axis)))
        # radial direction: from the aorta centroid outwards, in-plane
        radial = com - com_aorta
        radial_mm = radial * np.array([sz, sy, sx])
        nrm = np.linalg.norm(radial_mm)
        radial_u = radial_mm / nrm if nrm > 1e-6 else np.array([0.0, 1.0, 0.0])
        radiality = float(abs(np.dot(axis / np.linalg.norm(axis), radial_u)))
        # does this candidate's own axis run back into the aortic wall?
        hit, gap_mm, ost_vox, signed_u = link_to_aorta(
            pts, dist_out, np.asarray(axis, float), spacing,
            max_gap_mm=max_gap_mm, sub_mk=sub_mk)

        # ---- proximal segment only -------------------------------------
        # The task scores the first 10 mm past the ostium, nothing beyond. Whole
        # component features are contaminated by distal anatomy the scoring never
        # sees -- the SMA is the clean example: it leaves the aorta anteriorly,
        # then runs 10+ cm caudally in the mesentery and fans into jejunal
        # branches. Judged whole it looks like a companion vein; judged over its
        # first centimetre it is an ordinary radial daughter.
        prox = seed_vox = None
        prox_len = prox_travel = prox_hu = prox_rad = float("nan")
        prox_axis = np.asarray(axis, float)
        if hit and ost_vox is not None:
            o_mm = np.asarray(ost_vox, float) * np.array([sz, sy, sx])
            d_from_ost = np.linalg.norm(pts_mm - o_mm, axis=1)
            sel = d_from_ost <= proximal_mm
            if sel.sum() >= 5:
                prox = pts[sel]
                pm = pts_mm[sel] - pts_mm[sel].mean(axis=0)
                _, _, pv = np.linalg.svd(pm, full_matrices=False)
                prox_axis = pv[0] / max(np.linalg.norm(pv[0]), 1e-9)
                # orient outward, away from the ostium
                if np.dot(pts_mm[sel].mean(axis=0) - o_mm, prox_axis) < 0:
                    prox_axis = -prox_axis
                pr = pm @ prox_axis
                prox_len = float(pr.max() - pr.min())
                dsel = dist_out[tuple(prox.T.astype(int))]
                prox_travel = round(prox_len / max(
                    float(dsel.max() - dsel.min()), 0.5), 1)
                prox_hu = float(sub_ct[tuple(prox.T.astype(int))].mean())
                # seed: lumen centre ~5 mm out along the proximal path
                band = prox[(d_from_ost[sel] >= 3.5) & (d_from_ost[sel] <= 6.5)]
                if len(band):
                    seed_vox = band.mean(axis=0)
                    edt = ndi.distance_transform_edt(comp, sampling=(sz, sy, sx))
                    prox_rad = float(ndi.map_coordinates(
                        edt, np.asarray(seed_vox, float)[:, None], order=1)[0])

        idx_phys = (int(round(com[2])) + x0, int(round(com[1])) + y0,
                    int(round(com[0])) + z0)
        rows.append(dict(
            candidate_id=f"cand_{rank:03d}",
            volume_mm3=round(float(sizes[i] * vox_mm3), 2),
            mean_hu=round(float(sub_ct[comp].mean()), 1),
            hu_vs_lumen=round(float(sub_ct[comp].mean() - mu), 1),
            max_vesselness=round(float(vess[comp].max()), 4),
            mean_vesselness=round(float(vess[comp].mean()), 4),
            best_sigma_mm=round(float(np.median(scale[comp])), 2),
            min_dist_to_aorta_mm=round(float(dist_out[comp].min()), 2),
            max_dist_to_aorta_mm=round(float(dist_out[comp].max()), 2),
            extent_mm=round(float(dist_out[comp].max() - dist_out[comp].min()), 2),
            length_mm=round(length_mm, 1),
            # length travelled per mm of outward progress. A daughter leaves the
            # aorta, so this is near 1-2. A companion vein (IVC, renal vein,
            # ureter) runs 100+ mm while never getting further away -> huge.
            travel_ratio=round(length_mm / max(
                float(dist_out[comp].max() - dist_out[comp].min()), 0.5), 1),
            parallel_to_aorta=round(parallel, 3),
            prox_length_mm=round(prox_len, 1) if prox_len == prox_len else "",
            prox_travel_ratio=prox_travel if prox_travel == prox_travel else "",
            prox_mean_hu=round(prox_hu, 0) if prox_hu == prox_hu else "",
            radius_mm=round(prox_rad, 2) if prox_rad == prox_rad else "",
            elongation=round(elong, 2),
            radiality=round(radiality, 3),
            # Frangi is weak exactly at the ostium (a T-junction is not a tube),
            # so "touches the wall" is graded by distance rather than by overlap.
            axis_hits_aorta=int(bool(hit)),
            gap_to_wall_mm=round(float(gap_mm), 2) if hit else "",
            starts_near_wall=int(float(dist_out[comp].min()) <= 3.0),
            overlaps_wall_shell=int(bool((comp & wall).any())),
            n_voxels=int(sizes[i]),
            vox_z=idx_phys[2], vox_y=idx_phys[1], vox_x=idx_phys[0],
            label="",                       # <- you fill this in: 1 = real daughter
        ))
        if img is not None:
            # physical mm via SimpleITK -- the coordinate system your output needs
            px, py, pz = img.TransformIndexToPhysicalPoint(
                (int(idx_phys[0]), int(idx_phys[1]), int(idx_phys[2])))
            rows[-1].update(x_mm=round(px, 2), y_mm=round(py, 2), z_mm=round(pz, 2))
            if hit and ost_vox is not None:
                ox, oy, oz = img.TransformIndexToPhysicalPoint(
                    (int(round(ost_vox[2])) + x0, int(round(ost_vox[1])) + y0,
                     int(round(ost_vox[0])) + z0))
                rows[-1].update(ostium_x_mm=round(ox, 2), ostium_y_mm=round(oy, 2),
                                ostium_z_mm=round(oz, 2),
                                dir_x=round(float(prox_axis[2]), 3),
                                dir_y=round(float(prox_axis[1]), 3),
                                dir_z=round(float(prox_axis[0]), 3))
                if seed_vox is not None:
                    sxx, syy, szz = img.TransformIndexToPhysicalPoint(
                        (int(round(seed_vox[2])) + x0,
                         int(round(seed_vox[1])) + y0,
                         int(round(seed_vox[0])) + z0))
                    rows[-1].update(seed_x_mm=round(sxx, 2),
                                    seed_y_mm=round(syy, 2),
                                    seed_z_mm=round(szz, 2))
                else:
                    rows[-1].update(seed_x_mm="", seed_y_mm="", seed_z_mm="")
            else:
                rows[-1].update(ostium_x_mm="", ostium_y_mm="", ostium_z_mm="",
                                dir_x="", dir_y="", dir_z="",
                                seed_x_mm="", seed_y_mm="", seed_z_mm="")
        rows[-1]["_label_id"] = i
        rows[-1]["_ostium_vox"] = ost_vox if hit else None
        rows[-1]["_com_vox"] = com
        rows[-1]["_near_vox"] = pts[int(np.argmin(dist_out[comp]))]

    if rows:
        cols = [k for k in rows[0] if not k.startswith("_")]
        with open(out_csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    lines = [
        "=" * 82,
        f"FRANGI VESSELNESS — {case_id}",
        "=" * 82,
        f"  scales (mm)             : {', '.join(str(s) for s in sigmas_mm)}",
        f"  lumen HU                : mean {mu:.0f}, sd {sd:.0f}"
        + ("   *** LOW -- weak contrast, see note below ***" if mu < 200 else ""),
        f"  ROI                     : 0 < d <= {roi_mm:.0f} mm outside the aorta, "
        f"{mu - 4 * sd:.0f} < HU < {hi_cut:.0f}",
        f"  bone/calcium excluded   : {bone.sum() * vox_mm3 / 1000:.1f} mL "
        f"(> {hi_cut:.0f} HU, dilated 2 mm)",
        f"  ROI voxels              : {int(roi.sum()):,}  "
        f"({roi.sum() * vox_mm3 / 1000:.0f} mL)",
        f"  filter time             : {elapsed:.1f}s",
        f"  vesselness threshold    : {thr:.4f}  "
        f"({'percentile ' + str(vess_pct) if vess_pct else str(vess_frac) + ' x p99.5'})",
        f"  candidates >{min_component_mm3:.0f} mm3      : {len(keep)}",
        f"  feature table           : {os.path.basename(out_csv)} "
        f"({min(len(keep), 200)} rows)",
        "",
        f"  {'rank':>4} {'mm3':>8} {'HU':>6} {'vess':>7} {'sigma':>6} "
        f"{'d_min':>6} {'extent':>7} {'elong':>6} {'radial':>7} {'wall':>5}",
    ]
    for r in rows[:15]:
        lines.append(
            f"  {r['candidate_id'][-3:]:>4} {r['volume_mm3']:>8.1f} {r['mean_hu']:>6.0f} "
            f"{r['max_vesselness']:>7.3f} {r['best_sigma_mm']:>6.1f} "
            f"{r['min_dist_to_aorta_mm']:>6.1f} {r['extent_mm']:>7.1f} "
            f"{r['elongation']:>6.1f} {r['radiality']:>7.2f} {r['starts_near_wall']:>5d}")
    lines += [
        "",
        "  Columns that should separate real daughters from junk:",
        "    starts_near_wall  daughter begins at the wall (d_min <= 3 mm). Frangi",
        "                  is weak AT the ostium -- a T-junction is not a tube --",
        "                  so expect d_min of 1-4 mm even for real branches.",
        "    extent_mm     it must be followable >= 5 mm to be eligible",
        "    radiality     a branch leaves radially; a vein runs alongside",
        "    hu_vs_lumen   arterial contrast matches the aorta; veins are lower",
        "    best_sigma_mm the scale that responded = rough calibre",
        "",
        "  The 'label' column is empty on purpose. Fill it with 1/0 on a few cases",
        "  and that file is your classifier's training set.",
        "",
        "  If lumen HU is flagged LOW above (< ~200), this scan is not strongly",
        "  arterial-enhanced and the whole intensity-band approach is on thin ice.",
        "  Check 00_report.txt: a well-timed abdominal CTA sits at 250-400 HU in",
        "  the aorta. If yours is at 120, either the timing was late/venous or the",
        "  volume is not in Hounsfield units at all (missing rescale slope and",
        "  intercept from the DICOM conversion) -- verify before tuning anything.",
        "=" * 82,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    # ---------------- challenge JSON ----------------
    if out_json:
        zs_mask = np.where(sub_mk.any(axis=(1, 2)))[0]
        z_lo_mm = float(zs_mask[0] * sz) if len(zs_mask) else -1e9
        z_hi_mm = float(zs_mask[-1] * sz) if len(zs_mask) else 1e9

        picks = []
        for r in rows:
            if not r.get("axis_hits_aorta"):
                continue
            if r.get("ostium_x_mm", "") == "":
                continue
            # arterial contrast: veins sit far below the aortic lumen
            if abs(float(r["hu_vs_lumen"])) > hu_tol:
                continue
            # eligibility: followable at least 5 mm past the wall
            pl = r.get("prox_length_mm", "")
            if pl != "" and float(pl) < 5.0:
                continue
            # the flat cropped caps are not branch origins
            zc = float(r["_ostium_vox"][0]) * sz
            if (zc - z_lo_mm) < end_margin_mm or (z_hi_mm - zc) < end_margin_mm:
                continue
            picks.append(r)

        # one-to-one matching means duplicates are false positives: merge ostia
        # that sit within merge_mm of each other, keeping the stronger response.
        picks.sort(key=lambda r: -float(r["max_vesselness"]))
        kept = []
        for r in picks:
            p = np.array([float(r["ostium_x_mm"]), float(r["ostium_y_mm"]),
                          float(r["ostium_z_mm"])])
            if any(np.linalg.norm(p - k[1]) < merge_mm for k in kept):
                continue
            kept.append((r, p))

        kept.sort(key=lambda k: -k[1][2])          # cranial -> caudal
        daughters = []
        for n, (r, p) in enumerate(kept, start=1):
            d = np.array([float(r["dir_x"] or 0), float(r["dir_y"] or 0),
                          float(r["dir_z"] or 0)], float)
            nrm = np.linalg.norm(d)
            d = (d / nrm) if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
            seed = ([float(r["seed_x_mm"]), float(r["seed_y_mm"]),
                     float(r["seed_z_mm"])]
                    if r.get("seed_x_mm", "") != ""
                    else list(np.round(p + d * 5.0, 3)))
            daughters.append({
                "instance_id": f"branch_{n:03d}",
                "parent_instance_id": "aorta",
                "ostium_xyz_mm": [round(float(v), 3) for v in p],
                "seed_xyz_mm": [round(float(v), 3) for v in seed],
                "radius_mm": (round(float(r["radius_mm"]), 3)
                              if r.get("radius_mm", "") != "" else None),
                "direction_xyz": [round(float(v), 4) for v in d],
                "_source_candidate": r["candidate_id"],
            })

        import json as _json
        with open(out_json, "w") as fh:
            _json.dump({"case_id": case_id,
                        "parent": {"instance_id": "aorta"},
                        "daughters": daughters}, fh, indent=2)
        print(f"  wrote {out_json}  ({len(daughters)} daughters from "
              f"{len(rows)} candidates)")

    # ---------------- figure ----------------
    def surface_capped(binary, cap=250_000):
        return fine_surface(binary, spacing, cap=cap)

    big = np.isin(lab, keep[:max_components]) if keep else cand
    z_show = int(np.argmax((cand & ~sub_mk).sum(axis=(1, 2)))) if cand.any() \
        else sub_ct.shape[0] // 2

    fig = plt.figure(figsize=(17, 10.5), facecolor="white")

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(vess.max(axis=1), cmap="magma", aspect=sz / sx, origin="lower",
              vmin=0, vmax=max(thr * 2, 1e-6))
    ax.contour(sub_mk.max(axis=1), levels=[0.5], colors=["#57c7ff"],
               linewidths=0.8, origin="lower")
    ax.set_title("vesselness, coronal MIP", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(vess.max(axis=2), cmap="magma", aspect=sz / sy, origin="lower",
              vmin=0, vmax=max(thr * 2, 1e-6))
    ax.contour(sub_mk.max(axis=2), levels=[0.5], colors=["#57c7ff"],
               linewidths=0.8, origin="lower")
    ax.set_title("vesselness, sagittal MIP", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(window(sub_ct[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
    ov = np.zeros(sub_ct[z_show].shape + (4,), np.float32)
    ov[bone[z_show]] = (0.35, 0.45, 0.95, 0.30)          # excluded as bone
    ov[roi[z_show]] = (0.20, 0.80, 0.45, 0.22)           # searched
    ov[cand[z_show]] = (1.0, 0.78, 0.25, 0.95)           # candidates
    ov[sub_mk[z_show]] = (0.85, 0.29, 0.29, 0.45)
    ax.imshow(ov, aspect=sy / sx, interpolation="nearest")
    ax.set_title(f"slice {z_show}: blue = excluded bone,\n"
                 f"green = searched, gold = candidates", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(2, 1, 2, projection="3d")
    allv = []
    for binary, color, alpha in ((big, "#f2c14e", 0.95), (sub_mk, "#d94a4a", 0.45)):
        v, f = surface_capped(binary)
        if v is None:
            continue
        coll = Poly3DCollection(v[f], alpha=alpha)
        coll.set_facecolor(color)
        coll.set_edgecolor("none")
        ax.add_collection3d(coll)
        allv.append(v)
    if allv:
        allv = np.vstack(allv)
        ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
        ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
        ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
        try:
            ax.set_box_aspect([np.ptp(allv[:, k]) for k in range(3)])
        except Exception:
            pass
    ax.view_init(elev=14, azim=-75)
    ax.set_title(f"top {min(max_components, len(keep))} vesselness candidates "
                 f"(gold) on the aorta (red)", fontsize=11)
    ax.set_axis_off()

    fig.suptitle(f"{case_id} — Hessian/Frangi vesselness, scales "
                 f"{min(sigmas_mm)}-{max(sigmas_mm)} mm, "
                 f"{len(keep)} candidates above threshold {thr:.3f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)

    # ---------------- interactive 3D ----------------
    if out_html:
        # put vertices in (approximately) physical mm: crop offset + image origin.
        # Exact for axis-aligned volumes, which abdominal CT essentially always is.
        off = np.zeros(3)
        if img is not None:
            try:
                off = np.array(img.TransformIndexToPhysicalPoint(
                    (int(x0), int(y0), int(z0))), dtype=float)
            except Exception:
                off = np.zeros(3)
        note = "" if img is not None else "  [relative]"

        meshes = []
        v, f = _surface(sub_mk, spacing, 1.2)
        if v is not None:
            meshes.append(dict(verts=v + off, faces=f, name="aorta (supplied mask)",
                               color="#d94a4a", opacity=0.35,
                               text="parent aorta<br>supplied mask"))

        palette = ["#f2c14e", "#2f9e9e", "#7b6cd9", "#e07a5f", "#3fa34d",
                   "#c85b9b", "#4f86c6", "#b5852a", "#5bbfa5", "#d1495b"]
        for n, r in enumerate(rows[:max_components]):
            comp = lab == r["_label_id"]
            v, f = fine_surface(comp, spacing, cap=120_000)
            if v is None:
                continue
            pos = (f"({r['x_mm']}, {r['y_mm']}, {r['z_mm']}) mm"
                   if "x_mm" in r else f"voxel ({r['vox_x']}, {r['vox_y']}, {r['vox_z']})")
            txt = (f"<b>{r['candidate_id']}</b><br>"
                   f"volume {r['volume_mm3']} mm&#179;<br>"
                   f"mean {r['mean_hu']:.0f} HU ({r['hu_vs_lumen']:+.0f} vs lumen)<br>"
                   f"vesselness {r['max_vesselness']:.3f} at sigma "
                   f"{r['best_sigma_mm']} mm<br>"
                   f"starts {r['min_dist_to_aorta_mm']} mm from wall, "
                   f"extends {r['extent_mm']} mm<br>"
                   f"elongation {r['elongation']}, radiality {r['radiality']}<br>"
                   f"length {r['length_mm']} mm, travel ratio {r['travel_ratio']}<br>"
                   f"parallel to aorta {r['parallel_to_aorta']}<br>"
                   f"centroid {pos}")
            linked = bool(r.get("axis_hits_aorta"))
            meshes.append(dict(
                verts=v + off, faces=f,
                name=("* " if linked else "  ") + f"{r['candidate_id']}  "
                     f"(len {r['length_mm']:.0f}mm, travel {r['travel_ratio']}, "
                     f"r={r['radiality']:.2f})",
                color=palette[n % len(palette)], opacity=0.95,
                text=txt,
                # start clean: only candidates whose axis reaches the wall.
                # everything else is one legend click away.
                visible=True if linked else "legendonly"))

        # inferred ostia + the link segments that produced them
        vox_mm = np.array([sz, sy, sx])
        crop0 = np.array([z0, y0, x0], float)
        opts, olab, lseg = [], [], []
        bridges = np.zeros_like(sub_mk, bool)
        for r in rows[:max_components]:
            ov = r.get("_ostium_vox")
            if ov is None:
                continue
            o = (np.asarray(ov, float) * vox_mm)[[2, 1, 0]] + off
            c = (np.asarray(r.get("_near_vox", r["_com_vox"]), float)
                 * vox_mm)[[2, 1, 0]] + off
            opts.append(o)
            olab.append(f"<b>{r['candidate_id']}</b> ostium<br>"
                        f"gap bridged {r['gap_to_wall_mm']} mm<br>"
                        f"radiality {r['radiality']}")
            lseg += [c, o, [np.nan, np.nan, np.nan]]
            br = grow_bridge(sub_ct, sub_mk, lab == r["_label_id"], ov,
                             spacing, lo, hi)
            if br.any():
                bridges |= br
        if bridges.any():
            bv, bf = fine_surface(bridges, spacing, cap=80_000)
            if bv is not None:
                meshes.append(dict(
                    verts=bv + off, faces=bf,
                    name=f"bridging lumen ({bridges.sum() * vox_mm3 / 1000:.2f} mL)",
                    color="#111111", opacity=0.9,
                    text="in-band voxels found between a candidate and the wall"))
        if opts:
            meshes.append(dict(kind="points", points=np.array(opts, float),
                               name=f"inferred ostia ({len(opts)})",
                               color="#111111", size=7, labels=olab))
            meshes.append(dict(kind="lines",
                               points=np.asarray(lseg, dtype=float),
                               name="axis -> wall links", color="#111111", width=4))

        # dependency-free fallback: surfaces as OBJ, openable in any 3D viewer
        obj_dir = os.path.join(os.path.dirname(out_html), "meshes")
        os.makedirs(obj_dir, exist_ok=True)
        surf = [m for m in meshes if "verts" in m]
        for m in surf:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          m["name"].split("(")[0].strip()).strip("_") or "surface"
            safe = safe[:60]     # Windows rejects *, ?, :, and very long names
            write_obj(os.path.join(obj_dir, f"{safe}.obj"),
                      m["verts"], m["faces"], name=safe)
        print(f"  wrote {obj_dir}/ ({len(surf)} .obj surfaces)")

        ok = write_interactive_html(
            out_html, meshes,
            title=f"{case_id} — aorta and vesselness candidates",
            subtitle=f"{len(keep)} candidates above threshold {thr:.3f} · "
                     f"scales {min(sigmas_mm)}–{max(sigmas_mm)} mm · "
                     f"drag to rotate, scroll to zoom, click legend entries to "
                     f"isolate a candidate, hover a surface for its features",
            axis_note=note)
        if ok:
            print(f"  wrote {out_html}")


# -----------------------------------------------------------------------------
# MedSAM
# -----------------------------------------------------------------------------

def kernel_proxy_variants(vol, spacing, unsharp_sigma_mm=0.8):
    """Image-domain SHARPNESS PROXIES for CT reconstruction kernels.

    IMPORTANT, and the reason this is named "proxy": a reconstruction kernel
    (Siemens Bv40, Bv60, B30f, ...) is applied to the raw projection data during
    reconstruction. It cannot be applied to an already reconstructed volume.
    Smoothing here mimics a smoother kernel fairly well; sharpening does NOT
    reproduce a sharper kernel, because detail the original kernel discarded is
    not in the voxels any more -- you only amplify what survived, noise included.

    What this IS good for: sensitivity analysis. If your detector's output swings
    across this sweep, it will also swing across scanners in the hidden test set.
    """
    sx, sy, sz = spacing
    sp = (sz, sy, sx)

    def gauss(s_mm):
        return ndi.gaussian_filter(vol, tuple(s_mm / p for p in sp), mode="nearest")

    def unsharp(amount):
        return vol + amount * (vol - gauss(unsharp_sigma_mm))

    return [("smoother (sigma 1.0 mm)", gauss(1.0)),
            ("smooth (sigma 0.5 mm)", gauss(0.5)),
            ("as acquired", vol),
            ("sharper (unsharp 0.7)", unsharp(0.7)),
            ("sharpest (unsharp 1.5)", unsharp(1.5))]


def fig_kernel_sweep(ct, mk, spacing, out_png, out_csv, out_html, case_id,
                     sigmas_mm=(1.0, 1.5, 2.0), roi_mm=25.0, vess_pct=None, vess_frac=0.30,
                     margin_mm=40.0, min_component_mm3=8.0, top_n=8):
    """How sensitive is vesselness detection to image sharpness?"""
    import csv as _csv
    import time

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
           int(round(margin_mm / sx)))
    region = mask_bbox(mk, pad, crop=True)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)

    lo, hi, mu, sd = lumen_band(ct, mk, spacing=spacing)
    roi, dist_out, bone, hi_cut = branch_roi(sub_ct, sub_mk, spacing,
                                             roi_mm, mu, sd)
    core = ndi.binary_erosion(sub_mk, iterations=3)
    wall = ndi.binary_dilation(sub_mk, iterations=1) & ~ndi.binary_erosion(sub_mk)

    variants = kernel_proxy_variants(sub_ct, spacing)
    rows, per_variant = [], []

    for name, v in variants:
        t0 = time.time()
        noise = float(v[core].std()) if core.any() else float("nan")
        gz, gy, gx = np.gradient(v, sz, sy, sx)
        gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
        sharp = float(gmag[wall].mean()) if wall.any() else float("nan")
        del gz, gy, gx, gmag

        vess, _ = frangi_3d(v, spacing, sigmas_mm, roi=roi)
        thr = (vess_frac * float(np.percentile(vess[roi], 99.5))
               if vess_pct is None
               else float(np.percentile(vess[roi], vess_pct))) if roi.any() else 0.0
        cand = ndi.binary_opening((vess >= thr) & roi, iterations=1)
        lab, _ = ndi.label(cand)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        min_vox = max(int(min_component_mm3 / vox_mm3), 1)
        keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]

        # Run the REAL pipeline per variant: link each candidate back to the wall
        # and keep the inferred ostia. Counting raw components tells you nothing
        # about the output you actually submit.
        extents, coms = [], []
        for i in keep[:top_n]:
            comp = lab == i
            extents.append(float(dist_out[comp].max() - dist_out[comp].min()))
            pts = np.argwhere(comp).astype(float)
            pmm = pts * np.array([sz, sy, sx])
            cen = pmm - pmm.mean(axis=0)
            if len(pts) >= 3:
                _, _, vv = np.linalg.svd(cen[::max(len(cen) // 4000, 1)],
                                         full_matrices=False)
                ax_i = vv[0]
            else:
                ax_i = np.array([0.0, 0.0, 1.0])
            hit, _g, ost, _u = link_to_aorta(pts, dist_out, ax_i, spacing,
                                             sub_mk=sub_mk)
            if hit and ost is not None:
                coms.append(np.asarray(ost, float) * np.array([sz, sy, sx]))

        outside = roi & ~sub_mk
        rows.append(dict(
            variant=name,
            lumen_noise_sd_hu=round(noise, 1),
            wall_gradient_hu_per_mm=round(sharp, 1),
            cnr=round(float(abs(mu - v[outside].mean()) / max(noise, 1e-6)), 2)
            if outside.any() else 0.0,
            n_candidates=len(keep),
            n_ostia=len(coms),
            vess_threshold=round(thr, 4),
            median_extent_mm=round(float(np.median(extents)) if extents else 0.0, 1),
            max_extent_mm=round(float(max(extents)) if extents else 0.0, 1),
            runtime_s=round(time.time() - t0, 1)))
        per_variant.append((name, lab, keep, coms, vess))

    with open(out_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ref = per_variant[2][3]
    stability = []
    for _, _, _, coms, _ in per_variant:
        if not ref or not coms:
            stability.append(0.0)
            continue
        matched = sum(1 for r in ref
                      if min(np.linalg.norm(r - c) for c in coms) <= 5.0)
        stability.append(100.0 * matched / len(ref))

    print("\n" + "=" * 80)
    print(f"SHARPNESS SENSITIVITY SWEEP — {case_id}")
    print("=" * 80)
    print("  NOTE: image-domain sharpness proxies, NOT reconstruction kernels.")
    print("  A real Bv kernel acts on raw projection data, before a volume exists.")
    print("  Read this as a robustness test, not a kernel comparison.\n")
    print(f"  {'variant':<24}{'noise':>7}{'wallgrad':>10}{'cands':>7}"
          f"{'OSTIA':>7}{'agree%':>8}{'sec':>6}")
    for r, st in zip(rows, stability):
        print(f"  {r['variant']:<24}{r['lumen_noise_sd_hu']:>7.1f}"
              f"{r['wall_gradient_hu_per_mm']:>10.1f}{r['n_candidates']:>7d}"
              f"{r['n_ostia']:>7d}{st:>7.0f}%{r['runtime_s']:>6.1f}")
    print("\n  OSTIA = inferred branch origins, i.e. what actually reaches your")
    print("  prediction file. agree% = share of the 'as acquired' OSTIA that this")
    print("  variant recovers within 5 mm -- the same 5 mm tolerance a scorer")
    print("  would plausibly use. This is a label-free robustness number you can")
    print("  quote in the demo: it says how much your output depends on image")
    print("  sharpness, which is what varies across scanners and protocols.")
    print("=" * 80)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # show the slice with the most DETECTED candidate voxels, not the most ROI
    # voxels -- the latter just finds the slice nearest the vertebra.
    ref_lab, ref_keep = per_variant[2][1], per_variant[2][2]
    ref_cand = np.isin(ref_lab, ref_keep[:top_n]) if ref_keep else roi
    z_show = int(np.argmax(ref_cand.sum(axis=(1, 2)))) if ref_cand.any() else \
        sub_ct.shape[0] // 2
    n = len(variants)
    fig = plt.figure(figsize=(3.6 * n, 11.5), facecolor="white")

    # cache the aorta surface once -- it is the same in every panel
    av, af = _surface(sub_mk, spacing, 1.4)

    for col, ((name, v), (_, lab, keep, _, vess)) in enumerate(
            zip(variants, per_variant)):
        # row 1: the image itself
        ax = fig.add_subplot(3, n, col + 1)
        ax.imshow(window(v[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
        ax.contour(sub_mk[z_show], levels=[0.5], colors=["#ff4d4d"], linewidths=0.8)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

        # row 2: the STRUCTURES this variant actually found, in 3D
        ax = fig.add_subplot(3, n, n + col + 1, projection="3d")
        allv = []
        if av is not None:
            coll = Poly3DCollection(av[af], alpha=0.25)
            coll.set_facecolor("#d94a4a")
            coll.set_edgecolor("none")
            ax.add_collection3d(coll)
            allv.append(av)
        if keep:
            cv, cf = _surface(np.isin(lab, keep[:top_n]), spacing, 0.9)
            if cv is not None:
                coll = Poly3DCollection(cv[cf], alpha=0.98)
                coll.set_facecolor("#f2c14e")
                coll.set_edgecolor("none")
                ax.add_collection3d(coll)
                allv.append(cv)
        if allv:
            stacked = np.vstack(allv)
            ax.set_xlim(stacked[:, 0].min(), stacked[:, 0].max())
            ax.set_ylim(stacked[:, 1].min(), stacked[:, 1].max())
            ax.set_zlim(stacked[:, 2].min(), stacked[:, 2].max())
            try:
                ax.set_box_aspect([np.ptp(stacked[:, k]) for k in range(3)])
            except Exception:
                pass
        ax.view_init(elev=12, azim=-75)
        ax.set_title(f"{len(keep)} candidates / {rows[col]['n_ostia']} ostia",
                     fontsize=9)
        ax.set_axis_off()

    labels = [r["variant"].split("(")[0].strip() for r in rows]
    for k, (key, txt, colr) in enumerate(
            [("lumen_noise_sd_hu", "noise in lumen (HU sd)", "#d94a4a"),
             ("wall_gradient_hu_per_mm", "wall gradient (HU/mm)", "#2f9e9e"),
             ("n_ostia", "inferred ostia", "#7b6cd9")]):
        ax = fig.add_subplot(3, n, 2 * n + 1 + k)
        ax.plot(range(len(rows)), [r[key] for r in rows], "o-", color=colr, lw=2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(txt, fontsize=10)
        ax.grid(alpha=0.25)
        if key == "n_ostia":
            ax2 = ax.twinx()
            ax2.plot(range(len(rows)), stability, "s--", color="#888", lw=1.2)
            ax2.set_ylabel("agree % vs as-acquired", fontsize=8, color="#888")
            ax2.set_ylim(0, 105)

    fig.suptitle(f"{case_id} — sharpness sensitivity sweep "
                 f"(image-domain proxies, NOT reconstruction kernels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)

    if out_html:
        palette = ["#8fb8de", "#5b9bd5", "#d94a4a", "#f2c14e", "#e07a5f"]
        meshes = []
        v, f = _surface(sub_mk, spacing, 1.2)
        if v is not None:
            meshes.append(dict(verts=v, faces=f, name="aorta", color="#bbbbbb",
                               opacity=0.3, text="parent aorta"))
        for n_i, (name, lab, keep, _, _) in enumerate(per_variant):
            if not keep:
                continue
            v, f = _surface(np.isin(lab, keep[:top_n]), spacing, 0.9)
            if v is None:
                continue
            meshes.append(dict(
                verts=v, faces=f, name=f"{name} ({len(keep)} cands)",
                color=palette[n_i % len(palette)], opacity=0.9,
                visible=(n_i == 2),
                text=f"<b>{name}</b><br>{len(keep)} candidates<br>"
                     f"noise {rows[n_i]['lumen_noise_sd_hu']} HU · wall grad "
                     f"{rows[n_i]['wall_gradient_hu_per_mm']} HU/mm"))
        obj_dir = os.path.join(os.path.dirname(out_html), "meshes_kernel_sweep")
        os.makedirs(obj_dir, exist_ok=True)
        for m in meshes:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          m["name"].split("(")[0].strip()).strip("_") or "surface"
            safe = safe[:60]     # Windows rejects *, ?, :, and very long names
            write_obj(os.path.join(obj_dir, f"{safe}.obj"),
                      m["verts"], m["faces"], name=safe)
        print(f"  wrote {obj_dir}/ ({len(meshes)} .obj surfaces)")

        if write_interactive_html(
                out_html, meshes,
                title=f"{case_id} — candidate stability across sharpness",
                subtitle="Only 'as acquired' is shown at first. Click legend "
                         "entries to overlay smoother and sharper variants: "
                         "candidates that persist across all of them are the "
                         "trustworthy ones.",
                axis_note="  [relative]"):
            print(f"  wrote {out_html}")


def slice_boxes(mk, spacing, margin_mm=15.0):
    """Per-axial-slice bounding box around the aorta, expanded by margin_mm.

    Returns {z: (x0, y0, x1, y1)} in voxel coords, for every slice that has mask.
    The margin is what decides whether a branch is even inside the prompt: a
    branch 12 mm from the aortic wall is invisible to MedSAM if the box only
    extends 8 mm. Too wide, though, and the box swallows spine and bowel and the
    model has more reason to grab the wrong object.
    """
    sx, sy, _ = spacing
    mx, my = int(round(margin_mm / sx)), int(round(margin_mm / sy))
    ny, nx = mk.shape[1], mk.shape[2]
    boxes = {}
    for z in np.where(mk.any(axis=(1, 2)))[0]:
        ys, xs = np.where(mk[z])
        boxes[int(z)] = (
            max(int(xs.min()) - mx, 0), max(int(ys.min()) - my, 0),
            min(int(xs.max()) + mx + 1, nx), min(int(ys.max()) + my + 1, ny),
        )
    return boxes


def run_medsam(ct, mk, spacing, checkpoint, margin_mm=15.0, every=1,
               device="cpu", dry_run=False, verbose=True):
    """Prompt MedSAM with a per-slice box around the aorta; return a 3D mask.

    MedSAM is a 2D model: one forward pass per axial slice, box prompt in that
    slice's plane. The expensive part is the ViT-B image encoder, which runs once
    per slice regardless of how many boxes you give it.

    dry_run=True skips the network entirely and returns a crude stand-in (the HU
    band restricted to the box) so you can check the box geometry and the plumbing
    before downloading a 375 MB checkpoint.
    """
    import time
    from skimage.transform import resize as sk_resize

    boxes = slice_boxes(mk, spacing, margin_mm)
    z_list = sorted(boxes)[::every]
    out = np.zeros_like(mk, dtype=np.uint8)
    if not z_list:
        return out, boxes

    if dry_run:
        lo, hi = lumen_band(ct, mk, spacing=spacing)[:2]
        for z in z_list:
            x0, y0, x1, y1 = boxes[z]
            sl = np.zeros(ct.shape[1:], bool)
            sl[y0:y1, x0:x1] = ((ct[z, y0:y1, x0:x1] >= lo) &
                                (ct[z, y0:y1, x0:x1] <= hi))
            out[z] = sl
        return out, boxes

    import torch
    import torch.nn.functional as F
    from segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device).eval()
    torch.set_num_threads(max(os.cpu_count() or 1, 1))

    H, W = ct.shape[1], ct.shape[2]
    t0 = time.time()
    for i, z in enumerate(z_list):
        # window to the CTA range, to 3-channel uint8-ish, to 1024 square
        sl = window(ct[z])                                   # already 0..1
        img3 = np.repeat(sl[:, :, None], 3, axis=2)
        img1024 = sk_resize(img3, (1024, 1024), order=3,
                            preserve_range=True, anti_aliasing=True)
        rng = max(img1024.max() - img1024.min(), 1e-8)
        img1024 = (img1024 - img1024.min()) / rng
        tens = (torch.tensor(img1024).float().permute(2, 0, 1)
                .unsqueeze(0).to(device))

        with torch.no_grad():
            embed = model.image_encoder(tens)               # 1x256x64x64

            box = np.array(boxes[z], dtype=np.float32)
            box1024 = box / np.array([W, H, W, H]) * 1024.0
            bt = torch.as_tensor(box1024, dtype=torch.float, device=device)
            if bt.ndim == 1:
                bt = bt[None, None, :]
            elif bt.ndim == 2:
                bt = bt[:, None, :]

            sparse, dense = model.prompt_encoder(points=None, boxes=bt, masks=None)
            logits, _ = model.mask_decoder(
                image_embeddings=embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            prob = torch.sigmoid(logits)
            prob = F.interpolate(prob, size=(H, W), mode="bilinear",
                                 align_corners=False)
            out[z] = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        if verbose and i == 0:
            dt = time.time() - t0
            print(f"  MedSAM: {dt:.1f}s for the first slice -> "
                  f"~{dt * len(z_list):.0f}s for {len(z_list)} slices",
                  file=sys.stderr)
        elif verbose and (i + 1) % 25 == 0:
            print(f"  MedSAM: {i + 1}/{len(z_list)} slices "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)

    if every > 1:                    # fill skipped slices by nearest-neighbour
        done = sorted(z_list)
        for z in sorted(boxes):
            if z not in set(done):
                out[z] = out[min(done, key=lambda d: abs(d - z))]
    return out, boxes


def fig_medsam(ct, mk, spacing, out_png, out_txt, case_id, checkpoint,
               margin_mm=15.0, every=1, device="cpu", dry_run=False,
               min_component_mm3=20.0, max_components=15):
    """MedSAM prompted with a box around the aorta; the RESIDUAL is the point.

    MedSAM returns one mask per box. A box drawn around the aorta asks it for the
    aorta -- which you already have. What is worth looking at is
    (MedSAM output) minus (supplied aorta mask): if the model's notion of "the
    bright vessel in this box" extends into the branch stubs, that residual is a
    candidate set. If it just traces the aortic wall, this prompt buys nothing.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    pred, boxes = run_medsam(ct, mk, spacing, checkpoint, margin_mm=margin_mm,
                             every=every, device=device, dry_run=dry_run)
    pred = pred.astype(bool)
    mkb = mk.astype(bool)

    residual = pred & ~ndi.binary_dilation(mkb, iterations=1)
    residual = ndi.binary_opening(residual, iterations=1)

    lab, _ = ndi.label(residual)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]

    # does each residual component actually touch the aorta? (a branch must)
    touch = set(int(t) for t in np.unique(lab[ndi.binary_dilation(mkb, iterations=2)])
                if t > 0)

    inter = (pred & mkb).sum()
    dice = 2.0 * inter / max(pred.sum() + mkb.sum(), 1)

    lines = [
        "=" * 78,
        f"MEDSAM BOX-PROMPT EXPERIMENT — {case_id}",
        "=" * 78,
        f"  mode                    : {'DRY RUN (no network)' if dry_run else 'MedSAM vit_b'}",
        f"  checkpoint              : {checkpoint if not dry_run else '-'}",
        f"  box margin              : {margin_mm:.0f} mm beyond the aorta bbox, per slice",
        f"  slices prompted         : {len(sorted(boxes)[::every])} of {len(boxes)}"
        f"{f' (every {every})' if every > 1 else ''}",
        "",
        f"  aorta mask volume       : {mkb.sum() * vox_mm3 / 1000:.1f} mL",
        f"  MedSAM mask volume      : {pred.sum() * vox_mm3 / 1000:.1f} mL",
        f"  Dice vs supplied aorta  : {dice:.3f}",
        f"  residual (pred - aorta) : {residual.sum() * vox_mm3 / 1000:.2f} mL",
        f"  residual components >{min_component_mm3:.0f} mm3 : {len(keep)}",
        "",
        f"    {'rank':>4} {'volume mL':>10} {'mean HU':>8} {'touches aorta':>14}",
    ]
    for rank, i in enumerate(keep[:15], start=1):
        comp = lab == i
        lines.append(f"    {rank:>4} {sizes[i] * vox_mm3 / 1000:>10.3f} "
                     f"{ct[comp].mean():>8.0f} "
                     f"{('yes' if i in touch else 'no'):>14}")
    lines += [
        "",
        "  How to read this:",
        "   * Dice near 1.0 and a near-empty residual => MedSAM reproduced the aorta",
        "     you already had and found nothing new. The box prompt is not enough.",
        "   * A handful of small residual components that TOUCH the aorta and sit at",
        "     ~lumen HU => those are branch stubs, and this is worth pursuing.",
        "   * Many residual components that do NOT touch the aorta => the model is",
        "     grabbing other bright things in the box (spine, bowel, IVC).",
        "=" * 78,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    # ---------------- figure ----------------
    def surface_capped(binary, cap=120_000):
        for tv in (1.5, 2.0, 2.5, 3.0, 4.0):
            v, f = _surface(binary, spacing, tv)
            if v is None or len(f) <= cap:
                return v, f
        return v, f

    big = np.isin(lab, keep[:max_components]) if keep else residual
    z_show = max(boxes, key=lambda z: residual[z].sum()) if residual.any() \
        else sorted(boxes)[len(boxes) // 2]

    fig = plt.figure(figsize=(17, 10.5), facecolor="white")

    # top-left: the prompt itself, on the slice with the most residual
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(window(ct[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
    x0, y0, x1, y1 = boxes[z_show]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#f2c14e", linewidth=1.8))
    ax.contour(mk[z_show], levels=[0.5], colors=["#ff4d4d"], linewidths=1.0)
    ax.set_title(f"box prompt, slice {z_show}\n"
                 f"{x1 - x0} x {y1 - y0} vox "
                 f"({(x1 - x0) * sx:.0f} x {(y1 - y0) * sy:.0f} mm)", fontsize=10)
    ax.axis("off")

    # top-middle: MedSAM output on that slice
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(window(ct[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
    ov = np.zeros(ct[z_show].shape + (4,), np.float32)
    ov[pred[z_show]] = (0.18, 0.62, 0.62, 0.50)
    ov[mkb[z_show]] = (0.85, 0.29, 0.29, 0.50)
    ax.imshow(ov, aspect=sy / sx, interpolation="nearest")
    ax.set_title("MedSAM output (teal) vs supplied aorta (red)", fontsize=10)
    ax.axis("off")

    # top-right: residual only
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(window(ct[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
    ov = np.zeros(ct[z_show].shape + (4,), np.float32)
    ov[residual[z_show]] = (1.0, 0.78, 0.25, 0.85)
    ax.contour(mk[z_show], levels=[0.5], colors=["#ff4d4d"], linewidths=1.0)
    ax.imshow(ov, aspect=sy / sx, interpolation="nearest")
    ax.set_title("residual = MedSAM - aorta (gold)", fontsize=10)
    ax.axis("off")

    # bottom: 3D of aorta / MedSAM / residual
    panels = [("supplied aorta", [(mkb, "#d94a4a", 0.95)]),
              ("MedSAM mask", [(pred & ~mkb, "#2f9e9e", 0.6), (mkb, "#d94a4a", 0.9)]),
              (f"residual, largest {min(max_components, len(keep))} components",
               [(big, "#f2c14e", 0.95), (mkb, "#d94a4a", 0.35)])]
    for col, (title, layers) in enumerate(panels):
        ax = fig.add_subplot(2, 3, 4 + col, projection="3d")
        allv = []
        for binary, color, alpha in layers:
            v, f = surface_capped(binary)
            if v is None:
                continue
            coll = Poly3DCollection(v[f], alpha=alpha)
            coll.set_facecolor(color)
            coll.set_edgecolor("none")
            ax.add_collection3d(coll)
            allv.append(v)
        if allv:
            allv = np.vstack(allv)
            ax.set_xlim(allv[:, 0].min(), allv[:, 0].max())
            ax.set_ylim(allv[:, 1].min(), allv[:, 1].max())
            ax.set_zlim(allv[:, 2].min(), allv[:, 2].max())
            try:
                ax.set_box_aspect([np.ptp(allv[:, k]) for k in range(3)])
            except Exception:
                pass
        ax.view_init(elev=12, azim=-75)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    fig.suptitle(f"{case_id} — MedSAM with a per-slice box around the aorta "
                 f"(margin {margin_mm:.0f} mm)   Dice vs supplied mask = {dice:.3f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


def fig_wall_shell(ct, mk, spacing, out_png, case_id, shell_mm=(1.5, 5.0)):
    """Unrolled view of CT intensity in a shell just outside the aortic wall.

    For each axial slice: take the aorta centroid, sample the mean intensity in an
    annulus between shell_mm[0] and shell_mm[1] outside the mask, binned by angle.
    A daughter vessel leaving the wall lights up as a bright blob at a consistent
    angle over several slices -- which is essentially the signal any detector must
    latch onto, so this is a useful sanity plot before writing one.
    """
    sx, sy, sz = spacing
    zs = np.where(mk.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return

    n_bins = 120
    angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    prof = np.full((len(zs), n_bins), np.nan, dtype=np.float32)

    yy = np.arange(ct.shape[1])
    xx = np.arange(ct.shape[2])

    for row, z in enumerate(zs):
        slice_mk = mk[z]
        if slice_mk.sum() < 5:
            continue
        cy, cx = ndi.center_of_mass(slice_mk)
        # distance in mm from the lumen, outward
        dist = ndi.distance_transform_edt(1 - slice_mk, sampling=(sy, sx))
        shell = (dist >= shell_mm[0]) & (dist <= shell_mm[1])
        ys, xs = np.where(shell)
        if len(ys) == 0:
            continue
        theta = np.arctan2((ys - cy) * sy, (xs - cx) * sx)
        bins = np.clip(((theta + np.pi) / (2 * np.pi) * n_bins).astype(int), 0, n_bins - 1)
        vals = ct[z][ys, xs]
        sums = np.bincount(bins, weights=vals, minlength=n_bins)
        cnts = np.bincount(bins, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof[row] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(prof, aspect="auto", cmap="inferno",
                   vmin=np.nanpercentile(prof, 40),
                   vmax=np.nanpercentile(prof, 99.5),
                   extent=[-180, 180, zs[-1], zs[0]])
    ax.set_xlabel("angle around aorta  (0° = right/+x,  ±180° = left;  "
                  "anterior ≈ -90° in LPS)")
    ax.set_ylabel("axial slice index  (superior at top)")
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_title(f"{case_id} — mean HU in a {shell_mm[0]}–{shell_mm[1]} mm shell "
                 f"outside the aortic wall\nbright vertical streaks = candidate branch ostia")
    fig.colorbar(im, ax=ax, label="mean HU in shell")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# interactive scroller
# -----------------------------------------------------------------------------

def interactive_viewer(ct, mk, spacing, case_id, margin_mm=40.0, crop=False):
    """Scroll axial slices with the mouse wheel / arrow keys / slider."""
    from matplotlib.widgets import Slider

    sx, sy, sz = spacing
    pad = (0, int(round(margin_mm / sy)), int(round(margin_mm / sx)))
    _, sly, slx = mask_bbox(mk, pad, crop=crop)
    zs = np.where(mk.any(axis=(1, 2)))[0]
    z_lo, z_hi = (int(zs[0]), int(zs[-1])) if len(zs) else (0, ct.shape[0] - 1)

    state = {"z": (z_lo + z_hi) // 2}

    fig, ax = plt.subplots(figsize=(8, 8.6), facecolor="black")
    plt.subplots_adjust(bottom=0.12)
    img_h = ax.imshow(window(ct[state["z"], sly, slx]), cmap="gray",
                      vmin=0, vmax=1, aspect=sy / sx)
    cont = [ax.contour(mk[state["z"], sly, slx], levels=[0.5],
                       colors=["#ff4d4d"], linewidths=1.2)]
    ax.axis("off")
    title = ax.set_title("", color="white", fontsize=12)

    sax = fig.add_axes([0.15, 0.05, 0.7, 0.03], facecolor="#333333")
    slider = Slider(sax, "slice", z_lo, z_hi, valinit=state["z"], valstep=1,
                    color="#ff4d4d")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    def redraw(z):
        z = int(np.clip(z, z_lo, z_hi))
        state["z"] = z
        img_h.set_data(window(ct[z, sly, slx]))
        for c in cont:
            for coll in getattr(c, "collections", [c]):
                try:
                    coll.remove()
                except Exception:
                    pass
        cont[0] = ax.contour(mk[z, sly, slx], levels=[0.5],
                             colors=["#ff4d4d"], linewidths=1.2)
        n = int(mk[z].sum())
        title.set_text(f"{case_id}   slice {z}   |   aorta voxels {n}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        redraw(state["z"] + (1 if event.button == "up" else -1))
        slider.eventson = False
        slider.set_val(state["z"])
        slider.eventson = True

    def on_key(event):
        if event.key in ("up", "right", "j"):
            on_scroll(type("E", (), {"button": "up"})())
        elif event.key in ("down", "left", "k"):
            on_scroll(type("E", (), {"button": "down"})())

    slider.on_changed(redraw)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(state["z"])
    plt.show()


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="CT volume (.nii / .nii.gz)")
    ap.add_argument("--aorta-mask", required=True, help="binary parent-aorta mask")
    ap.add_argument("--outdir", default="viz", help="where to write PNGs")
    ap.add_argument("--case-id", default=None, help="label for the figures")
    ap.add_argument("--interactive", action="store_true",
                    help="open a scrollable axial viewer instead of only saving PNGs")
    ap.add_argument("--skip-3d", action="store_true", help="skip the surface renders")
    ap.add_argument("--hu-sd", type=float, default=2.5,
                    help="width of the lumen HU band for fig 06, in std devs "
                         "of the intensity inside the aorta (default 2.5)")
    ap.add_argument("--frangi", action="store_true",
                    help="run Hessian/Frangi vesselness (figure 09 + candidate CSV)")
    ap.add_argument("--sigmas", default="0.7,1.0,1.5,2.0,3.0",
                    help="Frangi scales in mm (comma separated)")
    ap.add_argument("--roi-mm", type=float, default=30.0,
                    help="how far outside the aorta to evaluate vesselness")
    ap.add_argument("--predict", default=None,
                    help="also write challenge-format predictions to this JSON "
                         "path (implies --frangi)")
    ap.add_argument("--hu-tol", type=float, default=45.0,
                    help="max |mean HU - aortic lumen HU| for a candidate to "
                         "count as arterial (rejects veins)")
    ap.add_argument("--merge-mm", type=float, default=6.0,
                    help="ostia closer than this are merged into one instance")
    ap.add_argument("--end-margin-mm", type=float, default=5.0,
                    help="ignore candidates this close to the cropped ends of "
                         "the aorta mask -- flat caps are not origins")
    ap.add_argument("--isotropic", type=float, default=None,
                    help="resample to this isotropic voxel size in mm before "
                         "anything else (e.g. 0.8). Removes the z-axis blur "
                         "asymmetry; costs runtime and memory.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Frangi alpha: plate-vs-line sensitivity (lower = "
                         "stricter about being a line, not a sheet)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Frangi beta: blob rejection (lower = stricter)")
    ap.add_argument("--max-gap-mm", type=float, default=15.0,
                    help="how far a candidate's axis may be extended to reach "
                         "the aortic wall when inferring an ostium")
    ap.add_argument("--vess-frac", type=float, default=0.30,
                    help="candidate threshold as a fraction of the 99.5th-pct "
                         "vesselness (lower = more candidates)")
    ap.add_argument("--bone-hu", type=float, default=None,
                    help="HU above which voxels are treated as bone/calcium and "
                         "excluded (default: max(400, lumen mean + 3 sd))")
    ap.add_argument("--vess-pct", type=float, default=None,
                    help="percentile of vesselness inside the ROI used as the "
                         "candidate threshold (lower = more candidates)")
    ap.add_argument("--all", action="store_true",
                    help="run every figure that needs no extra download "
                         "(01-07, 09, 10). MedSAM (08) still needs --medsam "
                         "because it requires a checkpoint file.")
    ap.add_argument("--kernel-sweep", action="store_true",
                    help="sharpness sensitivity sweep (figure 10). NOT a "
                         "reconstruction-kernel study -- see the docstring.")
    ap.add_argument("--medsam", action="store_true",
                    help="run the MedSAM box-prompt experiment (figure 08)")
    ap.add_argument("--medsam-checkpoint", default="medsam_vit_b.pth",
                    help="path to the MedSAM ViT-B checkpoint")
    ap.add_argument("--medsam-dry-run", action="store_true",
                    help="fig 08 without the network: check box geometry and "
                         "plumbing before downloading the 375 MB checkpoint")
    ap.add_argument("--box-margin-mm", type=float, default=15.0,
                    help="how far beyond the aorta bbox the prompt box extends "
                         "(default 15 mm)")
    ap.add_argument("--medsam-every", type=int, default=1,
                    help="prompt every Nth axial slice (N>1 is much faster and "
                         "much coarser)")
    ap.add_argument("--device", default="cpu", help="torch device for MedSAM")
    ap.add_argument("--kernel", type=int, default=3,
                    help="in-plane block size for fig 07 (default 3 -> 3x3)")
    ap.add_argument("--crop", action="store_true",
                    help="crop every figure to a margin around the aorta. "
                         "OFF by default -- figures show the whole field of view.")
    args = ap.parse_args()

    if args.predict:
        args.frangi = True

    if args.all:
        args.frangi = True
        args.kernel_sweep = True

    if not args.interactive:
        matplotlib.use("Agg")

    case_id = args.case_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.image))) or "case"
    os.makedirs(args.outdir, exist_ok=True)

    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    spacing = img.GetSpacing()

    if args.isotropic:
        n_before = ct.shape
        ct, mk, spacing = resample_isotropic(ct, mk, spacing, args.isotropic)
        img = _PhysMap(img.GetOrigin(), spacing, img.GetDirection())
        print(f"  resampled {n_before} -> {ct.shape} at "
              f"{args.isotropic} mm isotropic")

    print(f"  visualize_case {__version__}")

    report = geometry_report(img, ct, mk)
    print(report)
    with open(os.path.join(args.outdir, "00_report.txt"), "w") as fh:
        fh.write(report + "\n")

    jobs = [
        ("01_ortho.png",         lambda p: fig_ortho(ct, mk, spacing, p, case_id)),
        ("02_axial_montage.png", lambda p: fig_axial_montage(ct, mk, spacing, p,
                                                             case_id, crop=args.crop)),
        ("03_mip.png",           lambda p: fig_mip(ct, mk, spacing, p, case_id,
                                                   crop=args.crop)),
        ("05_wall_shell.png",    lambda p: fig_wall_shell(ct, mk, spacing, p, case_id)),
    ]
    if not args.skip_3d:
        jobs.insert(3, ("04_aorta_3d.png",
                        lambda p: fig_aorta_3d(
                            mk, spacing, p, case_id,
                            out_html=os.path.join(args.outdir, "04_aorta_3d.html"))))
        jobs.append(("06_intensity_render.png",
                     lambda p: fig_intensity_render(
                         ct, mk, spacing, p,
                         os.path.join(args.outdir, "06_intensity_report.txt"),
                         case_id, k_sd=args.hu_sd, crop=args.crop)))
        jobs.append(("07_kernel_stride.png",
                     lambda p: fig_kernel_stride(
                         ct, mk, spacing, p,
                         os.path.join(args.outdir, "07_kernel_report.txt"),
                         case_id, k_sd=args.hu_sd, ksize=args.kernel,
                         crop=args.crop)))

    if args.frangi:
        sig = tuple(float(s) for s in args.sigmas.split(",") if s.strip())
        jobs.append(("09_frangi.png",
                     lambda p: fig_frangi(
                         ct, mk, spacing, p,
                         os.path.join(args.outdir, "09_frangi_report.txt"),
                         os.path.join(args.outdir, "09_candidates.csv"),
                         case_id, sigmas_mm=sig, roi_mm=args.roi_mm,
                         vess_pct=args.vess_pct, vess_frac=args.vess_frac,
                         alpha=args.alpha, beta=args.beta,
                         max_gap_mm=args.max_gap_mm,
                         out_json=args.predict, hu_tol=args.hu_tol,
                         merge_mm=args.merge_mm,
                         end_margin_mm=args.end_margin_mm,
                         bone_floor_hu=args.bone_hu, img=img,
                         out_html=os.path.join(args.outdir,
                                               "09_interactive.html"))))

    if args.kernel_sweep:
        jobs.append(("10_kernel_sweep.png",
                     lambda p: fig_kernel_sweep(
                         ct, mk, spacing, p,
                         os.path.join(args.outdir, "10_kernel_sweep.csv"),
                         os.path.join(args.outdir, "10_interactive.html"),
                         case_id)))

    if args.medsam or args.medsam_dry_run:
        jobs.append(("08_medsam.png",
                     lambda p: fig_medsam(
                         ct, mk, spacing, p,
                         os.path.join(args.outdir, "08_medsam_report.txt"),
                         case_id, args.medsam_checkpoint,
                         margin_mm=args.box_margin_mm, every=args.medsam_every,
                         device=args.device, dry_run=args.medsam_dry_run)))

    written, failed = [], []
    for name, fn in sorted(jobs, key=lambda j: j[0]):
        path = os.path.join(args.outdir, name)
        try:
            fn(path)
            print(f"  wrote {path}")
            written.append(name)
        except Exception as exc:                      # keep going on one bad figure
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
            failed.append((name, str(exc)))

    # ---- manifest: say plainly what ran, what did not, and how to get it ----
    optional = [
        ("04_aorta_3d.png", not args.skip_3d, "--skip-3d was passed"),
        ("06_intensity_render.png", not args.skip_3d, "--skip-3d was passed"),
        ("07_kernel_stride.png", not args.skip_3d, "--skip-3d was passed"),
        ("08_medsam.png", args.medsam or args.medsam_dry_run,
         "add --medsam (needs a checkpoint) or --medsam-dry-run"),
        ("09_frangi.png", args.frangi, "add --frangi  (or --all)"),
        ("10_kernel_sweep.png", args.kernel_sweep, "add --kernel-sweep  (or --all)"),
    ]
    print("\n" + "-" * 68)
    print(f"  {len(written)} figure(s) written to {args.outdir}/")
    skipped = [(n, why) for n, on, why in optional if not on]
    if skipped:
        print("  not generated:")
        for n, why in skipped:
            print(f"    {n:<26} {why}")
        print("\n  Tip: `--all` runs everything except MedSAM.")
    if failed:
        print("  failed:")
        for n, exc in failed:
            print(f"    {n:<26} {exc}")
    print("-" * 68)

    if args.interactive:
        interactive_viewer(ct, mk, spacing, case_id, crop=args.crop)


if __name__ == "__main__":
    main()