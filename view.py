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
CTA_LEVEL, CTA_WIDTH = 200.0, 700.0

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


def fig_aorta_3d(mk, spacing, out_png, case_id, target_vox=1.5):
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


def fig_aorta_3d_interactive(mk, spacing, out_html, case_id, target_vox=1.5):
    """Write a browser-based interactive 3D surface of the supplied mask."""
    try:
        import plotly.graph_objects as go
        from skimage.measure import marching_cubes
    except ImportError:
        print("plotly or scikit-image not installed -- skipping interactive 3D", file=sys.stderr)
        return
    if not mk.any():
        return

    sx, sy, sz = spacing
    zoom = (sz / target_vox, sy / target_vox, sx / target_vox)
    small = ndi.zoom(mk.astype(np.float32), zoom, order=1)
    small = ndi.gaussian_filter(small, 0.8)
    if small.max() < 0.5:
        return

    verts, faces, _, _ = marching_cubes(small, level=0.5,
                                        spacing=(target_vox,) * 3)
    verts = verts[:, [2, 1, 0]]
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#d94a4a", opacity=0.9, flatshading=False,
        hoverinfo="skip",
    )
    fig = go.Figure(mesh)
    fig.update_layout(
        title=f"{case_id} - interactive aorta surface",
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


def _surface(binary, spacing, target_vox, smooth=0.8):
    """Downsample a binary volume to ~isotropic target_vox mm and marching-cube it.

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


def _write_interactive_layers(layers, out_html, title):
    """Write named groups of 3D surface layers with a Plotly selector."""
    import plotly.graph_objects as go

    traces = []
    groups = []
    for group_name, group_layers in layers:
        groups.append((group_name, group_layers))

    # Each layer stores the precomputed mesh so the selector only changes visibility.
    for group_name, group_layers in groups:
        for verts, faces, color, opacity in group_layers:
            traces.append(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=opacity, hoverinfo="skip", visible=False,
                name=group_name,
            ))

    buttons = []
    offset = 0
    for group_name, group_layers in groups:
        visible = [False] * len(traces)
        for index in range(len(group_layers)):
            visible[offset + index] = True
        buttons.append(dict(label=group_name, method="update",
                            args=[{"visible": visible}, {"title": f"{title} - {group_name}"}]))
        offset += len(group_layers)
    if traces:
        traces[0].visible = True
        initial = groups[0][0]
    else:
        initial = title
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{title} - {initial}",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=0.98,
                          xanchor="left", yanchor="top")],
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.write_html(out_html, include_plotlyjs=True, full_html=True)


def _mesh_layers(layer_groups, spacing, target_vox=3.0):
    meshes = []
    for group_name, binary_layers in layer_groups:
        group = []
        for binary, color, opacity in binary_layers:
            verts, faces = _surface(binary, spacing, target_vox)
            if verts is not None:
                group.append((verts, faces, color, opacity))
        meshes.append((group_name, group))
    return meshes


def fig_intensity_interactive(ct, mk, spacing, out_html, case_id,
                              k_sd=2.5, erode_mm=2.0, margin_mm=60.0,
                              crop=False, max_components=15, min_component_mm3=30.0):
    """Write an interactive version of the intensity-only 3D experiment."""
    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return
    lo, hi, _, _ = lumen_band(ct, mk, k_sd, erode_mm, spacing)
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
    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    keep = [int(i) for i in np.argsort(sizes)[::-1]
            if sizes[i] >= min_vox][:max_components]
    big = np.isin(lab, keep) if keep else band
    touch = set(int(i) for i in np.unique(lab[ndi.binary_dilation(sub_mk, iterations=1)]) if i > 0)
    connected = np.isin(lab, list(touch)) if touch else np.zeros_like(band)
    layers = _mesh_layers([
        ("A - supplied aorta mask", [(sub_mk, "#d94a4a", 0.95)]),
        ("B - HU band candidates", [(big & ~sub_mk, "#2f9e9e", 0.55),
                                     (sub_mk, "#d94a4a", 0.95)]),
        ("C - connected to aorta", [(connected & ~sub_mk, "#2f9e9e", 0.75),
                                     (sub_mk, "#d94a4a", 0.95)]),
    ], spacing)
    _write_interactive_layers(layers, out_html,
                              f"{case_id} - intensity-only experiment ({lo:.0f}-{hi:.0f} HU)")


def fig_kernel_interactive(ct, mk, spacing, out_html, case_id,
                           k_sd=2.5, ksize=3, strides=(1, 2), crop=False):
    """Write an interactive version of the block-mean kernel 3D experiment."""
    sx, sy, sz = spacing
    if not mk.any():
        return
    lo, hi, _, _ = lumen_band(ct, mk, k_sd, spacing=spacing)
    pad = (int(round(60.0 / sz)), int(round(60.0 / sy)), int(round(60.0 / sx)))
    region = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)
    per_voxel = (sub_ct >= lo) & (sub_ct <= hi)
    mean_ok = (ndi.uniform_filter(sub_ct, size=(1, ksize, ksize), mode="nearest") >= lo) & \
              (ndi.uniform_filter(sub_ct, size=(1, ksize, ksize), mode="nearest") <= hi)
    r = ksize // 2
    footprint = np.ones((1, ksize, ksize), bool)
    results = [("per-voxel (stride 1, 1x1)", per_voxel)]
    for stride in strides:
        centers = np.zeros_like(mean_ok)
        centers[:, r:-r or None:stride, r:-r or None:stride] = \
            mean_ok[:, r:-r or None:stride, r:-r or None:stride]
        results.append((f"{ksize}x{ksize} block mean, stride {stride}",
                        ndi.binary_dilation(centers, structure=footprint)))
    groups = []
    for name, volume in results:
        lab, _ = ndi.label(volume)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= 30][:15]
        big = np.isin(lab, keep) if keep else volume
        groups.append((name, [(big & ~sub_mk, "#2f9e9e", 0.6),
                              (sub_mk, "#d94a4a", 0.95)]))
    _write_interactive_layers(_mesh_layers(groups, spacing), out_html,
                              f"{case_id} - block-mean kernel experiment")


def fig_intensity_render(ct, mk, spacing, out_png, out_txt, case_id,
                         k_sd=2.5, erode_mm=2.0, margin_mm=60.0,
                         crop=False, max_components=15,
                         min_component_mm3=30.0, show=False):
    """EXPERIMENT: is intensity alone enough to isolate the arterial tree?

    Learns the lumen HU band from inside the supplied aorta mask, thresholds the
    whole (cropped) volume with it, and renders three surfaces side by side:

      A  the supplied aorta mask                      -- what you were given
      B  everything in the HU band                    -- what intensity alone buys
      C  the band, restricted to what touches the aorta

    Panel B is the answer to "is colour enough". Panel C is the answer to
    "is colour plus connectivity enough", which is the actually useful question.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sx, sy, sz = spacing
    vox_mm3 = sx * sy * sz
    if not mk.any():
        return

    # --- learn the band from the eroded lumen (dodging partial-volume at the wall)
    er_iter = max(int(round(erode_mm / min(sx, sy))), 1)
    core = ndi.binary_erosion(mk, iterations=er_iter)
    if core.sum() < 50:
        core = mk.astype(bool)
    lumen = ct[core]
    mu, sd = float(lumen.mean()), float(lumen.std())
    lo, hi = mu - k_sd * sd, mu + k_sd * sd

    pad = (int(round(margin_mm / sz)), int(round(margin_mm / sy)),
           int(round(margin_mm / sx)))
    region = mask_bbox(mk, pad, crop=crop)
    sub_ct, sub_mk = ct[region], mk[region].astype(bool)

    band = (sub_ct >= lo) & (sub_ct <= hi)
    band = ndi.binary_opening(band, iterations=1)          # drop single-voxel noise

    # --- connected components, and which of them touch the aorta
    lab, n_lab = ndi.label(band)
    if n_lab == 0:
        return
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0

    touch = np.unique(lab[ndi.binary_dilation(sub_mk, iterations=1)])
    touch = set(int(t) for t in touch if t > 0)
    connected = np.isin(lab, list(touch)) if touch else np.zeros_like(band)

    # --- text report: what else in this scan looks exactly like aortic blood?
    min_vox = max(int(min_component_mm3 / vox_mm3), 1)
    order = np.argsort(sizes)[::-1]
    keep = [int(i) for i in order if sizes[i] >= min_vox][:40]

    lines = [
        "=" * 74,
        f"INTENSITY-ONLY EXPERIMENT — {case_id}",
        "=" * 74,
        f"  lumen HU (eroded mask)  : mean {mu:.0f}, sd {sd:.0f}",
        f"  band used (mean +/- {k_sd} sd) : {lo:.0f} .. {hi:.0f} HU",
        f"  field of view           : {f'aorta bbox + {margin_mm:.0f} mm' if crop else 'whole volume'}",
        "",
        f"  aorta mask volume       : {sub_mk.sum() * vox_mm3 / 1000:.1f} mL",
        f"  in-band volume          : {band.sum() * vox_mm3 / 1000:.1f} mL "
        f"({band.sum() / max(sub_mk.sum(), 1):.1f}x the aorta)",
        f"  ... touching the aorta  : {connected.sum() * vox_mm3 / 1000:.1f} mL",
        f"  in-band components >{min_component_mm3:.0f} mm3 : "
        f"{sum(1 for i in np.flatnonzero(sizes) if sizes[i] >= min_vox)}",
        "",
        "  largest in-band components:",
        f"    {'rank':>4}  {'volume mL':>10}  {'mean HU':>8}  {'touches aorta':>13}",
    ]
    for rank, i in enumerate(keep[:15], start=1):
        comp = lab == i
        lines.append(f"    {rank:>4}  {sizes[i] * vox_mm3 / 1000:>10.2f}  "
                     f"{sub_ct[comp].mean():>8.0f}  "
                     f"{('yes' if i in touch else 'no'):>13}")
    lines += [
        "",
        "  Read this as: if the in-band volume is many times the aorta volume, raw",
        "  intensity is NOT separating arteries from bone, contrast-filled kidneys,",
        "  bowel or veins. Compare the 'touching the aorta' figure -- if that one is",
        "  close to the aorta volume plus a little, connectivity is doing the real",
        "  work and a band + region-grow is a viable candidate generator.",
        "=" * 74,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    # --- render. Coarsen until matplotlib can actually draw it.
    def surface_capped(binary, cap=120_000):
        for tv in (2.0, 2.5, 3.0, 4.0, 5.0):
            v, f = _surface(binary, spacing, tv)
            if v is None:
                return None, None
            if len(f) <= cap:
                return v, f
        return v, f

    # For the busy panel, keep only the biggest components so the figure stays legible.
    big = np.isin(lab, keep[:max_components]) if keep else band
    omitted = max(sum(1 for i in np.flatnonzero(sizes) if sizes[i] >= min_vox)
                  - max_components, 0)

    panels = [
        ("A — supplied aorta mask", [(sub_mk, "#d94a4a", 0.95)]),
        (f"B — HU band only ({lo:.0f}..{hi:.0f})\n"
         f"largest {min(max_components, len(keep))} components"
         + (f", {omitted} more omitted" if omitted else ""),
         [(big & ~sub_mk, "#2f9e9e", 0.55), (sub_mk, "#d94a4a", 0.95)]),
        ("C — HU band, connected to aorta",
         [(connected & ~sub_mk, "#2f9e9e", 0.75), (sub_mk, "#d94a4a", 0.95)]),
    ]

    fig = plt.figure(figsize=(17, 6.5), facecolor="white")
    for col, (title, layers) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
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
        ax.view_init(elev=12, azim=-75)          # near-anterior, slightly tilted
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    fig.suptitle(f"{case_id} — can intensity alone find the branches?   "
                 f"red = supplied aorta,  teal = everything else in the lumen HU band",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    if not show:
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
                      margin_mm=60.0, max_components=15, min_component_mm3=30.0,
                      show=False):
    """EXPERIMENT: threshold on a k x k in-plane block MEAN instead of single voxels.

    For every candidate block position (stepping by `stride`), take the mean of the
    k x k neighbourhood in that axial slice. If the mean falls in the lumen HU band,
    the whole k x k block is accepted. Stride 1 slides over every position; stride 2
    evaluates every other position, so the result is quantised onto a coarser lattice.

    Point of the test: block averaging is a low-pass filter. It kills isolated noisy
    voxels, but it also dilutes any vessel narrower than the kernel by mixing lumen
    with surrounding fat -- so small branches drop out of the band entirely. This
    figure shows you that tradeoff directly.
    """
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

    # per-voxel reference, and the k x k block mean (in-plane, per axial slice)
    per_voxel = (sub_ct >= lo) & (sub_ct <= hi)
    block_mean = ndi.uniform_filter(sub_ct, size=(1, ksize, ksize), mode="nearest")
    mean_ok = (block_mean >= lo) & (block_mean <= hi)

    r = ksize // 2
    footprint = np.ones((1, ksize, ksize), bool)

    results = [("per-voxel (stride 1, 1x1)", per_voxel)]
    for s in strides:
        centers = np.zeros_like(mean_ok)
        centers[:, r:-r or None:s, r:-r or None:s] = \
            mean_ok[:, r:-r or None:s, r:-r or None:s]
        # "graph the k x k": every accepted centre paints its whole block
        results.append((f"{ksize}x{ksize} block mean, stride {s}",
                        ndi.binary_dilation(centers, structure=footprint)))

    # ---------------- text report ----------------
    lines = [
        "=" * 78,
        f"BLOCK-MEAN KERNEL EXPERIMENT — {case_id}",
        "=" * 78,
        f"  lumen HU band           : {lo:.0f} .. {hi:.0f}  (mean {mu:.0f}, sd {sd:.0f})",
        f"  kernel                  : {ksize}x{ksize} in-plane "
        f"({ksize * sx:.1f} x {ksize * sy:.1f} mm at this spacing)",
        f"  field of view           : {f'aorta bbox + {margin_mm:.0f} mm' if crop else 'whole volume'}",
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
        "  'in aorta %' is how much of the supplied aorta each variant recovers --",
        "  a sanity floor. If block averaging drops it below ~95% the kernel is",
        "  already eroding a 20 mm vessel, and a 3 mm branch has no chance.",
        "  Watch the component count: fewer components = less speckle, but check the",
        "  volume at the same time, because losing small vessels also drops it.",
        "=" * 78,
    ]
    report = "\n".join(lines)
    print(report)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    # ---------------- pick a branch-rich axial slice for the 2D row ----------------
    dist_out = ndi.distance_transform_edt(~sub_mk, sampling=(sz, sy, sx))
    near = (dist_out > 0) & (dist_out < 20.0)
    score = (per_voxel & near).sum(axis=(1, 2))
    z_show = int(np.argmax(score)) if score.max() > 0 else sub_ct.shape[0] // 2

    def surface_capped(binary, cap=120_000):
        for tv in (2.0, 2.5, 3.0, 4.0, 5.0):
            v, f = _surface(binary, spacing, tv)
            if v is None or len(f) <= cap:
                return v, f
        return v, f

    n = len(results)
    fig = plt.figure(figsize=(5.6 * n, 10.5), facecolor="white")

    for col, (name, vol) in enumerate(results):
        # keep the render legible: biggest components only
        lab, _ = ndi.label(vol)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]
        big = np.isin(lab, keep[:max_components]) if keep else vol

        ax = fig.add_subplot(2, n, col + 1, projection="3d")
        allv = []
        for binary, color, alpha in ((big & ~sub_mk, "#2f9e9e", 0.6),
                                     (sub_mk, "#d94a4a", 0.95)):
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
        ax.set_title(name, fontsize=11)
        ax.set_axis_off()

        # bottom row: what the acceptance map looks like on one axial slice
        ax2 = fig.add_subplot(2, n, n + col + 1)
        ax2.imshow(window(sub_ct[z_show]), cmap="gray", vmin=0, vmax=1, aspect=sy / sx)
        overlay = np.zeros(sub_ct[z_show].shape + (4,), np.float32)
        overlay[vol[z_show]] = (0.18, 0.62, 0.62, 0.55)
        overlay[sub_mk[z_show]] = (0.85, 0.29, 0.29, 0.55)
        ax2.imshow(overlay, aspect=sy / sx, interpolation="nearest")
        ax2.set_title(f"axial slice {z_show}", fontsize=10)
        ax2.axis("off")

    fig.suptitle(
        f"{case_id} — block-mean thresholding, kernel {ksize}x{ksize}, band {lo:.0f}-{hi:.0f} HU\n"
        f"red = supplied aorta,  teal = accepted as lumen-like",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    if not show:
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
                    help="open the axial viewer and write an interactive 3D HTML model")
    ap.add_argument("--skip-3d", action="store_true", help="skip the surface renders")
    ap.add_argument("--hu-sd", type=float, default=2.5,
                    help="width of the lumen HU band for fig 06, in std devs "
                         "of the intensity inside the aorta (default 2.5)")
    ap.add_argument("--kernel", type=int, default=3,
                    help="in-plane block size for fig 07 (default 3 -> 3x3)")
    ap.add_argument("--crop", action="store_true",
                    help="crop every figure to a margin around the aorta. "
                         "OFF by default -- figures show the whole field of view.")
    args = ap.parse_args()

    if not args.interactive:
        matplotlib.use("Agg")

    case_id = args.case_id or os.path.basename(
        os.path.dirname(os.path.abspath(args.image))) or "case"
    os.makedirs(args.outdir, exist_ok=True)

    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    spacing = img.GetSpacing()

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
                        lambda p: fig_aorta_3d(mk, spacing, p, case_id)))
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

    for name, fn in jobs:
        path = os.path.join(args.outdir, name)
        try:
            fn(path)
            print(f"  wrote {path}")
        except Exception as exc:                      # keep going on one bad figure
            print(f"  FAILED {name}: {exc}", file=sys.stderr)

    if args.interactive:
        if not args.skip_3d:
            fig_intensity_render(
                ct, mk, spacing,
                os.path.join(args.outdir, "06_intensity_render.png"),
                os.path.join(args.outdir, "06_intensity_report.txt"),
                case_id, k_sd=args.hu_sd, crop=args.crop, show=True)
            fig_kernel_stride(
                ct, mk, spacing,
                os.path.join(args.outdir, "07_kernel_stride.png"),
                os.path.join(args.outdir, "07_kernel_report.txt"),
                case_id, k_sd=args.hu_sd, ksize=args.kernel,
                crop=args.crop, show=True)
        interactive_viewer(ct, mk, spacing, case_id, crop=args.crop)


if __name__ == "__main__":
    main()