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


def mask_bbox(mk: np.ndarray, pad_vox=(0, 0, 0)):
    """Return slices bounding the mask, padded and clipped."""
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


def fig_axial_montage(ct, mk, spacing, out_png, case_id, n_panels=24, margin_mm=35.0):
    """Evenly spaced axial slices spanning the aortic segment, cropped around it."""
    sx, sy, sz = spacing
    zs = np.where(mk.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return
    z_idx = np.linspace(zs[0], zs[-1], min(n_panels, len(zs))).round().astype(int)

    pad = (0, int(round(margin_mm / sy)), int(round(margin_mm / sx)))
    _, sly, slx = mask_bbox(mk, pad)

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


def fig_mip(ct, mk, spacing, out_png, case_id, slab_mm=45.0):
    """Coronal + sagittal maximum-intensity projections of a slab around the aorta.

    This is the view where branch anatomy becomes obvious: the celiac trunk and SMA
    project anteriorly on the sagittal MIP, the renals laterally on the coronal.
    """
    sx, sy, sz = spacing
    if not mk.any():
        return
    pad = (0, int(round(slab_mm / sy)), int(round(slab_mm / sx)))
    slz, sly, slx = mask_bbox(mk, pad)
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

    fig.suptitle(f"{case_id} — slab MIP ±{slab_mm:.0f} mm around the aorta "
                 f"(red = parent mask outline)", color="white", fontsize=13)
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

def interactive_viewer(ct, mk, spacing, case_id, margin_mm=40.0):
    """Scroll axial slices with the mouse wheel / arrow keys / slider."""
    from matplotlib.widgets import Slider

    sx, sy, sz = spacing
    pad = (0, int(round(margin_mm / sy)), int(round(margin_mm / sx)))
    _, sly, slx = mask_bbox(mk, pad)
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
    ap.add_argument("--skip-3d", action="store_true", help="skip the surface render")
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
        ("02_axial_montage.png", lambda p: fig_axial_montage(ct, mk, spacing, p, case_id)),
        ("03_mip.png",           lambda p: fig_mip(ct, mk, spacing, p, case_id)),
        ("05_wall_shell.png",    lambda p: fig_wall_shell(ct, mk, spacing, p, case_id)),
    ]
    if not args.skip_3d:
        jobs.insert(3, ("04_aorta_3d.png",
                        lambda p: fig_aorta_3d(mk, spacing, p, case_id)))

    for name, fn in jobs:
        path = os.path.join(args.outdir, name)
        try:
            fn(path)
            print(f"  wrote {path}")
        except Exception as exc:                      # keep going on one bad figure
            print(f"  FAILED {name}: {exc}", file=sys.stderr)

    if args.interactive:
        interactive_viewer(ct, mk, spacing, case_id)


if __name__ == "__main__":
     main()

