"""
outputs.py -- every artefact a case produces.

  prediction.json     the challenge schema, plus the reference annotation's
                      extra fields so ours is a superset of theirs
  daughters.nii.gz    label volume on the INPUT grid, one label per daughter
  snap_labels.txt     ITK-SNAP Label Description File matching those labels
  ostia.csv           the same numbers without parsing JSON
  check.html          self-contained 3D verification view

Physical coordinates come from SimpleITK throughout. The label volume copies the
input image's origin, spacing and direction verbatim, so it overlays the CT in
ITK-SNAP with no adjustment -- a mismatch here is invisible in the numbers and
obvious the moment a clinician opens it.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

__all__ = ["write_all", "build_prediction"]

PALETTE = [
    (255, 64, 64), (64, 176, 255), (96, 210, 120), (255, 176, 48),
    (186, 120, 255), (255, 112, 190), (96, 220, 220), (216, 160, 88),
    (140, 160, 255), (200, 200, 96), (255, 140, 100), (120, 200, 170),
]


def _round(v, n=4):
    return [round(float(x), n) for x in np.asarray(v, float).ravel()]


# -----------------------------------------------------------------------------
# JSON
# -----------------------------------------------------------------------------

def build_prediction(case_id, case, daughters, info, cfg, tc):
    """The challenge JSON. Required fields first, provenance after."""
    sx, sy, sz = case.spacing
    nz, ny, nx = case.full_shape
    recs = []
    for k, d in enumerate(daughters, 1):
        recs.append({
            # --- required by the challenge ---
            "instance_id": f"branch_{k:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": _round(d["ostium_mm"], 3),
            "seed_xyz_mm": _round(d["seed_mm"], 3),
            "radius_mm": round(float(d["radius_mm"]), 3),
            "direction_xyz": _round(d["direction_xyz"], 4),
            # --- links this record to daughters.nii.gz ---
            "label_value": k,
            # --- proximal path and how it was measured ---
            "centerline_xyz_mm": [_round(p, 3) for p in d["centerline_mm"]],
            "centerline_length_mm": d["centerline_length_mm"],
            "stopped_at_first_bifurcation": d["stopped_at_bifurcation"],
            "seed_truncated_to_path_end": d["seed_truncated"],
            "origin_diameter_estimate_mm": d["origin_diameter_mm"],
            "origin_diameter_method": "max(2 x inscribed radius at the ostium, "
                                      "area-equivalent diameter on a "
                                      "perpendicular plane 1.5 mm along the path)",
            "radius_measurement_status": d["radius_status"],
            "radius_method": "Area-equivalent radius sqrt(area/pi) on a 0.25 mm "
                             "interpolated plane perpendicular to the path at "
                             "the seed; parent-lumen voxels excluded.",
            "threshold_hu": d["threshold_hu"],
            "search_radius_mm": d["search_radius_mm"],
            "voxel_count": d["n_voxels"],
            "volume_mm3": round(d["volume_mm3"], 3),
            "traced_voxel_count": d["traced_voxel_count"],
        })

    return {
        "case_id": case_id,
        "annotation_status": "automatic_prediction",
        "coordinate_system": "SimpleITK physical LPS millimetres",
        "parent": {"instance_id": "aorta"},
        "shape_xyz": [int(nx), int(ny), int(nz)],
        "spacing_xyz_mm": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "policy": {
            "minimum_origin_diameter_mm": cfg.min_origin_diameter_mm,
            "minimum_visible_path_mm": cfg.min_reach_mm,
            "maximum_trace_mm": tc.max_trace_mm,
            "stop_at_first_bifurcation": True,
            "wall_definition": "boundary of supplied parent lumen mask",
            "diameter_resolution_warning":
                f"{min(case.spacing):.2f} mm voxels; borderline "
                f"{cfg.min_origin_diameter_mm} mm eligibility is uncertain",
        },
        "method": {
            "detector": "two-phase band growth with leak detection (Tahoces-style)",
            "band_hu": info["band_hu"],
            "band_rule": "floor absolute, ceiling mean + 3 sd",
            "lumen_mean_hu": info["lumen_mean_hu"],
            "lumen_sd_hu": info["lumen_sd_hu"],
            "bone_cut_hu": info["bone_cut_hu"],
            "leaks_blocked": info["leaks_blocked"],
        },
        "daughters": recs,
    }


# -----------------------------------------------------------------------------
# label volume + ITK-SNAP
# -----------------------------------------------------------------------------

def write_label_volume(path, case, daughters, labels):
    """Full-grid label volume: every voxel of every detected daughter."""
    nz, ny, nx = case.full_shape
    vol = np.zeros((nz, ny, nx), np.uint8)
    sub = vol[case.region]
    for k, d in enumerate(daughters, 1):
        sub[labels == d["label"]] = k
    vol[case.region] = sub

    out = sitk.GetImageFromArray(vol)
    out.CopyInformation(case.img)          # origin, spacing AND direction
    sitk.WriteImage(out, str(path), True)
    return path


def write_snap_labels(path, n):
    """ITK-SNAP Label Description File: IDX R G B A VIS MSH "LABEL"."""
    lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# File format:",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL",
        "# Fields:",
        "#    IDX:   Zero-based index",
        "#    -R-:   Red color component (0..255)",
        "#    -G-:   Green color component (0..255)",
        "#    -B-:   Blue color component (0..255)",
        "#    -A-:   Label transparency (0.00 .. 1.00)",
        "#    VIS:   Label visibility (0 or 1)",
        "#    MSH:   Label mesh visibility (0 or 1)",
        "#  LABEL:   Label description",
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
    ]
    for k in range(1, n + 1):
        r, g, b = PALETTE[(k - 1) % len(PALETTE)]
        lines.append(f'{k:5d} {r:5d} {g:4d} {b:4d}        1  1  1    '
                     f'"branch_{k:03d}"')
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_ostia_csv(path, pred):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "label_value", "parent_instance_id",
                    "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "seed_x_mm", "seed_y_mm", "seed_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm",
                    "origin_diameter_mm", "centerline_length_mm",
                    "stopped_at_bifurcation"])
        for d in pred["daughters"]:
            w.writerow([d["instance_id"], d["label_value"],
                        d["parent_instance_id"],
                        *d["ostium_xyz_mm"], *d["seed_xyz_mm"],
                        *d["direction_xyz"], d["radius_mm"],
                        d["origin_diameter_estimate_mm"],
                        d["centerline_length_mm"],
                        int(d["stopped_at_first_bifurcation"])])
    return path


# -----------------------------------------------------------------------------
# 3D verification view
# -----------------------------------------------------------------------------

def _smooth(verts, faces, iters=10, lam=0.55, mu=-0.58):
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


def _surface(binary, spacing, cap=120_000, iso=0.40, blur_vox=0.9):
    """Isotropic, lightly blurred marching cubes. The low isolevel is what keeps
    a two-voxel vessel from dissolving; blurring at 0.5 deletes them."""
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
            return _smooth(verts[:, [2, 1, 0]], faces), faces
    return None, None


def _to_world(case, v):
    if v is None:
        return None
    try:
        D = np.array(case.img.GetDirection(), float).reshape(3, 3)
    except Exception:
        D = np.eye(3)
    corner = case.to_mm((0, 0, 0))
    return corner + np.asarray(v, float) @ D.T


def write_check_html(path, case_id, case, daughters, labels, pred,
                     arrow_mm=8.0):
    """Aorta mask + detected daughters + ostia + direction arrows, in one view."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    traces = []
    av, af = _surface(case.mk, case.spacing)
    if av is not None:
        w = _to_world(case, av)
        traces.append(go.Mesh3d(
            x=w[:, 0], y=w[:, 1], z=w[:, 2],
            i=af[:, 0], j=af[:, 1], k=af[:, 2],
            color="#d94a4a", opacity=0.25, name="aorta (given mask)",
            showlegend=True, hovertemplate="parent aorta<extra></extra>",
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15)))

    dv = np.zeros_like(case.mk)
    for d in daughters:
        dv |= (labels == d["label"])
    bv, bf = _surface(dv, case.spacing)
    if bv is not None:
        w = _to_world(case, bv)
        traces.append(go.Mesh3d(
            x=w[:, 0], y=w[:, 1], z=w[:, 2],
            i=bf[:, 0], j=bf[:, 1], k=bf[:, 2],
            color="#f2c14e", opacity=0.95, name="detected daughters",
            showlegend=True, hovertemplate="daughter lumen<extra></extra>",
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15)))

    recs = pred["daughters"]
    if recs:
        O = np.array([r["ostium_xyz_mm"] for r in recs], float)
        U = np.array([r["direction_xyz"] for r in recs], float)
        S = np.array([r["seed_xyz_mm"] for r in recs], float)
        labels_txt = [
            f"{r['instance_id']}<br>ostium "
            f"({r['ostium_xyz_mm'][0]:.1f}, {r['ostium_xyz_mm'][1]:.1f}, "
            f"{r['ostium_xyz_mm'][2]:.1f}) mm<br>"
            f"radius {r['radius_mm']} mm<br>"
            f"origin &oslash; {r['origin_diameter_estimate_mm']} mm<br>"
            f"trace {r['centerline_length_mm']} mm"
            + ("<br><b>stopped at bifurcation</b>"
               if r["stopped_at_first_bifurcation"] else "")
            for r in recs]

        traces.append(go.Scatter3d(
            x=O[:, 0], y=O[:, 1], z=O[:, 2], mode="markers",
            marker=dict(size=7, color="#111111", symbol="diamond",
                        line=dict(width=1, color="white")),
            name=f"ostia ({len(recs)})", text=labels_txt,
            hovertemplate="%{text}<extra></extra>", showlegend=True))

        seg = np.full((len(recs) * 3, 3), np.nan)
        seg[0::3], seg[1::3] = O, O + U * arrow_mm
        traces.append(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(width=6, color="#111111"), name="direction",
            hoverinfo="skip", showlegend=True))
        tip = O + U * arrow_mm
        traces.append(go.Cone(
            x=tip[:, 0], y=tip[:, 1], z=tip[:, 2],
            u=U[:, 0], v=U[:, 1], w=U[:, 2],
            sizemode="absolute", sizeref=2.5, anchor="tail",
            showscale=False, colorscale=[[0, "#111111"], [1, "#111111"]],
            name="arrowheads", hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter3d(
            x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="markers",
            marker=dict(size=4, color="#2f9e9e"),
            name="seeds (5 mm)", text=[r["instance_id"] for r in recs],
            hovertemplate="%{text} seed<extra></extra>", showlegend=True))

        for r in recs:
            C = np.array(r["centerline_xyz_mm"], float)
            if len(C) > 1:
                traces.append(go.Scatter3d(
                    x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="lines",
                    line=dict(width=3, color="#2f9e9e"),
                    name="proximal path", legendgroup="path",
                    showlegend=(r is recs[0]), hoverinfo="skip"))

    if not traces:
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"<b>{case_id} &mdash; detected aortic daughters</b>"
                        f"<br><span style='font-size:13px'>{len(recs)} branches"
                        f" &middot; band {pred['method']['band_hu'][0]:.0f}"
                        f"-{pred['method']['band_hu'][1]:.0f} HU &middot; "
                        f"arrows are the reported direction, {arrow_mm:g} mm"
                        f"</span>", x=0.01, xanchor="left"),
        scene=dict(aspectmode="data",
                   xaxis_title="x (mm, LPS)", yaxis_title="y (mm, LPS)",
                   zaxis_title="z (mm, LPS)",
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))),
        legend=dict(itemsizing="constant", title="click to toggle"),
        margin=dict(l=0, r=0, t=74, b=0), template="plotly_white")
    fig.write_html(str(path), include_plotlyjs="inline", full_html=True)
    return path


# -----------------------------------------------------------------------------

def write_all(out_json, extras_dir, case_id, case, daughters, labels, pred,
              json_only=False):
    """Write prediction.json always; the rest into extras_dir unless suppressed."""
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(pred, fh, indent=2)
    written = [out_json]
    if json_only:
        return written

    os.makedirs(extras_dir, exist_ok=True)
    written.append(write_label_volume(
        os.path.join(extras_dir, "daughters.nii.gz"), case, daughters, labels))
    written.append(write_snap_labels(
        os.path.join(extras_dir, "snap_labels.txt"), len(daughters)))
    written.append(write_ostia_csv(
        os.path.join(extras_dir, "ostia.csv"), pred))
    html = write_check_html(os.path.join(extras_dir, "check.html"),
                            case_id, case, daughters, labels, pred)
    if html:
        written.append(html)
    return written