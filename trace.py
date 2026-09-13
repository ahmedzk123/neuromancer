"""
trace.py -- the proximal trace, per the challenge's own rule.

"For each eligible daughter, trace the proximal branch for up to 10 mm beyond
the ostium or until the first downstream bifurcation, whichever occurs first.
This proximal path should be used to estimate the daughter seed, local radius
and initial direction."

Three things this does that a fixed Euclidean window cannot:

  * The trace is GEODESIC inside the daughter. Euclidean shells cut across a
    curving vessel, so a branch that bends inside 10 mm gets its own far wall
    counted as proximal.
  * A bifurcation is a geodesic shell splitting into two limbs that BOTH
    persist. Requiring persistence stops a one-shell speckle, or a graze against
    an adjacent structure, from truncating the trace.
  * The radius is an area-equivalent radius on a plane perpendicular to the
    path, matching the reference annotation's stated method, rather than the
    distance transform of a binary blob. Those measure different things: a blob
    EDT on a 1.5 mm grid is quantised to half-voxel steps.

All of the heavy work happens inside the daughter's own bounding box. Running
`ndi.label` or a fast-marching front over the full crop once per shell per
daughter is what takes a case from under a second to over ten.

Reported directions are computed from PHYSICAL coordinates, so an oblique
direction matrix is handled correctly; sampling geometry uses index-aligned
millimetres, which differs only by a rotation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = ["TraceConfig", "trace_daughter"]


class TraceConfig:
    max_trace_mm = 10.0           # challenge: maximum proximal trace
    seed_mm = 5.0                 # challenge: seed is 5 mm along the path
    resample_mm = 0.25            # centreline pitch, matching the reference
    bifurcation_min_voxels = 2    # a limb smaller than this is speckle
    bifurcation_persist = 2       # consecutive split shells before we believe it
    bifurcation_skip_mm = 1.5     # ignore splits inside the junction itself
    plane_pitch_mm = 0.25         # cross-section sampling pitch
    envelope_k = 2.5              # envelope = k x inscribed radius, clipped
    envelope_min_mm = 1.5
    envelope_max_mm = 6.0
    origin_probe_mm = 1.5         # where the origin diameter is measured
    bbox_pad_vox = 2


# -----------------------------------------------------------------------------
# geodesic distance inside one component
# -----------------------------------------------------------------------------

def _geodesic(comp, start_vox, sp):
    """Distance in mm from `start_vox` to every voxel of `comp`, along `comp`.

    Falls back to Euclidean if skimage.graph is unavailable, which degrades the
    trace on a curving vessel but never breaks the pipeline.
    """
    start = np.round(np.asarray(start_vox, float)).astype(int)
    start = np.clip(start, 0, np.array(comp.shape) - 1)
    if not comp[tuple(start)]:
        pts = np.argwhere(comp)
        start = pts[int(np.argmin(np.linalg.norm((pts - start) * sp, axis=1)))]
    try:
        from skimage.graph import MCP_Geometric
        cost = np.where(comp, 1.0, np.inf)
        mcp = MCP_Geometric(cost, sampling=tuple(sp))
        geo, _ = mcp.find_costs([tuple(int(v) for v in start)])
        geo = np.asarray(geo, float)
        geo[~comp] = np.inf
        return geo, start
    except Exception:
        idx = np.argwhere(comp).astype(float)
        geo = np.full(comp.shape, np.inf)
        geo[comp] = np.linalg.norm((idx - start) * sp, axis=1)
        return geo, start


# -----------------------------------------------------------------------------
# shell walk: centreline + bifurcation stop
# -----------------------------------------------------------------------------

def _walk(comp, geo, edt, sp, tc):
    """Step outward in geodesic shells; stop at the first split that persists.

    Returns (raw centreline in sub-volume voxels, stop distance mm, stopped).
    """
    step = float(min(sp)) * 0.5
    thick = float(min(sp))
    struct = np.ones((3, 3, 3), bool)
    line, split_run, stop_mm, stopped = [], 0, tc.max_trace_mm, False

    d = 0.0
    while d <= tc.max_trace_mm + 1e-9:
        sel = comp & (geo >= d) & (geo < d + thick)
        if sel.any():
            lab, n = ndi.label(sel, structure=struct)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            limbs = int((sizes >= tc.bifurcation_min_voxels).sum())
            if d >= tc.bifurcation_skip_mm and limbs >= 2:
                split_run += 1
                if split_run >= tc.bifurcation_persist:
                    stop_mm = max(d - (tc.bifurcation_persist - 1) * step,
                                  tc.bifurcation_skip_mm)
                    stopped = True
                    break
            else:
                split_run = 0
            main = int(np.argmax(sizes))
            pts = np.argwhere(lab == main)
            w = edt[tuple(pts.T)] + 1e-6        # bias toward the lumen centre
            line.append((d, (pts * w[:, None]).sum(0) / w.sum()))
        d += step

    line = [(dd, p) for dd, p in line if dd <= stop_mm + 1e-9]
    return line, stop_mm, stopped


def _resample(line, sp, pitch, cap_mm):
    """Resample a polyline to a fixed pitch along arc length, in millimetres."""
    P = np.asarray([p for _d, p in line], float)
    if len(P) < 2:
        return P
    seg = np.linalg.norm(np.diff(P * sp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = min(float(s[-1]), cap_mm)
    if total <= 0:
        return P[:1]
    t = np.arange(0.0, total + 1e-9, pitch)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


# -----------------------------------------------------------------------------
# cross-section measurement
# -----------------------------------------------------------------------------

def _basis(u):
    a = (np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9
         else np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(u, a)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    return e1, np.cross(u, e1)


def _cross_section(ct, dout, sp, centre_vox, u_idx, thr_hu, r_env, tc):
    """Area-equivalent radius on a plane perpendicular to the path.

    Samples the CT on a disc at 0.25 mm, thresholds, keeps the region containing
    the centre, and returns sqrt(area/pi). Parent-lumen voxels are excluded so
    the measurement cannot bleed back into the aorta we started from.
    """
    pitch = tc.plane_pitch_mm
    e1, e2 = _basis(u_idx)
    k = int(round(r_env / pitch))
    g = np.arange(-k, k + 1) * pitch
    gi, gj = np.meshgrid(g, g, indexing="ij")
    rad = np.sqrt(gi ** 2 + gj ** 2)

    centre_mm = np.asarray(centre_vox, float) * sp
    pts_mm = (centre_mm[None, None, :]
              + gi[..., None] * e1[None, None, :]
              + gj[..., None] * e2[None, None, :])
    pts_vox = (pts_mm / sp).reshape(-1, 3).T

    hu = ndi.map_coordinates(ct, pts_vox, order=1, mode="constant", cval=-1000.0)
    do = ndi.map_coordinates(dout, pts_vox, order=1, mode="constant", cval=0.0)
    inside = ((hu >= thr_hu) & (do > 0.5 * float(min(sp)))
              & (rad.ravel() <= r_env)).reshape(gi.shape)

    c = (k, k)
    if not inside[c]:
        return None, "centre_below_threshold", False
    lab, n = ndi.label(inside)
    if n == 0:
        return None, "no_region", False
    reg = lab == lab[c]
    area = float(reg.sum()) * pitch * pitch
    touches = bool((rad[reg] > r_env - 1.5 * pitch).any())
    return (float(np.sqrt(area / np.pi)),
            "search_envelope_limited" if touches
            else "approximate_threshold_estimate", touches)


# -----------------------------------------------------------------------------

def trace_daughter(case, comp, ostium_vox, tc=TraceConfig(), thr_hu=None):
    """Everything downstream of the ostium for one daughter.

    `comp` is a boolean array on the crop grid. Returns a dict of measurements
    in crop and physical coordinates, or None if the component cannot be traced.
    """
    thr_hu = float(case.lo) if thr_hu is None else float(thr_hu)
    sp = case.sp

    # --- work inside the daughter's own bounding box -------------------------
    idx = np.where(comp)
    if len(idx[0]) == 0:
        return None
    pad = tc.bbox_pad_vox
    sl, off = [], []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - pad, 0)
        hi = min(int(idx[ax].max()) + pad + 1, comp.shape[ax])
        sl.append(slice(lo, hi))
        off.append(lo)
    sl, off = tuple(sl), np.array(off, float)

    sub = comp[sl]
    sub_ct = case.ct[sl]
    sub_do = case.dist_out[sl].astype(np.float32)
    ost_sub = np.asarray(ostium_vox, float) - off

    edt = ndi.distance_transform_edt(sub, sampling=tuple(sp))
    geo, _start = _geodesic(sub, ost_sub, sp)

    line, stop_mm, stopped = _walk(sub, geo, edt, sp, tc)
    if len(line) < 2:
        return None
    cl = _resample(line, sp, tc.resample_mm, min(stop_mm, tc.max_trace_mm))
    if len(cl) < 2:
        return None

    seg = np.linalg.norm(np.diff(cl * sp, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length_mm = float(s[-1])

    # --- seed: 5 mm along the path, or the path end if it stopped sooner -----
    s_seed = min(tc.seed_mm, length_mm)
    seed_sub = np.array([np.interp(s_seed, s, cl[:, k]) for k in range(3)])
    seed_truncated = bool(s_seed < tc.seed_mm - 1e-6)

    # --- direction: PCA in PHYSICAL space, oriented away from the ostium -----
    cl_crop = cl + off
    phys = np.array([case.to_mm(v) for v in cl_crop], float)
    if len(phys) >= 3:
        _, _, vv = np.linalg.svd(phys - phys.mean(axis=0), full_matrices=False)
        u_phys = vv[0]
        if np.dot(phys[-1] - phys[0], u_phys) < 0:
            u_phys = -u_phys
    else:
        u_phys = phys[-1] - phys[0]
    n = np.linalg.norm(u_phys)
    u_phys = u_phys / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    u_idx = (cl[-1] - cl[0]) * sp
    n = np.linalg.norm(u_idx)
    u_idx = u_idx / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    # --- radius at the seed, with an envelope sized to the vessel ------------
    r_ins = float(ndi.map_coordinates(edt, seed_sub[:, None], order=1)[0])
    r_env = float(np.clip(tc.envelope_k * max(r_ins, 0.5),
                          tc.envelope_min_mm, tc.envelope_max_mm))
    radius, r_status, r_touch = _cross_section(
        sub_ct, sub_do, sp, seed_sub, u_idx, thr_hu, r_env, tc)
    if radius is None:
        radius, r_status = r_ins, "inscribed_fallback"

    # --- origin diameter, for the 2 mm eligibility rule ----------------------
    d_edt = 2.0 * float(ndi.map_coordinates(
        edt, ost_sub[:, None], order=1)[0])
    s_probe = min(tc.origin_probe_mm, length_mm)
    probe = np.array([np.interp(s_probe, s, cl[:, k]) for k in range(3)])
    d_plane, _st, _t = _cross_section(
        sub_ct, sub_do, sp, probe, u_idx, thr_hu, r_env, tc)
    d_plane = 0.0 if d_plane is None else 2.0 * d_plane

    traced = int((sub & (geo <= stop_mm)).sum())
    return {
        "centerline_vox": cl_crop,
        "centerline_mm": phys,
        "centerline_length_mm": round(length_mm, 3),
        "stopped_at_bifurcation": stopped,
        "seed_vox": seed_sub + off,
        "seed_mm": case.to_mm(seed_sub + off),
        "seed_truncated": seed_truncated,
        "ostium_mm": case.to_mm(ostium_vox),
        "direction_xyz": u_phys,
        "radius_mm": round(float(radius), 3),
        "radius_status": r_status,
        "radius_touches_envelope": bool(r_touch),
        "threshold_hu": round(thr_hu, 2),
        "search_radius_mm": round(r_env, 3),
        "origin_diameter_mm": round(float(max(d_edt, d_plane)), 3),
        "origin_diameter_edt_mm": round(float(d_edt), 3),
        "origin_diameter_plane_mm": round(float(d_plane), 3),
        "traced_voxel_count": traced,
    }
