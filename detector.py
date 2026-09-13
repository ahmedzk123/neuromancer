"""
detector.py -- frozen daughter-artery detector.

Configuration: k_sd=3.0 ceiling, floor_hu=230 (absolute), min_reach_mm=5.0.
Validated on 5 annotated cases (EVAL_SET): pooled P=0.77, R=0.53, F1=0.63,
mean ostium error 2.34mm, 0.05-0.19s/case. See README.md for the regression
table this must keep reproducing.

Method, in order:
  1. crop to the aorta mask's bounding box, padded 45mm
  2. lumen model: erode mask 2mm in-plane, mu/sd = mean/std of CT inside
  3. band: floor_hu (absolute) to mu + k_sd*sd (relative)
  4. bone exclusion: > max(400, mu+3*sd), close x2, fill holes, dilate 2mm
  5. two-phase growth (Tahoces-style): grow to 20mm, block any component
     over leak_mL as organ leakage, regrow to 30mm avoiding blocked regions
  6. connected components >= 8mm3 -> one ostium each
  7. eligibility: reach >= min_reach_mm beyond the wall, origin diameter >= 2mm
  8. proximal trace: skeletonize the component, walk from the ostium, stop at
     10mm or the first real bifurcation (spurs under 1.5mm are pruned as
     skeletonization noise, not counted as forks)
  9. seed = point at 5mm arc-length along the trace; radius = local EDT value
     at the seed; direction = least-squares fit through the trace
  10. drop candidates within 6mm of the mask's cropped cranial/caudal end;
      merge candidates closer than 6mm
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

K_SD = 3.0
FLOOR_HU = 230.0
MIN_RECH_MM = 5.0
MIN_ORIGIN_DIAM_MM = 2.0
MIN_COMPONENT_MM3 = 8.0
PHASE1_MM = 20.0
PHASE2_MM = 30.0
LEAK_ML = 8.0
TRACE_MAX_MM = 10.0
SEED_ARC_MM = 5.0
SPUR_PRUNE_MM = 1.5
END_CAP_MARGIN_MM = 6.0
DEDUPE_MERGE_MM = 6.0
CROP_MARGIN_MM = 45.0
ERODE_MM = 2.0
BONE_HI_SD = 3.0
BONE_DILATE_MM = 2.0


def _mask_bbox(mk, pad_vox):
    idx = np.where(mk)
    out = []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - pad_vox[ax], 0)
        hi = min(int(idx[ax].max()) + pad_vox[ax] + 1, mk.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)


class Ctx:
    """Cropped volumes + the measured intensity model, shared by every step."""

    def __init__(self, img, ct, mk, spacing,
                 k_sd=K_SD, floor_hu=FLOOR_HU, min_reach_mm=MIN_RECH_MM):
        self.img = img
        sx, sy, sz = spacing
        self.sx, self.sy, self.sz = sx, sy, sz
        self.sp = np.array([sz, sy, sx])           # (z,y,x) mm-per-voxel, matches array order
        self.vox_mm3 = sx * sy * sz
        self.min_reach_mm = min_reach_mm

        pad = (int(round(CROP_MARGIN_MM / sz)), int(round(CROP_MARGIN_MM / sy)),
               int(round(CROP_MARGIN_MM / sx)))
        self.region = _mask_bbox(mk, pad)
        self.ct = ct[self.region]
        self.mk = mk[self.region]
        self.z0 = self.region[0].start
        self.y0 = self.region[1].start
        self.x0 = self.region[2].start

        # -- lumen model: erode 2mm IN-PLANE only (matches x/y resolution, not z) --
        er_iter = max(int(round(ERODE_MM / min(sx, sy))), 1)
        core = ndi.binary_erosion(self.mk, iterations=er_iter)
        if core.sum() < 50:
            core = self.mk
        lumen = self.ct[core]
        self.mu, self.sd = float(lumen.mean()), float(lumen.std())

        # -- band: absolute floor, relative ceiling --
        self.lo = float(floor_hu)
        self.hi = self.mu + k_sd * self.sd

        # -- bone exclusion --
        self.dist_out = ndi.distance_transform_edt(~self.mk, sampling=(sz, sy, sx))
        hi_cut = max(400.0, self.mu + BONE_HI_SD * self.sd)
        bone = ndi.binary_closing(self.ct > hi_cut, iterations=2)
        bone = ndi.binary_fill_holes(bone)
        bit = max(int(round(BONE_DILATE_MM / min(sx, sy))), 1)
        self.bone = ndi.binary_dilation(bone, iterations=bit)

        self.wall = (ndi.binary_dilation(self.mk, iterations=1)
                     & ~ndi.binary_erosion(self.mk, iterations=1))

    def to_mm(self, vox_zyx):
        """voxel (z,y,x) in the crop -> physical (x,y,z) mm."""
        v = np.asarray(vox_zyx, float)
        return np.array(self.img.TransformIndexToPhysicalPoint((
            int(round(v[2])) + self.x0, int(round(v[1])) + self.y0,
            int(round(v[0])) + self.z0)), float)

    def to_mm_continuous(self, vox_zyx):
        """Same as to_mm but sub-voxel (for interpolated points along a trace)."""
        v = np.asarray(vox_zyx, float)
        return np.array(self.img.TransformContinuousIndexToPhysicalPoint((
            v[2] + self.x0, v[1] + self.y0, v[0] + self.z0)), float)

    def mm_to_vox_continuous(self, xyz_mm):
        """Inverse of to_mm_continuous: physical mm -> crop-local (z,y,x) voxel."""
        xx, yy, zz = self.img.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in xyz_mm))
        return np.array([zz - self.z0, yy - self.y0, xx - self.x0], float)


def grow_expansion(ctx):
    """Two-phase Tahoces-style growth with leak detection. Returns (mask, n_leaks)."""
    band = (ctx.ct >= ctx.lo) & (ctx.ct <= ctx.hi) & ~ctx.bone

    def grow(limit_mm, blocked):
        allowed = band & (ctx.dist_out <= limit_mm) & ~blocked
        lab, _ = ndi.label(allowed | ctx.mk)
        touch = set(int(v) for v in np.unique(lab[ctx.mk]) if v > 0)
        return np.isin(lab, list(touch)) & ~ctx.mk

    g1 = grow(PHASE1_MM, np.zeros_like(band))
    lab1, n1 = ndi.label(g1)
    blocked = np.zeros_like(band)
    n_leak = 0
    if n1:
        sizes = np.bincount(lab1.ravel()) * ctx.vox_mm3 / 1000.0
        leaks = [i for i in range(1, n1 + 1) if sizes[i] > LEAK_ML]
        n_leak = len(leaks)
        if leaks:
            blocked = np.isin(lab1, leaks)

    g2 = grow(PHASE2_MM, blocked)
    return g2, n_leak


def _components(mask, ctx, min_mm3=MIN_COMPONENT_MM3):
    lab, _ = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(min_mm3 / ctx.vox_mm3), 1)
    keep = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]
    return lab, keep


def _link_to_aorta(pts_vox, ctx, axis_mm, max_gap_mm=15.0, step_mm=0.4,
                   hit_mm=0.9, end_mm=6.0, patch_mm=4.0):
    """March from a candidate's aorta-facing END along its LOCAL axis to the
    wall (ported unchanged from the validated algorithm).

    Three things matter for where the ostium lands: start from the CENTROID
    of the near end, not the single nearest voxel; use the axis of just the
    near end, not the whole component (a curved vessel's global axis points
    nowhere useful); once the ray hits, snap to the centre of the local wall
    patch, which is what "centre of the opening" means.
    """
    sp = ctx.sp
    pts_mm = pts_vox * sp
    d_here = ctx.dist_out[tuple(pts_vox.T.astype(int))]
    near_i = int(np.argmin(d_here))

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

    best = (False, np.inf, None)
    for sign in (1.0, -1.0):
        prev = None
        for t in np.arange(step_mm, max_gap_mm + step_mm, step_mm):
            p = p0 + sign * u * t / sp
            if np.any(p < 0) or np.any(p >= np.array(ctx.dist_out.shape) - 1):
                break
            d = float(ndi.map_coordinates(ctx.dist_out, p[:, None], order=1)[0])
            if prev is not None and d > prev + 0.6:
                break
            prev = d
            if d <= hit_mm:
                if t < best[1]:
                    ost = p
                    wp = np.argwhere(ctx.wall).astype(float)
                    if len(wp):
                        dd = np.linalg.norm((wp - p) * sp, axis=1)
                        loc = wp[dd <= patch_mm]
                        if len(loc) >= 3:
                            ost = loc.mean(axis=0)
                    best = (True, float(t), ost)
                break
    return best[2] if best[0] else None


def _find_ostium(comp, ctx, edt_full):
    """Contact-zone voxel maximising distance to the grown mask's edge (Tahoces),
    falling back to marching the component's local axis back to the wall."""
    near_wall = ndi.binary_dilation(ctx.mk, iterations=2)
    contact = comp & near_wall
    if contact.any():
        cpts = np.argwhere(contact)
        best = cpts[int(np.argmax(edt_full[tuple(cpts.T)]))]
        return best.astype(float)

    pts = np.argwhere(comp).astype(float)
    pmm = pts * ctx.sp
    cen = pmm - pmm.mean(axis=0)
    step = max(len(cen) // 4000, 1)
    _, _, vv = np.linalg.svd(cen[::step], full_matrices=False)
    axis = vv[0]
    ost = _link_to_aorta(pts, ctx, axis)
    return np.asarray(ost, float) if ost is not None else None


def _skeleton_graph(skel):
    pts = [tuple(p) for p in np.argwhere(skel)]
    pt_set = set(pts)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
               for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]
    graph = {}
    for p in pts:
        graph[p] = [tuple(np.add(p, o)) for o in offsets
                    if tuple(np.add(p, o)) in pt_set]
    return graph


def _prune_spurs(graph, ctx, min_len_mm=SPUR_PRUNE_MM):
    """Remove short leaf twigs (skeletonization noise) before bifurcation-testing."""
    graph = {k: list(v) for k, v in graph.items()}
    changed = True
    while changed:
        changed = False
        leaves = [p for p, nbrs in graph.items() if len(nbrs) == 1]
        for leaf in leaves:
            if leaf not in graph:
                continue
            path = [leaf]
            cur, prev = leaf, None
            length = 0.0
            while True:
                nbrs = [n for n in graph.get(cur, []) if n != prev]
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                length += float(np.linalg.norm(
                    (np.array(nxt) - np.array(cur)) * ctx.sp))
                path.append(nxt)
                prev, cur = cur, nxt
                if len(graph.get(cur, [])) != 2:
                    break
            if length < min_len_mm and len(graph.get(cur, [])) > 2:
                for p in path[:-1]:                # drop the twig, keep the fork node
                    for n in graph.get(p, []):
                        if p in graph.get(n, []):
                            graph[n].remove(p)
                    del graph[p]
                changed = True
    return graph


def _trace_proximal(comp, ostium_vox, ctx):
    """Skeletonize, prune noise spurs, walk from the ostium to 10mm or a fork.

    Returns a list of (z,y,x) voxel points from the ostium outward, or just
    [ostium_vox] if no usable skeleton exists.
    """
    skel = skeletonize(comp)
    if not skel.any():
        return [tuple(int(round(v)) for v in ostium_vox)]
    graph = _skeleton_graph(skel)
    graph = _prune_spurs(graph, ctx)
    if not graph:
        return [tuple(int(round(v)) for v in ostium_vox)]

    pts = np.array(list(graph.keys()))
    d = np.linalg.norm((pts - np.asarray(ostium_vox)) * ctx.sp, axis=1)
    start = tuple(pts[int(np.argmin(d))])

    path = [start]
    cur, prev, cum_mm = start, None, 0.0
    while True:
        nbrs = [n for n in graph.get(cur, []) if n != prev]
        if len(nbrs) != 1:
            break                                   # dead end or real bifurcation
        nxt = nbrs[0]
        step_mm = float(np.linalg.norm((np.array(nxt) - np.array(cur)) * ctx.sp))
        if cum_mm + step_mm > TRACE_MAX_MM:
            break                                    # truncate at 10mm
        cum_mm += step_mm
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _arc_length_point(path_mm, target_mm):
    """Point at `target_mm` arc-length along path_mm; clamps to the path end."""
    if len(path_mm) < 2:
        return path_mm[0], target_mm == 0
    cum = 0.0
    for a, b in zip(path_mm, path_mm[1:]):
        seg = float(np.linalg.norm(b - a))
        if cum + seg >= target_mm:
            t = (target_mm - cum) / max(seg, 1e-9)
            return a + t * (b - a), True
        cum += seg
    return path_mm[-1], False                        # trace shorter than target


def _fit_direction(path_mm, ostium_mm):
    if len(path_mm) < 2:
        return np.array([0.0, 0.0, 1.0])
    pts = np.array(path_mm)
    cen = pts - pts.mean(axis=0)
    _, _, vv = np.linalg.svd(cen, full_matrices=False)
    u = vv[0]
    if np.dot(pts[-1] - ostium_mm, u) < 0:
        u = -u
    return u / max(np.linalg.norm(u), 1e-9)


def _edt_radius_mm(edt_full, vox_zyx, ctx):
    zz, yy, xx = [int(round(v)) for v in vox_zyx]
    zz = np.clip(zz, 0, edt_full.shape[0] - 1)
    yy = np.clip(yy, 0, edt_full.shape[1] - 1)
    xx = np.clip(xx, 0, edt_full.shape[2] - 1)
    return float(edt_full[zz, yy, xx])


GEODESIC_HALF_WIDTH_MM = 5.0


def _truncate_to_trace(comp, path_vox, ctx):
    """daughters.nii.gz must hold only the traced proximal part, not the whole
    grown component (which can extend well past a bifurcation or the 10mm cap).
    Geodesic distance WITHIN the component from the trace, capped at a half-width
    generous enough for the branch's own thickness (~5mm) but far short of
    the remaining growth beyond a real fork -- so voxels past the truncation
    point naturally fall outside the cap without needing a second marker set.
    """
    from skimage.graph import MCP_Geometric
    cost = np.where(comp, 1.0, np.inf).astype(np.float64)
    mcp = MCP_Geometric(cost, sampling=tuple(ctx.sp))
    seeds = [list(p) for p in path_vox]
    geo, _ = mcp.find_costs(seeds)
    geo = np.nan_to_num(geo, nan=1e6, posinf=1e6)
    return comp & (geo <= GEODESIC_HALF_WIDTH_MM)


def _drop_end_caps(candidates, ctx, margin_mm=END_CAP_MARGIN_MM):
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return candidates
    zlo, zhi = zs[0] * ctx.sz, zs[-1] * ctx.sz
    return [c for c in candidates
            if (c["ostium_vox"][0] * ctx.sz - zlo) >= margin_mm
            and (zhi - c["ostium_vox"][0] * ctx.sz) >= margin_mm]


def _dedupe(candidates, ctx, merge_mm=DEDUPE_MERGE_MM):
    kept = []
    for c in candidates:
        p = np.asarray(c["ostium_vox"], float) * ctx.sp
        if any(np.linalg.norm(p - np.asarray(k["ostium_vox"], float) * ctx.sp) < merge_mm
               for k in kept):
            continue
        kept.append(c)
    return kept


def detect(img, ct, mk, spacing):
    """Run the full frozen pipeline. Returns a list of candidate dicts with
    voxel-space fields (ostium_vox, path_vox) plus physical-mm fields
    (ostium_mm, seed_mm, radius_mm, direction_xyz) ready for output writers.
    """
    ctx = Ctx(img, ct, mk, spacing)
    grown, n_leak = grow_expansion(ctx)
    edt_full = ndi.distance_transform_edt(grown, sampling=tuple(ctx.sp)) if grown.any() \
        else np.zeros_like(grown, float)

    lab, keep = _components(grown, ctx)
    candidates = []
    for i in keep:
        comp = lab == i
        if ctx.min_reach_mm > 0 and float(ctx.dist_out[comp].max()) < ctx.min_reach_mm:
            continue                                  # never leaves the wall

        ostium_vox = _find_ostium(comp, ctx, edt_full)
        if ostium_vox is None:
            continue

        origin_diam_mm = 2.0 * _edt_radius_mm(edt_full, ostium_vox, ctx)
        if origin_diam_mm < MIN_ORIGIN_DIAM_MM:
            continue                                  # origin too narrow to be eligible

        path_vox = _trace_proximal(comp, ostium_vox, ctx)
        path_mm = [ctx.to_mm_continuous(p) for p in path_vox]
        ostium_mm = ctx.to_mm_continuous(ostium_vox)
        seed_mm, _reached = _arc_length_point(path_mm, SEED_ARC_MM)
        direction = _fit_direction(path_mm, ostium_mm)
        seed_vox = ctx.mm_to_vox_continuous(seed_mm)   # same point as seed_mm, always
        radius_mm = _edt_radius_mm(edt_full, seed_vox, ctx)

        label_mask = _truncate_to_trace(comp, path_vox, ctx)

        candidates.append(dict(
            ostium_vox=ostium_vox, path_vox=path_vox,
            ostium_mm=ostium_mm, seed_mm=seed_mm,
            direction_xyz=direction, radius_mm=radius_mm,
            origin_diam_mm=origin_diam_mm, component_id=i,
            label_mask=label_mask, ctx_region=ctx.region,
        ))

    candidates = _drop_end_caps(candidates, ctx)
    candidates = _dedupe(candidates, ctx)
    for k, c in enumerate(candidates, start=1):
        c["instance_id"] = f"branch_{k:03d}"
    return candidates, ctx, grown, n_leak
