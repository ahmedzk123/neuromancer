"""
detector.py -- the FROZEN detector.

Configuration: sd:3 + absolute floor 230 HU + 5 mm reach, validated across five
annotated subjects. Every constant below is a measured choice. Do not tune them
without re-running score.py on the development set.

Pipeline (steps as specified):
  1 crop            bounding box of the aorta mask, padded 45 mm
  2 lumen model     erode mask 2 mm, mu = mean, sd = std of the CT inside
  3 band            [230 HU, mu + 3*sd]   -- floor ABSOLUTE, ceiling relative
  4 bone exclusion  > max(400, mu + 3*sd), close x2, fill holes, dilate 2 mm
  5 growth          two-phase: 20 mm, flag components > 8 mL as leaks, regrow 30 mm
  6 candidates      components >= 8 mm3; ostium = contact voxel maximising the
                    component distance transform
  7 eligibility     discard candidates never reaching 5 mm beyond the wall
  8 post            drop ostia within 6 mm of the mask ends; merge within 6 mm

The ostium POSITION comes from here. The seed, direction, radius and origin
diameter come from trace.py, which implements the challenge's own proximal-trace
rule (<=10 mm or first bifurcation).
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

__all__ = ["Config", "Case", "detect"]


# -----------------------------------------------------------------------------
# frozen constants
# -----------------------------------------------------------------------------

class Config:
    """Every number the detector uses. Frozen; exposed so run.py can print it."""

    # step 1
    margin_mm = 45.0
    # step 2
    erode_mm = 2.0
    # step 3
    k_sd = 3.0
    floor_hu = 230.0            # absolute lower edge; NOT mu - k*sd
    # step 4
    bone_floor_hu = 400.0       # actual cut is max(this, mu + bone_k_sd*sd)
    bone_k_sd = 3.0
    bone_dilate_mm = 2.0
    # step 5
    phase1_mm = 20.0
    phase2_mm = 30.0
    leak_mL = 8.0
    # step 6
    min_component_mm3 = 8.0
    max_candidates = 25
    # step 7
    min_reach_mm = 5.0
    # challenge eligibility: origin must be at least 2 mm across. Measured after
    # tracing, in run.py, because the diameter comes from the proximal path.
    min_origin_diameter_mm = 2.0
    # step 8
    end_margin_mm = 6.0
    merge_mm = 6.0

    def as_dict(self):
        return {k: v for k, v in vars(type(self)).items()
                if not k.startswith("_") and isinstance(v, (int, float))}


# -----------------------------------------------------------------------------
# I/O and case geometry
# -----------------------------------------------------------------------------

def _mask_bbox(mk, pad_vox):
    idx = np.where(mk > 0)
    if len(idx[0]) == 0:
        return tuple(slice(0, s) for s in mk.shape)
    out = []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - pad_vox[ax], 0)
        hi = min(int(idx[ax].max()) + pad_vox[ax] + 1, mk.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)


class Case:
    """One CT + aorta mask, cropped, with the intensity and bone models built.

    Array axes are [z, y, x]; `sp` is the matching (dz, dy, dx) spacing in mm.
    `to_mm` is the only place voxel indices become physical coordinates, and it
    goes through SimpleITK so origin, spacing and direction are all honoured.
    """

    def __init__(self, image_path, mask_path, cfg=Config()):
        self.cfg = cfg
        self.img = sitk.ReadImage(str(image_path))
        self.msk = sitk.ReadImage(str(mask_path))
        if self.img.GetSize() != self.msk.GetSize():
            raise ValueError(f"Grid mismatch: image {self.img.GetSize()} vs "
                             f"mask {self.msk.GetSize()}.")

        ct = sitk.GetArrayFromImage(self.img).astype(np.float32)
        mk = (sitk.GetArrayFromImage(self.msk) > 0)
        self.full_shape = ct.shape
        self.spacing = self.img.GetSpacing()                  # (sx, sy, sz)
        sx, sy, sz = self.spacing
        self.sp = np.array([sz, sy, sx], float)
        self.vox_mm3 = float(sx * sy * sz)

        # --- step 2: lumen model, on the FULL volume (not the crop) ----------
        er = max(int(round(cfg.erode_mm / min(sx, sy))), 1)
        core = ndi.binary_erosion(mk, iterations=er)
        if core.sum() < 50:
            core = mk
        lumen = ct[core]
        self.mu = float(lumen.mean())
        self.sd = float(lumen.std())

        # --- step 1: crop ----------------------------------------------------
        pad = (int(round(cfg.margin_mm / sz)), int(round(cfg.margin_mm / sy)),
               int(round(cfg.margin_mm / sx)))
        self.region = _mask_bbox(mk, pad)
        self.ct = ct[self.region]
        self.mk = mk[self.region]
        self.z0 = self.region[0].start or 0
        self.y0 = self.region[1].start or 0
        self.x0 = self.region[2].start or 0

        # --- step 3: band ----------------------------------------------------
        self.lo = float(cfg.floor_hu)
        self.hi = self.mu + cfg.k_sd * self.sd

        # --- step 4: bone ----------------------------------------------------
        self.dist_out = ndi.distance_transform_edt(~self.mk, sampling=self.sp)
        self.bone_cut = max(cfg.bone_floor_hu, self.mu + cfg.bone_k_sd * self.sd)
        bone = ndi.binary_closing(self.ct > self.bone_cut, iterations=2)
        bone = ndi.binary_fill_holes(bone)
        it = max(int(round(cfg.bone_dilate_mm / min(sx, sy))), 1)
        self.bone = ndi.binary_dilation(bone, iterations=it)

        self.band = (self.ct >= self.lo) & (self.ct <= self.hi) & ~self.bone

    # -- coordinates ---------------------------------------------------------

    def to_mm(self, vox_zyx):
        """Crop voxel (z, y, x), possibly fractional -> physical (x, y, z) mm.

        Fractional indices are handled by interpolating between the physical
        points of the two bracketing integer indices along each axis, which is
        exact for an affine index->physical map and keeps sub-voxel centreline
        points honest.
        """
        v = np.asarray(vox_zyx, float)
        base = np.floor(v).astype(int)
        frac = v - base
        o = np.array(self.img.TransformIndexToPhysicalPoint(
            (int(base[2]) + self.x0, int(base[1]) + self.y0,
             int(base[0]) + self.z0)), float)
        out = o.copy()
        for ax, step in ((2, (1, 0, 0)), (1, (0, 1, 0)), (0, (0, 0, 1))):
            if frac[ax] == 0.0:
                continue
            p = np.array(self.img.TransformIndexToPhysicalPoint(
                (int(base[2]) + self.x0 + step[0],
                 int(base[1]) + self.y0 + step[1],
                 int(base[0]) + self.z0 + step[2])), float)
            out = out + (p - o) * frac[ax]
        return out

    def full_index(self, vox_zyx):
        """Crop voxel -> index in the original, uncropped volume."""
        v = np.asarray(vox_zyx, float)
        return np.array([v[0] + self.z0, v[1] + self.y0, v[2] + self.x0])


# -----------------------------------------------------------------------------
# steps 5-8
# -----------------------------------------------------------------------------

def _grow(case, limit_mm, blocked):
    allowed = case.band & (case.dist_out <= limit_mm) & ~blocked
    lab, _ = ndi.label(allowed | case.mk)
    touch = [int(v) for v in np.unique(lab[case.mk]) if v > 0]
    return np.isin(lab, touch) & ~case.mk


def _two_phase_growth(case):
    """Step 5. Grow to 20 mm, block components that blow up, regrow to 30 mm."""
    cfg = case.cfg
    zeros = np.zeros_like(case.band)
    g1 = _grow(case, cfg.phase1_mm, zeros)
    lab1, n1 = ndi.label(g1)
    blocked, n_leak = zeros, 0
    if n1:
        mL = np.bincount(lab1.ravel()) * case.vox_mm3 / 1000.0
        leaks = [i for i in range(1, n1 + 1) if mL[i] > cfg.leak_mL]
        n_leak = len(leaks)
        if leaks:
            blocked = np.isin(lab1, leaks)
    return _grow(case, cfg.phase2_mm, blocked), n_leak


def _link_to_aorta(pts_vox, dist_out, axis_mm, sp, sub_mk,
                   max_gap_mm=15.0, step_mm=0.4, hit_mm=0.9, end_mm=6.0,
                   patch_mm=4.0):
    """Step 6 fallback: march a detached component's local axis back to the wall.

    Starts from the CENTROID of the aorta-facing end and uses that end's own
    axis, because a curving branch's global axis points nowhere useful.
    """
    pts_mm = pts_vox * sp
    d_here = dist_out[tuple(pts_vox.T.astype(int))]
    near_i = int(np.argmin(d_here))
    dsel = np.linalg.norm(pts_mm - pts_mm[near_i], axis=1) <= end_mm
    if dsel.sum() >= 4:
        end_pts = pts_mm[dsel]
        start_mm = end_pts.mean(axis=0)
        _, _, vv = np.linalg.svd(end_pts - start_mm, full_matrices=False)
        u = vv[0]
    else:
        start_mm, u = pts_mm[near_i], np.asarray(axis_mm, float)
    u = u / max(np.linalg.norm(u), 1e-9)
    p0 = start_mm / sp

    best = (False, np.inf, None)
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
                    wall = (ndi.binary_dilation(sub_mk, iterations=1)
                            & ~ndi.binary_erosion(sub_mk, iterations=1))
                    wp = np.argwhere(wall).astype(float)
                    if len(wp):
                        dd = np.linalg.norm((wp - p) * sp, axis=1)
                        loc = wp[dd <= patch_mm]
                        if len(loc) >= 3:
                            ost = loc.mean(axis=0)
                    best = (True, float(t), ost)
                break
    return best


def _candidates(case, grown):
    """Steps 6 and 7. One ostium per eligible component.

    Returns a list of dicts with the component label and the ostium in crop
    voxel coordinates, largest component first.
    """
    cfg = case.cfg
    lab, n = ndi.label(grown)
    if n == 0:
        return lab, []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    min_vox = max(int(cfg.min_component_mm3 / case.vox_mm3), 1)
    order = [int(i) for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]

    edt = ndi.distance_transform_edt(grown, sampling=tuple(case.sp))
    near_wall = ndi.binary_dilation(case.mk, iterations=2)

    out = []
    for i in order[:cfg.max_candidates]:
        comp = lab == i
        # step 7: eligibility -- must leave the wall by at least 5 mm
        if float(case.dist_out[comp].max()) < cfg.min_reach_mm:
            continue
        contact = comp & near_wall
        if contact.any():
            cp = np.argwhere(contact)
            ost = cp[int(np.argmax(edt[tuple(cp.T)]))].astype(float)
        else:
            pts = np.argwhere(comp).astype(float)
            pmm = pts * case.sp
            cen = pmm - pmm.mean(axis=0)
            st = max(len(cen) // 4000, 1)
            _, _, vv = np.linalg.svd(cen[::st], full_matrices=False)
            hit, _gap, ost = _link_to_aorta(pts, case.dist_out, vv[0],
                                            case.sp, case.mk)
            if not hit or ost is None:
                continue
            ost = np.asarray(ost, float)
        out.append({"label": i, "ostium_vox": ost,
                    "n_voxels": int(comp.sum()),
                    "volume_mm3": float(comp.sum()) * case.vox_mm3})
    return lab, out


def _drop_end_caps(case, cands):
    """Step 8a. The flat cropped ends of the mask are not branch origins."""
    zs = np.where(case.mk.any(axis=(1, 2)))[0]
    if not len(zs):
        return cands
    sz = case.spacing[2]
    zlo, zhi = zs[0] * sz, zs[-1] * sz
    m = case.cfg.end_margin_mm
    return [c for c in cands
            if (c["ostium_vox"][0] * sz - zlo) >= m
            and (zhi - c["ostium_vox"][0] * sz) >= m]


def _dedupe(case, cands):
    """Step 8b. Two ostia closer than 6 mm are one origin."""
    kept = []
    for c in cands:
        p = np.asarray(c["ostium_vox"], float) * case.sp
        if any(np.linalg.norm(p - np.asarray(k["ostium_vox"], float) * case.sp)
               < case.cfg.merge_mm for k in kept):
            continue
        kept.append(c)
    return kept


def detect(case):
    """Run the frozen detector. Returns (component_labels, candidates, info)."""
    grown, n_leak = _two_phase_growth(case)
    lab, cands = _candidates(case, grown)
    cands = _dedupe(case, _drop_end_caps(case, cands))
    info = {
        "lumen_mean_hu": round(case.mu, 1),
        "lumen_sd_hu": round(case.sd, 1),
        "band_hu": [round(case.lo, 1), round(case.hi, 1)],
        "bone_cut_hu": round(case.bone_cut, 1),
        "leaks_blocked": n_leak,
        "grown_mL": round(float(grown.sum()) * case.vox_mm3 / 1000.0, 2),
        "crop_shape": list(case.ct.shape),
    }
    return lab, cands, info