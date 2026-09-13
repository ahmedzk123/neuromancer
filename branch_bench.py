#!/usr/bin/env python3
"""
branch_bench.py -- SINGLE FILE. Four independent branch-detection methods on one
CT + aorta-mask pair. Nothing else is needed; no other module is imported.

    pip install SimpleITK numpy scipy scikit-image plotly
    python branch_bench.py --image orig1.nii --aorta-mask mask1.nii \
        --outdir bench --case-id subject001

    --band sd   (default) lumen mean +/- 2.5 sd, exactly as before.
    --band v2   absolute floor 250 HU, ceiling median + 2.5 MAD, bone cut
                relative to the ceiling AND required to be bulky. Opt-in, so
                the default behaviour of this file is unchanged.

THE METHODS
  M1 frangi        Hessian vesselness + geometric linking back to the wall.
  M2 band_connect  Naive control: intensity band connected to the aorta.
  M3 expansion     Tahoces-style two-phase growing with leak detection.
  M4 wall_flux     Unrolled shell around the wall; peaks are openings.

OUTPUT  (one folder per method, each render opens in its own browser tab)
  <outdir>/M1_frangi/ostia.csv   render.html   pred.json
  <outdir>/M2_band_connect/...
  <outdir>/M3_expansion/...
  <outdir>/M4_wall_flux/...
  <outdir>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "2026-09-14.M1-M4+band"

# --band v2 constants. Every one of these is a measurement, not a preference:
#   FLOOR_HU      subject001's own working floor (246.5) rounded, and inside
#                 subject019's measured [<=300] window for keeping all 3 GT
#                 daughters. One number that satisfies both subjects.
#   K_HI          median + 2.5*MAD reproduces subject001's ceiling (357.8 -> 357)
#                 and gives subject019 631 against a GT maximum of 625.
#   BONE_*        the bone cut must clear the lumen ceiling, so it is relative;
#                 the size test is what actually separates a vertebra from a
#                 daughter, and it holds at any contrast level.
FLOOR_HU = 250.0
K_HI = 2.5
BONE_MARGIN_HU = 60.0
MIN_BONE_ML = 2.0


# -----------------------------------------------------------------------------
# I/O and geometry
# -----------------------------------------------------------------------------

def mask_bbox(mk, pad_vox=(0, 0, 0), crop=True):
    """Slices bounding the mask, padded and clipped to the volume."""
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


def load_case(image_path, mask_path):
    """Read image + mask, verify they share a grid, return (img, msk, ct, mk)."""
    img = sitk.ReadImage(image_path)
    msk = sitk.ReadImage(mask_path)

    if img.GetSize() != msk.GetSize():
        raise ValueError(
            f"Grid mismatch: image {img.GetSize()} vs mask {msk.GetSize()}. "
            "Resample the mask onto the image before continuing.")
    if not np.allclose(img.GetSpacing(), msk.GetSpacing(), atol=1e-4):
        print("WARNING: spacing differs slightly between image and mask",
              file=sys.stderr)

    ct = sitk.GetArrayFromImage(img).astype(np.float32)          # [z, y, x]
    mk = (sitk.GetArrayFromImage(msk) > 0).astype(np.uint8)      # [z, y, x]
    return img, msk, ct, mk


# -----------------------------------------------------------------------------
# meshing for the 3D render
# -----------------------------------------------------------------------------

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
    level 0.5 is exactly what deletes them.

    Returns (verts_xyz_mm_from_crop_corner, faces), or (None, None).
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
            verts = verts[:, [2, 1, 0]]          # (z,y,x) -> (x,y,z)
            return smooth_mesh(verts, faces, iters=smooth_iters), faces
    return None, None


def write_interactive_html(out_html, meshes, title, subtitle="", axis_note=""):
    """Self-contained rotatable 3D page (plotly). Opens in any browser tab."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        if not getattr(write_interactive_html, "_warned", False):
            write_interactive_html._warned = True
            print("  NOTE: plotly not installed -- render.html skipped."
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


# -----------------------------------------------------------------------------
# intensity model, search region, vesselness, wall linking
# -----------------------------------------------------------------------------

def lumen_band(ct, mk, k_sd=2.5, erode_mm=2.0, spacing=(1, 1, 1),
               rule="sd", floor_hu=FLOOR_HU, k_hi=K_HI):
    """Learn the lumen HU band from inside the (eroded) aorta mask.

    rule="sd" is the original: mu +/- k_sd*sd, symmetric. It is kept as the
    default because it is what produced subject001's good result.

    rule="v2" is an ABSOLUTE floor with a RELATIVE ceiling, which is the
    asymmetry the two subjects actually demand:

      * the floor cannot be relative. Subject019's daughters sit 185 HU under
        its lumen -- seven MADs -- because 1.5 mm voxels partial-volume a 2 mm
        vessel down toward fat. That offset is set by resolution and vessel
        calibre, not by the lumen's variance, so no multiple of sd or MAD lands
        on it in both subjects. Measured: subject001 works at 246.5 HU,
        subject019 needs <= 300 HU. 250 satisfies both.

      * the ceiling cannot be absolute. Subject019's lumen is 564 HU, brighter
        than the level that is bone in subject001. But the offsets agree:
        subject001's working ceiling is lumen+56, subject019's brightest GT
        voxel is lumen+61. median + 2.5*MAD reproduces both.

    mu and sd are still returned as the mean and sd whatever the rule, because
    M4 and the ROI are calibrated against them and this patch is not about M4.
    """
    sx, sy, sz = spacing
    er_iter = max(int(round(erode_mm / min(sx, sy))), 1)
    core = ndi.binary_erosion(mk, iterations=er_iter)
    if core.sum() < 50:
        core = mk.astype(bool)
    lumen = ct[core]
    mu, sd = float(lumen.mean()), float(lumen.std())
    if rule == "v2":
        med = float(np.median(lumen))
        mad = max(float(np.median(np.abs(lumen - med))) * 1.4826, 8.0)
        return float(floor_hu), med + k_hi * mad, mu, sd
    return mu - k_sd * sd, mu + k_sd * sd, mu, sd


def branch_roi(sub_ct, sub_mk, spacing, roi_mm, mu, sd,
               hi_sd=3.0, lo_sd=4.0, bone_dilate_mm=2.0, bone_floor_hu=None,
               min_bone_mL=None):
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
    # Bright alone cannot identify bone once the lumen is bright: subject019's
    # blood is 564 HU. Bone is also BULKY, and a daughter vessel never is, so
    # requiring volume holds at any contrast level. Off by default.
    if min_bone_mL is not None:
        lab_b, n_b = ndi.label(bone)
        if n_b:
            mL = np.bincount(lab_b.ravel()) * (sx * sy * sz) / 1000.0
            big = mL > min_bone_mL
            big[0] = False
            bone = big[lab_b]
        else:
            bone = np.zeros_like(bone)
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


# -----------------------------------------------------------------------------
# shared plumbing
# -----------------------------------------------------------------------------

class Ctx:
    """Everything the methods share: cropped volumes, geometry, intensity model."""

    def __init__(self, img, ct, mk, spacing, margin_mm=45.0, roi_mm=30.0,
                 band_rule="sd", k_sd=2.5, min_reach_mm=0.0, floor_hu=None):
        self.img, self.spacing = img, spacing
        self.band_rule, self.k_sd = band_rule, k_sd
        self.min_reach_mm = min_reach_mm
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

        self.lo, self.hi, self.mu, self.sd = lumen_band(
            ct, mk, k_sd=k_sd, spacing=spacing, rule=band_rule)
        # Override ONLY the floor, leaving the ceiling and the bone cut exactly
        # as the sd rule computed them. The required floor measured per subject
        # is 246 / <=300 / <=260 / <=260 on 001 / 019 / 020 / 023, while
        # mean - 3*sd hands those cases 235 / 289 / 404 / 286 -- so the floor is
        # the one term that must stop being a multiple of sd.
        if floor_hu is not None:
            self.lo = float(floor_hu)
        v2 = (band_rule == "v2")
        # Take the LOWER of the old cut and the ceiling-relative one, so v2 can
        # never exclude less bone by threshold than the current code does.
        # subject001: min(400, 418) = 400, unchanged. subject019:
        # min(869, 691) = 691, which is the difference between excluding a
        # vertebra and excluding nothing at all.
        bone_floor = (min(max(400.0, self.mu + 3.0 * self.sd),
                          self.hi + BONE_MARGIN_HU) if v2 else None)
        self.roi, self.dist_out, self.bone, self.hi_cut = branch_roi(
            self.ct, self.mk, spacing, roi_mm, self.mu, self.sd,
            bone_floor_hu=bone_floor,
            min_bone_mL=MIN_BONE_ML if v2 else None)
        self.roi_mm = roi_mm
        self.wall = (ndi.binary_dilation(self.mk, iterations=1)
                     & ~ndi.binary_erosion(self.mk, iterations=1))
        self.sp = np.array([sz, sy, sx])

        # aorta centreline: centroid per slice, for angles and radial geometry
        self.cl = {}
        for z in np.where(self.mk.any(axis=(1, 2)))[0]:
            cy, cx = ndi.center_of_mass(self.mk[z])
            self.cl[int(z)] = (cy, cx)

        # world placement for meshes: a mesh lives in crop-voxel-mm space, and
        # the volume may be rotated, so the direction matrix has to be applied
        # or the surfaces land somewhere the ostium markers are not.
        try:
            self.D = np.array(img.GetDirection(), float).reshape(3, 3)
        except Exception:
            self.D = np.eye(3)
        self._corner = None

    def to_mm(self, vox):
        """voxel (z, y, x) in the CROP -> physical (x, y, z) mm."""
        v = np.asarray(vox, float)
        return np.array(self.img.TransformIndexToPhysicalPoint(
            (int(round(v[2])) + self.x0, int(round(v[1])) + self.y0,
             int(round(v[0])) + self.z0)), float)

    def verts_to_world(self, v):
        """Mesh vertices (x, y, z mm from the crop corner) -> physical mm."""
        if v is None:
            return None
        if self._corner is None:
            self._corner = self.to_mm((0, 0, 0))
        return self._corner + np.asarray(v, float) @ self.D.T


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


def ostia_from_mask(mask, ctx, min_mm3=8.0, max_n=25, use_contact=True,
                    min_reach_mm=None):
    """Shared post-step: components -> one ostium each.

    min_reach_mm applies the challenge's own eligibility rule: a daughter must
    be followable at least 5 mm beyond the aortic wall. Without it the only
    filter is 8 mm3 of volume, which passes any wall-hugging speck -- and on
    subject019 and subject021 at sd:3 the recall is already 1.0, so every point
    of F1 still missing is a false positive. Measured GT reach is 6.2-9.8 mm,
    so a 5 mm floor cannot cost a true positive on those cases.
    Default 0.0 -- off, so existing behaviour is unchanged.

    If the component already touches the aorta, take the contact zone and pick
    the voxel that maximises distance to the edge of the ramification -- the
    centre of the opening, following Tahoces. Otherwise fall back to extending
    the component's local axis back to the wall.
    """
    lab, keep = components(mask, ctx, min_mm3)
    edt = ndi.distance_transform_edt(mask, sampling=tuple(ctx.sp)) if mask.any() \
        else np.zeros_like(mask, float)
    near_wall = ndi.binary_dilation(ctx.mk, iterations=2)

    reach_mm = (getattr(ctx, "min_reach_mm", 0.0) if min_reach_mm is None
                else min_reach_mm)
    ostia, dirs, radii = [], [], []
    for i in keep[:max_n]:
        comp = lab == i
        if reach_mm > 0 and float(ctx.dist_out[comp].max()) < reach_mm:
            continue                       # never leaves the wall -> ineligible
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
            hit, _g, ost, _u = link_to_aorta(pts, ctx.dist_out, vv[0],
                                             ctx.spacing, sub_mk=ctx.mk)
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
# per-method output: one folder with a CSV, a challenge JSON and a 3D render
# -----------------------------------------------------------------------------

def band_tag(ctx):
    """'sd:3' / 'v2' -- stamped on every render so two of them can't be mixed up."""
    return (f"sd:{ctx.k_sd:g}" if getattr(ctx, "band_rule", "sd") == "sd"
            else getattr(ctx, "band_rule", "sd"))


def daughters(res, ctx):
    """Challenge records: ostium, 5 mm seed, radius, unit direction."""
    out = []
    for k, (o, d, rad) in enumerate(zip(res.ostia_vox, res.dirs, res.radii), 1):
        p = ctx.to_mm(o)
        vec = ([float(d[2]), float(d[1]), float(d[0])] if d is not None
               else [0.0, 0.0, 1.0])
        vec = [float(v) for v in
               np.round(np.array(vec) / max(np.linalg.norm(vec), 1e-9), 4)]
        out.append({"instance_id": f"branch_{k:03d}",
                    "parent_instance_id": "aorta",
                    "ostium_xyz_mm": [round(float(v), 3) for v in p],
                    "seed_xyz_mm": [round(float(v), 3)
                                    for v in (p + np.array(vec) * 5.0)],
                    "radius_mm": round(float(rad), 3) if rad else None,
                    "direction_xyz": vec})
    return out


def save_method(res, ctx, outdir, case_id):
    """Write <outdir>/<method>/{ostia.csv, pred.json, render.html}."""
    d = os.path.join(outdir, res.name.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    recs = daughters(res, ctx)

    with open(os.path.join(d, "ostia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "parent_instance_id",
                    "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "seed_x_mm", "seed_y_mm", "seed_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"])
        for r in recs:
            w.writerow([r["instance_id"], r["parent_instance_id"],
                        *[round(v, 2) for v in r["ostium_xyz_mm"]],
                        *[round(v, 2) for v in r["seed_xyz_mm"]],
                        *r["direction_xyz"],
                        r["radius_mm"] if r["radius_mm"] is not None else ""])

    with open(os.path.join(d, "pred.json"), "w") as fh:
        json.dump({"case_id": case_id, "method": res.name,
                   "parent": {"instance_id": "aorta"}, "daughters": recs},
                  fh, indent=2)

    # ---- 3D render -------------------------------------------------------
    meshes = []
    av, af = fine_surface(ctx.mk, ctx.spacing, cap=120_000)
    if av is not None:
        meshes.append(dict(verts=ctx.verts_to_world(av), faces=af,
                           name="aorta (given mask)", color="#d94a4a",
                           opacity=0.30, text="parent aorta"))
    bv, bf = fine_surface(res.mask & ~ctx.mk, ctx.spacing, cap=120_000)
    if bv is not None:
        meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                           name=f"{res.name} branches", color="#f2c14e",
                           opacity=0.95, text=res.name))
    if recs:
        pts = np.array([r["ostium_xyz_mm"] for r in recs], float)
        labels = [f"{r['instance_id']}<br>"
                  f"({r['ostium_xyz_mm'][0]:.1f}, {r['ostium_xyz_mm'][1]:.1f}, "
                  f"{r['ostium_xyz_mm'][2]:.1f}) mm<br>"
                  f"r = {r['radius_mm']} mm" for r in recs]
        meshes.append(dict(kind="points", points=pts, name="ostia",
                           color="#111111", size=6, labels=labels))
        seeds = np.array([r["seed_xyz_mm"] for r in recs], float)
        seg = np.full((len(recs) * 3, 3), np.nan)
        seg[0::3], seg[1::3] = pts, seeds
        meshes.append(dict(kind="lines", points=seg,
                           name="ostium -> 5 mm seed", color="#111111",
                           width=4))

    ok = write_interactive_html(
        os.path.join(d, "render.html"), meshes,
        f"{case_id} -- {res.name}",
        subtitle=(f"{len(recs)} ostia &middot; {res.runtime:.1f} s &middot; "
                  f"{res.note} &middot; band {ctx.lo:.0f}-{ctx.hi:.0f} HU "
                  f"[{band_tag(ctx)}] &middot; lumen {ctx.mu:.0f} +/- "
                  f"{ctx.sd:.0f} HU"),
        axis_note=" [patient/world]")
    return d, ok


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
    ap.add_argument("--floor-hu", type=float, default=None,
                    help="absolute lower edge of the band in HU, overriding "
                         "mean - k*sd. Ceiling and bone cut are untouched.")
    ap.add_argument("--min-reach-mm", type=float, default=0.0,
                    help="discard candidates that never get this far beyond "
                         "the aortic wall (the challenge's own 5 mm "
                         "eligibility rule). Default 0 = off.")
    ap.add_argument("--k-sd", type=float, default=2.5,
                    help="band half-width in standard deviations "
                         "(--band sd only). Default 2.5, unchanged.")
    ap.add_argument("--band", choices=("sd", "v2"), default="sd",
                    help="sd = mu +/- 2.5 sd (default, unchanged). "
                         "v2 = absolute floor, relative ceiling, bulk-tested "
                         "bone cut.")
    ap.add_argument("--skip", default="",
                    help="comma-separated method ids to skip, e.g. M1,M4")
    args = ap.parse_args()

    case_id = args.case_id or os.path.splitext(
        os.path.basename(args.image))[0] or "case"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"branch_bench {__version__}   case {case_id}")

    img, _, ct, mk = load_case(args.image, args.aorta_mask)
    ctx = Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm,
              band_rule=args.band, k_sd=args.k_sd,
              min_reach_mm=args.min_reach_mm, floor_hu=args.floor_hu)
    tag = args.band + (f":{args.k_sd:g}" if args.band == "sd" else "")
    print(f"  lumen {ctx.mu:.0f} +/- {ctx.sd:.0f} HU   "
          f"band {ctx.lo:.0f}-{ctx.hi:.0f} [{tag}]   "
          f"bone > {ctx.hi_cut:.0f}"
          f"{f' and > {MIN_BONE_ML:g} mL' if args.band == 'v2' else ''}   "
          f"crop {ctx.ct.shape}")

    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    plan = [("M1", m1_frangi), ("M2", m2_band_connect),
            ("M3", m3_expansion), ("M4", m4_wall_flux)]

    rows = []
    for mid, fn in plan:
        if mid in skip:
            print(f"  {mid} skipped")
            continue
        try:
            res = dedupe(drop_end_caps(fn(ctx), ctx), ctx)
        except Exception as exc:
            print(f"  {mid} FAILED: {exc}", file=sys.stderr)
            continue
        folder, ok = save_method(res, ctx, args.outdir, case_id)
        print(f"  {res.name:<18} {len(res.ostia_vox):>3} ostia  "
              f"{res.runtime:>6.1f}s  {res.note}")
        print(f"      -> {os.path.join(folder, 'ostia.csv')}")
        print(f"      -> {os.path.join(folder, 'render.html')}"
              f"{'' if ok else '   (skipped: plotly missing)'}")
        rows.append([res.name, len(res.ostia_vox), round(res.runtime, 2),
                     res.note])

    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_ostia", "runtime_s", "note"])
        w.writerows(rows)
    print(f"  wrote {os.path.join(args.outdir, 'summary.csv')}")
    print("  open each render.html in its own browser tab to compare.")


if __name__ == "__main__":
    main()