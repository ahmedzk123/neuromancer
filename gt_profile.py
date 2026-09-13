#!/usr/bin/env python3
"""
gt_profile.py -- measure the ground-truth daughters instead of guessing at them.

Answers, per GT branch, the four questions the sweep could not:

  1. WHERE is it?        distance from each end of the aorta mask, so we can see
                         whether drop_end_caps() is deleting correct answers.
  2. Is it CONNECTED?    does the GT branch actually touch the supplied aorta
                         mask, or is there a gap the growth can never cross?
  3. How BRIGHT is it?   median / p10 / p90 HU in 1 mm shells outward from the
                         wall -- i.e. exactly what intensity floor is needed to
                         follow it, and how far that floor holds.
  4. Is the BAND to blame? what fraction of the branch falls inside the band at
                         each k_sd, under BOTH the mean/sd rule the code uses now
                         and a robust median/MAD rule.

    python gt_profile.py --image orig19.nii --aorta-mask mask19.nii \
        --gt gt19.nii --outdir diag --case-id subject019

The GT file is a label volume on the same grid as the image. Labels that sit
mostly inside the aorta mask are treated as the parent and reported separately;
everything else is a daughter. Override with --parent-label / --branch-labels.

OUTPUT
  <outdir>/gt_summary.csv         one row per GT branch: the headline numbers
  <outdir>/gt_shell_profiles.csv  HU by distance from the wall, 1 mm shells
  <outdir>/gt_band_recall.csv     branch x k_sd x estimator -> fraction in band
  <outdir>/gt_floor_reach.csv     intensity floor -> how far each branch follows
  <outdir>/gt_profiles.html       those three tables as charts
  <outdir>/gt_render.html         aorta + GT daughters in 3D, ostia marked
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

__version__ = "2026-09-14.gtprofile1"

PALETTE = ["#f2c14e", "#2f9e9e", "#7b6cd9", "#e07a5f", "#3fa34d",
           "#c85b9b", "#4f86c6", "#b07d2b", "#5f7d95"]


def load_bench(explicit=None):
    """Import branch_bench.py from beside this file, the cwd, or --bench."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [explicit] if explicit else []
    cands += [os.path.join(here, "branch_bench.py"),
              os.path.join(os.getcwd(), "branch_bench.py")]
    for p in cands:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("_bench", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_bench"] = mod
            spec.loader.exec_module(mod)
            return mod, p
    raise SystemExit(
        "\nCould not find branch_bench.py.\n"
        "  gt_profile.py reuses its loader, context and renderer, so it needs\n"
        "  branch_bench.py in the same folder (or pass --bench /path/to/it).\n"
        f"  Looked in: {here}\n        and: {os.getcwd()}\n")


# -----------------------------------------------------------------------------
# intensity model, measured two ways
# -----------------------------------------------------------------------------

def lumen_stats(ct, mk, spacing, erode_mm=2.0):
    """Both estimators side by side, plus what is contaminating the mask.

    mean/sd is what lumen_band() uses. A single calcified plaque or a stent
    inside the mask drags the mean up and inflates the sd, and since the band,
    the bone cut and everything downstream are expressed in sd, one dense object
    moves every threshold in the pipeline at once. median/MAD cannot be moved
    that way, so the gap between the two columns IS the contamination.
    """
    sx, sy, sz = spacing
    er = max(int(round(erode_mm / min(sx, sy))), 1)
    core = ndi.binary_erosion(mk.astype(bool), iterations=er)
    if core.sum() < 50:
        core = mk.astype(bool)
    v = ct[core].astype(np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    n = float(v.size)
    return dict(
        n_voxels=int(n),
        mean=float(v.mean()), sd=float(v.std()),
        median=med, mad=max(mad, 1e-6),
        p01=float(np.percentile(v, 1)), p05=float(np.percentile(v, 5)),
        p50=med, p95=float(np.percentile(v, 95)),
        p99=float(np.percentile(v, 99)), vmax=float(v.max()),
        frac_over_600=float((v > 600).sum() / n),
        frac_over_800=float((v > 800).sum() / n),
        frac_over_1000=float((v > 1000).sum() / n),
        values=v)


# -----------------------------------------------------------------------------
# per-branch measurements
# -----------------------------------------------------------------------------

def shell_profile(ct, comp, dist_out, step=1.0, max_mm=20.0):
    """HU percentiles in 1 mm shells outward from the aortic wall."""
    rows = []
    reach = float(dist_out[comp].max()) if comp.any() else 0.0
    for d0 in np.arange(0.0, min(max_mm, reach) + step, step):
        sel = comp & (dist_out >= d0) & (dist_out < d0 + step)
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(d_lo=float(d0), d_hi=float(d0 + step), n=0,
                             p10=None, p50=None, p90=None, vmin=None))
            continue
        v = ct[sel]
        rows.append(dict(d_lo=float(d0), d_hi=float(d0 + step), n=n,
                         p10=float(np.percentile(v, 10)),
                         p50=float(np.median(v)),
                         p90=float(np.percentile(v, 90)),
                         vmin=float(v.min())))
    return rows


def floor_reach(ctx, comps, floors, ceiling):
    """How far each branch can be followed from the aorta at a given HU floor.

    This is the number the whole argument is about. For each floor F, keep the
    voxels in [F, ceiling], connect them to the aorta mask, and ask how far
    along each GT branch that connected set still reaches. The largest F whose
    reach clears 5 mm is the floor the band actually needs -- measured on this
    subject's own anatomy rather than picked as a multiple of sd.

    A ceiling is applied so a branch cannot be reported as reachable via a
    bridge through bone, which would flatter the floor.
    """
    out = {}
    for F in floors:
        allowed = (ctx.ct >= F) & (ctx.ct <= ceiling)
        lab, _ = ndi.label(allowed | ctx.mk)
        ids = [int(v) for v in np.unique(lab[ctx.mk]) if v > 0]
        conn = np.isin(lab, ids)
        for name, comp in comps.items():
            sel = comp & conn
            out.setdefault(name, []).append(
                (float(F), float(ctx.dist_out[sel].max()) if sel.any() else 0.0,
                 float(sel.sum()) / max(float(comp.sum()), 1.0)))
    return out


def required_floor(series, target_mm):
    """Largest floor in the sweep whose reach still clears target_mm."""
    ok = [F for F, reach, _ in series if reach >= target_mm]
    return max(ok) if ok else None


def bridge_min_hu(ct, wall_pts, start_vox, sp, n=40):
    """Darkest voxel on the straight line from the wall to the branch.

    A GT branch that does not overlap the mask is not necessarily unreachable --
    the mask simply stops a voxel short. What decides it is whether the voxels
    in between are bright enough to be in the band. This is that number, so the
    gap and the bridge can be judged separately.
    """
    if not len(wall_pts):
        return None, None
    d = np.linalg.norm((wall_pts - start_vox) * sp, axis=1)
    w = wall_pts[int(np.argmin(d))]
    t = np.linspace(0.0, 1.0, n)
    pts = (w[None, :] * (1 - t)[:, None] + start_vox[None, :] * t[:, None])
    v = ndi.map_coordinates(ct, pts.T, order=1)
    return float(v.min()), float(d.min())


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------

def write_profiles_html(path, case_id, branches, profiles, reaches, lum, ks):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.09,
        subplot_titles=(
            "HU along each GT daughter, by distance from the aortic wall "
            "(solid = median, dashed = 10th percentile)",
            "how far each daughter can be followed, by intensity floor "
            "(a daughter must clear 5 mm to be eligible)",
            "HU inside the eroded aorta mask -- where the lumen estimate "
            "comes from"))

    for i, b in enumerate(branches):
        col = PALETTE[i % len(PALETTE)]
        rows = [r for r in profiles[b["name"]] if r["n"] > 0]
        if rows:
            d = [(r["d_lo"] + r["d_hi"]) / 2 for r in rows]
            fig.add_trace(go.Scatter(x=d, y=[r["p50"] for r in rows],
                                     mode="lines+markers", name=b["name"],
                                     legendgroup=b["name"],
                                     line=dict(color=col, width=3)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=d, y=[r["p10"] for r in rows],
                                     mode="lines", name=b["name"] + " p10",
                                     legendgroup=b["name"], showlegend=False,
                                     line=dict(color=col, width=2,
                                               dash="dash")), row=1, col=1)
        ser = reaches[b["name"]]
        fig.add_trace(go.Scatter(x=[F for F, _, _ in ser],
                                 y=[r for _, r, _ in ser],
                                 mode="lines+markers", name=b["name"],
                                 legendgroup=b["name"], showlegend=False,
                                 line=dict(color=col, width=3)), row=2, col=1)

    # the band floors the current rule would impose, drawn over the profiles
    dmax = 20.0
    for k in ks:
        lo = lum["mean"] - k * lum["sd"]
        fig.add_trace(go.Scatter(
            x=[0, dmax], y=[lo, lo], mode="lines",
            name=f"floor, mean-{k:g}sd = {lo:.0f}", legendgroup="floors",
            line=dict(color="#888888", width=1, dash="dot")), row=1, col=1)
    lo_rob = lum["median"] - 2.5 * lum["mad"]
    fig.add_trace(go.Scatter(
        x=[0, dmax], y=[lo_rob, lo_rob], mode="lines",
        name=f"floor, median-2.5MAD = {lo_rob:.0f}", legendgroup="floors",
        line=dict(color="#c1121f", width=2, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=[0, 0], y=[0, 0], mode="lines",
                             showlegend=False, hoverinfo="skip",
                             line=dict(color="rgba(0,0,0,0)")), row=2, col=1)
    xs = [F for F, _, _ in reaches[branches[0]["name"]]] if branches else [0, 1]
    fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[5, 5], mode="lines",
                             name="5 mm eligibility",
                             line=dict(color="#111111", width=2, dash="dot")),
                  row=2, col=1)

    hist, edges = np.histogram(lum["values"], bins=80)
    centres = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centres, y=hist, name="lumen voxels",
                         marker=dict(color="#d94a4a")), row=3, col=1)

    fig.update_xaxes(title_text="distance from aortic wall (mm)", row=1, col=1)
    fig.update_yaxes(title_text="HU", row=1, col=1)
    fig.update_xaxes(title_text="intensity floor (HU)", row=2, col=1)
    fig.update_yaxes(title_text="reach (mm)", row=2, col=1)
    fig.update_xaxes(title_text="HU", row=3, col=1)
    fig.update_yaxes(title_text="voxels", row=3, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{case_id} -- ground-truth daughters, measured"
                        "</b><br><span style='font-size:13px'>"
                        f"lumen mean {lum['mean']:.0f} &plusmn; {lum['sd']:.0f}"
                        f" &nbsp;vs&nbsp; median {lum['median']:.0f} "
                        f"&plusmn; {lum['mad']:.0f} (MAD)</span>",
                   x=0.01, xanchor="left"),
        height=1050, template="plotly_white", bargap=0.02,
        margin=dict(l=70, r=20, t=100, b=50))
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return True


def write_gt_render(bb, ctx, path, case_id, branches, comps, parent_note):
    meshes = []
    av, af = bb.fine_surface(ctx.mk, ctx.spacing, cap=120_000)
    if av is not None:
        meshes.append(dict(verts=ctx.verts_to_world(av), faces=af,
                           name="aorta (given mask)", color="#d94a4a",
                           opacity=0.28, text="parent aorta"))
    pts, labels = [], []
    for i, b in enumerate(branches):
        bv, bf = bb.fine_surface(comps[b["name"]], ctx.spacing, cap=90_000)
        if bv is not None:
            meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                               name=f"{b['name']} (GT)",
                               color=PALETTE[i % len(PALETTE)], opacity=0.95,
                               text=b["name"]))
        pts.append(b["ostium_mm"])
        labels.append(
            f"{b['name']}<br>({b['ostium_mm'][0]:.1f}, {b['ostium_mm'][1]:.1f},"
            f" {b['ostium_mm'][2]:.1f}) mm<br>"
            f"volume {b['volume_mm3']:.0f} mm&sup3;<br>"
            f"reach {b['reach_mm']:.1f} mm<br>"
            f"{b['gap_mm']:.2f} mm from the mask<br>"
            f"{b['from_caudal_mm']:.1f} mm from the caudal end")
    if pts:
        meshes.append(dict(kind="points", points=np.array(pts, float),
                           name="GT ostia", color="#111111", size=7,
                           labels=labels))
    return bb.write_interactive_html(
        path, meshes, f"{case_id} -- ground truth",
        subtitle=f"{len(branches)} daughters &middot; {parent_note}",
        axis_note=" [patient/world]")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--gt", required=True,
                    help="label volume of the ground-truth branches")
    ap.add_argument("--outdir", default="diag")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--k", default="1.5,2.0,2.5,3.0,3.5,4.0")
    ap.add_argument("--roi-mm", type=float, default=30.0)
    ap.add_argument("--end-margin-mm", type=float, default=6.0,
                    help="the drop_end_caps margin, to flag GT branches it "
                         "would delete")
    ap.add_argument("--floors", default="100,140,180,220,260,300,340,380,420,"
                                        "460,500",
                    help="intensity floors to test reachability at")
    ap.add_argument("--ceiling", type=float, default=600.0,
                    help="ceiling used during the floor sweep, so a branch "
                         "cannot be reached through bone")
    ap.add_argument("--parent-label", type=int, default=None)
    ap.add_argument("--branch-labels", default=None,
                    help="comma-separated labels to treat as daughters")
    args = ap.parse_args()

    bb, bench_path = load_bench(args.bench)
    ks = [float(s) for s in args.k.split(",") if s.strip()]
    floors = [float(s) for s in args.floors.split(",") if s.strip()]
    case_id = args.case_id or os.path.splitext(
        os.path.basename(args.image))[0] or "case"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"gt_profile {__version__}   case {case_id}")
    print(f"  using {bench_path}  (branch_bench {bb.__version__})")

    img, _, ct, mk = bb.load_case(args.image, args.aorta_mask)
    gt_img = sitk.ReadImage(args.gt)
    if gt_img.GetSize() != img.GetSize():
        raise SystemExit(f"GT grid {gt_img.GetSize()} != image "
                         f"{img.GetSize()}; resample the GT first.")
    gt = sitk.GetArrayFromImage(gt_img).astype(np.int32)

    ctx = bb.Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm)
    gtc = gt[ctx.region]

    # ---- the intensity model, both ways ------------------------------------
    lum = lumen_stats(ct, mk, img.GetSpacing())
    print("\n  LUMEN ESTIMATE")
    print(f"    mean/sd        : {lum['mean']:7.1f} / {lum['sd']:6.1f}"
          f"    -> band at k=2.5 is "
          f"{lum['mean'] - 2.5 * lum['sd']:.0f}-"
          f"{lum['mean'] + 2.5 * lum['sd']:.0f} HU")
    print(f"    median/MAD     : {lum['median']:7.1f} / {lum['mad']:6.1f}"
          f"    -> band at k=2.5 is "
          f"{lum['median'] - 2.5 * lum['mad']:.0f}-"
          f"{lum['median'] + 2.5 * lum['mad']:.0f} HU")
    print(f"    percentiles    : p1 {lum['p01']:.0f}  p5 {lum['p05']:.0f}  "
          f"p50 {lum['p50']:.0f}  p95 {lum['p95']:.0f}  p99 {lum['p99']:.0f}  "
          f"max {lum['vmax']:.0f}")
    print(f"    dense fraction : >600 HU {lum['frac_over_600'] * 100:.2f}%   "
          f">800 HU {lum['frac_over_800'] * 100:.2f}%   "
          f">1000 HU {lum['frac_over_1000'] * 100:.2f}%")
    bone_cut_now = max(400.0, lum["mean"] + 3.0 * lum["sd"])
    print(f"    bone cut now   : max(400, mean+3sd) = {bone_cut_now:.0f} HU"
          + ("   <-- ABOVE cortical bone, so bone is NOT being excluded"
             if bone_cut_now > 500 else ""))
    print(f"    bone mask      : "
          f"{ctx.bone.sum() * ctx.vox_mm3 / 1000.0:.1f} mL excluded in the crop")

    # ---- which GT labels are daughters -------------------------------------
    ids = [int(v) for v in np.unique(gtc) if v > 0]
    if not ids:
        raise SystemExit("The GT volume has no non-zero labels inside the crop.")
    inside = {}
    for L in ids:
        comp = gtc == L
        inside[L] = float((comp & ctx.mk).sum()) / max(float(comp.sum()), 1.0)
    if args.branch_labels:
        blabels = [int(s) for s in args.branch_labels.split(",") if s.strip()]
    else:
        blabels = [L for L in ids
                   if inside[L] < 0.5 and L != (args.parent_label or -1)]
    parents = [L for L in ids if L not in blabels]
    parent_note = ("parent label(s) " + ", ".join(str(L) for L in parents)
                   if parents else "no parent label in the GT")
    print("\n  GT LABELS")
    for L in ids:
        comp = gtc == L
        print(f"    label {L:<3} {comp.sum() * ctx.vox_mm3:9.0f} mm3   "
              f"{inside[L] * 100:5.1f}% inside the aorta mask   "
              f"{'PARENT' if L in parents else 'daughter'}")

    # ---- per-branch measurements -------------------------------------------
    zs = np.where(ctx.mk.any(axis=(1, 2)))[0]
    mid = (ctx.mk.shape[1] // 2, ctx.mk.shape[2] // 2)
    ez = [ctx.to_mm((int(zs[0]),) + mid)[2], ctx.to_mm((int(zs[-1]),) + mid)[2]]
    z_caudal, z_cranial = min(ez), max(ez)     # by PHYSICAL z, not array order
    wall_pts = np.argwhere(ctx.wall).astype(float)
    comps, branches, profiles = {}, [], {}

    for i, L in enumerate(blabels):
        comp = gtc == L
        name = f"label_{L}"
        comps[name] = comp
        d = ctx.dist_out[comp]
        near = np.argwhere(comp)[int(np.argmin(d))].astype(float)
        ost_mm = ctx.to_mm(near)
        v = ct[ctx.region][comp]
        near5 = comp & (ctx.dist_out <= 5.0)
        bmin, _wd = bridge_min_hu(ctx.ct, wall_pts, near, ctx.sp)
        b = dict(
            name=name, label=L,
            volume_mm3=float(comp.sum()) * ctx.vox_mm3,
            n_voxels=int(comp.sum()),
            frac_inside_mask=inside[L],
            gap_mm=float(d.min()),
            bridge_min_hu=bmin,
            reach_mm=float(d.max()),
            ostium_mm=[float(x) for x in ost_mm],
            from_caudal_mm=float(ost_mm[2]) - z_caudal,
            from_cranial_mm=z_cranial - float(ost_mm[2]),
            hu_p10=float(np.percentile(v, 10)),
            hu_p50=float(np.median(v)),
            hu_p90=float(np.percentile(v, 90)),
            hu_min=float(v.min()), hu_max=float(v.max()),
            # how much dimmer than the parent: negative means a symmetric band
            # centred on the lumen is spending half its width on nothing
            hu_p50_minus_lumen=float(np.median(v)) - lum["median"],
            hu_p10_within_5mm=(float(np.percentile(ct[ctx.region][near5], 10))
                               if near5.any() else None),
            frac_in_bone_mask=float((comp & ctx.bone).sum())
            / max(float(comp.sum()), 1.0))
        b["end_cap_dropped"] = bool(
            min(b["from_caudal_mm"], b["from_cranial_mm"]) < args.end_margin_mm)
        branches.append(b)
        profiles[name] = shell_profile(ct[ctx.region], comp, ctx.dist_out)

    if not branches:
        raise SystemExit("No daughter labels found; pass --branch-labels.")

    # ---- band recall, both estimators --------------------------------------
    recall = []
    for b in branches:
        v = ct[ctx.region][comps[b["name"]]]
        for k in ks:
            for rule, c, s in (("mean_sd", lum["mean"], lum["sd"]),
                               ("median_mad", lum["median"], lum["mad"])):
                lo, hi = c - k * s, c + k * s
                recall.append(dict(
                    branch=b["name"], k_sd=k, rule=rule,
                    lo_hu=round(lo, 1), hi_hu=round(hi, 1),
                    frac_in_band=round(float(((v >= lo) & (v <= hi)).mean()), 4),
                    frac_below=round(float((v < lo).mean()), 4),
                    frac_above=round(float((v > hi).mean()), 4)))

    # ---- floor reachability -------------------------------------------------
    print(f"\n  floor sweep at ceiling {args.ceiling:.0f} HU "
          f"({len(floors)} thresholds)...")
    reaches = floor_reach(ctx, comps, floors, args.ceiling)
    for b in branches:
        ser = reaches[b["name"]]
        b["floor_for_5mm"] = required_floor(ser, 5.0)
        b["floor_for_10mm"] = required_floor(ser, 10.0)

    # ---- verdicts -----------------------------------------------------------
    print("\n  GT DAUGHTERS")
    for b in branches:
        print(f"    {b['name']}  {b['volume_mm3']:7.0f} mm3  "
              f"reach {b['reach_mm']:5.1f} mm  "
              f"HU p10/p50 {b['hu_p10']:.0f}/{b['hu_p50']:.0f}  "
              f"ostium ({b['ostium_mm'][0]:.1f}, {b['ostium_mm'][1]:.1f}, "
              f"{b['ostium_mm'][2]:.1f}) mm")
        print(f"        {b['from_caudal_mm']:.1f} mm from the caudal end of the "
              f"mask, {b['from_cranial_mm']:.1f} mm from the cranial end"
              + ("   <-- drop_end_caps WOULD DELETE THIS"
                 if b["end_cap_dropped"] else ""))
        print(f"        {b['hu_p50_minus_lumen']:+.0f} HU vs the lumen median "
              f"-- a band centred on the lumen spends half its width above the "
              f"branch")
        if b["gap_mm"] > 0:
            bm = b["bridge_min_hu"]
            far = b["gap_mm"] > 1.2 * float(min(ctx.sp))
            print(f"        {b['gap_mm']:.2f} mm clear of the mask"
                  + (f", darkest voxel on the way in {bm:.0f} HU"
                     if bm is not None else "")
                  + ("   <-- a real gap, not just a voxel of rounding"
                     if far else ""))
        if b["frac_in_bone_mask"] > 0.05:
            print(f"        {b['frac_in_bone_mask'] * 100:.0f}% of it sits "
                  f"inside the bone-exclusion mask")
        f5, f10 = b["floor_for_5mm"], b["floor_for_10mm"]
        print(f"        floor needed: {f5 if f5 is not None else 'NONE TESTED'}"
              f" HU to reach 5 mm, "
              f"{f10 if f10 is not None else 'NONE TESTED'} HU to reach 10 mm")
        for k in (2.5,):
            lo = lum["mean"] - k * lum["sd"]
            r = [x for x in recall if x["branch"] == b["name"]
                 and x["k_sd"] == k and x["rule"] == "mean_sd"]
            if r:
                print(f"        at k={k:g} the current band is "
                      f"{r[0]['lo_hu']:.0f}-{r[0]['hi_hu']:.0f} and holds "
                      f"{r[0]['frac_in_band'] * 100:.0f}% of it "
                      f"({r[0]['frac_below'] * 100:.0f}% below, "
                      f"{r[0]['frac_above'] * 100:.0f}% above)")

    # ---- CSVs ---------------------------------------------------------------
    p = os.path.join(args.outdir, "gt_summary.csv")
    cols = ["name", "label", "n_voxels", "volume_mm3", "frac_inside_mask",
            "gap_mm", "bridge_min_hu", "reach_mm", "from_caudal_mm",
            "from_cranial_mm", "end_cap_dropped", "hu_min", "hu_p10", "hu_p50",
            "hu_p90", "hu_max", "hu_p50_minus_lumen", "hu_p10_within_5mm",
            "frac_in_bone_mask", "floor_for_5mm", "floor_for_10mm",
            "ostium_x_mm", "ostium_y_mm", "ostium_z_mm"]
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for b in branches:
            row = dict(b)
            row["ostium_x_mm"], row["ostium_y_mm"], row["ostium_z_mm"] = \
                [round(x, 2) for x in b["ostium_mm"]]
            w.writerow([round(row[c], 3) if isinstance(row.get(c), float)
                        else row.get(c, "") for c in cols])
    print(f"\n  wrote {p}")

    p = os.path.join(args.outdir, "gt_shell_profiles.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["branch", "dist_lo_mm", "dist_hi_mm", "n_voxels",
                    "hu_p10", "hu_p50", "hu_p90", "hu_min"])
        for b in branches:
            for r in profiles[b["name"]]:
                w.writerow([b["name"], r["d_lo"], r["d_hi"], r["n"],
                            "" if r["p10"] is None else round(r["p10"], 1),
                            "" if r["p50"] is None else round(r["p50"], 1),
                            "" if r["p90"] is None else round(r["p90"], 1),
                            "" if r["vmin"] is None else round(r["vmin"], 1)])
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "gt_band_recall.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["branch", "k_sd", "rule", "lo_hu",
                                           "hi_hu", "frac_in_band",
                                           "frac_below", "frac_above"])
        w.writeheader()
        w.writerows(recall)
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "gt_floor_reach.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["branch", "floor_hu", "ceiling_hu", "reach_mm",
                    "frac_of_branch_connected"])
        for b in branches:
            for F, reach, frac in reaches[b["name"]]:
                w.writerow([b["name"], F, args.ceiling, round(reach, 2),
                            round(frac, 4)])
    print(f"  wrote {p}")

    # ---- figures ------------------------------------------------------------
    p = os.path.join(args.outdir, "gt_profiles.html")
    if write_profiles_html(p, case_id, branches, profiles, reaches, lum, ks):
        print(f"  wrote {p}")
    else:
        print("  gt_profiles.html skipped: plotly missing", file=sys.stderr)

    p = os.path.join(args.outdir, "gt_render.html")
    if write_gt_render(bb, ctx, p, case_id, branches, comps, parent_note):
        print(f"  wrote {p}")

    print("\n  Read gt_profiles.html first: the top panel shows what floor the "
          "band needs,\n  the middle panel shows how far that floor carries, "
          "and the bottom panel shows\n  whether the lumen estimate is being "
          "poisoned by something dense.")


if __name__ == "__main__":
    main()