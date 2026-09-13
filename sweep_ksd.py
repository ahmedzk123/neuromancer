#!/usr/bin/env python3
"""
sweep_ksd.py -- sweep the ONE parameter that differs most between subjects:
k_sd, the half-width of the lumen intensity band in standard deviations.

    band = lumen_mean +/- k_sd * lumen_sd        (branch_bench.lumen_band)

M3 reads that band and nothing else about intensity, so sweeping k_sd is a
direct read-out of how sensitive the result is to the band -- which is the
difference between subject001 working and subject019 finding nothing.

    python sweep_ksd.py --image orig1.nii --aorta-mask mask1.nii \
        --outdir sweep --case-id subject001 --k 1.5,2.0,2.5,3.0,3.5,4.0

This file does NOT contain a copy of M3. It loads branch_bench.py and calls the
real one, so the sweep can never drift from what the benchmark actually runs.

OUTPUT
  <outdir>/sweep_summary.csv       one row per k: band, counts, volume, runtime
  <outdir>/sweep_ostia.csv         one row per (k, ostium), with its cluster id
  <outdir>/sweep_persistence.csv   one row per ostium CLUSTER: how many k found it
  <outdir>/sweep_curves.html       n_ostia / volume / band width against k
  <outdir>/render_persistence.html all k at once, ostia coloured by persistence
  <outdir>/k_2.50/render.html      one 3D render per k value
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

__version__ = "2026-09-14.sweep1"


# -----------------------------------------------------------------------------

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
            spec.loader.exec_module(mod)          # main() does not run on import
            return mod, p
    raise SystemExit(
        "\nCould not find branch_bench.py.\n"
        "  sweep_ksd.py calls the real M3 instead of copying it, so it needs\n"
        "  branch_bench.py in the same folder (or pass --bench /path/to/it).\n"
        f"  Looked in: {here}\n        and: {os.getcwd()}\n")


# -----------------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------------

def mask_stats(mask, ctx, min_mm3=8.0):
    """Volume, component count, largest component, how far it reaches."""
    if not mask.any():
        return dict(grown_mL=0.0, n_components=0, largest_comp_mL=0.0,
                    max_reach_mm=0.0)
    lab, n = ndi.label(mask)
    mm3 = np.bincount(lab.ravel()) * ctx.vox_mm3
    mm3[0] = 0.0
    return dict(
        grown_mL=float(mask.sum()) * ctx.vox_mm3 / 1000.0,
        n_components=int((mm3 >= min_mm3).sum()),
        largest_comp_mL=float(mm3.max()) / 1000.0,
        max_reach_mm=float(ctx.dist_out[mask].max()))


def cluster_ostia(rows, tol_mm=6.0):
    """Group ostia found at different k into one cluster each.

    An ostium that survives the whole sweep is a structure the band is not
    arguing about; one that appears at a single k is a threshold artefact. This
    is the label-free confidence signal the sweep exists to produce.

    Greedy, in k order: a point joins the nearest cluster within tol_mm, and a
    cluster takes at most one point per k value.
    """
    clusters = []                       # {"c": centre mm, "pts": [...], "ks": set}
    for r in sorted(rows, key=lambda r: (r["k"], -(r["radius"] or 0.0))):
        p = np.array(r["pos"], float)
        best, bestd = -1, tol_mm
        for j, cl in enumerate(clusters):
            if r["k"] in cl["ks"]:
                continue
            d = float(np.linalg.norm(p - cl["c"]))
            if d < bestd:
                best, bestd = j, d
        if best < 0:
            clusters.append({"c": p.copy(), "pts": [r], "ks": {r["k"]}})
            r["cluster"] = len(clusters)
        else:
            cl = clusters[best]
            cl["pts"].append(r)
            cl["ks"].add(r["k"])
            P = np.array([q["pos"] for q in cl["pts"]], float)
            cl["c"] = P.mean(axis=0)
            r["cluster"] = best + 1
    return clusters


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------

def write_curves(path, summary, case_id):
    """Three curves against k: ostia found, volume grown, band width."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    k = [s["k_sd"] for s in summary]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("ostia reported", "volume grown outside the aorta (mL)",
                        "band width (HU) and its edges"))
    fig.add_trace(go.Scatter(x=k, y=[s["n_ostia"] for s in summary],
                             mode="lines+markers", name="n ostia",
                             line=dict(color="#1f4e79", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["n_components"] for s in summary],
                             mode="lines+markers", name="n components",
                             line=dict(color="#7b6cd9", width=2,
                                       dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["grown_mL"] for s in summary],
                             mode="lines+markers", name="grown mL",
                             line=dict(color="#3fa34d", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["largest_comp_mL"] for s in summary],
                             mode="lines+markers", name="largest comp mL",
                             line=dict(color="#e07a5f", width=2,
                                       dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["width_hu"] for s in summary],
                             mode="lines+markers", name="band width HU",
                             line=dict(color="#111111", width=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["lo_hu"] for s in summary],
                             mode="lines+markers", name="lower edge HU",
                             line=dict(color="#c85b9b", width=2,
                                       dash="dash")), row=3, col=1)
    fig.add_trace(go.Scatter(x=k, y=[s["hi_hu"] for s in summary],
                             mode="lines+markers", name="upper edge HU",
                             line=dict(color="#4f86c6", width=2,
                                       dash="dash")), row=3, col=1)
    fig.update_xaxes(title_text="k_sd", row=3, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{case_id} -- M3 sensitivity to the lumen band</b>"
                        "<br><span style='font-size:13px'>band = lumen mean "
                        "&plusmn; k_sd &times; lumen sd</span>",
                   x=0.01, xanchor="left"),
        height=820, template="plotly_white",
        margin=dict(l=60, r=20, t=90, b=50))
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return True


def persistence_meshes(bb, ctx, clusters, aorta_mesh, n_k):
    """Aorta + every ostium cluster, grouped by how many k values found it."""
    meshes = []
    av, af = aorta_mesh
    if av is not None:
        meshes.append(dict(verts=av, faces=af, name="aorta (given mask)",
                           color="#d94a4a", opacity=0.30, text="parent aorta"))
    # warm = survives the sweep, cold = one threshold only
    palette = ["#cfd8dc", "#b0bec5", "#90a4ae", "#f2c14e", "#e07a5f", "#c1121f",
               "#7f0d16"]
    by_n = {}
    for i, cl in enumerate(clusters, 1):
        by_n.setdefault(len(cl["ks"]), []).append((i, cl))
    for n in sorted(by_n, reverse=True):
        items = by_n[n]
        pts = np.array([cl["c"] for _, cl in items], float)
        labels = []
        for i, cl in items:
            rr = [q["radius"] for q in cl["pts"] if q["radius"]]
            labels.append(
                f"cluster {i:03d}<br>seen at {n}/{n_k} k values<br>"
                f"k = {', '.join(f'{x:g}' for x in sorted(cl['ks']))}<br>"
                f"({cl['c'][0]:.1f}, {cl['c'][1]:.1f}, {cl['c'][2]:.1f}) mm<br>"
                f"r = {np.median(rr):.2f} mm" if rr else
                f"cluster {i:03d}<br>seen at {n}/{n_k} k values")
        frac = n / max(n_k, 1)
        meshes.append(dict(kind="points", points=pts,
                           name=f"seen at {n}/{n_k} k",
                           color=palette[min(int(frac * (len(palette) - 1)),
                                             len(palette) - 1)],
                           size=4 + 6 * frac, labels=labels))
    return meshes


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--outdir", default="sweep")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--bench", default=None,
                    help="path to branch_bench.py if it is not beside this file")
    ap.add_argument("--k", default="1.5,2.0,2.5,3.0,3.5,4.0",
                    help="comma-separated k_sd values to sweep")
    ap.add_argument("--roi-mm", type=float, default=30.0)
    ap.add_argument("--tol-mm", type=float, default=6.0,
                    help="two ostia from different k count as the same one "
                         "within this distance")
    ap.add_argument("--no-renders", action="store_true",
                    help="skip the per-k 3D renders (much faster)")
    args = ap.parse_args()

    bb, bench_path = load_bench(args.bench)
    ks = [float(s) for s in args.k.split(",") if s.strip()]
    case_id = args.case_id or os.path.splitext(
        os.path.basename(args.image))[0] or "case"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"sweep_ksd {__version__}   case {case_id}")
    print(f"  using {bench_path}  (branch_bench {bb.__version__})")

    img, _, ct, mk = bb.load_case(args.image, args.aorta_mask)
    ctx = bb.Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm)

    # mu and sd do not depend on k_sd, and neither does the bone mask or the
    # distance transform -- only lo/hi do. So build the context once and move
    # the band, which is both faster and provably the same band lumen_band()
    # would have returned for that k.
    mu, sd = ctx.mu, ctx.sd
    print(f"  lumen {mu:.0f} +/- {sd:.0f} HU   bone > {ctx.hi_cut:.0f}   "
          f"crop {ctx.ct.shape}")
    print(f"  sweeping k_sd = {', '.join(f'{k:g}' for k in ks)}")

    aorta_mesh = (None, None)
    if not args.no_renders:
        av, af = bb.fine_surface(ctx.mk, ctx.spacing, cap=120_000)
        aorta_mesh = (ctx.verts_to_world(av) if av is not None else None, af)

    summary, ostia_rows = [], []
    for k in ks:
        ctx.lo, ctx.hi = mu - k * sd, mu + k * sd
        t0 = time.time()
        res = bb.dedupe(bb.drop_end_caps(bb.m3_expansion(ctx), ctx), ctx)
        wall = time.time() - t0
        recs = bb.daughters(res, ctx)
        st = mask_stats(res.mask, ctx)

        radii = [r["radius_mm"] for r in recs if r["radius_mm"]]
        summary.append(dict(
            k_sd=k, lo_hu=round(ctx.lo, 1), hi_hu=round(ctx.hi, 1),
            width_hu=round(ctx.hi - ctx.lo, 1), n_ostia=len(recs),
            n_components=st["n_components"],
            grown_mL=round(st["grown_mL"], 2),
            largest_comp_mL=round(st["largest_comp_mL"], 2),
            max_reach_mm=round(st["max_reach_mm"], 1),
            median_radius_mm=round(float(np.median(radii)), 2) if radii else "",
            runtime_s=round(wall, 2), note=res.note))

        for r in recs:
            ostia_rows.append(dict(k=k, id=r["instance_id"],
                                   pos=r["ostium_xyz_mm"],
                                   seed=r["seed_xyz_mm"],
                                   dir=r["direction_xyz"],
                                   radius=r["radius_mm"], cluster=None))

        print(f"  k={k:<4g} band {ctx.lo:7.0f}-{ctx.hi:<7.0f} "
              f"({ctx.hi - ctx.lo:5.0f} HU)  {len(recs):>3} ostia  "
              f"{st['n_components']:>3} comps  {st['grown_mL']:7.1f} mL  "
              f"{wall:5.1f}s  {res.note}")

        if not args.no_renders:
            d = os.path.join(args.outdir, f"k_{k:.2f}")
            os.makedirs(d, exist_ok=True)
            meshes = []
            if aorta_mesh[0] is not None:
                meshes.append(dict(verts=aorta_mesh[0], faces=aorta_mesh[1],
                                   name="aorta (given mask)", color="#d94a4a",
                                   opacity=0.30, text="parent aorta"))
            bv, bf = bb.fine_surface(res.mask & ~ctx.mk, ctx.spacing,
                                     cap=120_000)
            if bv is not None:
                meshes.append(dict(verts=ctx.verts_to_world(bv), faces=bf,
                                   name=f"M3 grown (k={k:g})", color="#f2c14e",
                                   opacity=0.95, text=f"k_sd = {k:g}"))
            if recs:
                pts = np.array([r["ostium_xyz_mm"] for r in recs], float)
                labels = [f"{r['instance_id']}<br>"
                          f"({r['ostium_xyz_mm'][0]:.1f}, "
                          f"{r['ostium_xyz_mm'][1]:.1f}, "
                          f"{r['ostium_xyz_mm'][2]:.1f}) mm<br>"
                          f"r = {r['radius_mm']} mm" for r in recs]
                meshes.append(dict(kind="points", points=pts, name="ostia",
                                   color="#111111", size=6, labels=labels))
            ok = bb.write_interactive_html(
                os.path.join(d, "render.html"), meshes,
                f"{case_id} -- M3 at k_sd = {k:g}",
                subtitle=(f"band {ctx.lo:.0f}-{ctx.hi:.0f} HU &middot; "
                          f"{len(recs)} ostia &middot; {st['grown_mL']:.1f} mL "
                          f"&middot; {res.note}"),
                axis_note=" [patient/world]")
            if not ok:
                print("      (render skipped: plotly missing)", file=sys.stderr)

    # ---- clustering across k ------------------------------------------------
    clusters = cluster_ostia(ostia_rows, args.tol_mm)
    n_k = len(ks)
    stable = sum(1 for c in clusters if len(c["ks"]) == n_k)
    once = sum(1 for c in clusters if len(c["ks"]) == 1)
    print(f"  {len(clusters)} distinct ostia across the sweep: "
          f"{stable} found at every k, {once} at exactly one")

    # ---- CSVs ---------------------------------------------------------------
    p = os.path.join(args.outdir, "sweep_summary.csv")
    cols = ["k_sd", "lo_hu", "hi_hu", "width_hu", "n_ostia", "n_components",
            "grown_mL", "largest_comp_mL", "max_reach_mm", "median_radius_mm",
            "runtime_s", "note"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "sweep_ostia.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k_sd", "instance_id", "cluster_id",
                    "ostium_x_mm", "ostium_y_mm", "ostium_z_mm",
                    "seed_x_mm", "seed_y_mm", "seed_z_mm",
                    "dir_x", "dir_y", "dir_z", "radius_mm"])
        for r in sorted(ostia_rows, key=lambda r: (r["k"], r["cluster"])):
            w.writerow([r["k"], r["id"], r["cluster"],
                        *[round(v, 2) for v in r["pos"]],
                        *[round(v, 2) for v in r["seed"]],
                        *r["dir"], r["radius"] if r["radius"] else ""])
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "sweep_persistence.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "n_k_seen", "n_k_total", "persistence",
                    "k_values", "x_mm", "y_mm", "z_mm", "spread_mm",
                    "median_radius_mm"])
        order = sorted(range(len(clusters)),
                       key=lambda i: (-len(clusters[i]["ks"]),
                                      clusters[i]["c"][2]))
        for i in order:
            cl = clusters[i]
            P = np.array([q["pos"] for q in cl["pts"]], float)
            spread = float(np.linalg.norm(P - cl["c"], axis=1).max())
            rr = [q["radius"] for q in cl["pts"] if q["radius"]]
            w.writerow([i + 1, len(cl["ks"]), n_k,
                        round(len(cl["ks"]) / n_k, 3),
                        " ".join(f"{x:g}" for x in sorted(cl["ks"])),
                        *[round(float(v), 2) for v in cl["c"]],
                        round(spread, 2),
                        round(float(np.median(rr)), 2) if rr else ""])
    print(f"  wrote {p}")

    # ---- figures ------------------------------------------------------------
    p = os.path.join(args.outdir, "sweep_curves.html")
    if write_curves(p, summary, case_id):
        print(f"  wrote {p}")
    else:
        print("  sweep_curves.html skipped: plotly missing", file=sys.stderr)

    if not args.no_renders:
        p = os.path.join(args.outdir, "render_persistence.html")
        meshes = persistence_meshes(bb, ctx, clusters, aorta_mesh, n_k)
        if bb.write_interactive_html(
                p, meshes, f"{case_id} -- ostia persistence across k_sd",
                subtitle=(f"{len(clusters)} distinct ostia over {n_k} bands "
                          f"&middot; {stable} survive every k &middot; "
                          f"{once} appear once &middot; matched within "
                          f"{args.tol_mm:g} mm"),
                axis_note=" [patient/world]"):
            print(f"  wrote {p}")

    print("  open sweep_curves.html and render_persistence.html first; the "
          "per-k renders are in k_*/render.html.")


if __name__ == "__main__":
    main()