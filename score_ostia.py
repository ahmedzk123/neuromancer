#!/usr/bin/env python3
"""
score_ostia.py -- match predicted ostia against ground truth and report numbers
instead of impressions.

Subject019 looked like a total failure in the renders and was in fact finding
three of three daughters at k_sd = 3. The GT branches are 14-34 voxels against
several mL of grown lumen, so a correct detection is invisible next to a wrong
one. Nothing should be judged by eye again.

    # score a whole k_sd sweep
    python score_ostia.py --gt-csv diag/gt_summary.csv --pred sweep --outdir score

    # score the four methods
    python score_ostia.py --gt-csv diag/gt_summary.csv --pred bench --outdir score

    # GT straight from the label volume instead of gt_profile's output
    python score_ostia.py --gt gt19.nii --image orig19.nii \
        --aorta-mask mask19.nii --pred bench --outdir score

Matching is one-to-one and optimal (Hungarian assignment on the distance
matrix), not greedy nearest-neighbour: greedy lets one lucky prediction claim a
GT point that a better prediction should have had, which inflates the error and
can turn a true positive into a pair of mistakes.

OUTPUT
  <outdir>/scores.csv       group x tolerance -> tp/fp/fn, P/R/F1, ostium error
  <outdir>/matches.csv      every GT, every prediction, matched or not
  <outdir>/per_branch.csv   which GT daughter each group found, and by how far
  <outdir>/score_curves.html   P/R/F1 across groups, and F1 against tolerance
  <outdir>/score_render.html   GT / hits / misses / false alarms in 3D
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

import numpy as np

__version__ = "2026-09-14.score1"


def load_bench(explicit=None, required=False):
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
    if required:
        raise SystemExit("Could not find branch_bench.py; pass --bench.")
    return None, None


# -----------------------------------------------------------------------------
# reading ground truth and predictions
# -----------------------------------------------------------------------------

def gt_from_csv(path):
    """Ostia from gt_profile.py's gt_summary.csv."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(dict(
                name=r.get("name") or r.get("label") or f"gt_{len(out) + 1}",
                xyz=np.array([float(r["ostium_x_mm"]), float(r["ostium_y_mm"]),
                              float(r["ostium_z_mm"])])))
    return out


def gt_from_volume(bb, image, aorta_mask, gt_path, roi_mm, branch_labels=None):
    """Ostium per GT label = its voxel nearest the aortic wall, as in gt_profile."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi                            # noqa: F401

    img, _, ct, mk = bb.load_case(image, aorta_mask)
    g = sitk.ReadImage(gt_path)
    if g.GetSize() != img.GetSize():
        raise SystemExit(f"GT grid {g.GetSize()} != image {img.GetSize()}.")
    gt = sitk.GetArrayFromImage(g).astype(np.int32)
    ctx = bb.Ctx(img, ct, mk, img.GetSpacing(), roi_mm=roi_mm)
    gtc = gt[ctx.region]

    ids = [int(v) for v in np.unique(gtc) if v > 0]
    if branch_labels:
        ids = [L for L in ids if L in branch_labels]
    else:
        ids = [L for L in ids
               if float(((gtc == L) & ctx.mk).sum())
               / max(float((gtc == L).sum()), 1.0) < 0.5]
    out = []
    for L in ids:
        comp = gtc == L
        near = np.argwhere(comp)[int(np.argmin(ctx.dist_out[comp]))]
        out.append(dict(name=f"label_{L}",
                        xyz=np.asarray(ctx.to_mm(near.astype(float)), float)))
    return out, ctx


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_predictions(path):
    """Predictions as {group_name: [ {id, xyz, radius} ]}.

    Understands three layouts: a sweep's sweep_ostia.csv (one group per k_sd),
    a branch_bench outdir (one group per method folder), and a bare ostia.csv.
    """
    files = []
    if os.path.isdir(path):
        s = os.path.join(path, "sweep_ostia.csv")
        if os.path.isfile(s):
            files.append(s)
        files += sorted(glob.glob(os.path.join(path, "*", "ostia.csv")))
        b = os.path.join(path, "ostia.csv")
        if os.path.isfile(b):
            files.append(b)
    elif os.path.isfile(path):
        files.append(path)
    if not files:
        raise SystemExit(f"No ostia.csv or sweep_ostia.csv found under {path}")

    groups = {}
    for f in files:
        rows = _rows(f)
        if not rows:
            # A method that found nothing must still be scored, at recall 0 --
            # dropping it would quietly flatter the comparison.
            g = os.path.basename(os.path.dirname(os.path.abspath(f)))
            groups.setdefault(g or os.path.basename(f), [])
            continue
        if "k_sd" in rows[0]:
            for r in rows:
                g = f"k={float(r['k_sd']):g}"
                groups.setdefault(g, []).append(_one(r, len(groups.get(g, []))))
        else:
            g = os.path.basename(os.path.dirname(os.path.abspath(f)))
            if g in ("", ".", os.path.basename(path)):
                g = os.path.splitext(os.path.basename(f))[0]
            for r in rows:
                groups.setdefault(g, []).append(_one(r, len(groups.get(g, []))))
    return groups


def _one(r, i):
    rad = r.get("radius_mm", "")
    return dict(id=r.get("instance_id") or f"pred_{i + 1:03d}",
                xyz=np.array([float(r["ostium_x_mm"]), float(r["ostium_y_mm"]),
                              float(r["ostium_z_mm"])]),
                radius=float(rad) if rad not in ("", None) else None)


# -----------------------------------------------------------------------------
# matching
# -----------------------------------------------------------------------------

def match(gt, pred, tol_mm):
    """Optimal one-to-one assignment, then drop pairs beyond the tolerance.

    Returns (pairs, unmatched_gt_idx, unmatched_pred_idx) where each pair is
    (gt_index, pred_index, distance_mm).
    """
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))
    G = np.array([g["xyz"] for g in gt], float)
    P = np.array([p["xyz"] for p in pred], float)
    D = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=2)

    try:
        from scipy.optimize import linear_sum_assignment
        big = D.max() * 10 + 1e6
        C = np.where(D <= tol_mm, D, big)
        ri, ci = linear_sum_assignment(C)
        cand = [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci)]
    except ImportError:                       # greedy fallback
        cand, used_g, used_p = [], set(), set()
        for i, j in sorted(((i, j) for i in range(len(gt))
                            for j in range(len(pred))),
                           key=lambda t: D[t[0], t[1]]):
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            cand.append((i, j, float(D[i, j])))

    pairs = [(i, j, d) for i, j, d in cand if d <= tol_mm]
    mg = {i for i, _, _ in pairs}
    mp = {j for _, j, _ in pairs}
    return (pairs, [i for i in range(len(gt)) if i not in mg],
            [j for j in range(len(pred)) if j not in mp])


def metrics(gt, pred, pairs, tol_mm):
    tp, fp, fn = len(pairs), len(pred) - len(pairs), len(gt) - len(pairs)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    err = [d for _, _, d in pairs]
    return dict(
        n_gt=len(gt), n_pred=len(pred), tol_mm=tol_mm, tp=tp, fp=fp, fn=fn,
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        mean_err_mm=round(float(np.mean(err)), 3) if err else "",
        median_err_mm=round(float(np.median(err)), 3) if err else "",
        max_err_mm=round(float(np.max(err)), 3) if err else "",
        # 1 at a perfect hit, 0 at the tolerance edge -- a crude stand-in for
        # the challenge's separate ostium-localisation term
        loc_score=round(float(np.mean([1.0 - d / tol_mm for d in err])), 4)
        if err else 0.0)


def sort_key(name):
    if name.startswith("k="):
        try:
            return (0, float(name[2:]), "")
        except ValueError:
            pass
    return (1, 0.0, name)


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------

def write_curves(path, case_id, rows, groups, main_tol, tols):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    names = sorted(groups, key=sort_key)
    numeric = all(n.startswith("k=") for n in names) and len(names) > 1
    xs = [float(n[2:]) for n in names] if numeric else names

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.14,
        subplot_titles=(f"precision / recall / F1 at {main_tol:g} mm tolerance",
                        "F1 against the matching tolerance"))
    at = {r["group"]: r for r in rows if r["tol_mm"] == main_tol}
    mode = "lines+markers" if numeric else "markers"
    for key, col in (("precision", "#4f86c6"), ("recall", "#3fa34d"),
                     ("f1", "#1f4e79")):
        fig.add_trace(go.Scatter(
            x=xs, y=[at[n][key] for n in names], mode=mode, name=key,
            line=dict(color=col, width=3),
            marker=dict(size=10, color=col)), row=1, col=1)
    for n in names:
        ys = [next(r["f1"] for r in rows if r["group"] == n
                   and r["tol_mm"] == t) for t in tols]
        fig.add_trace(go.Scatter(x=list(tols), y=ys, mode="lines+markers",
                                 name=n), row=2, col=1)

    fig.update_xaxes(title_text="k_sd" if numeric else "method", row=1, col=1)
    fig.update_yaxes(title_text="score", row=1, col=1)
    fig.update_xaxes(title_text="tolerance (mm)", row=2, col=1)
    fig.update_yaxes(title_text="F1", row=2, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{case_id} -- ostium detection scored</b>",
                   x=0.01, xanchor="left"),
        height=760, template="plotly_white",
        margin=dict(l=60, r=20, t=90, b=50))
    fig.write_html(path, include_plotlyjs="inline", full_html=True)
    return True


def write_render(bb, ctx, path, case_id, gt, groups, results, main_tol, best):
    """GT, hits, misses and false alarms for the best-scoring group."""
    try:
        import plotly.graph_objects as go                        # noqa: F401
    except ImportError:
        return False
    pairs, ug, up = results[best]["detail"][main_tol]
    pred = groups[best]

    meshes = []
    if ctx is not None:
        av, af = bb.fine_surface(ctx.mk, ctx.spacing, cap=120_000)
        if av is not None:
            meshes.append(dict(verts=ctx.verts_to_world(av), faces=af,
                               name="aorta (given mask)", color="#d94a4a",
                               opacity=0.25, text="parent aorta"))

    if pairs:
        meshes.append(dict(
            kind="points", points=np.array([pred[j]["xyz"] for _, j, _ in pairs]),
            name=f"hit ({len(pairs)})", color="#2f9e9e", size=8,
            labels=[f"{pred[j]['id']} -> {gt[i]['name']}<br>{d:.2f} mm"
                    for i, j, d in pairs]))
        seg = np.full((len(pairs) * 3, 3), np.nan)
        seg[0::3] = np.array([gt[i]["xyz"] for i, _, _ in pairs])
        seg[1::3] = np.array([pred[j]["xyz"] for _, j, _ in pairs])
        meshes.append(dict(kind="lines", points=seg, name="error", width=4,
                           color="#2f9e9e"))
    if ug:
        meshes.append(dict(
            kind="points", points=np.array([gt[i]["xyz"] for i in ug]),
            name=f"missed GT ({len(ug)})", color="#c1121f", size=9,
            labels=[gt[i]["name"] for i in ug]))
    if up:
        meshes.append(dict(
            kind="points", points=np.array([pred[j]["xyz"] for j in up]),
            name=f"false alarm ({len(up)})", color="#b0bec5", size=6,
            labels=[pred[j]["id"] for j in up]))
    meshes.append(dict(
        kind="points", points=np.array([g["xyz"] for g in gt]),
        name="ground truth", color="#111111", size=5,
        labels=[g["name"] for g in gt]))

    m = results[best]["rows"][main_tol]
    return bb.write_interactive_html(
        path, meshes, f"{case_id} -- {best} at {main_tol:g} mm",
        subtitle=(f"TP {m['tp']} &middot; FP {m['fp']} &middot; FN {m['fn']}"
                  f" &middot; P {m['precision']:.2f} R {m['recall']:.2f} "
                  f"F1 {m['f1']:.2f} &middot; mean error "
                  f"{m['mean_err_mm']} mm"),
        axis_note=" [patient/world]")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="a sweep/bench outdir, or one ostia.csv")
    ap.add_argument("--gt-csv", default=None,
                    help="gt_summary.csv written by gt_profile.py")
    ap.add_argument("--gt", default=None, help="GT label volume (needs --image "
                                               "and --aorta-mask)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--aorta-mask", default=None)
    ap.add_argument("--branch-labels", default=None)
    ap.add_argument("--outdir", default="score")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--tol-mm", type=float, default=5.0,
                    help="headline matching tolerance")
    ap.add_argument("--tolerances", default="3,5,6,8,10",
                    help="tolerances to tabulate, so the choice is visible")
    ap.add_argument("--roi-mm", type=float, default=30.0)
    args = ap.parse_args()

    if not args.gt_csv and not args.gt:
        raise SystemExit("Pass --gt-csv (from gt_profile.py) or --gt.")
    tols = sorted({float(s) for s in args.tolerances.split(",") if s.strip()}
                  | {args.tol_mm})
    case_id = args.case_id or os.path.basename(
        os.path.abspath(args.pred)) or "case"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"score_ostia {__version__}   case {case_id}")

    ctx, bb = None, None
    if args.gt_csv:
        gt = gt_from_csv(args.gt_csv)
        if args.image and args.aorta_mask:
            bb, _ = load_bench(args.bench)
            if bb is not None:
                import SimpleITK as sitk                          # noqa: F401
                img, _m, ct, mk = bb.load_case(args.image, args.aorta_mask)
                ctx = bb.Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm)
    else:
        if not (args.image and args.aorta_mask):
            raise SystemExit("--gt needs --image and --aorta-mask.")
        bb, _ = load_bench(args.bench, required=True)
        labels = ([int(s) for s in args.branch_labels.split(",")]
                  if args.branch_labels else None)
        gt, ctx = gt_from_volume(bb, args.image, args.aorta_mask, args.gt,
                                 args.roi_mm, labels)
    if not gt:
        raise SystemExit("No ground-truth daughters found.")
    if bb is None:
        bb, _ = load_bench(args.bench)

    groups = read_predictions(args.pred)
    print(f"  {len(gt)} GT daughters, {len(groups)} prediction group(s): "
          f"{', '.join(sorted(groups, key=sort_key))}")

    # ---- score every group at every tolerance ------------------------------
    rows, results = [], {}
    for g in sorted(groups, key=sort_key):
        results[g] = dict(rows={}, detail={})
        for t in tols:
            pairs, ug, up = match(gt, groups[g], t)
            m = metrics(gt, groups[g], pairs, t)
            m["group"] = g
            rows.append(m)
            results[g]["rows"][t] = m
            results[g]["detail"][t] = (pairs, ug, up)

    print(f"\n  AT {args.tol_mm:g} mm TOLERANCE")
    print(f"    {'group':<18}{'n':>4}{'TP':>4}{'FP':>4}{'FN':>4}"
          f"{'P':>7}{'R':>7}{'F1':>7}{'err':>8}")
    for g in sorted(groups, key=sort_key):
        m = results[g]["rows"][args.tol_mm]
        print(f"    {g:<18}{m['n_pred']:>4}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}"
              f"{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
              f"{(m['mean_err_mm'] if m['mean_err_mm'] != '' else '-'):>8}")

    best = max(sorted(groups, key=sort_key),
               key=lambda g: (results[g]["rows"][args.tol_mm]["f1"],
                              -(results[g]["rows"][args.tol_mm]["fp"])))
    bm = results[best]["rows"][args.tol_mm]
    print(f"\n  best at {args.tol_mm:g} mm: {best}  "
          f"F1 {bm['f1']:.2f}  (P {bm['precision']:.2f} / R {bm['recall']:.2f})")

    # ---- which daughter is hard --------------------------------------------
    print("\n  PER GT DAUGHTER (at the headline tolerance)")
    found_by = {g0["name"]: [] for g0 in gt}
    for g in sorted(groups, key=sort_key):
        pairs, _, _ = results[g]["detail"][args.tol_mm]
        for i, _j, d in pairs:
            found_by[gt[i]["name"]].append((g, d))
    for g0 in gt:
        hits = found_by[g0["name"]]
        if hits:
            bestd = min(d for _, d in hits)
            print(f"    {g0['name']:<12} found by {len(hits)}/{len(groups)} "
                  f"group(s), best {bestd:.2f} mm  "
                  f"[{', '.join(n for n, _ in hits)}]")
        else:
            print(f"    {g0['name']:<12} NEVER FOUND by any group at "
                  f"{args.tol_mm:g} mm")

    # ---- CSVs ---------------------------------------------------------------
    p = os.path.join(args.outdir, "scores.csv")
    cols = ["group", "n_gt", "n_pred", "tol_mm", "tp", "fp", "fn", "precision",
            "recall", "f1", "mean_err_mm", "median_err_mm", "max_err_mm",
            "loc_score"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in rows])
    print(f"\n  wrote {p}")

    p = os.path.join(args.outdir, "matches.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "tol_mm", "status", "gt_name", "gt_x_mm", "gt_y_mm",
                    "gt_z_mm", "pred_id", "pred_x_mm", "pred_y_mm", "pred_z_mm",
                    "error_mm"])
        for g in sorted(groups, key=sort_key):
            for t in tols:
                pairs, ug, up = results[g]["detail"][t]
                for i, j, d in pairs:
                    w.writerow([g, t, "TP", gt[i]["name"],
                                *[round(v, 2) for v in gt[i]["xyz"]],
                                groups[g][j]["id"],
                                *[round(v, 2) for v in groups[g][j]["xyz"]],
                                round(d, 3)])
                for i in ug:
                    w.writerow([g, t, "FN", gt[i]["name"],
                                *[round(v, 2) for v in gt[i]["xyz"]],
                                "", "", "", "", ""])
                for j in up:
                    w.writerow([g, t, "FP", "", "", "", "",
                                groups[g][j]["id"],
                                *[round(v, 2) for v in groups[g][j]["xyz"]], ""])
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "per_branch.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gt_name", "group", "tol_mm", "matched", "error_mm"])
        for g in sorted(groups, key=sort_key):
            for t in tols:
                pairs, ug, _ = results[g]["detail"][t]
                hit = {gt[i]["name"]: d for i, _j, d in pairs}
                for g0 in gt:
                    w.writerow([g0["name"], g, t,
                                int(g0["name"] in hit),
                                round(hit[g0["name"]], 3)
                                if g0["name"] in hit else ""])
    print(f"  wrote {p}")

    # ---- figures ------------------------------------------------------------
    p = os.path.join(args.outdir, "score_curves.html")
    if write_curves(p, case_id, rows, groups, args.tol_mm, tols):
        print(f"  wrote {p}")
    else:
        print("  score_curves.html skipped: plotly missing", file=sys.stderr)

    if bb is not None:
        p = os.path.join(args.outdir, "score_render.html")
        if write_render(bb, ctx, p, case_id, gt, groups, results,
                        args.tol_mm, best):
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()