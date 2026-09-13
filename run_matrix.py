#!/usr/bin/env python3
"""
run_matrix.py -- every subject x every band setting x every method, scored, in
one table. One command, one answer, no more pasting CSVs back and forth.

    python run_matrix.py --manifest cases.csv --settings sd:2.5,sd:3.0 \
        --outdir matrix

cases.csv is four columns with a header:

    case_id,image,aorta_mask,gt
    subject001,data/orig1.nii,data/mask1.nii,data/gt1.nii
    subject019,data/orig19.nii,data/mask19.nii,data/gt19.nii
    subject021,data/orig21.nii,data/mask21.nii,data/gt21.nii

or pass cases inline, repeatably:

    --case subject019,orig19.nii,mask19.nii,gt19.nii

A SETTING is a band rule and, for the sd rule, its k:

    sd:2.5   lumen mean +/- 2.5 sd   (the current default)
    sd:3.0   lumen mean +/- 3.0 sd
    v2       absolute floor, relative ceiling

Volumes are loaded once per case and the context once per band rule, so adding
k values is nearly free -- only lo/hi move between them.

OUTPUT
  <outdir>/matrix.csv          case x setting x method x tolerance, full scores
  <outdir>/by_setting.csv      mean F1 per setting, which is the decision
  <outdir>/per_branch.csv      which GT daughter each configuration found
  <outdir>/matrix.html         the same table, readable, no dependencies
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import os
import sys
import time

import numpy as np

__version__ = "2026-09-14.matrix1"


def load_module(fname, alias, explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([explicit] if explicit else []) + [os.path.join(here, fname),
                                                 os.path.join(os.getcwd(),
                                                              fname)]:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(alias, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[alias] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"Could not find {fname} beside this script or in {os.getcwd()}")


def parse_settings(s):
    """'sd:2.5,sd:3.0,v2' -> [('sd', 2.5), ('sd', 3.0), ('v2', None)]."""
    out = []
    for tok in (t.strip() for t in s.split(",") if t.strip()):
        if ":" in tok:
            rule, val = tok.split(":", 1)
            out.append((rule.strip(), float(val)))
        else:
            out.append((tok, None))
    for rule, _ in out:
        if rule not in ("sd", "v2"):
            raise SystemExit(f"Unknown band rule '{rule}'; use sd or v2.")
    return out


def setting_name(rule, k):
    return f"{rule}:{k:g}" if k is not None else rule


def read_cases(args):
    cases = []
    if args.manifest:
        with open(args.manifest, newline="") as fh:
            for r in csv.DictReader(fh):
                cases.append((r["case_id"], r["image"], r["aorta_mask"],
                              r["gt"]))
    for c in args.case or []:
        parts = [p.strip() for p in c.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"--case needs id,image,mask,gt -- got '{c}'")
        cases.append(tuple(parts))
    if not cases:
        raise SystemExit("Pass --manifest or at least one --case.")
    for cid, im, mk, gt in cases:
        for p in (im, mk, gt):
            if not os.path.isfile(p):
                raise SystemExit(f"{cid}: missing file {p}")
    return cases


# -----------------------------------------------------------------------------

def f1_cell(v):
    """Colour a cell by F1 so the table can be read at a glance."""
    if v == "" or v is None:
        return "background:#f5f5f5"
    x = float(v)
    # grey -> amber -> green
    if x <= 0:
        return "background:#f1f3f4;color:#9aa0a6"
    r, g, b = (int(241 - 100 * x), int(243 - 40 * x), int(244 - 180 * x))
    return f"background:rgb({r},{g},{b})"


def write_html(path, case_ids, settings, methods, rows, tol, by_setting):
    idx = {(r["case_id"], r["setting"], r["method"]): r for r in rows
           if r["tol_mm"] == tol}
    p = ['<meta charset="utf-8"><title>branch bench matrix</title>',
         "<style>body{font:14px/1.45 system-ui,sans-serif;margin:24px;"
         "color:#202124}h1{font-size:20px;margin:0 0 4px}"
         "h2{font-size:15px;margin:26px 0 8px}"
         "p.sub{color:#5f6368;margin:0 0 18px}"
         "table{border-collapse:collapse;margin-bottom:8px}"
         "th,td{border:1px solid #dadce0;padding:5px 9px;text-align:right;"
         "font-variant-numeric:tabular-nums}"
         "th{background:#f8f9fa;font-weight:600;text-align:center}"
         "td.l,th.l{text-align:left}"
         "small{color:#5f6368}</style>",
         f"<h1>branch_bench matrix</h1>",
         f'<p class="sub">F1 at {tol:g} mm tolerance. '
         f'{len(case_ids)} case(s) &times; {len(settings)} setting(s) '
         f'&times; {len(methods)} method(s).</p>']

    p.append("<h2>F1 by case and setting</h2><table><tr><th class='l'>case</th>"
             "<th class='l'>setting</th>"
             + "".join(f"<th>{html.escape(m)}</th>" for m in methods)
             + "</tr>")
    for cid in case_ids:
        for sname in settings:
            p.append(f"<tr><td class='l'>{html.escape(cid)}</td>"
                     f"<td class='l'>{html.escape(sname)}</td>")
            for m in methods:
                r = idx.get((cid, sname, m))
                v = r["f1"] if r else ""
                txt = f"{v:.2f}" if v != "" else "&ndash;"
                extra = (f"<br><small>{r['tp']}/{r['n_gt']} &middot; "
                         f"{r['n_pred']} pred</small>") if r else ""
                p.append(f"<td style='{f1_cell(v)}'>{txt}{extra}</td>")
            p.append("</tr>")
    p.append("</table>")

    p.append("<h2>mean F1 across cases &mdash; this is the decision</h2>"
             "<table><tr><th class='l'>setting</th>"
             + "".join(f"<th>{html.escape(m)}</th>" for m in methods)
             + "</tr>")
    for sname in settings:
        p.append(f"<tr><td class='l'>{html.escape(sname)}</td>")
        for m in methods:
            v = by_setting.get((sname, m), "")
            txt = f"{v:.3f}" if v != "" else "&ndash;"
            p.append(f"<td style='{f1_cell(v)}'>{txt}</td>")
        p.append("</tr>")
    p.append("</table>")
    with open(path, "w") as fh:
        fh.write("\n".join(p))


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--case", action="append",
                    help="id,image,aorta_mask,gt  (repeatable)")
    ap.add_argument("--settings", default="sd:2.5,sd:3.0")
    ap.add_argument("--floors", default="auto",
                    help="absolute band floors in HU as a third axis, e.g. "
                         "'auto,250,280'. 'auto' keeps mean - k*sd. Only the "
                         "floor moves; ceiling and bone cut are untouched. "
                         "Shown as '+f250'.")
    ap.add_argument("--min-reach", default="0",
                    help="comma-separated eligibility floors in mm, as a "
                         "second axis: '0,5' scores every band setting with "
                         "and without the 5 mm rule. Shown as '+r5'.")
    ap.add_argument("--methods", default="M1,M2,M3,M4")
    ap.add_argument("--outdir", default="matrix")
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--tolerances", default="3,5,6,8,10")
    ap.add_argument("--roi-mm", type=float, default=30.0)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--save-ostia", action="store_true",
                    help="also write every configuration's ostia.csv")
    args = ap.parse_args()

    bb = load_module("branch_bench.py", "_bench", args.bench)
    so = load_module("score_ostia.py", "_score", args.scorer)
    cases = read_cases(args)
    settings = parse_settings(args.settings)
    reaches = [float(s) for s in args.min_reach.split(",") if s.strip()]
    floors = [None if s.strip().lower() in ("auto", "none", "")
              else float(s) for s in args.floors.split(",") if s.strip()]
    want = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    tols = sorted({float(s) for s in args.tolerances.split(",") if s.strip()}
                  | {args.tol_mm})
    os.makedirs(args.outdir, exist_ok=True)
    print(f"run_matrix {__version__}   branch_bench {bb.__version__}")
    print(f"  {len(cases)} case(s) x {len(settings)} setting(s) x "
          f"{len(want)} method(s)")

    plan = dict(M1=bb.m1_frangi, M2=bb.m2_band_connect,
                M3=bb.m3_expansion, M4=bb.m4_wall_flux)
    rows, branch_rows, method_names = [], [], []
    t_start = time.time()

    for cid, image, amask, gtpath in cases:
        print(f"\n  == {cid} ==")
        gt, _ctx0 = so.gt_from_volume(bb, image, amask, gtpath, args.roi_mm)
        print(f"     {len(gt)} GT daughters: "
              f"{', '.join(g['name'] for g in gt)}")
        img, _m, ct, mk = bb.load_case(image, amask)

        for rule in sorted({r for r, _ in settings}):
            ctx = bb.Ctx(img, ct, mk, img.GetSpacing(), roi_mm=args.roi_mm,
                         band_rule=rule)
            base_lo, base_hi = ctx.lo, ctx.hi
            for (r2, k), floor, reach in [((r, k), f, q)
                                          for r, k in settings if r == rule
                                          for f in floors for q in reaches]:
                if rule == "sd" and k is not None:
                    ctx.lo, ctx.hi = ctx.mu - k * ctx.sd, ctx.mu + k * ctx.sd
                else:
                    ctx.lo, ctx.hi = base_lo, base_hi
                if floor is not None:          # floor only; ceiling untouched
                    ctx.lo = float(floor)
                ctx.min_reach_mm = reach
                sname = (setting_name(r2, k)
                         + (f"+f{floor:g}" if floor is not None else "")
                         + (f"+r{reach:g}" if reach else ""))
                print(f"     {sname:<11} band {ctx.lo:7.0f}-{ctx.hi:<7.0f}",
                      end="")

                for mid in want:
                    if mid not in plan:
                        continue
                    t0 = time.time()
                    try:
                        res = bb.dedupe(bb.drop_end_caps(plan[mid](ctx), ctx),
                                        ctx)
                    except Exception as exc:
                        print(f"\n       {mid} FAILED: {exc}", file=sys.stderr)
                        continue
                    if res.name not in method_names:
                        method_names.append(res.name)
                    recs = bb.daughters(res, ctx)
                    pred = [dict(id=r["instance_id"],
                                 xyz=np.array(r["ostium_xyz_mm"], float),
                                 radius=r["radius_mm"]) for r in recs]
                    if args.save_ostia:
                        d = os.path.join(args.outdir, cid,
                                         sname.replace(":", "_"))
                        os.makedirs(d, exist_ok=True)
                        bb.save_method(res, ctx, d, f"{cid} {sname}")

                    for t in tols:
                        pairs, ug, _up = so.match(gt, pred, t)
                        m = so.metrics(gt, pred, pairs, t)
                        m.update(case_id=cid, setting=sname, method=res.name,
                                 runtime_s=round(time.time() - t0, 2))
                        rows.append(m)
                        hit = {gt[i]["name"]: d for i, _j, d in pairs}
                        for g0 in gt:
                            branch_rows.append(dict(
                                case_id=cid, setting=sname, method=res.name,
                                tol_mm=t, gt_name=g0["name"],
                                matched=int(g0["name"] in hit),
                                error_mm=round(hit[g0["name"]], 3)
                                if g0["name"] in hit else ""))
                    at = next(x for x in rows if x["case_id"] == cid
                              and x["setting"] == sname
                              and x["method"] == res.name
                              and x["tol_mm"] == args.tol_mm)
                    print(f"   {mid} F1 {at['f1']:.2f}", end="")
                print()

    # ---- aggregate ---------------------------------------------------------
    case_ids = [c[0] for c in cases]
    snames = [setting_name(r, k)
              + (f"+f{f:g}" if f is not None else "")
              + (f"+r{q:g}" if q else "")
              for r, k in settings for f in floors for q in reaches]
    by_setting = {}
    for sname in snames:
        for m in method_names:
            vals = [r["f1"] for r in rows if r["setting"] == sname
                    and r["method"] == m and r["tol_mm"] == args.tol_mm]
            if vals:
                by_setting[(sname, m)] = float(np.mean(vals))

    print(f"\n  MEAN F1 ACROSS {len(case_ids)} CASE(S) AT {args.tol_mm:g} mm")
    print("    " + "setting".ljust(10)
          + "".join(m.split()[0].rjust(8) for m in method_names))
    for sname in snames:
        print("    " + sname.ljust(10)
              + "".join(f"{by_setting.get((sname, m), float('nan')):>8.2f}"
                        for m in method_names))
    best = max(by_setting.items(), key=lambda kv: kv[1]) if by_setting else None
    if best:
        print(f"\n  best configuration: {best[0][1]} at {best[0][0]}  "
              f"mean F1 {best[1]:.3f}")

    # ---- CSVs ---------------------------------------------------------------
    cols = ["case_id", "setting", "method", "tol_mm", "n_gt", "n_pred", "tp",
            "fp", "fn", "precision", "recall", "f1", "mean_err_mm",
            "median_err_mm", "max_err_mm", "loc_score", "runtime_s"]
    p = os.path.join(args.outdir, "matrix.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {p}")

    p = os.path.join(args.outdir, "by_setting.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "method", "tol_mm", "n_cases", "mean_f1",
                    "mean_precision", "mean_recall", "mean_err_mm"])
        for sname in snames:
            for m in method_names:
                sel = [r for r in rows if r["setting"] == sname
                       and r["method"] == m and r["tol_mm"] == args.tol_mm]
                if not sel:
                    continue
                errs = [r["mean_err_mm"] for r in sel if r["mean_err_mm"] != ""]
                w.writerow([sname, m, args.tol_mm, len(sel),
                            round(float(np.mean([r["f1"] for r in sel])), 4),
                            round(float(np.mean([r["precision"] for r in sel])), 4),
                            round(float(np.mean([r["recall"] for r in sel])), 4),
                            round(float(np.mean(errs)), 3) if errs else ""])
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "per_branch.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "setting", "method",
                                           "tol_mm", "gt_name", "matched",
                                           "error_mm"])
        w.writeheader()
        w.writerows(branch_rows)
    print(f"  wrote {p}")

    p = os.path.join(args.outdir, "matrix.html")
    write_html(p, case_ids, snames, method_names, rows, args.tol_mm, by_setting)
    print(f"  wrote {p}")
    print(f"  {time.time() - t_start:.0f}s total")


if __name__ == "__main__":
    main()