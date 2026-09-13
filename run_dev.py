#!/usr/bin/env python3
"""
run_dev.py -- run and score the whole development set in one command.

    python run_dev.py --manifest cases.csv --outdir dev

cases.csv, four columns with a header:

    case_id,image,aorta_mask,ref
    subject019,data/subject019/orig19.nii,data/subject019/mask19.nii,data/subject019/annotation.json
    ...

`ref` is the reference annotation JSON. A label volume also works but scores
against a DERIVED ostium rather than the declared one -- see score.py.

`--gates 0,2` runs the whole set at several origin-diameter thresholds so the
cost of the 2 mm eligibility rule is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from outputs import write_all
from run import process
from score import load_prediction, load_reference, score_case


def read_manifest(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("image", "aorta_mask", "ref"):
            if not os.path.isfile(r[k]):
                raise SystemExit(f"{r['case_id']}: missing {k} -> {r[k]}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default="dev")
    ap.add_argument("--gates", default="2",
                    help="origin-diameter thresholds in mm, e.g. '0,2'")
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--tolerances", default="3,5,6,8,10")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args()

    cases = read_manifest(args.manifest)
    gates = [float(s) for s in args.gates.split(",") if s.strip()]
    tols = sorted({float(s) for s in args.tolerances.split(",") if s.strip()}
                  | {args.tol_mm})
    os.makedirs(args.outdir, exist_ok=True)

    rows, runtimes = [], []
    for gate in gates:
        tag = f"g{gate:g}"
        pred_dir = os.path.join(args.outdir, f"predictions_{tag}")
        os.makedirs(pred_dir, exist_ok=True)
        print(f"\n=== origin-diameter gate {gate:g} mm ===")
        for c in cases:
            t0 = time.time()
            pred, case, daughters, labels = process(
                c["image"], c["aorta_mask"], c["case_id"],
                min_origin_diameter_mm=gate, verbose=False)
            out_json = os.path.join(pred_dir, f"{c['case_id']}.json")
            write_all(out_json,
                      os.path.join(pred_dir, f"{c['case_id']}_files"),
                      c["case_id"], case, daughters, labels, pred,
                      json_only=args.json_only)
            wall = time.time() - t0
            runtimes.append(wall)

            p, _ = load_prediction(out_json)
            ref, _ = load_reference(c["ref"], c["image"], c["aorta_mask"])
            for t in tols:
                s, _ = score_case(ref, p, t)
                s.update(case_id=c["case_id"], gate_mm=gate,
                         runtime_s=round(wall, 2))
                rows.append(s)
            at = next(r for r in rows if r["case_id"] == c["case_id"]
                      and r["gate_mm"] == gate and r["tol_mm"] == args.tol_mm)
            print(f"  {c['case_id']:<12} ref {at['n_ref']}  pred {at['n_pred']}"
                  f"  TP {at['tp']}  F1 {at['f1']:.2f}"
                  f"  ost {at['mean_ostium_err_mm'] or '-'} mm"
                  f"  {wall:.1f}s")

    print(f"\n  MEAN F1 AT {args.tol_mm:g} mm")
    print(f"    {'gate':<8}{'F1':>8}{'P':>8}{'R':>8}{'ostium mm':>12}"
          f"{'dir deg':>10}{'seed mm':>10}")
    for gate in gates:
        at = [r for r in rows if r["gate_mm"] == gate
              and r["tol_mm"] == args.tol_mm]

        def avg(key):
            v = [r[key] for r in at if r[key] != ""]
            return f"{np.mean(v):.2f}" if v else "-"

        print(f"    {gate:<8g}{np.mean([r['f1'] for r in at]):>8.3f}"
              f"{np.mean([r['precision'] for r in at]):>8.3f}"
              f"{np.mean([r['recall'] for r in at]):>8.3f}"
              f"{avg('mean_ostium_err_mm'):>12}"
              f"{avg('mean_direction_deg'):>10}"
              f"{avg('mean_seed_to_path_mm'):>10}")

    if runtimes:
        print(f"\n  runtime per case: mean {np.mean(runtimes):.1f}s  "
              f"max {np.max(runtimes):.1f}s  (target <= 60s)")

    p = os.path.join(args.outdir, "scores.csv")
    cols = ["case_id", "gate_mm", "tol_mm", "n_ref", "n_pred", "tp", "fp", "fn",
            "precision", "recall", "f1", "mean_ostium_err_mm",
            "median_ostium_err_mm", "mean_direction_deg",
            "mean_seed_to_path_mm", "mean_radius_err_mm", "runtime_s"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
