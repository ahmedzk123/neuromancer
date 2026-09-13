#!/usr/bin/env python3
"""
run.py -- detect every eligible artery arising directly from the supplied aorta.

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz \
        --output prediction.json

Writes prediction.json at --output. Unless --json-only is given it also writes,
into a sibling folder named after the output file:

    daughters.nii.gz   label volume on the input grid, one label per daughter
    snap_labels.txt    ITK-SNAP Label Description File for that volume
    ostia.csv          the same numbers in a flat table
    check.html         3D verification view: mask, ostia, direction arrows

No manual input, no case-specific constants, CPU only, deterministic.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from detector import Config, Case, detect
from outputs import build_prediction, write_all
from trace import TraceConfig, trace_daughter


def process(image_path, mask_path, case_id, cfg=None, tc=None,
            min_origin_diameter_mm=None, verbose=True):
    """Detect, trace and package one case. Returns (pred, case, daughters, labels)."""
    cfg = cfg or Config()
    tc = tc or TraceConfig()
    gate = (cfg.min_origin_diameter_mm if min_origin_diameter_mm is None
            else float(min_origin_diameter_mm))

    t0 = time.time()
    case = Case(image_path, mask_path, cfg)
    t_load = time.time() - t0

    t0 = time.time()
    labels, cands, info = detect(case)
    t_detect = time.time() - t0

    t0 = time.time()
    daughters, rejected = [], []
    for c in cands:
        comp = labels == c["label"]
        tr = trace_daughter(case, comp, c["ostium_vox"], tc, thr_hu=case.lo)
        if tr is None:
            rejected.append((c["label"], "untraceable"))
            continue
        if tr["origin_diameter_mm"] < gate:
            rejected.append((c["label"],
                             f"origin {tr['origin_diameter_mm']:.2f} mm < {gate}"))
            continue
        tr.update(c)
        daughters.append(tr)
    t_trace = time.time() - t0

    info["rejected_by_origin_diameter"] = len(rejected)
    info["min_origin_diameter_mm"] = gate
    cfg_out = type("C", (), dict(vars(Config), min_origin_diameter_mm=gate))
    pred = build_prediction(case_id, case, daughters, info, cfg_out, tc)
    pred["method"]["runtime_s"] = {
        "load": round(t_load, 2), "detect": round(t_detect, 2),
        "trace": round(t_trace, 2),
        "total": round(t_load + t_detect + t_trace, 2)}

    if verbose:
        print(f"  lumen {info['lumen_mean_hu']:.0f} +/- {info['lumen_sd_hu']:.0f} HU"
              f"   band {info['band_hu'][0]:.0f}-{info['band_hu'][1]:.0f}"
              f"   bone > {info['bone_cut_hu']:.0f}"
              f"   crop {info['crop_shape']}")
        print(f"  {len(cands)} candidate(s) -> {len(daughters)} daughter(s)"
              f"   ({len(rejected)} rejected)"
              f"   {info['leaks_blocked']} leak(s) blocked"
              f"   {t_load + t_detect + t_trace:.1f}s")
        for r, why in rejected:
            print(f"      rejected component {r}: {why}")
    return pred, case, daughters, labels


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-id", default=None,
                    help="defaults to the image filename stem")
    ap.add_argument("--json-only", action="store_true",
                    help="write only --output, no label volume or view")
    ap.add_argument("--extras-dir", default=None,
                    help="where the extra files go "
                         "(default: <output stem>_files beside --output)")
    ap.add_argument("--min-origin-diameter-mm", type=float, default=None,
                    help="challenge eligibility gate; default 2.0. Pass 0 to "
                         "disable and measure its cost.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    for p in (args.image, args.aorta_mask):
        if not os.path.isfile(p):
            raise SystemExit(f"No such file: {p}")

    stem = os.path.basename(args.image)
    for ext in (".nii.gz", ".nii", ".mha", ".mhd"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    case_id = args.case_id or stem
    extras = args.extras_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)),
        os.path.splitext(os.path.basename(args.output))[0] + "_files")

    if not args.quiet:
        print(f"case {case_id}")
    pred, case, daughters, labels = process(
        args.image, args.aorta_mask, case_id,
        min_origin_diameter_mm=args.min_origin_diameter_mm,
        verbose=not args.quiet)

    written = write_all(args.output, extras, case_id, case, daughters, labels,
                        pred, json_only=args.json_only)
    if not args.quiet:
        for w in written:
            print(f"  wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())