#!/usr/bin/env python3
"""
run.py -- daughter-artery detection, one case at a time.

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

Writes prediction.json at --output, and alongside it in the same directory:
daughters.nii.gz, snap_labels.txt, ostia.csv, check.html.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import io_utils
import outputs
from detector import detect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--aorta-mask", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    outdir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(outdir, exist_ok=True)
    case_id = os.path.splitext(os.path.splitext(
        os.path.basename(args.image))[0])[0]

    img, ct, mk = io_utils.load_case(args.image, args.aorta_mask)
    if not mk.any():
        print("WARNING: aorta mask is empty -- no daughters possible", file=sys.stderr)

    candidates, ctx, grown, n_leak = detect(img, ct, mk, img.GetSpacing())

    outputs.write_prediction_json(args.output, case_id, candidates)
    io_utils.write_daughter_labels(
        os.path.join(outdir, "daughters.nii.gz"), img, mk.shape, candidates)
    io_utils.write_snap_labels(os.path.join(outdir, "snap_labels.txt"), candidates)
    outputs.write_ostia_csv(os.path.join(outdir, "ostia.csv"), candidates)
    outputs.write_check_html(os.path.join(outdir, "check.html"), case_id, ctx,
                             mk.shape, candidates, grown)

    dt = time.time() - t0
    print(f"{case_id}: {len(candidates)} daughter(s), {n_leak} leak(s) blocked, "
          f"{dt:.2f}s -> {args.output}")


if __name__ == "__main__":
    main()
