#!/usr/bin/env python3
"""
score.py -- score predictions against reference annotations.

    python score.py --pred predictions/subject020.json \
        --ref data/subject020/annotation.json

    python score.py --manifest cases.csv --pred-dir predictions --out score

Matching is optimal one-to-one (Hungarian on the ostium distance matrix), not
greedy nearest-neighbour: greedy lets one lucky prediction claim a reference
point a closer prediction should have had, turning one hit into two mistakes.

TWO REFERENCE MODES, and they disagree:

  --ref x.json   uses the annotation's declared `ostium_xyz_mm`. This is what
                 the organisers score against.
  --ref x.nii.gz uses the labelled voxel nearest the aortic wall, DERIVED from
                 the label volume. On subject020 that proxy sits 1.3-7.2 mm from
                 the declared ostium, because for a daughter running alongside
                 the aorta the nearest-to-wall voxel can be anywhere along it.

Always prefer the JSON. The volume mode exists only to quantify how wrong the
proxy was.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

__all__ = ["load_reference", "load_prediction", "score_case"]


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------

def load_prediction(path):
    with open(path) as fh:
        p = json.load(fh)
    out = []
    for d in p.get("daughters", []):
        out.append({
            "id": d.get("instance_id", f"branch_{len(out)+1:03d}"),
            "ostium": np.array(d["ostium_xyz_mm"], float),
            "seed": np.array(d.get("seed_xyz_mm", d["ostium_xyz_mm"]), float),
            "direction": np.array(d.get("direction_xyz", [0, 0, 1]), float),
            "radius": d.get("radius_mm"),
            "centerline": np.array(d.get("centerline_xyz_mm", []), float),
        })
    return out, p.get("case_id", os.path.basename(path))


def _ref_from_json(path):
    with open(path) as fh:
        a = json.load(fh)
    out = []
    for d in a.get("daughters", []):
        out.append({
            "id": d.get("instance_id", f"ref_{len(out)+1:03d}"),
            "ostium": np.array(d["ostium_xyz_mm"], float),
            "seed": (np.array(d["seed_xyz_mm"], float)
                     if d.get("seed_xyz_mm") else None),
            "direction": (np.array(d["direction_xyz"], float)
                          if d.get("direction_xyz") else None),
            "radius": d.get("radius_mm"),
            "centerline": np.array(d.get("centerline_xyz_mm", []), float),
        })
    return out, a.get("case_id", os.path.basename(path))


def _ref_from_volume(gt_path, image, mask, roi_mm=30.0):
    """Fallback: ostium = labelled voxel nearest the wall. A proxy, not truth."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from detector import Case, Config

    case = Case(image, mask, Config())
    g = sitk.ReadImage(str(gt_path))
    if g.GetSize() != case.img.GetSize():
        raise SystemExit(f"GT grid {g.GetSize()} != image {case.img.GetSize()}")
    gt = sitk.GetArrayFromImage(g).astype(np.int32)[case.region]

    ids = [int(v) for v in np.unique(gt) if v > 0]
    ids = [L for L in ids
           if float(((gt == L) & case.mk).sum()) / max((gt == L).sum(), 1) < 0.5]
    out = []
    for L in ids:
        comp = gt == L
        near = np.argwhere(comp)[int(np.argmin(case.dist_out[comp]))]
        out.append({"id": f"label_{L}",
                    "ostium": np.asarray(case.to_mm(near.astype(float)), float),
                    "seed": None, "direction": None, "radius": None,
                    "centerline": np.zeros((0, 3))})
    return out, "volume"


def load_reference(path, image=None, mask=None):
    if str(path).endswith(".json"):
        return _ref_from_json(path)
    if not (image and mask):
        raise SystemExit("A label-volume reference needs --image and --aorta-mask.")
    return _ref_from_volume(path, image, mask)


# -----------------------------------------------------------------------------
# matching and metrics
# -----------------------------------------------------------------------------

def match(ref, pred, tol_mm):
    if not ref or not pred:
        return [], list(range(len(ref))), list(range(len(pred)))
    R = np.array([r["ostium"] for r in ref], float)
    P = np.array([p["ostium"] for p in pred], float)
    D = np.linalg.norm(R[:, None, :] - P[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        big = D.max() * 10 + 1e6
        ri, ci = linear_sum_assignment(np.where(D <= tol_mm, D, big))
        cand = [(int(i), int(j), float(D[i, j])) for i, j in zip(ri, ci)]
    except ImportError:
        cand, ug, up = [], set(), set()
        for i, j in sorted(((i, j) for i in range(len(ref))
                            for j in range(len(pred))),
                           key=lambda t: D[t[0], t[1]]):
            if i in ug or j in up:
                continue
            ug.add(i)
            up.add(j)
            cand.append((i, j, float(D[i, j])))
    pairs = [(i, j, d) for i, j, d in cand if d <= tol_mm]
    mg = {i for i, _, _ in pairs}
    mp = {j for _, j, _ in pairs}
    return (pairs, [i for i in range(len(ref)) if i not in mg],
            [j for j in range(len(pred)) if j not in mp])


def _seed_to_path(seed, ref):
    """Distance from a predicted seed to the reference proximal centreline."""
    C = ref.get("centerline")
    if C is None or len(C) == 0:
        return None
    return float(np.min(np.linalg.norm(C - seed[None, :], axis=1)))


def score_case(ref, pred, tol_mm):
    pairs, ug, up = match(ref, pred, tol_mm)
    tp, fp, fn = len(pairs), len(pred) - len(pairs), len(ref) - len(pairs)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    err = [d for _, _, d in pairs]
    ang, seedd, rerr = [], [], []
    for i, j, _d in pairs:
        r, p = ref[i], pred[j]
        if r.get("direction") is not None:
            a = float(np.clip(np.dot(r["direction"] / np.linalg.norm(r["direction"]),
                                     p["direction"] / max(np.linalg.norm(p["direction"]), 1e-9)),
                              -1, 1))
            ang.append(float(np.degrees(np.arccos(a))))
        s = _seed_to_path(p["seed"], r)
        if s is not None:
            seedd.append(s)
        if r.get("radius") is not None and p.get("radius") is not None:
            rerr.append(abs(float(r["radius"]) - float(p["radius"])))

    def m(v):
        return round(float(np.mean(v)), 3) if v else ""

    return dict(
        n_ref=len(ref), n_pred=len(pred), tol_mm=tol_mm, tp=tp, fp=fp, fn=fn,
        precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
        mean_ostium_err_mm=m(err), median_ostium_err_mm=(
            round(float(np.median(err)), 3) if err else ""),
        mean_direction_deg=m(ang), mean_seed_to_path_mm=m(seedd),
        mean_radius_err_mm=m(rerr)), (pairs, ug, up)


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None, help="one prediction.json")
    ap.add_argument("--ref", default=None, help="one annotation .json or label volume")
    ap.add_argument("--image", default=None)
    ap.add_argument("--aorta-mask", default=None)
    ap.add_argument("--manifest", default=None,
                    help="csv: case_id,image,aorta_mask,ref")
    ap.add_argument("--pred-dir", default=None,
                    help="with --manifest: folder of <case_id>.json predictions")
    ap.add_argument("--out", default=None, help="folder for scores.csv/matches.csv")
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--tolerances", default="3,5,6,8,10")
    args = ap.parse_args()

    tols = sorted({float(s) for s in args.tolerances.split(",") if s.strip()}
                  | {args.tol_mm})

    cases = []
    if args.manifest:
        with open(args.manifest, newline="") as fh:
            for r in csv.DictReader(fh):
                pj = os.path.join(args.pred_dir or ".", f"{r['case_id']}.json")
                cases.append((r["case_id"], pj, r["ref"],
                              r.get("image"), r.get("aorta_mask")))
    elif args.pred and args.ref:
        cases.append((None, args.pred, args.ref, args.image, args.aorta_mask))
    else:
        raise SystemExit("Pass --pred and --ref, or --manifest and --pred-dir.")

    rows, detail = [], []
    print(f"{'case':<14}{'ref':>4}{'pred':>5}{'TP':>4}{'FP':>4}{'FN':>4}"
          f"{'P':>7}{'R':>7}{'F1':>7}{'ost mm':>9}{'dir deg':>9}")
    for cid, pj, rp, im, mk in cases:
        if not os.path.isfile(pj):
            print(f"{str(cid):<14}  no prediction at {pj}")
            continue
        pred, pid = load_prediction(pj)
        ref, _rid = load_reference(rp, im, mk)
        cid = cid or pid
        for t in tols:
            s, (pairs, ug, up) = score_case(ref, pred, t)
            s["case_id"] = cid
            rows.append(s)
            if t == args.tol_mm:
                for i, j, d in pairs:
                    detail.append(dict(case_id=cid, tol_mm=t, status="TP",
                                       ref_id=ref[i]["id"], pred_id=pred[j]["id"],
                                       error_mm=round(d, 3)))
                for i in ug:
                    detail.append(dict(case_id=cid, tol_mm=t, status="FN",
                                       ref_id=ref[i]["id"], pred_id="",
                                       error_mm=""))
                for j in up:
                    detail.append(dict(case_id=cid, tol_mm=t, status="FP",
                                       ref_id="", pred_id=pred[j]["id"],
                                       error_mm=""))
                print(f"{cid:<14}{s['n_ref']:>4}{s['n_pred']:>5}{s['tp']:>4}"
                      f"{s['fp']:>4}{s['fn']:>4}{s['precision']:>7.2f}"
                      f"{s['recall']:>7.2f}{s['f1']:>7.2f}"
                      f"{str(s['mean_ostium_err_mm'] or '-'):>9}"
                      f"{str(s['mean_direction_deg'] or '-'):>9}")

    at = [r for r in rows if r["tol_mm"] == args.tol_mm]
    if at:
        print(f"\n  MEAN over {len(at)} case(s) at {args.tol_mm:g} mm: "
              f"P {np.mean([r['precision'] for r in at]):.3f}  "
              f"R {np.mean([r['recall'] for r in at]):.3f}  "
              f"F1 {np.mean([r['f1'] for r in at]):.3f}")
        tp = sum(r["tp"] for r in at)
        np_, nr = sum(r["n_pred"] for r in at), sum(r["n_ref"] for r in at)
        print(f"  POOLED over {nr} reference daughters: "
              f"P {tp / max(np_, 1):.3f}  R {tp / max(nr, 1):.3f}  "
              f"F1 {2 * tp / max(np_ + nr, 1):.3f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        cols = ["case_id", "tol_mm", "n_ref", "n_pred", "tp", "fp", "fn",
                "precision", "recall", "f1", "mean_ostium_err_mm",
                "median_ostium_err_mm", "mean_direction_deg",
                "mean_seed_to_path_mm", "mean_radius_err_mm"]
        with open(os.path.join(args.out, "scores.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out, "matches.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["case_id", "tol_mm", "status",
                                               "ref_id", "pred_id", "error_mm"])
            w.writeheader()
            w.writerows(detail)
        print(f"  wrote {args.out}/scores.csv and matches.csv")


if __name__ == "__main__":
    main()
