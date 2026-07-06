#!/usr/bin/env python3
"""Pool per-scene results.json and report pooled selective-gate metrics."""
from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ricloc.config import FROZEN
from ricloc import evaluation as ev
from ricloc.gate import calibrate_sigjoint, decide_sigjoint_selective_policy


def _load(files: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        p = Path(fp)
        if not p.exists():
            continue
        scene = p.parent.name
        for r in json.loads(p.read_text()):
            r["_scene"] = scene
            rows.append(r)
    return rows


def _relabel(rows: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    tm, rd = FROZEN.success_threshold_for(dataset)
    thr = {"profile": f"{dataset}_strict", "source": "frozen_config", "trans_m": tm, "rot_deg": rd}
    for r in rows:
        if r.get("translation_error") is not None and r.get("rotation_error_deg") is not None:
            ev.add_success_label(r, thr)
    return thr


def _sigma_joint(r: Dict[str, Any], calib: Dict[str, Any]):
    mc = r.get("mapfree_cov") or {}
    d = decide_sigjoint_selective_policy(
        mc.get("sigma_new"), mc.get("sigma_disp_cov"),
        median_cons=calib.get("median_cons", 0.0), median_disp=calib.get("median_disp", 0.0),
        tau_accept=calib.get("tau_accept", float("inf")))
    return d.get("sigma_joint"), bool(d.get("accepted", False))


def _report(tag: str, rows: List[Dict[str, Any]], thr: Dict[str, Any], target_cov: float) -> None:
    labeled = [r for r in rows if r.get("success_label") is not None]
    if not labeled:
        print(f"[{tag}] no GT-labeled queries — accuracy/AUROC unavailable"); return
    succ = np.asarray([bool(r["success_label"]) for r in labeled], dtype=bool)
    sj = np.asarray([float(r.get("_sigma_joint")) if r.get("_sigma_joint") is not None else np.inf
                     for r in labeled], dtype=np.float64)
    has_sj = np.isfinite(sj)
    te = np.asarray([float(r["translation_error"]) for r in labeled], dtype=np.float64)
    re = np.asarray([float(r["rotation_error_deg"]) for r in labeled], dtype=np.float64)
    n = len(labeled)
    n_all = len(rows)                       # denom for full-set recall (incl. non-localized queries)
    n_succ = int(np.sum(succ))
    cat_t, cat_r = ev.catastrophic_threshold_for(thr.get("profile"))
    print(f"\n[{tag}] n_all={n_all} n_labeled={n}")
    print(f"        full-set strict_recall@{thr['trans_m']}m/{thr['rot_deg']}deg = {n_succ / max(n_all, 1):.4f}"
          f"  (conditional over localized = {np.mean(succ):.4f})")
    print(f"        median trans/rot = {np.median(te):.4f} m / {np.median(re):.4f} deg")
    for tmb, rdb in [(0.25, 2.0), (0.5, 5.0), (1.0, 10.0)]:
        print(f"        full-set recall@{tmb}m/{rdb}deg = {np.sum((te <= tmb) & (re <= rdb)) / max(n_all, 1):.4f}")
    print(f"        catastrophic_rate (>{cat_t}m or >{cat_r}deg, full-set) = "
          f"{np.sum((te > cat_t) | (re > cat_r)) / max(n_all, 1):.4f}")
    # sigma_joint failure-detection AUROC over the covariance-eligible set: reliability = -sigma_joint
    if has_sj.sum() >= 2 and len(set(succ[has_sj].tolist())) == 2:
        auroc = ev._rank_auc(succ[has_sj], -sj[has_sj])
        print(f"        sigma_joint failure-AUROC (held-out, eligible) = {auroc:.4f}  "
              f"(eligible coverage = {has_sj.mean():.3f})")
    else:
        print("        sigma_joint failure-AUROC: unavailable (need both classes + >=2 valid sigma)")
    # risk@target coverage WITHIN the eligible set: accept lowest-sigma_joint first
    elig_idx = np.where(has_sj)[0]
    order_e = elig_idx[np.argsort(sj[elig_idx])]
    k = max(1, int(np.ceil(target_cov * len(elig_idx))))
    chosen = order_e[:k]
    acc_succ = succ[chosen]
    acc_cat = (te[chosen] > cat_t) | (re[chosen] > cat_r)
    print(f"        @eligible-coverage~{target_cov:.2f} (k={k}/{len(elig_idx)}): "
          f"accepted_recall={np.mean(acc_succ):.4f} strict_risk={1.0 - np.mean(acc_succ):.4f} "
          f"catastrophic={np.mean(acc_cat):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="pool per-scene RIC-Loc results + held-out selective metrics")
    ap.add_argument("--glob", required=True, help="glob of results.json files to evaluate")
    ap.add_argument("--dataset", default="", help="7scenes|cambridge|naver|aachen")
    ap.add_argument("--loso_key", default=None, choices=[None, "scene"],
                    help="leave-one-scene-out gate calibration (public benchmarks)")
    ap.add_argument("--calib_glob", default=None, help="results.json glob to calibrate the gate (cross-fold)")
    ap.add_argument("--target_coverage", type=float, default=FROZEN.gate_target_coverage)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    rows = _load(sorted(globlib.glob(args.glob)))
    if not rows:
        raise SystemExit(f"no results matched: {args.glob}")
    thr = _relabel(rows, args.dataset)
    tag = args.tag or f"{args.dataset}|{args.loso_key or ('cross-fold' if args.calib_glob else 'self')}"

    if args.calib_glob:                                   # cross-fold (e.g. NAVER cross-building)
        calib_rows = _load(sorted(globlib.glob(args.calib_glob)))
        calib = calibrate_sigjoint(calib_rows, target_coverage=args.target_coverage, source=args.calib_glob)
        for r in rows:
            r["_sigma_joint"], _ = _sigma_joint(r, calib)
    elif args.loso_key == "scene":                        # leave-one-scene-out
        scenes = sorted({r["_scene"] for r in rows})
        for held in scenes:
            others = [r for r in rows if r["_scene"] != held]
            try:
                calib = calibrate_sigjoint(others, target_coverage=args.target_coverage, source=f"loso!={held}")
            except ValueError:
                continue
            for r in (r for r in rows if r["_scene"] == held):
                r["_sigma_joint"], _ = _sigma_joint(r, calib)
    else:                                                 # self-calibration (single fold)
        calib = calibrate_sigjoint(rows, target_coverage=args.target_coverage, source="self")
        for r in rows:
            r["_sigma_joint"], _ = _sigma_joint(r, calib)

    _report(tag, rows, thr, args.target_coverage)


if __name__ == "__main__":
    main()
