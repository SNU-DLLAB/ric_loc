#!/usr/bin/env python3
"""Evaluate RIC-Loc on a posed dataset and write metrics plus selective-gate calibration."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ricloc.config import FROZEN
from ricloc.localizer import FrozenLocalizer
from ricloc.helpers import _sorted_image_list_from_path
from ricloc.gate import apply_gate, calibrate_sigjoint
from ricloc import evaluation as ev


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (json.dumps(v, ensure_ascii=False, default=_json_default) if isinstance(v, (dict, list)) else v)
            for k, v in row.items()}


def _label_errors(results: List[Dict[str, Any]], gt_poses: Dict[str, Any], thr: Dict[str, Any]) -> bool:
    has_gt = bool(gt_poses)
    for r in results:
        r["translation_error"] = None
        r["rotation_error_deg"] = None
        if not has_gt or r.get("status") != "ok":
            continue
        gt = ev.match_gt_pose(gt_poses, r["query_name"]) or ev.match_gt_pose(gt_poses, r["query_path"])
        if gt is None:
            continue
        err = ev.compute_pose_errors(r["pose_world"], gt)
        if err:
            r["translation_error"] = err["translation_error"]
            r["rotation_error_deg"] = err["rotation_error_deg"]
            ev.add_success_label(r, thr)
    return has_gt


def main() -> None:
    ap = argparse.ArgumentParser(description="RIC-Loc frozen-method dataset evaluation")
    ap.add_argument("--reference_root", required=True)
    ap.add_argument("--vggt_ckpt", required=True)
    ap.add_argument("--query_path", required=True, help="query image folder")
    ap.add_argument("--gt", default=None, help="GT poses (text: name qw qx qy qz tx ty tz, or JSON/JSONL)")
    ap.add_argument("--dataset", default="", help="7scenes|cambridge|naver|aachen (selects strict threshold)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target_coverage", type=float, default=FROZEN.gate_target_coverage)
    ap.add_argument("--calib_results", default=None,
                    help="held-out fold results.json to calibrate the gate (default: self-calibrate)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of queries (debug)")
    ap.add_argument("--num_refs", type=int, default=None,
                    help="SENSITIVITY SWEEP ONLY (tab:supp_ksweep): override frozen K (active refs to VGGT). "
                         "Main experiments use the frozen K=8; see scripts/run_ksweep.sh.")
    ap.add_argument("--topk", type=int, default=None,
                    help="SENSITIVITY SWEEP ONLY: override frozen retrieval pool L.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    trans_m, rot_deg = FROZEN.success_threshold_for(args.dataset)
    thr = {"profile": f"{args.dataset or 'custom'}_strict", "source": "frozen_config",
           "trans_m": float(trans_m), "rot_deg": float(rot_deg)}

    queries = _sorted_image_list_from_path(Path(args.query_path).expanduser())
    if args.limit > 0:
        queries = queries[: args.limit]

    cfg = FROZEN
    if args.num_refs is not None or args.topk is not None:
        from dataclasses import replace
        cfg = replace(FROZEN, num_refs=(args.num_refs or FROZEN.num_refs), topk=(args.topk or FROZEN.topk))
        print(f"[ricloc] SENSITIVITY-SWEEP override: K={cfg.num_refs} L={cfg.topk} (NOT the frozen headline config)")
    loc = FrozenLocalizer(reference_root=args.reference_root, vggt_ckpt=args.vggt_ckpt,
                          device=args.device, config=cfg)
    print(f"[ricloc] map={loc.reference_dir} colmap={loc.colmap_dir} refs={len(loc.reference_images)} "
          f"device={loc.device} | {len(queries)} queries | strict={trans_m}m/{rot_deg}deg")

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(queries):
        r = loc.localize(q)
        results.append(r)
        if (i + 1) % 25 == 0 or i + 1 == len(queries):
            ok = sum(1 for x in results if x.get("status") == "ok")
            print(f"  [{i+1}/{len(queries)}] ok={ok}")

    gt_poses = ev.load_gt_poses(Path(args.gt)) if args.gt else {}
    has_gt = _label_errors(results, gt_poses, thr)

    # selective gate: calibrate then apply
    calib_src = json.loads(Path(args.calib_results).read_text()) if args.calib_results else results
    try:
        calib = calibrate_sigjoint(calib_src, target_coverage=args.target_coverage,
                                   source=(args.calib_results or "self"))
        apply_gate(results, calib)
    except ValueError as e:
        calib = {"error": str(e)}
        print(f"[ricloc] WARN gate calibration skipped: {e}")

    # summarize
    summary = ev.summarize_results(results, has_gt=has_gt, success_threshold=thr)
    summary["dataset"] = args.dataset
    summary["frozen_strict_trans_m"] = float(trans_m)
    summary["frozen_strict_rot_deg"] = float(rot_deg)
    if has_gt:
        labeled = [r for r in results if r.get("success_label") is not None]
        labels = np.asarray([bool(r["success_label"]) for r in labeled], dtype=bool)
        n_success_elig = sum(1 for r in labeled
                             if (r.get("mapfree_cov") or {}).get("sigma_new") is not None
                             and bool(r.get("success_label")))
        n_success = int(np.sum(labels)) if labels.size else 0
        summary["strict_recall"] = float(n_success_elig / max(len(results), 1))      # full-set (eligible-gated)
        summary["strict_recall_localized"] = float(n_success / max(len(results), 1))  # any-localized (diagnostic)
        summary["strict_recall_conditional"] = float(np.mean(labels)) if labels.size else None
        # sigma_joint failure-detection AUROC over the covariance-eligible labeled set
        elig = ev._eligible_rows(labeled)
        el_labels = np.asarray([bool(r["success_label"]) for r in elig], dtype=bool)
        el_scores = np.asarray([ev._score(r) for r in elig], dtype=np.float64)
        summary["sigjoint_success_auroc"] = (
            ev._rank_auc(el_labels, el_scores) if el_labels.size else None)
        summary["covariance_eligible_coverage"] = float(len(elig) / max(len(labeled), 1)) if labeled else None
        cov = sum(1 for r in results if r.get("accepted"))
        n_elig_all = sum(1 for r in results if np.isfinite(ev._score(r)))
        summary["selective_coverage"] = float(cov / max(len(results), 1))            # full-set acceptance
        summary["selective_coverage_eligible"] = float(cov / max(n_elig_all, 1)) if n_elig_all else None
        acc = [r for r in results if r.get("accepted") and r.get("success_label") is not None]
        summary["selective_accepted_recall"] = (
            float(np.mean([bool(r["success_label"]) for r in acc])) if acc else None)
    summary["sigjoint_calib"] = calib

    # write artifacts
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))
    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    rows = [_flatten_for_csv(r) for r in results]
    if rows:
        fields = sorted({k for r in rows for k in r})
        with (out_dir / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    if isinstance(calib, dict) and "error" not in calib:
        (out_dir / "sigjoint_calib.json").write_text(json.dumps(calib, indent=2))
    if has_gt:
        curve = ev.coverage_risk_curve(results, success_threshold=thr)
        if curve:
            with (out_dir / "coverage_risk.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
                w.writeheader(); w.writerows(curve)

    print(f"\n[ricloc] === {args.dataset or 'dataset'} summary ===")
    if has_gt:
        print(f"  strict recall@{trans_m}m/{rot_deg}deg : {summary.get('strict_recall')}")
        print(f"  median trans / rot          : {summary.get('median_translation_error')} m / "
              f"{summary.get('median_rotation_error')} deg")
        print(f"  recall@0.25/0.5/1.0         : {summary.get('recall@0.25m/2.0deg')} / "
              f"{summary.get('recall@0.5m/5.0deg')} / {summary.get('recall@1.0m/10.0deg')}")
        print(f"  sigma_joint success-AUROC   : {summary.get('sigjoint_success_auroc')}")
        print(f"  selective coverage / recall : {summary.get('selective_coverage')} / "
              f"{summary.get('selective_accepted_recall')}")
    print(f"  wrote -> {out_dir}")


if __name__ == "__main__":
    main()
