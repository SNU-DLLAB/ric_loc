# Pose accuracy and selective localization metrics.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SUCCESS_THRESHOLD_PROFILES: Dict[str, Tuple[float, float]] = {
    "7scenes_standard": (0.10, 10.0),
    "inloc_standard": (0.50, 10.0),
    "aachen_standard": (0.50, 5.0),
    "naver_standard": (0.50, 10.0),
    "indoor_standard": (0.50, 10.0),
}

DATASET_PROFILE_HINTS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("7scenes", "7-scenes", "seven_scenes", "seven-scenes"), "7scenes_standard"),
    (("inloc",), "inloc_standard"),
    (("aachen",), "aachen_standard"),
    (("naver", "gangnam", "hyundai"), "naver_standard"),
)


def _qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(qvec), dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q.tolist()
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _camera_center(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    return -(Rcw.T @ tcw.reshape(3))


def load_gt_poses(gt_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load GT poses from JSONL/JSON or whitespace text: name qw qx qy qz tx ty tz."""
    if gt_path is None:
        return {}
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"GT pose file not found: {path}")
    poses: Dict[str, Dict[str, Any]] = {}
    if path.suffix.lower() in {".json", ".jsonl"}:
        rows = []
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            blob = json.loads(path.read_text(encoding="utf-8"))
            rows = blob if isinstance(blob, list) else blob.get("poses", [])
        for row in rows:
            name = str(row.get("query_name", row.get("name", row.get("image_name", ""))))
            if name:
                poses[name] = row
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < 8:
            continue
        name = toks[0]
        vals = [float(x) for x in toks[1:8]]
        poses[name] = {"qvec_wxyz_world_to_cam": vals[:4], "t_world_to_cam": vals[4:7]}
    return poses


def match_gt_pose(gt_poses: Dict[str, Dict[str, Any]], query_name: str) -> Optional[Dict[str, Any]]:
    if query_name in gt_poses:
        return gt_poses[query_name]
    base = Path(query_name).name
    if base in gt_poses:
        return gt_poses[base]
    for k, v in gt_poses.items():
        if Path(k).name == base or str(query_name).endswith(str(k)):
            return v
    return None


def compute_pose_errors(pred_pose: Dict[str, Any], gt_pose: Dict[str, Any]) -> Dict[str, float]:
    # translation error: distance between predicted and GT camera centers (m)
    pred_q = pred_pose.get("qvec_wxyz_world_to_cam", pred_pose.get("qwqxqyqz"))
    pred_t = pred_pose.get("t_world_to_cam", pred_pose.get("t_xyz"))
    gt_q = gt_pose.get("qvec_wxyz_world_to_cam", gt_pose.get("qwqxqyqz"))
    gt_t = gt_pose.get("t_world_to_cam", gt_pose.get("t_xyz"))
    if pred_q is None or pred_t is None or gt_q is None or gt_t is None:
        return {}
    R_pred = _qvec_to_rotmat(pred_q)
    R_gt = _qvec_to_rotmat(gt_q)
    C_pred = np.asarray(pred_pose.get("camera_center_world", _camera_center(R_pred, np.asarray(pred_t))), dtype=np.float64)
    C_gt = np.asarray(gt_pose.get("camera_center_world", _camera_center(R_gt, np.asarray(gt_t))), dtype=np.float64)
    trans = float(np.linalg.norm(C_pred.reshape(3) - C_gt.reshape(3)))
    # rotation error: geodesic angle between predicted and GT rotations (deg)
    dR = R_pred @ R_gt.T
    cos = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    rot = float(np.degrees(np.arccos(cos)))
    return {"translation_error": trans, "rotation_error_deg": rot}


def infer_success_threshold_profile(paths: Sequence[Any]) -> Optional[str]:
    text = " ".join(str(p).lower() for p in paths if p is not None)
    for tokens, profile in DATASET_PROFILE_HINTS:
        if any(tok in text for tok in tokens):
            return profile
    return None


def resolve_success_threshold(
    profile: Optional[str] = "auto",
    reference_root: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    trans_override: Optional[float] = None,
    rot_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve evaluation success thresholds."""
    requested = str(profile or "auto").strip().lower()
    if requested and requested != "auto":
        if requested not in SUCCESS_THRESHOLD_PROFILES:
            known = ", ".join(["auto"] + sorted(SUCCESS_THRESHOLD_PROFILES))
            raise ValueError(f"unknown success threshold profile '{profile}'. Known profiles: {known}")
        resolved_profile = requested
        source = "explicit_profile"
    else:
        inferred = infer_success_threshold_profile([reference_root, output_dir])
        if inferred is not None:
            resolved_profile = inferred
            source = "auto_dataset_inference"
        else:
            resolved_profile = "indoor_standard"
            source = "indoor_standard_fallback"

    trans_m, rot_deg = SUCCESS_THRESHOLD_PROFILES[resolved_profile]
    if trans_override is not None or rot_override is not None:
        if trans_override is not None:
            trans_m = float(trans_override)
        if rot_override is not None:
            rot_deg = float(rot_override)
        source = "explicit_override"

    return {
        "profile": resolved_profile,
        "source": source,
        "trans_m": float(trans_m),
        "rot_deg": float(rot_deg),
    }


def add_success_label(row: Dict[str, Any], success_threshold: Dict[str, Any]) -> None:
    row["success_threshold_profile"] = success_threshold.get("profile")
    row["success_threshold_source"] = success_threshold.get("source")
    row["success_threshold_trans_m"] = float(success_threshold.get("trans_m", 0.5))
    row["success_threshold_rot_deg"] = float(success_threshold.get("rot_deg", 5.0))
    if row.get("translation_error") is None or row.get("rotation_error_deg") is None:
        row["success_label"] = None
        return
    row["success_label"] = bool(
        float(row["translation_error"]) <= row["success_threshold_trans_m"]
        and float(row["rotation_error_deg"]) <= row["success_threshold_rot_deg"]
    )


def catastrophic_threshold_for(profile_or_dataset: Optional[str]) -> Tuple[float, float]:
    """Catastrophic-failure thresholds: 0.5 m / 10 deg on 7-Scenes, 5 m / 20 deg elsewhere."""
    s = str(profile_or_dataset or "").lower()
    if "7scenes" in s or "7-scenes" in s or "seven" in s:
        return (0.5, 10.0)
    return (5.0, 20.0)


def _summarize_scalar_field(summary: Dict[str, Any], results: List[Dict[str, Any]], key: str) -> None:
    vals = []
    for r in results:
        if r.get(key) is None:
            continue
        try:
            v = float(r[key])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=np.float64)
    summary[f"mean_{key}"] = float(np.mean(arr)) if arr.size else None
    summary[f"median_{key}"] = float(np.median(arr)) if arr.size else None
    summary[f"p90_{key}"] = float(np.percentile(arr, 90.0)) if arr.size else None


def _score(row: Dict[str, Any], score_key: str = "reliability_score") -> float:
    """Reliability score for selective ranking (higher = more reliable)."""
    try:
        v = float(row.get(score_key, None))
    except (TypeError, ValueError):
        return float("-inf")
    return float(v) if np.isfinite(v) else float("-inf")


def _eligible_rows(rows: List[Dict[str, Any]], score_key: str = "reliability_score") -> List[Dict[str, Any]]:
    """Covariance-eligible queries: those with a finite reliability score."""
    return [r for r in rows if np.isfinite(_score(r, score_key=score_key))]


def _probability_from_score(score: float, eps: float = 1e-6) -> float:
    return float(np.clip(float(score), eps, 1.0 - eps))


def _valid_labeled_rows(
    results: List[Dict[str, Any]],
    success_threshold: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        if r.get("translation_error") is None or r.get("rotation_error_deg") is None:
            continue
        row = dict(r)
        if success_threshold is not None:
            add_success_label(row, success_threshold)
        elif row.get("success_label") is None:
            threshold = success_threshold or {
                "profile": "indoor_standard",
                "source": "legacy_default",
                "trans_m": 0.5,
                "rot_deg": 5.0,
            }
            add_success_label(row, threshold)
        if row.get("success_label") is not None:
            rows.append(row)
    return rows


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels_bool = labels.astype(bool)
    n_pos = int(np.sum(labels_bool))
    n_neg = int(labels_bool.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * float(start + 1 + end)
        start = end
    ranks = np.empty_like(ranks_sorted, dtype=np.float64)
    ranks[order] = ranks_sorted
    pos_rank_sum = float(np.sum(ranks[labels_bool]))
    auc = (pos_rank_sum - float(n_pos * (n_pos + 1)) * 0.5) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    labels_bool = labels.astype(bool)
    positives = int(np.sum(labels_bool))
    if positives == 0:
        return None
    if scores.size and float(np.max(scores)) == float(np.min(scores)):
        return float(positives / max(labels_bool.size, 1))
    order = np.argsort(-scores, kind="mergesort")
    y = labels_bool[order]
    cum_pos = np.cumsum(y.astype(np.float64))
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    precision_at_pos = cum_pos[y] / ranks[y]
    return float(np.mean(precision_at_pos)) if precision_at_pos.size else None


def _probability_metrics(labels: np.ndarray, scores: np.ndarray, num_bins: int = 10) -> Dict[str, Optional[float]]:
    if labels.size == 0:
        return {"ece": None, "brier": None, "nll": None}
    probs = np.asarray([_probability_from_score(s) for s in scores], dtype=np.float64)
    y = labels.astype(np.float64)
    brier = float(np.mean((probs - y) ** 2))
    nll = float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    ece = 0.0
    bins = np.minimum((probs * float(num_bins)).astype(np.int64), int(num_bins) - 1)
    for b in range(int(num_bins)):
        mask = bins == b
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return {"ece": float(ece), "brier": brier, "nll": nll}


def _risk_at_coverage(
    rows: List[Dict[str, Any]],
    coverage: float,
    score_key: str = "reliability_score",
) -> Dict[str, Optional[float]]:
    rows = _eligible_rows(rows, score_key=score_key)   # selective coverage within the eligible set
    if not rows:
        return {
            "risk_translation_mean": None,
            "risk_rotation_mean": None,
            "success_rate": None,
            "failure_rate": None,
            "threshold": None,
        }
    sorted_rows = sorted(rows, key=lambda r: _score(r, score_key=score_key), reverse=True)
    k = int(np.ceil(float(coverage) * len(sorted_rows)))
    k = max(1, min(k, len(sorted_rows)))
    chosen = sorted_rows[:k]
    labels = [r.get("success_label") for r in chosen if r.get("success_label") is not None]
    success_rate = float(np.mean(labels)) if labels else None
    return {
        "risk_translation_mean": float(np.mean([float(r["translation_error"]) for r in chosen])),
        "risk_rotation_mean": float(np.mean([float(r.get("rotation_error_deg", 0.0)) for r in chosen])),
        "success_rate": success_rate,
        "failure_rate": float(1.0 - success_rate) if success_rate is not None else None,
        "threshold": _score(chosen[-1], score_key=score_key),
    }


def _aurc(rows: List[Dict[str, Any]], field: str, score_key: str = "reliability_score") -> Optional[float]:
    if not rows:
        return None
    curve = coverage_risk_curve(rows, score_key=score_key)
    if not curve:
        return None
    coverage = np.asarray([r["coverage"] for r in curve], dtype=np.float64)
    risk = np.asarray([r[field] for r in curve], dtype=np.float64)
    if coverage.size == 1:
        return float(risk[0])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(risk, coverage))


def compute_selective_reliability_metrics(
    results: List[Dict[str, Any]],
    success_threshold: Optional[Dict[str, Any]] = None,
    score_key: str = "reliability_score",
    probability_source: str = "untrained_heuristic_probability",
) -> Dict[str, Any]:
    rows = _eligible_rows(_valid_labeled_rows(results, success_threshold=success_threshold),
                          score_key=score_key)   # AUROC over covariance-eligible queries only
    labels = np.asarray([bool(r.get("success_label")) for r in rows], dtype=bool)
    scores = np.asarray([_score(r, score_key=score_key) for r in rows], dtype=np.float64)
    success_count = int(np.sum(labels)) if labels.size else 0
    failure_count = int(labels.size - success_count)
    out: Dict[str, Any] = {
        "reliability_probability_source": str(probability_source),
        "reliability_score_key": str(score_key),
        "reliability_label_count": int(labels.size),
        "reliability_success_count": success_count,
        "reliability_failure_count": failure_count,
        "reliability_metrics_available": bool(labels.size > 0 and len(set(labels.tolist())) == 2),
    }
    metric_keys = {
        "auroc_success_failure": None,
        "auprc_success_failure": None,
        "ece_success_failure": None,
        "brier_score_success_failure": None,
        "nll_success_failure": None,
        "failure_rate_at_80pct_coverage": None,
        "failure_rate_at_90pct_coverage": None,
    }
    if not out["reliability_metrics_available"]:
        out.update(metric_keys)
        return out

    prob = _probability_metrics(labels, scores)
    cov80 = _risk_at_coverage(rows, 0.80, score_key=score_key)
    cov90 = _risk_at_coverage(rows, 0.90, score_key=score_key)
    out.update(
        {
            "auroc_success_failure": _rank_auc(labels, scores),
            "auprc_success_failure": _average_precision(labels, scores),
            "ece_success_failure": prob["ece"],
            "brier_score_success_failure": prob["brier"],
            "nll_success_failure": prob["nll"],
            "failure_rate_at_80pct_coverage": cov80["failure_rate"],
            "failure_rate_at_90pct_coverage": cov90["failure_rate"],
        }
    )
    return out


def summarize_results(
    results: List[Dict[str, Any]],
    has_gt: bool,
    success_threshold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_queries": int(len(results)), "has_gt": bool(has_gt)}
    threshold = success_threshold or {
        "profile": "indoor_standard",
        "source": "legacy_default",
        "trans_m": 0.5,
        "rot_deg": 5.0,
    }
    summary["success_threshold_profile"] = threshold.get("profile")
    summary["success_threshold_source"] = threshold.get("source")
    summary["success_threshold_trans_m"] = float(threshold.get("trans_m", 0.5))
    summary["success_threshold_rot_deg"] = float(threshold.get("rot_deg", 5.0))
    for key in [
        "aurc_translation",
        "aurc_rotation",
        "risk_translation_at_80pct_coverage",
        "risk_rotation_at_80pct_coverage",
        "reliability_threshold_at_80pct_coverage",
        "risk_translation_at_90pct_coverage",
        "risk_rotation_at_90pct_coverage",
        "reliability_threshold_at_90pct_coverage",
    ]:
        summary[key] = None
    summary["coverage_risk_points"] = 0
    accepted = [r for r in results if bool(r.get("accepted", False))]
    valid_outputs = [r for r in results if bool(r.get("pose_output_valid", bool(r.get("accepted", False))))]
    accept_current = [r for r in valid_outputs if str(r.get("selective_decision", "")) == "accept_current"]
    # coverage: fraction of queries that produced a pose (not rejected)
    summary["accept_reject_coverage"] = float(len(accepted) / max(len(results), 1))
    summary["accepted_count"] = int(len(accepted))
    summary["rejected_count"] = int(len(results) - len(accepted))
    summary["pose_output_valid_coverage"] = float(len(valid_outputs) / max(len(results), 1))
    summary["pose_output_valid_count"] = int(len(valid_outputs))
    summary["pose_output_invalid_count"] = int(len(results) - len(valid_outputs))
    summary["selective_accept_current_count"] = int(len(accept_current))
    for key in [
        "reliability_score",
        "align_condition_number",
    ]:
        _summarize_scalar_field(summary, results, key)
    if not has_gt:
        summary.update(compute_selective_reliability_metrics(results, success_threshold=threshold))
        return summary
    trans = [float(r["translation_error"]) for r in results if r.get("translation_error") is not None]
    rot = [float(r["rotation_error_deg"]) for r in results if r.get("rotation_error_deg") is not None]
    if not trans or not rot:
        summary.update(compute_selective_reliability_metrics(results, success_threshold=threshold))
        return summary
    arr_t = np.asarray(trans, dtype=np.float64)
    arr_r = np.asarray(rot, dtype=np.float64)
    # recall: fraction of queries within (translation_m, rotation_deg) thresholds
    for tm, rd in [(0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 5.0), (1.0, 10.0)]:
        ok = (arr_t <= tm) & (arr_r <= rd)
        summary[f"recall@{tm}m/{rd}deg"] = float(np.mean(ok))
    summary["median_translation_error"] = float(np.median(arr_t))
    summary["median_rotation_error"] = float(np.median(arr_r))
    # catastrophic failure rate: fraction with very large translation/rotation error
    cat_t, cat_r = catastrophic_threshold_for(threshold.get("profile"))
    summary["catastrophic_threshold_trans_m"] = float(cat_t)
    summary["catastrophic_threshold_rot_deg"] = float(cat_r)
    summary["catastrophic_failure_rate"] = float(np.mean((arr_t > cat_t) | (arr_r > cat_r)))
    # full-set strict recall: denominator is all labeled queries; missing pose/error counts as failure
    n_all = int(len(results))
    n_success = int(np.sum((arr_t <= float(threshold.get("trans_m", 0.5)))
                           & (arr_r <= float(threshold.get("rot_deg", 5.0)))))
    summary["full_set_strict_recall"] = float(n_success / max(n_all, 1))
    summary["conditional_strict_recall"] = float(n_success / max(len(arr_t), 1))
    summary["num_labeled_with_error"] = int(len(arr_t))
    # risk: mean pose error among accepted queries
    acc_err = [float(r["translation_error"]) for r in accepted if r.get("translation_error") is not None]
    summary["risk_among_accepted_poses"] = float(np.mean(acc_err)) if acc_err else None
    labeled_rows = _valid_labeled_rows(results, success_threshold=threshold)
    curve = coverage_risk_curve(labeled_rows)
    cov80 = _risk_at_coverage(labeled_rows, 0.80)
    cov90 = _risk_at_coverage(labeled_rows, 0.90)
    summary["aurc_translation"] = _aurc(labeled_rows, "risk_translation_mean")
    summary["aurc_rotation"] = _aurc(labeled_rows, "risk_rotation_mean")
    summary["risk_translation_at_80pct_coverage"] = cov80["risk_translation_mean"]
    summary["risk_rotation_at_80pct_coverage"] = cov80["risk_rotation_mean"]
    summary["reliability_threshold_at_80pct_coverage"] = cov80["threshold"]
    summary["risk_translation_at_90pct_coverage"] = cov90["risk_translation_mean"]
    summary["risk_rotation_at_90pct_coverage"] = cov90["risk_rotation_mean"]
    summary["reliability_threshold_at_90pct_coverage"] = cov90["threshold"]
    summary["coverage_risk_points"] = int(len(curve))
    summary.update(compute_selective_reliability_metrics(results, success_threshold=threshold))
    return summary


def compute_auroc_for_success(
    results: List[Dict[str, Any]],
    success_threshold: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    rows = _valid_labeled_rows(results, success_threshold=success_threshold)
    labels = np.asarray([bool(r.get("success_label")) for r in rows], dtype=bool)
    scores = np.asarray([_score(r) for r in rows], dtype=np.float64)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return None
    return _rank_auc(labels, scores)


def coverage_risk_curve(
    results: List[Dict[str, Any]],
    success_threshold: Optional[Dict[str, Any]] = None,
    score_key: str = "reliability_score",
    catastrophic: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, float]]:
    # selective curve within the covariance-eligible set, accepting highest-reliability first
    valid = _eligible_rows(_valid_labeled_rows(results, success_threshold=success_threshold),
                           score_key=score_key)
    valid.sort(key=lambda r: _score(r, score_key=score_key), reverse=True)
    ct_t, ct_r = catastrophic or catastrophic_threshold_for((success_threshold or {}).get("profile"))
    rows: List[Dict[str, float]] = []
    for k in range(1, len(valid) + 1):
        chosen = valid[:k]
        labels = [bool(r["success_label"]) for r in chosen if r.get("success_label") is not None]
        success_rate = float(np.mean(labels)) if labels else float("nan")
        cat = float(np.mean([
            1.0 if (float(r["translation_error"]) > ct_t
                    or float(r.get("rotation_error_deg", 0.0)) > ct_r) else 0.0
            for r in chosen
        ]))
        rows.append(
            {
                "coverage": float(k / max(len(valid), 1)),
                "risk_translation_mean": float(np.mean([float(r["translation_error"]) for r in chosen])),
                "risk_rotation_mean": float(np.mean([float(r.get("rotation_error_deg", 0.0)) for r in chosen])),
                "success_rate": success_rate,
                "failure_rate": float(1.0 - success_rate) if np.isfinite(success_rate) else float("nan"),
                "catastrophic_rate": cat,
                "threshold": _score(chosen[-1], score_key=score_key),
            }
        )
    return rows
