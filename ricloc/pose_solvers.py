from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ricloc.alignment import WeightedSim3Aligner, weighted_umeyama_alignment


@dataclass
class AlignResult:
    ok: bool
    transform_type: str
    scale: float = 1.0
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    rmse: float = float("inf")
    num_pairs: int = 0
    inliers: int = 0
    inlier_mask: Optional[np.ndarray] = None
    message: str = ""
    residuals: Optional[np.ndarray] = None
    residual_median: float = float("inf")
    residual_max: float = float("inf")
    residual_rmse_all: float = float("inf")
    residual_median_all: float = float("inf")
    residual_max_all: float = float("inf")
    residual_stats_source: str = "inliers"
    has_inlier_residual_stats: bool = False
    has_all_ref_residual_stats: bool = False
    normal_matrix: Optional[np.ndarray] = None
    condition_number: float = float("inf")
    logdet_info: float = float("-inf")
    logdet_cov: float = float("inf")


def umeyama_alignment(src: np.ndarray, dst: np.ndarray, estimate_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    return weighted_umeyama_alignment(
        src=src_arr,
        dst=np.asarray(dst, dtype=np.float64).reshape(-1, 3),
        weights=np.ones(src_arr.shape[0], dtype=np.float64),
        estimate_scale=estimate_scale,
    )


def ransac_align(
    src: np.ndarray,
    dst: np.ndarray,
    transform_type: str,
    iters: int,
    thresh: float,
    min_inliers: int,
    seed: int,
    weights: Optional[np.ndarray] = None,
) -> AlignResult:
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = int(src_arr.shape[0])
    if src_arr.shape != dst_arr.shape:
        return AlignResult(
            ok=False,
            transform_type=transform_type,
            num_pairs=min(n, int(dst_arr.shape[0])),
            message=f"Alignment point shape mismatch: {src_arr.shape} vs {dst_arr.shape}",
        )
    estimate_scale = str(transform_type).upper() == "SIM3"
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        try:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception:
            w = np.ones(n, dtype=np.float64)
        if w.size != n:
            fixed = np.ones(n, dtype=np.float64)
            fixed[: min(n, w.size)] = w[: min(n, w.size)]
            w = fixed
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        if float(np.sum(w)) <= 1e-12:
            w = np.ones(n, dtype=np.float64)
        med = float(np.median(w[np.isfinite(w) & (w > 0.0)])) if np.any(np.isfinite(w) & (w > 0.0)) else 1.0
        if np.isfinite(med) and med > 0.0:
            w = w / med
    aligner = WeightedSim3Aligner(
        estimate_scale=estimate_scale,
        ransac_threshold_m=float(thresh),
        ransac_max_iters=int(iters),
        ransac_min_inliers=int(min_inliers),
        seed=int(seed),
        huber_delta=max(float(thresh), 1e-6),
    )
    res = aligner.fit(
        src_points_v=src_arr,
        dst_points_w=dst_arr,
        weights=w,
        robust="huber",
        ransac=True,
    )
    all_res = np.linalg.norm(res.residuals, axis=1) if res.residuals is not None and res.residuals.size else np.zeros((0,))
    active_res = np.linalg.norm(res.residuals[res.inlier_mask], axis=1) if np.any(res.inlier_mask) else np.zeros((0,))
    all_rmse = float(np.sqrt(np.mean(all_res * all_res))) if all_res.size else float("inf")
    all_median = float(np.median(all_res)) if all_res.size else float("inf")
    all_max = float(np.max(all_res)) if all_res.size else float("inf")
    active_median = float(np.median(active_res)) if active_res.size else float("inf")
    active_max = float(np.max(active_res)) if active_res.size else float("inf")
    if not bool(res.success):
        return AlignResult(
            ok=False,
            transform_type=transform_type,
            num_pairs=n,
            inliers=int(np.count_nonzero(res.inlier_mask)),
            inlier_mask=res.inlier_mask,
            message=res.message,
            residuals=res.residuals,
            residual_median=active_median,
            residual_max=active_max,
            residual_rmse_all=all_rmse,
            residual_median_all=all_median,
            residual_max_all=all_max,
            residual_stats_source="inliers",
            has_inlier_residual_stats=bool(active_res.size),
            has_all_ref_residual_stats=bool(all_res.size),
            normal_matrix=None,
            condition_number=float("inf"),
            logdet_info=float("-inf"),
            logdet_cov=float("inf"),
        )

    logdet_cov = -float(res.logdet_info) if np.isfinite(float(res.logdet_info)) else float("inf")
    return AlignResult(
        ok=True,
        transform_type=transform_type,
        scale=float(res.scale),
        R=res.R_wv,
        t=res.t_wv,
        rmse=float(res.rmse),
        num_pairs=n,
        inliers=int(np.count_nonzero(res.inlier_mask)),
        inlier_mask=res.inlier_mask,
        message=res.message,
        residuals=res.residuals,
        residual_median=active_median,
        residual_max=active_max,
        residual_rmse_all=all_rmse,
        residual_median_all=all_median,
        residual_max_all=all_max,
        residual_stats_source="inliers",
        has_inlier_residual_stats=bool(active_res.size),
        has_all_ref_residual_stats=bool(all_res.size),
        normal_matrix=res.normal_matrix,
        condition_number=float(res.condition_number),
        logdet_info=float(res.logdet_info),
        logdet_cov=float(logdet_cov),
    )


def estimate_alignment_with_fallback(
    src: np.ndarray,
    dst: np.ndarray,
    transform_type: str,
    seed: int,
    primary_iters: int,
    primary_thresh: float,
    primary_min_inliers: int,
    fallback_iters: int,
    fallback_thresh: float,
    fallback_min_inliers: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[AlignResult, List[Dict[str, Any]], bool]:
    attempts_cfg = [
        ("primary", primary_iters, primary_thresh, primary_min_inliers),
        ("fallback", fallback_iters, fallback_thresh, fallback_min_inliers),
    ]

    attempts_log: List[Dict[str, Any]] = []
    last_result: Optional[AlignResult] = None

    for mode, iters, thresh, min_inliers in attempts_cfg:
        res = ransac_align(
            src=src,
            dst=dst,
            transform_type=transform_type,
            iters=iters,
            thresh=thresh,
            min_inliers=min_inliers,
            seed=seed,
            weights=weights,
        )
        attempts_log.append(
            {
                "mode": mode,
                "iters": int(iters),
                "thresh_m": float(thresh),
                "min_inliers_req": int(min_inliers),
                "ok": bool(res.ok),
                "inliers": int(res.inliers),
                "num_pairs": int(res.num_pairs),
                "rmse": None if not np.isfinite(res.rmse) else float(res.rmse),
                "residual_median": None if not np.isfinite(res.residual_median) else float(res.residual_median),
                "residual_max": None if not np.isfinite(res.residual_max) else float(res.residual_max),
                "residual_rmse_all": None if not np.isfinite(res.residual_rmse_all) else float(res.residual_rmse_all),
                "residual_median_all": (
                    None if not np.isfinite(res.residual_median_all) else float(res.residual_median_all)
                ),
                "residual_max_all": None if not np.isfinite(res.residual_max_all) else float(res.residual_max_all),
                "residual_stats_source": str(res.residual_stats_source),
                "has_inlier_residual_stats": bool(res.has_inlier_residual_stats),
                "has_all_ref_residual_stats": bool(res.has_all_ref_residual_stats),
                "condition_number": None if not np.isfinite(res.condition_number) else float(res.condition_number),
                "logdet_info": None if not np.isfinite(res.logdet_info) else float(res.logdet_info),
                "logdet_cov": None if not np.isfinite(res.logdet_cov) else float(res.logdet_cov),
                "message": str(res.message),
            }
        )
        last_result = res
        if res.ok:
            res.message = f"ok({mode})"
            return res, attempts_log, mode != "primary"

    assert last_result is not None
    last_result.message = " | ".join([f"{a['mode']}: {a['message']}" for a in attempts_log])
    return last_result, attempts_log, True
