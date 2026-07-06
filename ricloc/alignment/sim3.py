from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _skew(x: np.ndarray) -> np.ndarray:
    x0, x1, x2 = np.asarray(x, dtype=np.float64).reshape(3).tolist()
    return np.array(
        [
            [0.0, -x2, x1],
            [x2, 0.0, -x0],
            [-x1, x0, 0.0],
        ],
        dtype=np.float64,
    )


def _sanitize_points(src_points_v: np.ndarray, dst_points_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src_points_v, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_points_w, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    return src[finite], dst[finite], finite


def _sanitize_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size < n:
        fixed = np.ones(n, dtype=np.float64)
        fixed[: w.size] = w
        w = fixed
    else:
        w = w[:n].copy()
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        w = np.ones(n, dtype=np.float64)
    return w.astype(np.float64, copy=False)


def _weighted_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    estimate_scale: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src_arr.shape != dst_arr.shape or src_arr.shape[0] < 3:
        raise ValueError("weighted Umeyama requires at least 3 paired 3D points")

    w = _sanitize_weights(weights, src_arr.shape[0])
    w_sum = float(np.sum(w))
    w_norm = w / max(w_sum, 1e-12)

    mu_src = np.sum(src_arr * w_norm[:, None], axis=0)
    mu_dst = np.sum(dst_arr * w_norm[:, None], axis=0)
    X = src_arr - mu_src.reshape(1, 3)
    Y = dst_arr - mu_dst.reshape(1, 3)
    cov = Y.T @ (X * w_norm[:, None])

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    if estimate_scale:
        var_src = float(np.sum(w_norm * np.sum(X * X, axis=1)))
        scale = float(np.sum(D * np.diag(S)) / max(var_src, 1e-12))
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R.astype(np.float64), t.astype(np.float64)


def weighted_umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
    estimate_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    src_arr, dst_arr, finite_mask = _sanitize_points(src, dst)
    if weights is None:
        w = np.ones(src_arr.shape[0], dtype=np.float64)
    else:
        w_all = _sanitize_weights(weights, int(finite_mask.shape[0]))
        w = _sanitize_weights(w_all[finite_mask], src_arr.shape[0])
    return _weighted_umeyama(src_arr, dst_arr, w, estimate_scale=estimate_scale)


def _predict(src: np.ndarray, scale: float, R_wv: np.ndarray, t_wv: np.ndarray) -> np.ndarray:
    return (float(scale) * (np.asarray(src, dtype=np.float64).reshape(-1, 3) @ R_wv.T)) + t_wv.reshape(1, 3)


def _residuals(src: np.ndarray, dst: np.ndarray, scale: float, R_wv: np.ndarray, t_wv: np.ndarray) -> np.ndarray:
    return np.asarray(dst, dtype=np.float64).reshape(-1, 3) - _predict(src, scale, R_wv, t_wv)


def _logdet_spd(matrix: np.ndarray, damping: float) -> float:
    M = np.asarray(matrix, dtype=np.float64)
    eye = np.eye(M.shape[0], dtype=np.float64)
    try:
        L = np.linalg.cholesky(M + float(damping) * eye)
        return float(2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-300))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(M + float(damping) * eye)
        return float(logdet if sign > 0 else -np.inf)


@dataclass
class Sim3AlignmentResult:
    success: bool
    scale: float
    R_wv: np.ndarray
    t_wv: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray
    rmse: float
    J: np.ndarray
    W: np.ndarray
    normal_matrix: np.ndarray
    condition_number: float
    logdet_info: float
    robust_weights: np.ndarray
    message: str = ""


class WeightedSim3Aligner:
    def __init__(
        self,
        estimate_scale: bool = True,
        ransac_threshold_m: float = 0.3,
        ransac_max_iters: int = 2000,
        ransac_min_inliers: int = 5,
        seed: int = 42,
        huber_delta: float = 1.0,
        robust_iters: int = 3,
        covariance_damping: float = 1e-6,
    ):
        self.estimate_scale = bool(estimate_scale)
        self.ransac_threshold_m = float(ransac_threshold_m)
        self.ransac_max_iters = int(max(1, ransac_max_iters))
        self.ransac_min_inliers = int(max(3, ransac_min_inliers))
        self.seed = int(seed)
        self.huber_delta = float(max(huber_delta, 1e-12))
        self.robust_iters = int(max(0, robust_iters))
        self.covariance_damping = float(max(covariance_damping, 0.0))

    def fit(
        self,
        src_points_v: np.ndarray,
        dst_points_w: np.ndarray,
        weights: Optional[np.ndarray] = None,
        robust: str = "huber",
        ransac: bool = True,
    ) -> Sim3AlignmentResult:
        src, dst, finite_mask = _sanitize_points(src_points_v, dst_points_w)
        n = int(src.shape[0])
        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w_all = _sanitize_weights(weights, int(finite_mask.shape[0]))
            w = w_all[finite_mask]
            w = _sanitize_weights(w, n)
        if n < 3:
            return self._failure_result(src, dst, w, f"Not enough correspondences for alignment: {n} < 3")

        if ransac:
            init = self._fit_ransac(src, dst, w)
            if init is None:
                return self._failure_result(
                    src,
                    dst,
                    w,
                    (
                        "RANSAC alignment failed. "
                        f"iters={self.ransac_max_iters}, thresh={self.ransac_threshold_m}, "
                        f"min_inliers={min(self.ransac_min_inliers, n)}, pairs={n}"
                    ),
                )
            scale, R_wv, t_wv, inlier_mask = init
        else:
            try:
                scale, R_wv, t_wv = _weighted_umeyama(src, dst, w, estimate_scale=self.estimate_scale)
            except Exception as exc:
                return self._failure_result(src, dst, w, f"Umeyama alignment failed: {exc}")
            inlier_mask = np.ones(n, dtype=bool)

        if int(np.count_nonzero(inlier_mask)) < 3:
            return self._failure_result(src, dst, w, "Alignment produced fewer than 3 inliers")

        scale, R_wv, t_wv, robust_weights = self._refine_robust(src, dst, w, inlier_mask, scale, R_wv, t_wv, robust)
        return self._build_result(src, dst, w, inlier_mask, scale, R_wv, t_wv, robust_weights, "ok")

    def _fit_ransac(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        weights: np.ndarray,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        n = int(src.shape[0])
        sample_size = 3
        req = int(max(3, min(self.ransac_min_inliers, n)))
        effective_iters = 1 if n == sample_size else self.ransac_max_iters
        rng = np.random.default_rng(self.seed)

        best_count = -1
        best_rmse = float("inf")
        best_model: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        best_inliers: Optional[np.ndarray] = None

        for _ in range(effective_iters):
            idx = np.arange(sample_size) if n == sample_size else rng.choice(n, size=sample_size, replace=False)
            try:
                scale, R_wv, t_wv = _weighted_umeyama(src[idx], dst[idx], weights[idx], self.estimate_scale)
            except Exception:
                continue
            err = np.linalg.norm(_residuals(src, dst, scale, R_wv, t_wv), axis=1)
            inliers = np.isfinite(err) & (err <= self.ransac_threshold_m)
            count = int(np.count_nonzero(inliers))
            if count < req:
                continue
            try:
                scale_ref, R_ref, t_ref = _weighted_umeyama(src[inliers], dst[inliers], weights[inliers], self.estimate_scale)
            except Exception:
                continue
            err_ref = np.linalg.norm(_residuals(src[inliers], dst[inliers], scale_ref, R_ref, t_ref), axis=1)
            rmse = float(np.sqrt(np.mean(err_ref * err_ref))) if err_ref.size else float("inf")
            if count > best_count or (count == best_count and rmse < best_rmse):
                best_count = count
                best_rmse = rmse
                best_model = (scale_ref, R_ref, t_ref)
                best_inliers = inliers.copy()

        if best_model is None or best_inliers is None:
            return None
        return best_model[0], best_model[1], best_model[2], best_inliers

    def _refine_robust(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        base_weights: np.ndarray,
        inlier_mask: np.ndarray,
        scale: float,
        R_wv: np.ndarray,
        t_wv: np.ndarray,
        robust: str,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        robust_weights = np.ones(src.shape[0], dtype=np.float64)
        use_huber = str(robust or "").strip().lower() == "huber"
        if not use_huber or self.robust_iters <= 0:
            return scale, R_wv, t_wv, robust_weights

        active = np.asarray(inlier_mask, dtype=bool)
        for _ in range(self.robust_iters):
            res_norm = np.linalg.norm(_residuals(src, dst, scale, R_wv, t_wv), axis=1)
            rw = np.ones_like(res_norm, dtype=np.float64)
            large = np.isfinite(res_norm) & (res_norm > self.huber_delta)
            rw[large] = self.huber_delta / np.maximum(res_norm[large], 1e-12)
            rw[~np.isfinite(res_norm)] = 0.0
            robust_weights = rw
            fit_weights = base_weights[active] * robust_weights[active]
            if float(np.sum(fit_weights)) <= 1e-12:
                fit_weights = base_weights[active]
            scale, R_wv, t_wv = _weighted_umeyama(src[active], dst[active], fit_weights, self.estimate_scale)
        return scale, R_wv, t_wv, robust_weights

    def _build_result(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        base_weights: np.ndarray,
        inlier_mask: np.ndarray,
        scale: float,
        R_wv: np.ndarray,
        t_wv: np.ndarray,
        robust_weights: np.ndarray,
        message: str,
    ) -> Sim3AlignmentResult:
        residuals = _residuals(src, dst, scale, R_wv, t_wv)
        active = np.asarray(inlier_mask, dtype=bool)
        err = np.linalg.norm(residuals[active], axis=1) if np.any(active) else np.zeros((0,), dtype=np.float64)
        rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("inf")

        J = self._build_jacobian(src[active], scale, R_wv)
        scalar_w = base_weights[active] * robust_weights[active]
        scalar_w = _sanitize_weights(scalar_w, J.shape[0] // 3 if J.size else 0)
        w_diag = np.repeat(scalar_w, 3)
        W = np.diag(w_diag.astype(np.float64)) if w_diag.size else np.zeros((0, 0), dtype=np.float64)
        normal = J.T @ (J * w_diag[:, None]) if J.size else np.zeros((7, 7), dtype=np.float64)

        trace = float(np.trace(normal)) if normal.size else 0.0
        damping = self.covariance_damping * trace / float(normal.shape[0]) if trace > 0.0 else self.covariance_damping
        if damping <= 0.0:
            damping = 1e-12
        damped = normal + np.eye(normal.shape[0], dtype=np.float64) * damping
        try:
            condition_number = float(np.linalg.cond(damped))
        except np.linalg.LinAlgError:
            condition_number = float("inf")
        logdet_info = _logdet_spd(normal, damping)

        return Sim3AlignmentResult(
            success=True,
            scale=float(scale),
            R_wv=np.asarray(R_wv, dtype=np.float64).reshape(3, 3),
            t_wv=np.asarray(t_wv, dtype=np.float64).reshape(3),
            inlier_mask=np.asarray(inlier_mask, dtype=bool),
            residuals=residuals.astype(np.float64),
            rmse=rmse,
            J=J.astype(np.float64),
            W=W.astype(np.float64),
            normal_matrix=normal.astype(np.float64),
            condition_number=condition_number,
            logdet_info=logdet_info,
            robust_weights=np.asarray(robust_weights, dtype=np.float64).reshape(-1),
            message=message,
        )

    def _build_jacobian(self, src: np.ndarray, scale: float, R_wv: np.ndarray) -> np.ndarray:
        rows = []
        R = np.asarray(R_wv, dtype=np.float64).reshape(3, 3)
        s = float(scale)
        for c in np.asarray(src, dtype=np.float64).reshape(-1, 3):
            Rc = R @ c.reshape(3)
            rows.append(
                np.concatenate(
                    [
                        s * R @ _skew(c),
                        -np.eye(3, dtype=np.float64),
                        (-s * Rc).reshape(3, 1),
                    ],
                    axis=1,
                )
            )
        if not rows:
            return np.zeros((0, 7), dtype=np.float64)
        return np.concatenate(rows, axis=0)

    def _failure_result(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        weights: np.ndarray,
        message: str,
    ) -> Sim3AlignmentResult:
        n = int(np.asarray(src, dtype=np.float64).reshape(-1, 3).shape[0])
        return Sim3AlignmentResult(
            success=False,
            scale=1.0,
            R_wv=np.eye(3, dtype=np.float64),
            t_wv=np.zeros(3, dtype=np.float64),
            inlier_mask=np.zeros(n, dtype=bool),
            residuals=np.asarray(dst, dtype=np.float64).reshape(-1, 3) - np.asarray(src, dtype=np.float64).reshape(-1, 3),
            rmse=float("inf"),
            J=np.zeros((0, 7), dtype=np.float64),
            W=np.zeros((0, 0), dtype=np.float64),
            normal_matrix=np.zeros((7, 7), dtype=np.float64),
            condition_number=float("inf"),
            logdet_info=float("-inf"),
            robust_weights=_sanitize_weights(weights, n),
            message=message,
        )
