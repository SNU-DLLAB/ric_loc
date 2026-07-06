"""Map-point-free uncertainty-propagated reference consensus (center side).

Per reference, builds a first-order information matrix H_i = sum_j J_ij^T W_ij J_ij from VGGT
query<->reference track reprojection geometry, inverts to a relative-pose covariance
Sigma_rel,i = (H_i + lambda I)^-1, propagates to a map-frame center covariance Sigma_C,i, and runs
a robust Student-t covariance-weighted IRLS center consensus -> C_cons, posterior Sigma_post, with
reliability sigma_new = sqrt(lambda_max(phi * Sigma_post)), phi = max(1, chi2_red).

Coordinate convention (COLMAP world->camera): x_c = R_cw x_w + t_cw, C_w = -R_cw^T t_cw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues exponential, SO(3)."""
    w = np.asarray(w, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3, dtype=np.float64) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential for a 6-vector xi = [rho(3), phi(3)] (translation-first). Returns 4x4."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho, phi = xi[:3], xi[3:]
    R = exp_so3(phi)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = rho  # small-perturbation translation
    return T


def lift_rotation(R_q_v: np.ndarray, R_i_v: np.ndarray, R_i_w: np.ndarray) -> np.ndarray:
    """R_qi = R_q^v (R_i^v)^T R_i^w  (map-frame query rotation induced by reference i)."""
    return np.asarray(R_q_v, np.float64) @ np.asarray(R_i_v, np.float64).T @ np.asarray(R_i_w, np.float64)


def lift_center(C_q_v: np.ndarray, C_i_v: np.ndarray, R_i_v: np.ndarray,
                R_i_w: np.ndarray, C_i_w: np.ndarray, scale_s: float) -> np.ndarray:
    """C_qi = C_i^w + s (R_i^w)^T R_i^v (C_q^v - C_i^v)."""
    C_q_v = np.asarray(C_q_v, np.float64).reshape(3)
    C_i_v = np.asarray(C_i_v, np.float64).reshape(3)
    M = np.asarray(R_i_w, np.float64).T @ np.asarray(R_i_v, np.float64)
    return np.asarray(C_i_w, np.float64).reshape(3) + float(scale_s) * (M @ (C_q_v - C_i_v))


def project(K: np.ndarray, X_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of a 3D point in camera frame. X_cam: [...,3] -> [...,2]."""
    K = np.asarray(K, np.float64)
    X = np.asarray(X_cam, np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = X[..., 2]
    u = fx * X[..., 0] / Z + cx
    v = fy * X[..., 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _proj_jac_point(K: np.ndarray, X_cam: np.ndarray) -> np.ndarray:
    """d pi / d X_cam, 2x3, at point X_cam (in camera frame)."""
    K = np.asarray(K, np.float64)
    fx, fy = K[0, 0], K[1, 1]
    X, Y, Z = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])
    return np.array([[fx / Z, 0.0, -fx * X / (Z * Z)],
                     [0.0, fy / Z, -fy * Y / (Z * Z)]], dtype=np.float64)


def reprojection_jacobian_analytic(K_i: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Jacobian of the reprojection residual w.r.t. a LEFT SE(3) perturbation of the relative pose
    T_{i<-q}, at the point X_ref in reference camera frame.
    d e / d xi = - d pi/d X * [ I | -[X_ref]_x ]   (xi = [rho, phi]).  Returns 2x6."""
    dpi = _proj_jac_point(K_i, X_ref)              # 2x3
    dX = np.hstack([np.eye(3), -skew(X_ref)])      # 3x6  (left perturbation)
    return -dpi @ dX                               # 2x6


def reprojection_residual(K_i: np.ndarray, T_i_from_q: np.ndarray, X_q_cam: np.ndarray,
                          x_ij: np.ndarray) -> np.ndarray:
    """e = x_ij - pi(K_i, T_{i<-q} X_q_cam). X_q_cam in query camera frame, T 4x4."""
    Xh = np.append(np.asarray(X_q_cam, np.float64).reshape(3), 1.0)
    X_ref = (np.asarray(T_i_from_q, np.float64) @ Xh)[:3]
    return np.asarray(x_ij, np.float64).reshape(2) - project(K_i, X_ref)


def robust_mad_scale(residual_norms: np.ndarray, min_scale: float = 1.0) -> float:
    r = np.asarray(residual_norms, np.float64).reshape(-1)
    if r.size == 0:
        return float(min_scale)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return float(max(1.4826 * mad, min_scale))


def invert_information(H: np.ndarray, min_eig: float = 1e-8, max_cov_eig: float = 1e6) -> np.ndarray:
    """PSD inverse with eigenvalue clamping. Degenerate dirs -> large covariance."""
    H = 0.5 * (np.asarray(H, np.float64) + np.asarray(H, np.float64).T)
    w, V = np.linalg.eigh(H)
    w = np.clip(w, min_eig, None)
    cov_eig = np.clip(1.0 / w, None, max_cov_eig)
    return (V * cov_eig) @ V.T


@dataclass
class RefHypothesis:
    R_qi: np.ndarray
    C_qi: np.ndarray
    Sigma_C: np.ndarray          # 3x3 map-frame center covariance
    num_tracks: int
    sigma_pix: float
    H_rel: np.ndarray            # 6x6 relative-pose information
    cond_H: float
    lever_u: np.ndarray          # 3-vector dC_qi/ds (scale lever arm)
    weak: bool = False


def build_reference_hypothesis(
    *, R_q_v, C_q_v, R_i_v, C_i_v, R_i_w, C_i_w, scale_s,
    K_q, K_i, X_q_cam, x_ref, conf=None,
    T_i_from_q_v, var_scale: float = 0.0,
    sigma_pix_min: float = 1.0, lambda_damp: float = 1e-6,
    min_tracks: int = 6, weak_cov_scale: float = 1e4,
) -> RefHypothesis:
    """Build one reference's covariance-bearing hypothesis.

    X_q_cam: [N,3] query-local 3D points in QUERY camera frame.
    x_ref:   [N,2] observed reference pixels of the same tracks.
    T_i_from_q_v: 4x4 relative transform query-cam -> reference-cam in VGGT local frame.
    conf:    [N] optional VGGT track confidence.
    """
    R_qi = lift_rotation(R_q_v, R_i_v, R_i_w)
    C_qi = lift_center(C_q_v, C_i_v, R_i_v, R_i_w, C_i_w, scale_s)
    X_q_cam = np.asarray(X_q_cam, np.float64).reshape(-1, 3)
    x_ref = np.asarray(x_ref, np.float64).reshape(-1, 2)
    n = int(X_q_cam.shape[0])
    M = np.asarray(R_i_w, np.float64).T @ np.asarray(R_i_v, np.float64)   # lift rotation (R_i^w)^T R_i^v
    u_i = M @ (np.asarray(C_q_v, np.float64).reshape(3) - np.asarray(C_i_v, np.float64).reshape(3))

    if n < int(min_tracks):
        Sigma_C = (weak_cov_scale ** 2) * np.eye(3) + (u_i[:, None] * u_i[None, :]) * float(var_scale)
        return RefHypothesis(R_qi, C_qi, Sigma_C, n, float(sigma_pix_min),
                             np.zeros((6, 6)), float("inf"), lever_u=u_i, weak=True)

    # residuals + Jacobians at the VGGT relative pose
    res = np.zeros((n, 2)); Js = np.zeros((n, 2, 6))
    for j in range(n):
        Xh = np.append(X_q_cam[j], 1.0)
        X_ref = (np.asarray(T_i_from_q_v, np.float64) @ Xh)[:3]
        res[j] = x_ref[j] - project(K_i, X_ref)
        Js[j] = reprojection_jacobian_analytic(K_i, X_ref)
    sigma_pix = robust_mad_scale(np.linalg.norm(res, axis=1), min_scale=sigma_pix_min)
    c = np.ones(n) if conf is None else np.clip(np.asarray(conf, np.float64).reshape(-1), 0.0, None)
    # H = sum_j J^T W J,  W = c_j / sigma_pix^2 I
    H = np.zeros((6, 6))
    for j in range(n):
        W = (c[j] / (sigma_pix ** 2)) * np.eye(2)
        H += Js[j].T @ W @ Js[j]
    Sigma_rel = invert_information(H + lambda_damp * np.eye(6))   # 6x6 rel-pose cov, xi=[rho, phi]
    # FULL lift Jacobian of the map-frame centre w.r.t. the SAME left-SE(3) perturbation used by H.
    # C_qi = C_i^w + s (R_i^w)^T t_{i<-q}^v ;  u_i = (R_i^w)^T t_{i<-q}^v = M (C_q^v - C_i^v).
    # Left perturbation: t' ~ t + rho - [t]x phi  =>  dC/d(rho,phi) = s (R_i^w)^T [ I | -[t]x ]
    #                                                              = s [ (R_i^w)^T | -[u_i]x (R_i^w)^T ].
    Riw_T = np.asarray(R_i_w, np.float64).T
    G = float(scale_s) * np.hstack([Riw_T, -skew(u_i) @ Riw_T])   # 3x6
    Sigma_C = G @ Sigma_rel @ G.T + (u_i[:, None] * u_i[None, :]) * float(var_scale)
    Sigma_C = 0.5 * (Sigma_C + Sigma_C.T)
    cond = float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else float("inf")
    return RefHypothesis(R_qi, C_qi, Sigma_C, n, float(sigma_pix), H, cond, lever_u=u_i, weak=False)


@dataclass
class ConsensusResult:
    C_cons: np.ndarray
    omega: np.ndarray
    Sigma_post: np.ndarray
    Q: float
    chi2_red: float
    phi: float
    Sigma_inflated: np.ndarray
    sigma_new: float
    dof: int


def _student_t_irls(C_hyp: np.ndarray, Info: np.ndarray, nu: float, d: int,
                    max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner Student-t IRLS center fit for a GIVEN set of per-reference information matrices.
    Returns (C, omega, m2)."""
    K = int(C_hyp.shape[0])
    A0 = Info.sum(0); b0 = np.einsum("kij,kj->i", Info, C_hyp)
    C = np.linalg.solve(A0 + 1e-12 * np.eye(3), b0)
    for _ in range(int(max_iter)):
        diff = C_hyp - C[None, :]
        m2 = np.einsum("ki,kij,kj->k", diff, Info, diff)
        omega = (nu + d) / (nu + np.maximum(m2, 0.0))
        A = np.einsum("k,kij->ij", omega, Info)
        b = np.einsum("k,kij,kj->i", omega, Info, C_hyp)
        C_new = np.linalg.solve(A + 1e-12 * np.eye(3), b)
        if float(np.linalg.norm(C_new - C)) < tol:
            C = C_new; break
        C = C_new
    diff = C_hyp - C[None, :]
    m2 = np.einsum("ki,kij,kj->k", diff, Info, diff)
    omega = (nu + d) / (nu + np.maximum(m2, 0.0))
    return C, omega, m2


def student_t_center_consensus(C_hyp: np.ndarray, Sigma_C: np.ndarray,
                               nu: float = 5.0, max_iter: int = 20, tol: float = 1e-7,
                               lever_arms: Optional[np.ndarray] = None,
                               var_s: float = 0.0) -> ConsensusResult:
    """IRLS for the Student-t covariance-weighted center MAP estimate.

    Sim(3) scale uncertainty is common-mode (shifts every C_qi along lever arm u_i = dC_qi/ds),
    propagated into the posterior: dC_cons/ds = Sigma_post (sum_i w_i Info_i u_i),
    Sigma_cons = phi*Sigma_post + var_s (dC_cons/ds)(dC_cons/ds)^T -> sigma_cons.
    """
    C_hyp = np.asarray(C_hyp, np.float64).reshape(-1, 3)
    Sigma_C = np.asarray(Sigma_C, np.float64).reshape(-1, 3, 3)
    K = int(C_hyp.shape[0]); d = 3
    dof = max(d * K - d, 1)
    Info = np.stack([invert_information(s, min_eig=1e-12, max_cov_eig=1e12) for s in Sigma_C], axis=0)
    C, omega, m2 = _student_t_irls(C_hyp, Info, nu, d, max_iter, tol)
    Q = float(np.einsum("k,k->", omega, m2))
    chi2_red = Q / dof
    Sigma_post = invert_information(np.einsum("k,kij->ij", omega, Info), min_eig=1e-12, max_cov_eig=1e12)
    phi = max(1.0, chi2_red)
    Sigma_inflated = phi * Sigma_post
    # common-mode Sim(3) scale uncertainty
    if lever_arms is not None and float(var_s) > 0.0:
        U = np.asarray(lever_arms, np.float64).reshape(-1, 3)
        dCds = Sigma_post @ np.einsum("k,kij,kj->i", omega, Info, U)   # dC_cons/ds
        Sigma_inflated = Sigma_inflated + float(var_s) * np.outer(dCds, dCds)
    sigma_new = float(np.sqrt(max(np.linalg.eigvalsh(Sigma_inflated)[-1], 0.0)))
    return ConsensusResult(C, omega, Sigma_post, Q, float(chi2_red), float(phi),
                           Sigma_inflated, sigma_new, dof)
