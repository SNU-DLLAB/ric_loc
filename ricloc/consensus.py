"""Reference-induced SE(3) consensus — R_cons + scalar C_cons.

Lift math (per reference i, COLMAP world->cam convention):
    R_qi = R_q^v (R_i^v)^T R_i^w
    C_qi = C_i^w + s (R_i^w)^T R_i^v (C_q^v - C_i^v)
R_cons = weighted quaternion mean of {R_qi} with one MAD angular-inlier refit.
C_cons = weight-normalized mean of the rotation-inlier {C_qi}.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .geometry import qvec_to_rotmat, rotmat_to_qvec_wxyz


def mapfree_consensus_weight(
    track_conf_mean: Optional[float],
    retrieval_rank: Optional[int],
    retrieval_count: Optional[int],
    retrieval_score_uncertainty: Optional[float],
    sigma_base: float = 0.05,
) -> float:
    """Reference-consensus weight w = 1 / sigma^2.

    sigma^2 = sigma_base^2 + sigma_vggt^2 + sigma_retrieval^2, from VGGT track confidence
    and retrieval uncertainty (rank + score gap).
    """
    sigma_vggt = 0.0
    if track_conf_mean is not None:
        try:
            conf = float(track_conf_mean)
            if np.isfinite(conf):
                conf = float(np.clip(conf, 0.05, 1.0))
                sigma_vggt = 0.05 * np.sqrt(max(0.0, 1.0 - conf) / conf)
        except Exception:
            sigma_vggt = 0.0
    retrieval_unc = 0.0
    if retrieval_rank is not None:
        try:
            rank = max(1, int(retrieval_rank))
            count = max(1, int(retrieval_count if retrieval_count is not None else rank))
            retrieval_unc += float(np.sqrt(max(rank - 1, 0) / max(count - 1, 1)))
        except Exception:
            pass
    if retrieval_score_uncertainty is not None:
        try:
            score_unc = float(retrieval_score_uncertainty)
            if np.isfinite(score_unc):
                retrieval_unc += float(np.clip(score_unc, 0.0, 1.0))
        except Exception:
            pass
    sigma_retrieval = 0.05 * float(np.clip(retrieval_unc, 0.0, 2.0))
    sigma_sq = sigma_base * sigma_base + sigma_vggt * sigma_vggt + sigma_retrieval * sigma_retrieval
    weight = 1.0 / max(sigma_sq, 1e-9)
    return float(weight) if np.isfinite(weight) and weight > 0.0 else 1.0


def weighted_rotation_mean(rotations: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    """Weighted quaternion (Markley) mean of a set of rotations. Returns 3x3 or None."""
    if not rotations:
        return None
    A = np.zeros((4, 4), dtype=np.float64)
    ref_q: Optional[np.ndarray] = None
    for R, w in zip(rotations, weights):
        q = rotmat_to_qvec_wxyz(np.asarray(R, dtype=np.float64).reshape(3, 3))
        if ref_q is None:
            ref_q = q.copy()
        elif float(np.dot(ref_q, q)) < 0.0:
            q = -q
        weight = float(w) if np.isfinite(float(w)) and float(w) > 0.0 else 1.0
        A += weight * np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q_mean = vecs[:, int(np.argmax(vals))]
    if q_mean[0] < 0.0:
        q_mean = -q_mean
    return qvec_to_rotmat(q_mean.astype(np.float64))


def reference_consensus_pose(
    ref_hyps: List[Dict[str, Any]],
    query_R_v: np.ndarray,
    query_C_v: np.ndarray,
    scale_s: float,
    dump: bool = False,
) -> Optional[Dict[str, Any]]:
    """Reference-induced consensus pose (R_cons, C_cons).

    ref_hyps: per-reference {"R_v","C_v","R_w","C_w","weight","image_id"}; *_v are
    VGGT-local world->cam, *_w are COLMAP map-frame world->cam. Returns None when
    fewer than 2 usable references.
    """
    Rq_v = np.asarray(query_R_v, dtype=np.float64).reshape(3, 3)
    Cq_v = np.asarray(query_C_v, dtype=np.float64).reshape(3)
    s = float(scale_s)

    Rqi: List[np.ndarray] = []
    Cqi: List[np.ndarray] = []
    ws: List[float] = []
    ref_ids: List[int] = []
    for h in ref_hyps:
        Rv = np.asarray(h["R_v"], dtype=np.float64).reshape(3, 3)
        Cv = np.asarray(h["C_v"], dtype=np.float64).reshape(3)
        Rw = np.asarray(h["R_w"], dtype=np.float64).reshape(3, 3)
        Cw = np.asarray(h["C_w"], dtype=np.float64).reshape(3)
        Rqi.append(Rq_v @ Rv.T @ Rw)
        Cqi.append(Cw + s * (Rw.T @ Rv) @ (Cq_v - Cv))
        ws.append(float(h.get("weight", 1.0)))
        ref_ids.append(int(h.get("image_id", -1)))

    if len(Rqi) < 2:
        return None
    R_cons = weighted_rotation_mean(Rqi, ws)
    if R_cons is None:
        return None

    def _geo(Ra, Rb):
        c = np.clip(0.5 * (float(np.trace(Ra @ Rb.T)) - 1.0), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    ang = np.asarray([_geo(R, R_cons) for R in Rqi], dtype=np.float64)
    med = float(np.median(ang))
    mad = float(np.median(np.abs(ang - med)))
    thr = max(5.0, med + 2.5 * 1.4826 * mad)
    keep = ang <= thr
    if int(keep.sum()) >= 2 and int(keep.sum()) < len(Rqi):
        R_cons = weighted_rotation_mean(
            [R for R, k in zip(Rqi, keep.tolist()) if k],
            [w for w, k in zip(ws, keep.tolist()) if k],
        )

    Cq_arr = np.asarray(Cqi, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)
    Cin = Cq_arr[keep] if keep.any() else Cq_arr
    win = w_arr[keep] if keep.any() else w_arr
    C_rotmean = (win[:, None] * Cin).sum(0) / max(win.sum(), 1e-9)
    disp = float(np.sqrt(((Cin - C_rotmean) ** 2).sum(1).mean())) if len(Cin) > 1 else 0.05
    # C_cons = weighted mean over the rotation-inlier set
    C_cons = (win[:, None] * Cin).sum(0) / max(win.sum(), 1e-9)

    Rcw = np.asarray(R_cons, dtype=np.float64).reshape(3, 3)
    tcw = (-Rcw @ C_cons.reshape(3)).astype(np.float64)
    out: Dict[str, Any] = {
        "Rcw": Rcw,
        "tcw": tcw,
        "C": C_cons,
        "source": "se3_consensus",
        "center_dispersion_m": disp,
        "num_refs": int(len(Rqi)),
        "num_inliers": int(keep.sum()),
    }
    if dump:
        out["consensus_hyp"] = {
            "ref_ids": [int(i) for i in ref_ids],
            "Cqi": Cq_arr.tolist(),
            "ang_to_rcons_deg": ang.tolist(),
            "w": w_arr.tolist(),
            "keep_rot": [bool(k) for k in keep.tolist()],
            "C_cons": [float(x) for x in C_cons.reshape(3)],
            "scale": float(s),
        }
    return out
