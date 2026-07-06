"""Adapter: VGGT forward outputs -> map-point-free covariance consensus.

Extracts, per reference view, the query<->reference VGGT tracks + query-local 3D from the VGGT
pointmap, builds a covariance-bearing reference-induced hypothesis, and runs the robust Student-t
center consensus.

Frame conventions:
  - poses[v]["Rcw"], ["tcw"]: VGGT-local world->camera for view v (v=0 query, v>=1 references).
  - poses[v]["K"]: intrinsics in the ORIGINAL image frame (K_scaled).
  - points_vggt[v]: [Hm, Wm, 3] per-pixel 3D in the VGGT-local WORLD frame, at model resolution.
  - track_xy[v]: [T, 2] track pixels in MODEL frame; frame_xy_to_original maps model xy -> original xy.
  - points_vggt is indexed by the MODEL track pixel; reprojection into references is in the ORIGINAL
    frame using K (K_scaled), compared to frame_xy_to_original(ref track pixel).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ricloc.mapfree_consensus import (
    build_reference_hypothesis, student_t_center_consensus, project, ConsensusResult,
)


def _w2c(Rcw, tcw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rcw, np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(tcw, np.float64).reshape(3)
    return T


def _apply_xy_to_original(xy_model: np.ndarray, frame_xy_to_original: Any, view: int) -> np.ndarray:
    """Map model-frame track xy -> original-frame xy: x_orig = (x_model - crop[0]) * scale[0].
    frame_xy_to_original: per-view dicts (crop_box_model_xyxy, scale_to_orig_xy, valid). Identity fallback."""
    xy = np.asarray(xy_model, np.float64).reshape(-1, 2)
    f = frame_xy_to_original
    if isinstance(f, list) and 0 <= view < len(f) and isinstance(f[view], dict):
        meta = f[view]
        if not bool(meta.get("valid", False)):
            return xy
        crop = np.asarray(meta.get("crop_box_model_xyxy", [0.0, 0.0, 0.0, 0.0]), np.float64).reshape(-1)
        scale = np.asarray(meta.get("scale_to_orig_xy", [1.0, 1.0]), np.float64).reshape(-1)
        if crop.size < 2 or scale.size < 2 or scale[0] <= 0 or scale[1] <= 0:
            return xy
        return np.stack([(xy[:, 0] - crop[0]) * scale[0], (xy[:, 1] - crop[1]) * scale[1]], axis=1)
    return xy


def mapfree_consensus_from_vggt(
    *,
    poses: List[Dict[str, Any]],            # [V] pose dicts (Rcw,tcw,K); 0=query, 1..K=refs
    points_vggt: np.ndarray,                # [V, Hm, Wm, 3] VGGT-local pointmap
    track_xy: np.ndarray,                   # [V, T, 2] model-frame track pixels
    track_vis: np.ndarray,                  # [V, T]
    track_conf: Optional[np.ndarray],       # [V, T] or None
    ref_R_w: List[np.ndarray],              # [K] known map-frame ref world->cam rotations
    ref_C_w: List[np.ndarray],              # [K] known map-frame ref centers
    ref_view_indices: List[int],            # [K] which pose/track view each ref maps to (>=1)
    scale_s: float,
    frame_xy_to_original: Optional[np.ndarray] = None,
    vis_thresh: float = 0.5,
    conf_thresh: float = 0.0,
    min_tracks: int = 6,
    nu: float = 5.0,
    diag: Optional[Dict[str, Any]] = None,
) -> Optional[ConsensusResult]:
    """Return the Student-t covariance-weighted center consensus, or None if too few usable refs."""
    P = points_vggt
    Hm, Wm = int(P.shape[1]), int(P.shape[2])
    Rq = np.asarray(poses[0]["Rcw"], np.float64).reshape(3, 3)
    tq = np.asarray(poses[0]["tcw"], np.float64).reshape(3)
    Cq_v = -Rq.T @ tq                                       # query center in VGGT-local frame
    Tq = _w2c(Rq, tq)
    vis = np.asarray(track_vis, np.float64)
    conf = None if track_conf is None else np.asarray(track_conf, np.float64)
    q_xy_model = np.asarray(track_xy[0], np.float64).reshape(-1, 2)
    # query-local 3D per track from the VGGT pointmap (model-pixel index), then to query CAMERA frame
    qx = np.clip(np.round(q_xy_model[:, 0]).astype(int), 0, Wm - 1)
    qy = np.clip(np.round(q_xy_model[:, 1]).astype(int), 0, Hm - 1)
    Xq_world = P[0][qy, qx]                                  # [T,3] VGGT-local world
    Xq_cam_all = (Rq @ Xq_world.T).T + tq[None, :]          # [T,3] query camera frame

    hyps = []
    resid_med = []
    hyp_Cv = []      # VGGT-local ref centers (for Var(s) estimate)
    hyp_Cw = []      # map-frame ref centers
    # Keep references with too few usable tracks as weak (inflated-covariance) hypotheses;
    # set UNIFIED_MAPFREE_KEEP_WEAK=0 to drop them instead.
    _keep_weak = str(os.environ.get("UNIFIED_MAPFREE_KEEP_WEAK", "1")).strip().lower() in {"1", "true", "yes", "on"}

    def _emit_weak(Xq_w, xr_w):
        h = build_reference_hypothesis(
            R_q_v=Rq, C_q_v=Cq_v, R_i_v=Ri_v, C_i_v=Ci_v,
            R_i_w=np.asarray(ref_R_w[k], np.float64), C_i_w=np.asarray(ref_C_w[k], np.float64),
            scale_s=scale_s, K_q=np.asarray(poses[0]["K"], np.float64), K_i=K_i,
            X_q_cam=Xq_w, x_ref=xr_w, conf=None, T_i_from_q_v=T_i_from_q, min_tracks=int(min_tracks),
        )
        hyps.append(h); hyp_Cv.append(Ci_v); hyp_Cw.append(np.asarray(ref_C_w[k], np.float64).reshape(3))

    for k, v in enumerate(ref_view_indices):
        Ri_v = np.asarray(poses[v]["Rcw"], np.float64).reshape(3, 3)
        ti_v = np.asarray(poses[v]["tcw"], np.float64).reshape(3)
        Ci_v = -Ri_v.T @ ti_v
        Ti = _w2c(Ri_v, ti_v)
        T_i_from_q = Ti @ np.linalg.inv(Tq)                 # query cam -> ref cam (VGGT local)
        K_i = np.asarray(poses[v]["K"], np.float64).reshape(3, 3)
        m = (vis[0] >= vis_thresh) & (vis[v] >= vis_thresh) & np.isfinite(Xq_cam_all).all(1) & (Xq_cam_all[:, 2] > 1e-6)
        if conf is not None and conf_thresh > 0:
            m = m & (conf[v] >= conf_thresh)
        idx = np.where(m)[0]
        if idx.size < int(min_tracks):
            if _keep_weak:
                _emit_weak(Xq_cam_all[idx], np.zeros((int(idx.size), 2)))
            continue
        Xq_cam = Xq_cam_all[idx]
        x_ref = _apply_xy_to_original(np.asarray(track_xy[v], np.float64)[idx], frame_xy_to_original, v)
        c = None if conf is None else np.clip(conf[v][idx], 0.0, None)
        Xref = (T_i_from_q @ np.c_[Xq_cam, np.ones(len(Xq_cam))].T).T[:, :3]
        ok = Xref[:, 2] > 1e-6
        if int(ok.sum()) < int(min_tracks):
            if _keep_weak:
                _emit_weak(Xq_cam[ok], x_ref[ok])
            continue
        rr = np.linalg.norm(x_ref[ok] - project(K_i, Xref[ok]), axis=1)
        resid_med.append(float(np.median(rr)))
        h = build_reference_hypothesis(
            R_q_v=Rq, C_q_v=Cq_v, R_i_v=Ri_v, C_i_v=Ci_v,            # Rcw (world->cam)
            R_i_w=np.asarray(ref_R_w[k], np.float64), C_i_w=np.asarray(ref_C_w[k], np.float64),
            scale_s=scale_s, K_q=np.asarray(poses[0]["K"], np.float64), K_i=K_i,
            X_q_cam=Xq_cam[ok], x_ref=x_ref[ok], conf=(None if c is None else c[ok]),
            T_i_from_q_v=T_i_from_q, min_tracks=min_tracks,
        )
        hyps.append(h)
        hyp_Cv.append(Ci_v); hyp_Cw.append(np.asarray(ref_C_w[k], np.float64).reshape(3))

    # common-mode Sim(3) scale variance from pairwise ref-center ratios:
    # s_ij = ||C_i^w - C_j^w|| / ||C_i^v - C_j^v|| ; Var(s) ~ (1.4826 MAD(s_ij))^2 / K.
    var_s = 0.0
    if len(hyp_Cv) >= 2:
        Cv = np.stack(hyp_Cv); Cw = np.stack(hyp_Cw); ratios = []
        for a in range(len(Cv)):
            for b in range(a + 1, len(Cv)):
                dv = float(np.linalg.norm(Cv[a] - Cv[b]))
                if dv > 1e-6:
                    ratios.append(float(np.linalg.norm(Cw[a] - Cw[b])) / dv)
        if len(ratios) >= 2:
            rr = np.asarray(ratios); mad = float(np.median(np.abs(rr - np.median(rr))))
            var_s = float((1.4826 * mad) ** 2 / max(len(hyp_Cv), 1))
    if diag is not None:
        diag["num_ref_hyps"] = len(hyps)
        diag["sanity_resid_median_px"] = float(np.median(resid_med)) if resid_med else None
        diag["var_s"] = var_s
    if len(hyps) < 2:
        return None
    res = student_t_center_consensus(
        np.stack([h.C_qi for h in hyps]), np.stack([h.Sigma_C for h in hyps]),
        nu=nu, lever_arms=np.stack([h.lever_u for h in hyps]), var_s=var_s,
    )
    if diag is not None:
        diag["C_cons"] = res.C_cons.tolist()
        diag["sigma_new"] = res.sigma_new          # posterior-covariance reliability
        diag["chi2_red"] = res.chi2_red
        # RMS dispersion of the covariance hypotheses
        Cq = np.stack([h.C_qi for h in hyps])
        diag["sigma_disp_cov"] = float(np.sqrt(np.mean(np.sum((Cq - res.C_cons) ** 2, axis=1))))
    return res
