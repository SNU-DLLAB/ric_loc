"""Sanity tests for ricloc/mapfree_consensus.py (map-point-free uncertainty + FD Jacobian)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ricloc.mapfree_consensus import (
    skew, exp_se3, exp_so3, lift_rotation, lift_center, project,
    reprojection_jacobian_analytic, build_reference_hypothesis,
    student_t_center_consensus,
)

rng = np.random.RandomState(0)
PASS = []


def check(name, cond, extra=""):
    PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


def K_default():
    return np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


# ---- finite-difference Jacobian ----
def test_jacobian_fd():
    print("test_jacobian_fd")
    K = K_default()
    T = np.eye(4); T[:3, :3] = exp_so3([0.1, -0.2, 0.05]); T[:3, 3] = [0.3, -0.1, 0.2]
    Xq = np.array([0.4, -0.3, 2.5])
    Xh = np.append(Xq, 1.0)
    X_ref = (T @ Xh)[:3]
    x_obs = project(K, X_ref) + np.array([0.5, -0.4])  # arbitrary observed pixel

    def resid(xi):
        Xr = (exp_se3(xi) @ T @ Xh)[:3]
        return x_obs - project(K, Xr)

    J_an = reprojection_jacobian_analytic(K, X_ref)
    J_num = np.zeros((2, 6)); d = 1e-6
    for k in range(6):
        e = np.zeros(6); e[k] = d
        J_num[:, k] = (resid(e) - resid(-e)) / (2 * d)
    err = float(np.max(np.abs(J_an - J_num)))
    check("analytic vs finite-diff Jacobian", err < 1e-4, f"max|dJ|={err:.2e}")


# ---- lift-center covariance Jacobian (FD-verified) ----
def test_lift_jacobian_fd():
    print("test_lift_jacobian_fd")
    R_i_v = exp_so3([0.3, -0.4, 0.1]); R_i_w = exp_so3([-0.2, 0.15, 0.5])  # distinct R_i^v and R_i^w
    C_q_v = np.array([0.7, -0.2, 0.4]); C_i_v = np.array([-0.3, 0.5, -0.1]); C_i_w = np.array([1.0, 2.0, -0.5])
    s = 1.3
    M = R_i_w.T @ R_i_v
    u_i = M @ (C_q_v - C_i_v)                      # lever arm = (R_i^w)^T t_{i<-q}^v
    t_iq = R_i_v @ (C_q_v - C_i_v)                 # relative translation in ref-cam frame
    G = s * np.hstack([R_i_w.T, -skew(u_i) @ R_i_w.T])   # the implemented full Jacobian

    def lifted_center(xi):
        tp = (exp_se3(xi) @ np.append(t_iq, 1.0))[:3]
        return C_i_w + s * (R_i_w.T @ tp)

    G_num = np.zeros((3, 6)); d = 1e-6
    for k in range(6):
        e = np.zeros(6); e[k] = d
        G_num[:, k] = (lifted_center(e) - lifted_center(-e)) / (2 * d)
    err = float(np.max(np.abs(G - G_num)))
    check("full lift Jacobian vs finite-diff (R_i^v != R_i^w)", err < 1e-5, f"max|dG|={err:.2e}")
    G_old = np.hstack([s * M, np.zeros((3, 3))])
    check("old s*M form is provably wrong here", float(np.max(np.abs(G_old - G_num))) > 0.1,
          f"|G_old-G_num|={float(np.max(np.abs(G_old - G_num))):.2f}")


# ---- lift convention: if VGGT frame == world frame and s=1, lift recovers the true query pose ----
def test_lift_identity():
    print("test_lift_identity")
    R_q_w = exp_so3([0.2, 0.1, -0.3]); C_q_w = np.array([1.0, 2.0, 3.0])
    R_i_w = exp_so3([-0.1, 0.05, 0.2]); C_i_w = np.array([0.5, -1.0, 2.0])
    # VGGT frame == world frame
    C_qi = lift_center(C_q_w, C_i_w, R_i_w, R_i_w, C_i_w, 1.0)
    R_qi = lift_rotation(R_q_w, R_i_w, R_i_w)
    check("lift_center recovers query center", np.allclose(C_qi, C_q_w, atol=1e-9))
    check("lift_rotation recovers query rotation", np.allclose(R_qi, R_q_w, atol=1e-9))


def _synthetic_ref_hyps(K_refs, center_noise=0.02, parallax=True, outlier_idx=None, common_shift=None,
                        n_tracks=40, seed=1):
    """Build K reference hypotheses around a known query center via the real builder, map-point-free.
    World==VGGT frame, s=1. Returns (C_true, hyps)."""
    r = np.random.RandomState(seed)
    K = K_default()
    C_q = np.array([0.0, 0.0, 0.0]); R_q = np.eye(3)
    hyps = []
    for i in range(K_refs):
        # reference placed around the query, looking toward it
        ang = 2 * np.pi * i / K_refs
        rad = 2.0
        C_i = np.array([rad * np.cos(ang), rad * np.sin(ang), -3.0])
        R_i = exp_so3(r.randn(3) * 0.1)
        # parallax=False -> far tracks (low parallax) -> large center covariance
        z_lo, z_hi = (2.0, 5.0) if parallax else (80.0, 120.0)
        Xq = np.stack([r.uniform(-1, 1, n_tracks), r.uniform(-1, 1, n_tracks), r.uniform(z_lo, z_hi, n_tracks)], 1)
        # relative transform query-cam -> ref-cam (world==vggt): T_i T_q^-1 with T = [R| -R C]
        Tq = np.eye(4); Tq[:3, :3] = R_q; Tq[:3, 3] = -R_q @ C_q
        Ti = np.eye(4); Ti[:3, :3] = R_i; Ti[:3, 3] = -R_i @ C_i
        T_i_from_q = Ti @ np.linalg.inv(Tq)
        # observed ref pixels = projection + pixel noise
        x_ref = np.zeros((n_tracks, 2))
        for j in range(n_tracks):
            Xr = (T_i_from_q @ np.append(Xq[j], 1.0))[:3]
            x_ref[j] = project(K, Xr) + r.randn(2) * 0.5
        # inject a wrong reference center to create an outlier hypothesis
        C_i_w = C_i.copy()
        if outlier_idx is not None and i == outlier_idx:
            C_i_w = C_i + np.array([5.0, -4.0, 3.0])
        if common_shift is not None:
            C_i_w = C_i + common_shift  # all refs biased same way (common-mode)
        h = build_reference_hypothesis(
            R_q_v=R_q, C_q_v=C_q, R_i_v=R_i, C_i_v=C_i, R_i_w=R_i, C_i_w=C_i_w, scale_s=1.0,
            K_q=K, K_i=K, X_q_cam=Xq, x_ref=x_ref, conf=None, T_i_from_q_v=T_i_from_q,
        )
        hyps.append(h)
    C_true = C_q if common_shift is None else C_q  # truth is still query center
    return C_true, hyps


# ---- 2. synthetic perfect ----
def test_perfect():
    print("test_perfect")
    for K_refs in (4, 8, 16):
        C_true, hyps = _synthetic_ref_hyps(K_refs, seed=2)
        res = student_t_center_consensus(np.stack([h.C_qi for h in hyps]),
                                         np.stack([h.Sigma_C for h in hyps]))
        err = float(np.linalg.norm(res.C_cons - C_true))
        if K_refs == 4:
            sig4 = res.sigma_new; err4 = err
        if K_refs == 16:
            check("C_cons accurate (16 refs)", err < 0.15, f"err={err:.3f}m chi2_red={res.chi2_red:.2f}")
            check("sigma_new decreases with K", res.sigma_new < sig4 + 1e-9,
                  f"sig(4)={sig4:.3f} sig(16)={res.sigma_new:.3f}")
        check(f"PSD Sigma_C (K={K_refs})", all(np.linalg.eigvalsh(h.Sigma_C)[0] > -1e-9 for h in hyps))


# ---- 3. degenerate geometry ----
def test_degenerate():
    print("test_degenerate")
    _, good = _synthetic_ref_hyps(8, parallax=True, seed=3)
    _, deg = _synthetic_ref_hyps(8, parallax=False, seed=3)
    sig_good = np.median([np.sqrt(np.linalg.eigvalsh(h.Sigma_C)[-1]) for h in good if not h.weak])
    sig_deg = np.median([np.sqrt(np.linalg.eigvalsh(h.Sigma_C)[-1]) for h in deg if not h.weak])
    check("degenerate geometry -> larger center covariance", sig_deg > sig_good,
          f"good={sig_good:.3f} deg={sig_deg:.3f}")


# ---- 4. outlier reference ----
def test_outlier():
    print("test_outlier")
    C_true, hyps = _synthetic_ref_hyps(8, outlier_idx=3, seed=4)
    res = student_t_center_consensus(np.stack([h.C_qi for h in hyps]),
                                     np.stack([h.Sigma_C for h in hyps]))
    err = float(np.linalg.norm(res.C_cons - C_true))
    check("outlier down-weighted (small omega)", res.omega[3] < np.median(res.omega),
          f"omega_outlier={res.omega[3]:.3f} median={np.median(res.omega):.3f}")
    check("C_cons stays stable under 1 outlier", err < 0.5, f"err={err:.3f}m")


# ---- common-mode bias ----
def test_common_mode():
    print("test_common_mode")
    shift = np.array([0.6, -0.4, 0.3])
    C_true, hyps = _synthetic_ref_hyps(8, common_shift=shift, seed=5)
    res = student_t_center_consensus(np.stack([h.C_qi for h in hyps]),
                                     np.stack([h.Sigma_C for h in hyps]))
    err = float(np.linalg.norm(res.C_cons - C_true))
    # error ~ shift magnitude but chi2_red stays small -> sigma underestimates
    check("common-mode bias produces real error", err > 0.3, f"err={err:.3f}m (~shift {np.linalg.norm(shift):.3f})")
    check("common-mode NOT flagged by chi2 (documented limitation)", res.chi2_red < 3.0,
          f"chi2_red={res.chi2_red:.2f} sigma_new={res.sigma_new:.3f} << err -> known blind spot")


def test_scale_variance():
    """Common-mode Sim(3) scale variance must inflate sigma_new (reliability) WITHOUT moving C_cons."""
    print("test_scale_variance")
    C_true, hyps = _synthetic_ref_hyps(8, seed=9)
    C = np.stack([h.C_qi for h in hyps]); S = np.stack([h.Sigma_C for h in hyps]); U = np.stack([h.lever_u for h in hyps])
    r0 = student_t_center_consensus(C, S, lever_arms=U, var_s=0.0)
    r1 = student_t_center_consensus(C, S, lever_arms=U, var_s=1e-2)
    check("scale variance inflates sigma_new", r1.sigma_new > r0.sigma_new + 1e-9,
          f"sig {r0.sigma_new:.4f}->{r1.sigma_new:.4f}")
    check("scale variance does NOT move C_cons", float(np.linalg.norm(r0.C_cons - r1.C_cons)) < 1e-9)


def test_adapter_synthetic():
    """End-to-end glue check: synthetic VGGT-like outputs (world==VGGT frame, s=1) -> adapter ->
    C_cons must recover the true query center and the sanity reprojection residual must be ~0."""
    print("test_adapter_synthetic")
    from ricloc.mapfree_adapter import mapfree_consensus_from_vggt
    K = K_default(); Hm, Wm = 480, 640
    r = np.random.RandomState(7)
    C_q = np.array([0.3, -0.2, 0.1]); R_q = exp_so3([0.05, -0.03, 0.02])  # query Rcw, world->cam
    tq = -R_q @ C_q
    Xw = np.stack([r.uniform(-1.5, 1.5, 200), r.uniform(-1.5, 1.5, 200), r.uniform(3.0, 7.0, 200)], 1)

    def proj(Rcw, tcw, X):
        Xc = (Rcw @ X.T).T + tcw[None, :]
        return project(K, Xc), Xc[:, 2]

    qxy, qz = proj(R_q, tq, Xw)
    inb = (qxy[:, 0] > 1) & (qxy[:, 0] < Wm - 2) & (qxy[:, 1] > 1) & (qxy[:, 1] < Hm - 2) & (qz > 0.1)
    Xw, qxy = Xw[inb], qxy[inb]
    T = len(Xw)
    # pointmap[0]: fill world point at each query track pixel
    P = np.full((1, Hm, Wm, 3), np.nan)  # only query view used by adapter
    qx = np.clip(np.round(qxy[:, 0]).astype(int), 0, Wm - 1); qy = np.clip(np.round(qxy[:, 1]).astype(int), 0, Hm - 1)
    P[0, qy, qx] = Xw
    Vviews = 1 + 6
    P = np.concatenate([P, np.zeros((Vviews - 1, Hm, Wm, 3))], 0)
    track_xy = np.zeros((Vviews, T, 2)); track_xy[0] = qxy
    track_vis = np.ones((Vviews, T))
    poses = [{"Rcw": R_q, "tcw": tq, "K": K}]
    ref_R_w = []; ref_C_w = []; ref_idx = []
    for i in range(6):
        ang = 2 * np.pi * i / 6
        C_i = np.array([1.5 * np.cos(ang), 1.5 * np.sin(ang), -2.0]); R_i = exp_so3(r.randn(3) * 0.1)
        ti = -R_i @ C_i
        rxy, _ = proj(R_i, ti, Xw)
        track_xy[1 + i] = rxy
        poses.append({"Rcw": R_i, "tcw": ti, "K": K})
        ref_R_w.append(R_i); ref_C_w.append(C_i); ref_idx.append(1 + i)
    diag = {}
    res = mapfree_consensus_from_vggt(
        poses=poses, points_vggt=P, track_xy=track_xy, track_vis=track_vis, track_conf=None,
        ref_R_w=ref_R_w, ref_C_w=ref_C_w, ref_view_indices=ref_idx, scale_s=1.0, diag=diag,
    )
    check("adapter returns consensus", res is not None)
    if res is not None:
        _sr = diag.get("sanity_resid_median_px")
        check("adapter sanity residual ~0 (convention OK)", (_sr if _sr is not None else 9) < 1e-3,
              f"resid={_sr}")
        check("adapter recovers true query center", float(np.linalg.norm(res.C_cons - C_q)) < 0.05,
              f"err={float(np.linalg.norm(res.C_cons - C_q)):.4f}m")


if __name__ == "__main__":
    test_jacobian_fd(); test_lift_jacobian_fd(); test_lift_identity(); test_perfect()
    test_degenerate(); test_outlier(); test_common_mode(); test_scale_variance(); test_adapter_synthetic()
    n = len(PASS); p = sum(PASS)
    print(f"\n=== {p}/{n} checks passed ===")
    sys.exit(0 if p == n else 1)
