"""Smoke tests for RIC-Loc — no GPU / model / data required."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pure_python_imports():
    import ricloc.config, ricloc.geometry, ricloc.consensus, ricloc.gate            # noqa: F401
    import ricloc.mapfree_consensus, ricloc.mapfree_adapter, ricloc.evaluation     # noqa: F401
    from ricloc.config import FROZEN
    assert FROZEN.num_refs == 8 and FROZEN.topk == 20 and FROZEN.keep_weak is True
    assert FROZEN.env_flags["UNIFIED_MAPFREE_COV"] == "1"


def test_consensus_recovers_clean_geometry():
    """With perfect VGGT==COLMAP frames (scale 1, identity offset) the consensus pose must equal GT."""
    from ricloc.consensus import reference_consensus_pose
    from ricloc.geometry import qvec_to_rotmat

    rng = np.random.default_rng(0)
    # ground-truth query world->cam
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    Rq_w = qvec_to_rotmat(q)
    Cq_w = rng.normal(size=3)
    # references with known world->cam; VGGT-local frame == world frame (perfect)
    ref_hyps = []
    for _ in range(5):
        qr = rng.normal(size=4); qr /= np.linalg.norm(qr)
        Rw = qvec_to_rotmat(qr)
        Cw = rng.normal(size=3)
        ref_hyps.append({"R_v": Rw, "C_v": Cw, "R_w": Rw, "C_w": Cw, "weight": 1.0, "image_id": 1})
    out = reference_consensus_pose(ref_hyps, query_R_v=Rq_w, query_C_v=Cq_w, scale_s=1.0)
    assert out is not None
    C_err = float(np.linalg.norm(np.asarray(out["C"]) - Cq_w))
    dR = np.asarray(out["Rcw"]) @ Rq_w.T
    rot_err = np.degrees(np.arccos(np.clip(0.5 * (np.trace(dR) - 1.0), -1, 1)))
    assert C_err < 1e-9, f"center error {C_err}"
    assert rot_err < 1e-6, f"rotation error {rot_err} deg"


def test_sigjoint_gate():
    from ricloc.gate import calibrate_sigjoint, decide_sigjoint_selective_policy
    # synthetic batch: reliable queries have small sigma, unreliable large
    results = []
    for i in range(20):
        s = 0.05 if i < 16 else 0.5
        results.append({"mapfree_cov": {"status": "ok", "sigma_new": s, "sigma_disp_cov": s}})
    calib = calibrate_sigjoint(results, target_coverage=0.8)
    assert calib["median_cons"] > 0 and np.isfinite(calib["tau_accept"])
    d_good = decide_sigjoint_selective_policy(0.05, 0.05, **{k: calib[k] for k in
                ("median_cons", "median_disp", "tau_accept")})
    d_bad = decide_sigjoint_selective_policy(0.5, 0.5, **{k: calib[k] for k in
                ("median_cons", "median_disp", "tau_accept")})
    assert d_good["accepted"] is True
    assert d_bad["accepted"] is False


if __name__ == "__main__":
    test_pure_python_imports()
    test_consensus_recovers_clean_geometry()
    test_sigjoint_gate()
    print("smoke tests passed")
