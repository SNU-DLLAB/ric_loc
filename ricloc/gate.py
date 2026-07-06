"""Calibrated selective gate.

  calibrate_sigjoint(results, target_coverage)  -> calibration artifact
      median_cons = median of mapfree_cov.sigma_new
      median_disp = median of mapfree_cov.sigma_disp_cov
      tau_accept  = quantile of sigma_joint at the target accept fraction

  decide_sigjoint_selective_policy(...)  -> accept / reject per query
      sigma_joint = max(sigma_cons/median_cons, sigma_disp/median_disp)
      accept iff sigma_joint <= tau_accept
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return default
    return float(v) if np.isfinite(v) else default


def collect_sigmas(results: List[Dict[str, Any]]):
    """Extract (sigma_new, sigma_disp_cov) over queries whose mapfree_cov.status == 'ok'."""
    sc: List[float] = []
    sd: List[float] = []
    for r in results:
        mc = r.get("mapfree_cov") or {}
        if mc.get("status") != "ok":
            continue
        a, b = mc.get("sigma_new"), mc.get("sigma_disp_cov")
        if a is None or b is None:
            continue
        sc.append(float(a))
        sd.append(float(b))
    return np.asarray(sc, dtype=np.float64), np.asarray(sd, dtype=np.float64)


def calibrate_sigjoint(
    results: List[Dict[str, Any]],
    target_coverage: float = 0.8,
    source: str = "",
) -> Dict[str, Any]:
    """Build the runtime sigma_joint calibration artifact from a calibration set's results."""
    sc, sd = collect_sigmas(results)
    if len(sc) < 2:
        raise ValueError(f"too few calibration queries with mapfree_cov.status=='ok' ({len(sc)})")
    mc, md = float(np.median(sc)), float(np.median(sd))
    sj = np.maximum(sc / mc, sd / md)
    tau_accept = float(np.quantile(sj, float(target_coverage)))      # accept lowest-sigma_joint first
    return {
        "method": "sigjoint",
        "eq": "selective_joint",
        "median_cons": mc,
        "median_disp": md,
        "tau_accept": tau_accept,
        "target_coverage": float(target_coverage),
        "n_calib": int(len(sc)),
        "source": str(source),
    }


def decide_sigjoint_selective_policy(
    sigma_cons: Any,
    sigma_disp: Any,
    median_cons: float,
    median_disp: float,
    tau_accept: float,
) -> Dict[str, Any]:
    """Apply the sigma_joint policy to one query. Lower sigma_joint = more reliable.

    sigma_joint = max(sigma_cons/median_cons, sigma_disp/median_disp); accept iff <= tau_accept.
    """
    sc = _finite_float(sigma_cons, default=None)
    sd = _finite_float(sigma_disp, default=None)
    mc = _finite_float(median_cons, default=None)
    md = _finite_float(median_disp, default=None)
    ta = float(tau_accept)
    base = {"sigma_joint": None, "median_cons": mc, "median_disp": md,
            "tau_accept": ta}
    if sc is None or sd is None or mc is None or md is None or mc <= 0.0 or md <= 0.0:
        return {**base, "accepted": False, "decision": "reject", "reason": "sigjoint_inputs_unavailable"}
    sj = max(sc / mc, sd / md)
    base["sigma_joint"] = float(sj)
    if sj <= ta:
        return {**base, "accepted": True, "decision": "accept_current", "reason": ""}
    return {**base, "accepted": False, "decision": "reject",
            "reason": f"sigjoint_above_threshold:{sj:.4f}>{ta:.4f}"}


def apply_gate(results: List[Dict[str, Any]], calib: Dict[str, Any]) -> None:
    """Annotate each result row in-place with the sigjoint decision under the given calibration."""
    for r in results:
        mc = r.get("mapfree_cov") or {}
        d = decide_sigjoint_selective_policy(
            mc.get("sigma_new"), mc.get("sigma_disp_cov"),
            median_cons=calib.get("median_cons", 0.0), median_disp=calib.get("median_disp", 0.0),
            tau_accept=calib.get("tau_accept", float("inf")),
        )
        r["selective_sigma_joint"] = d.get("sigma_joint")
        r["selective_decision"] = d.get("decision")
        r["selective_decision_reason"] = d.get("reason")
        r["accepted"] = bool(d.get("accepted", False))
        # selective gate ranks by reliability = -sigma_joint (higher score = more reliable)
        sj = d.get("sigma_joint")
        r["reliability_score"] = float(-sj) if sj is not None else None
