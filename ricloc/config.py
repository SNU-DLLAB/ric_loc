"""Pipeline configuration.

apply_env() exports two environment flags:
    UNIFIED_MAPFREE_COV=1        use covariance as the center + posterior sigma
    UNIFIED_MAPFREE_KEEP_WEAK=1  keep low-track refs as weak hypotheses

Pose convention (COLMAP): x_c = R_cw x_w + t_cw ; C_w = -R_cw^T t_cw. Output quaternions wxyz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class FrozenConfig:
    # retrieval + reference selection
    topk: int = 20                       # MegaLoc retrieval depth
    num_refs: int = 8                    # references kept for the VGGT forward pass
    align_type: str = "Sim3"             # VGGT-local -> COLMAP-world similarity transform

    # Sim(3) RANSAC alignment
    align_ransac_iters: int = 2000
    align_inlier_thresh_m: float = 0.30
    align_min_inliers: int = 5
    align_fallback_ransac_iters: int = 6000
    align_fallback_inlier_thresh_m: float = 0.40
    align_fallback_min_inliers: int = 4
    align_seed: int = 42

    # map-point-free covariance consensus
    mapfree_vis_thresh: float = 0.5      # per-track visibility gate
    mapfree_conf_thresh: float = 0.0     # track confidence gate (0 = off)
    mapfree_min_tracks: int = 6          # below this a reference becomes a weak hypothesis
    mapfree_nu: float = 5.0              # Student-t robustness dof
    keep_weak: bool = True               # keep low-track refs as weak hypotheses

    # consensus weight
    weight_sigma_base: float = 0.05      # base variance floor

    # selective gate
    gate_target_coverage: float = 0.8    # target accept fraction

    # evaluation success thresholds per dataset (translation_m, rotation_deg)
    success_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "12scenes":  (0.05, 5.0),    # strict indoor
        "7scenes":   (0.05, 5.0),    # strict indoor
        "cambridge": (0.25, 2.0),    # strict outdoor landmark
        "naver":     (0.25, 2.0),    # large indoor
        "aachen":    (0.50, 5.0),    # city-scale day-night
    })

    # env flags
    env_flags: Dict[str, str] = field(default_factory=lambda: {
        "UNIFIED_MAPFREE_COV": "1",
        "UNIFIED_MAPFREE_KEEP_WEAK": "1",
    })

    def apply_env(self) -> None:
        """Export the env flags; values already set in the environment are respected."""
        for k, v in self.env_flags.items():
            os.environ.setdefault(k, v)

    def success_threshold_for(self, dataset: str) -> Tuple[float, float]:
        key = str(dataset or "").strip().lower()
        for name, thr in self.success_thresholds.items():
            if name in key:
                return thr
        return (0.5, 5.0)


FROZEN = FrozenConfig()
