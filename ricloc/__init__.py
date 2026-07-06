"""RIC-Loc: Reference-Induced Consensus for Selective Posed-Reference Visual Localization.

Public entry points:
    ricloc.localizer.FrozenLocalizer              - load model+map once, localize queries
    ricloc.gate.calibrate_sigjoint                - build the selective-gate calibration artifact
    ricloc.gate.decide_sigjoint_selective_policy  - apply the gate at runtime
"""

from .config import FROZEN, FrozenConfig  # noqa: F401

__all__ = ["FROZEN", "FrozenConfig"]
__version__ = "1.0.0"
