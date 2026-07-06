from __future__ import annotations

import sys


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr)
