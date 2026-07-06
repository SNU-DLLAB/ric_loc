#!/usr/bin/env python3
"""Preflight check: verify the environment, models, and optional reference map are ready."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, BAD, WARN = "  [ok] ", "  [MISSING] ", "  [warn] "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_root", default=None)
    ap.add_argument("--vggt_ckpt", default=os.environ.get("VGGT_CKPT", ""))
    args = ap.parse_args()
    problems = 0

    print("== python packages ==")
    for mod in ("numpy", "torch", "h5py", "PIL"):
        try:
            m = importlib.import_module(mod)
            print(OK + f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(BAD + f"{mod}: {e}"); problems += 1
    try:
        import torch
        print(OK + f"cuda available: {torch.cuda.is_available()}")
    except Exception:
        pass

    print("== ricloc package ==")
    try:
        import ricloc
        from ricloc.localizer import FrozenLocalizer  # noqa: F401
        print(OK + f"ricloc {ricloc.__version__} imports cleanly")
    except Exception as e:
        print(BAD + f"ricloc import failed: {e}"); problems += 1

    print("== VGGT source tree ==")
    src = os.environ.get("RICLOC_VGGT_SRC", "")
    cands = ([Path(src)] if src else []) + [
        Path(os.environ.get("RICLOC_REPO_ROOT", ".")) / "thirdparty" / "vggt",
        Path(os.environ.get("RICLOC_REPO_ROOT", ".")) / "vggt", Path("vggt")]
    found = next((c for c in cands if c.exists()), None)
    if found:
        print(OK + f"vggt source: {found}")
    else:
        print(BAD + f"vggt source not found (set RICLOC_VGGT_SRC). searched: {[str(c) for c in cands]}"); problems += 1

    print("== checkpoints ==")
    ck = Path(args.vggt_ckpt) if args.vggt_ckpt else None
    if ck and ck.exists():
        print(OK + f"VGGT ckpt: {ck}")
    else:
        print((BAD if ck else WARN) + f"VGGT ckpt: {ck or '(pass --vggt_ckpt or set VGGT_CKPT)'}")
        if ck:
            problems += 1
    models_dir = Path(os.environ.get("RICLOC_MODELS_DIR", "models"))
    megaloc = models_dir / "megaloc_model.safetensors"
    print((OK if megaloc.exists() else WARN) + f"MegaLoc weights: {megaloc} "
          f"({'present' if megaloc.exists() else 'auto-download at first run unless disabled'})")

    if args.reference_root:
        print("== reference map ==")
        try:
            from ricloc.localizer import (find_reference_image_dir, find_colmap_dir, find_retrieval_features)
            rr = Path(args.reference_root).expanduser().resolve()
            print(OK + f"images: {find_reference_image_dir(rr)}")
            print(OK + f"colmap: {find_colmap_dir(rr)}")
            print(OK + f"retrieval h5: {find_retrieval_features(rr)}")
        except Exception as e:
            print(BAD + f"reference map incomplete: {e}"); problems += 1

    print()
    if problems:
        print(f"FAILED: {problems} hard requirement(s) missing.")
        sys.exit(1)
    print("All hard requirements satisfied.")


if __name__ == "__main__":
    main()
