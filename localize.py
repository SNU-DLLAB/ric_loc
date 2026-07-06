#!/usr/bin/env python3
"""Localize query images against a prebuilt reference map and write poses.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `ricloc` importable

from ricloc.localizer import FrozenLocalizer
from ricloc.helpers import _sorted_image_list_from_path
from ricloc.gate import apply_gate


def main() -> None:
    ap = argparse.ArgumentParser(description="RIC-Loc frozen-method localization")
    ap.add_argument("--reference_root", required=True, help="map root (dense/images, dense/sparse, hloc_out/global)")
    ap.add_argument("--vggt_ckpt", required=True, help="VGGT checkpoint path (e.g. models/vggt_1B.pt)")
    ap.add_argument("--query_path", required=True, help="query image file or folder")
    ap.add_argument("--out", default="poses.json", help="output JSON path")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--sigjoint_calib", default=None, help="optional sigjoint calib JSON to apply the selective gate")
    args = ap.parse_args()

    query_path = Path(args.query_path).expanduser()
    queries = _sorted_image_list_from_path(query_path) if query_path.is_dir() else [query_path]

    loc = FrozenLocalizer(reference_root=args.reference_root, vggt_ckpt=args.vggt_ckpt, device=args.device)
    print(f"[ricloc] loaded map={loc.reference_dir} colmap={loc.colmap_dir} device={loc.device} "
          f"refs={len(loc.reference_images)} | localizing {len(queries)} query(ies)")

    results = []
    for i, q in enumerate(queries):
        r = loc.localize(q)
        results.append(r)
        src = r.get("final_pose_source", r.get("status"))
        mf = (r.get("mapfree_cov") or {})
        print(f"  [{i+1}/{len(queries)}] {Path(q).name}: {r['status']} src={src} "
              f"sigma_new={mf.get('sigma_new')}")

    calib = None
    if args.sigjoint_calib:
        calib = json.loads(Path(args.sigjoint_calib).read_text())
        apply_gate(results, calib)
        n_acc = sum(1 for r in results if r.get("accepted"))
        print(f"[ricloc] selective gate applied: {n_acc}/{len(results)} accepted "
              f"(tau_accept={calib.get('tau_accept'):.4f})")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))
    print(f"[ricloc] wrote {args.out}")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


if __name__ == "__main__":
    main()
