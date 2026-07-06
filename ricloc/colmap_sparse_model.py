from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ricloc.geometry import camera_center_from_world_to_camera, qvec_to_rotmat
from ricloc.helpers import _normalize_path_str
from ricloc.logging_utils import _info


def map_vggt_ref_to_map_candidates(vggt_name: str, camera_id_hint: Optional[int] = None) -> List[str]:
    nm = vggt_name.replace("\\", "/")

    m = re.match(r"^([1-9])_(\d+\.[A-Za-z0-9]+)$", nm)
    if m:
        rig = int(m.group(1))
        frame = m.group(2)
        return [f"origin_{rig}/{frame}", f"origin{rig}/{frame}", nm]

    m = re.match(r"^origin_?([1-9])/(.+)$", nm)
    if m:
        rig = int(m.group(1))
        frame = m.group(2)
        return [f"origin_{rig}/{frame}", f"origin{rig}/{frame}", nm]

    m = re.match(r"^(\d+\.[A-Za-z0-9]+)$", nm)
    if m and camera_id_hint is not None and camera_id_hint >= 1:
        frame = m.group(1)
        return [f"origin_{camera_id_hint}/{frame}", f"origin{camera_id_hint}/{frame}", nm]

    return [nm]


class ColmapSparseModel:
    def __init__(self, sparse_dir: Path):
        self.sparse_dir = sparse_dir
        self.cameras: Dict[int, Any] = {}
        self.images: Dict[int, Any] = {}
        self.points3D: Dict[int, Any] = {}
        self.image_centers: Dict[int, np.ndarray] = {}

        self.image_name_to_id: Dict[str, int] = {}
        self.basename_to_ids: Dict[str, List[int]] = {}
        self._load()

    def _load(self) -> None:
        from ricloc.colmap_read_write_model import read_model  # type: ignore

        cams, imgs, pts = read_model(str(self.sparse_dir))
        self.cameras = cams
        self.images = imgs
        self.points3D = pts

        for image_id, im in self.images.items():
            norm = _normalize_path_str(im.name).lstrip("./")
            self.image_name_to_id[norm] = image_id
            base = Path(norm).name
            self.basename_to_ids.setdefault(base, []).append(image_id)
            Rcw = qvec_to_rotmat(np.asarray(im.qvec, dtype=np.float64))
            tcw = np.asarray(im.tvec, dtype=np.float64)
            self.image_centers[image_id] = camera_center_from_world_to_camera(Rcw, tcw)

        _info(
            f"Loaded COLMAP model: dir={self.sparse_dir}, "
            f"cams={len(self.cameras)}, images={len(self.images)}, points3D={len(self.points3D)}"
        )

    def resolve_ref_image_id(self, ref_path: Path, reference_dir: Path) -> Optional[int]:
        full_norm = _normalize_path_str(ref_path).lstrip("./")
        rel_norm = full_norm
        if ref_path.is_relative_to(reference_dir):
            rel_norm = _normalize_path_str(ref_path.relative_to(reference_dir)).lstrip("./")
        base = ref_path.name

        candidates = [
            rel_norm,
            full_norm,
            base,
        ]

        for c in candidates:
            if c in self.image_name_to_id:
                return self.image_name_to_id[c]

        for name, image_id in self.image_name_to_id.items():
            if name.endswith(rel_norm):
                return image_id
            if name.endswith(base):
                return image_id

        ids = self.basename_to_ids.get(base, [])
        if len(ids) == 1:
            return ids[0]
        return None

    def resolve_ref_image_id_with_vggt_name(
        self,
        ref_path: Path,
        reference_dir: Path,
        vggt_name: str,
        camera_id_hint: Optional[int],
    ) -> Optional[int]:
        image_id = self.resolve_ref_image_id(ref_path, reference_dir)
        if image_id is not None:
            return image_id

        for cand in map_vggt_ref_to_map_candidates(vggt_name, camera_id_hint):
            cand_norm = _normalize_path_str(cand).lstrip("./")
            if cand_norm in self.image_name_to_id:
                return self.image_name_to_id[cand_norm]
            for name, iid in self.image_name_to_id.items():
                if name.endswith(cand_norm):
                    return iid

        return None
