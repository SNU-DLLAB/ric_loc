# Path, boolean, and image-orientation helper functions.
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image

from ricloc.constants import QUERY_ORIENTATION_DEFAULT, QUERY_ORIENTATION_VALUES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_bool(s: str) -> bool:
    x = str(s).strip().lower()
    if x in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {s}")


def _normalize_query_orientation(raw: Any) -> str:
    txt = str(raw).strip()
    if not txt:
        return QUERY_ORIENTATION_DEFAULT
    key = re.sub(r"[^a-z]", "", txt.lower())
    alias_map = {
        "original": "Original",
        "none": "Original",
        "norotate": "Original",
        "noorientation": "Original",
        "portrait": "Portrait",
        "landscapeleft": "LandscapeLeft",
        "landscaperight": "LandscapeRight",
        "portraitupsidedown": "PortraitUpsideDown",
    }
    normalized = alias_map.get(key)
    if normalized is None:
        raise ValueError(
            "orientation must be one of: "
            + ", ".join(QUERY_ORIENTATION_VALUES)
        )
    return normalized


def _apply_query_orientation_to_image(im: Image.Image, orientation: str) -> Tuple[Image.Image, bool, int]:
    transpose_ns = getattr(Image, "Transpose", Image)
    if orientation == "Original":
        return im, False, 0
    if orientation == "Portrait":
        rot = getattr(transpose_ns, "ROTATE_270", getattr(Image, "ROTATE_270"))
        return im.transpose(rot), True, 90
    if orientation == "LandscapeLeft":
        rot = getattr(transpose_ns, "ROTATE_270", getattr(Image, "ROTATE_270"))
        return im.transpose(rot), True, 90
    if orientation == "LandscapeRight":
        rot = getattr(transpose_ns, "ROTATE_90", getattr(Image, "ROTATE_90"))
        return im.transpose(rot), True, 270
    if orientation == "PortraitUpsideDown":
        rot = getattr(transpose_ns, "ROTATE_180", getattr(Image, "ROTATE_180"))
        return im.transpose(rot), True, 180
    return im, False, 0


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _normalize_path_str(p: Any) -> str:
    return str(p).replace("\\", "/")


def _safe_stem(p: Path) -> str:
    s = p.stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in s)


def _sorted_image_list_from_path(path: Path) -> List[Path]:
    if path.is_file():
        if not _is_image_file(path):
            raise ValueError(f"query_path is a file but not an image: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"query_path does not exist: {path}")

    imgs = [p for p in path.rglob("*") if _is_image_file(p)]
    if not imgs:
        raise FileNotFoundError(f"No images found under: {path}")
    imgs.sort(key=lambda x: (x.name.lower(), _normalize_path_str(x)))
    return imgs


def _sorted_reference_images(reference_dir: Path) -> List[Path]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference_dir does not exist: {reference_dir}")
    imgs = [p for p in reference_dir.rglob("*") if _is_image_file(p)]
    if not imgs:
        raise FileNotFoundError(f"No reference images under: {reference_dir}")
    imgs.sort(key=lambda x: (x.name.lower(), _normalize_path_str(x)))
    return imgs
