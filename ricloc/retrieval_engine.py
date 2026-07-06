from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ricloc.helpers import _parse_bool
from ricloc.logging_utils import _info, _warn

TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    import torchvision.transforms as T
except Exception as _e:
    torch = None  # type: ignore
    T = None  # type: ignore
    TORCH_IMPORT_ERROR = str(_e)

H5PY_IMPORT_ERROR: Optional[str] = None
try:
    import h5py  # type: ignore
except Exception as _e:
    h5py = None  # type: ignore
    H5PY_IMPORT_ERROR = str(_e)

FAISS_IMPORT_ERROR: Optional[str] = None
try:
    import faiss  # type: ignore
except Exception as _e:
    faiss = None  # type: ignore
    FAISS_IMPORT_ERROR = str(_e)

SAFETENSORS_IMPORT_ERROR: Optional[str] = None
try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception as _e:
    safetensors_load_file = None  # type: ignore
    SAFETENSORS_IMPORT_ERROR = str(_e)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MEGALOC_LOCAL_MODEL_PY = PACKAGE_ROOT / "retrieval" / "megaloc_model.py"
MEGALOC_LOCAL_WEIGHTS = MODELS_DIR / "megaloc_model.safetensors"
DEFAULT_MEGALOC_WEIGHTS_URL = "https://huggingface.co/gberton/MegaLoc/resolve/main/model.safetensors?download=true"
MEGALOC_WEIGHTS_URL_ENV = "UNIFIED_VGGT_MEGALOC_WEIGHTS_URL"
MEGALOC_AUTO_DOWNLOAD_ENV = "UNIFIED_VGGT_MEGALOC_AUTO_DOWNLOAD"

RETRIEVAL_RESIZE_MAX = 1024
RETRIEVAL_BATCH_SIZE_DB = 1
USE_AMP = True
def _download_to_path(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "visual-localization/1.0"})
    try:
        with urllib.request.urlopen(req) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f)
        tmp.replace(dst)
    finally:
        with contextlib.suppress(FileNotFoundError):
            if tmp.exists():
                tmp.unlink()


def _ensure_megaloc_weights() -> None:
    if MEGALOC_LOCAL_WEIGHTS.is_file() and MEGALOC_LOCAL_WEIGHTS.stat().st_size > 0:
        return

    auto_download = _parse_bool(os.environ.get(MEGALOC_AUTO_DOWNLOAD_ENV, "1"))
    if not auto_download:
        raise FileNotFoundError(
            f"MegaLoc weights not found: {MEGALOC_LOCAL_WEIGHTS} "
            f"(enable auto download with {MEGALOC_AUTO_DOWNLOAD_ENV}=1, "
            f"or set custom URL via {MEGALOC_WEIGHTS_URL_ENV})"
        )

    url = os.environ.get(MEGALOC_WEIGHTS_URL_ENV, DEFAULT_MEGALOC_WEIGHTS_URL).strip()
    if not url:
        raise RuntimeError(
            f"MegaLoc weights URL is empty. Set {MEGALOC_WEIGHTS_URL_ENV}."
        )

    _info(f"MegaLoc weights not found; downloading from {url}")
    try:
        _download_to_path(url, MEGALOC_LOCAL_WEIGHTS)
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Failed to download MegaLoc weights "
            f"(url={url}, reason={e}). "
            f"Set local file at {MEGALOC_LOCAL_WEIGHTS} or override URL with {MEGALOC_WEIGHTS_URL_ENV}."
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Failed to prepare MegaLoc weights "
            f"(target={MEGALOC_LOCAL_WEIGHTS}): {e}"
        ) from e

    if not MEGALOC_LOCAL_WEIGHTS.is_file() or MEGALOC_LOCAL_WEIGHTS.stat().st_size == 0:
        raise RuntimeError(f"Downloaded MegaLoc weights file is invalid: {MEGALOC_LOCAL_WEIGHTS}")
    _info(
        "MegaLoc weights ready: "
        f"{MEGALOC_LOCAL_WEIGHTS} ({MEGALOC_LOCAL_WEIGHTS.stat().st_size / (1024 ** 2):.1f} MB)"
    )


class RetrievalEngine:
    def __init__(
        self,
        reference_images: List[Path],
        device: str,
        reference_dir: Optional[Path] = None,
        precomputed_features_path: Optional[Path] = None,
        require_precomputed_features: bool = False,
    ):
        self.reference_images = reference_images
        self.device = device
        self.reference_dir = reference_dir
        self.precomputed_features_path = precomputed_features_path
        self.require_precomputed_features = bool(require_precomputed_features)
        self.available = False
        self.error_message = ""
        self.model = None
        self.db_desc: Optional[np.ndarray] = None
        self.faiss_index = None
        self.score_type = "l2_sq"

        self.transform = None
        if T is not None:
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self._preload()

    def _preload(self) -> None:
        try:
            if torch is None or self.transform is None:
                raise RuntimeError(f"torch/torchvision unavailable: {TORCH_IMPORT_ERROR}")
            if safetensors_load_file is None:
                raise RuntimeError(f"safetensors unavailable: {SAFETENSORS_IMPORT_ERROR}")
            if not MEGALOC_LOCAL_MODEL_PY.is_file():
                raise FileNotFoundError(f"MegaLoc model code not found: {MEGALOC_LOCAL_MODEL_PY}")
            _ensure_megaloc_weights()

            spec = importlib.util.spec_from_file_location("local_megaloc_model", MEGALOC_LOCAL_MODEL_PY)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load MegaLoc module spec: {MEGALOC_LOCAL_MODEL_PY}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            megaloc_cls = getattr(module, "MegaLoc", None)
            if megaloc_cls is None:
                raise AttributeError(f"'MegaLoc' class not found in {MEGALOC_LOCAL_MODEL_PY}")

            self.model = megaloc_cls()
            state = safetensors_load_file(str(MEGALOC_LOCAL_WEIGHTS), device="cpu")
            load_result = self.model.load_state_dict(state, strict=True)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise RuntimeError(
                    "MegaLoc state_dict mismatch: "
                    f"missing={len(load_result.missing_keys)}, "
                    f"unexpected={len(load_result.unexpected_keys)}"
                )
            self.model = self.model.eval().to(self.device)
            _info(
                "Retrieval backend: local MegaLoc "
                f"(code={MEGALOC_LOCAL_MODEL_PY}, weights={MEGALOC_LOCAL_WEIGHTS})"
            )

            self.db_desc = self._load_precomputed_db_descriptors()
            if self.db_desc is None and self.require_precomputed_features and self.precomputed_features_path is not None:
                raise RuntimeError(
                    f"Required precomputed retrieval features could not be loaded: {self.precomputed_features_path}"
                )
            if self.db_desc is None:
                self.db_desc = self._encode_images(self.reference_images, batch_size=RETRIEVAL_BATCH_SIZE_DB)
            if faiss is not None and self.db_desc is not None and self.db_desc.size > 0:
                self.faiss_index = faiss.IndexFlatL2(self.db_desc.shape[1])  # type: ignore[attr-defined]
                self.faiss_index.add(self.db_desc.astype(np.float32))
                _info("Retrieval search index: faiss.IndexFlatL2")
            else:
                self.faiss_index = None
                if faiss is None:
                    _warn(f"faiss unavailable; using numpy L2 search. reason={FAISS_IMPORT_ERROR}")
                else:
                    _warn("faiss index not built; using numpy L2 search.")
            _info(
                f"Retrieval preload done: db={len(self.reference_images)} images, "
                f"desc_dim={self.db_desc.shape[1] if self.db_desc.size else 0}"
            )
            self.available = True
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            _warn(f"Retrieval model preload failed: {e}")

    def _encode_images(self, paths: List[Path], batch_size: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Retrieval model is not initialized")
        if self.transform is None or torch is None:
            raise RuntimeError(f"torch/torchvision unavailable: {TORCH_IMPORT_ERROR}")

        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                imgs = []
                for p in chunk:
                    imgs.append(self._load_image_for_megaloc(p))
                batch = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)
                amp_enabled = USE_AMP and self.device.startswith("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    desc = self.model(batch)

                if isinstance(desc, dict):
                    if "global_descriptor" in desc:
                        desc = desc["global_descriptor"]
                    else:
                        raise RuntimeError("Retrieval model returned dict without global_descriptor")

                desc_np = desc.detach().float().cpu().numpy().astype(np.float32)
                outputs.append(desc_np)

        if not outputs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _load_image_for_megaloc(self, path: Path) -> Any:
        if self.transform is None:
            raise RuntimeError(f"torchvision unavailable: {TORCH_IMPORT_ERROR}")
        with Image.open(path) as im_raw:
            im = im_raw.convert("RGB")
            w, h = int(im.width), int(im.height)
            max_side = max(w, h)
            if max_side > int(RETRIEVAL_RESIZE_MAX):
                scale = float(RETRIEVAL_RESIZE_MAX) / float(max_side)
                new_size = (int(round(w * scale)), int(round(h * scale)))
                if hasattr(Image, "Resampling"):
                    resample = Image.Resampling.BOX  # type: ignore[attr-defined]
                else:
                    resample = Image.BOX  # type: ignore[attr-defined]
                im = im.resize(new_size, resample=resample)
            return self.transform(im)

    def _candidate_feature_keys(self, ref_path: Path) -> List[str]:
        keys: List[str] = []
        if self.reference_dir is not None:
            try:
                keys.append(ref_path.relative_to(self.reference_dir).as_posix())
            except Exception:
                pass
        keys.append(ref_path.as_posix())
        keys.append(ref_path.name)
        out: List[str] = []
        seen = set()
        for k in keys:
            kk = str(k).replace("\\", "/").lstrip("./")
            if kk and kk not in seen:
                seen.add(kk)
                out.append(kk)
        return out

    def _load_precomputed_db_descriptors(self) -> Optional[np.ndarray]:
        path = self.precomputed_features_path
        if path is None or not Path(path).is_file():
            return None
        if h5py is None:
            if self.require_precomputed_features:
                raise RuntimeError(f"h5py unavailable, cannot read required retrieval H5: {H5PY_IMPORT_ERROR}")
            _warn(f"Precomputed retrieval features ignored: h5py unavailable ({H5PY_IMPORT_ERROR})")
            return None

        try:
            feature_by_key: Dict[str, np.ndarray] = {}

            def collect(name: str, obj: Any) -> None:
                if h5py is not None and isinstance(obj, h5py.Group) and "global_descriptor" in obj:
                    desc = np.asarray(obj["global_descriptor"][...], dtype=np.float32).reshape(-1)
                    key = str(name).replace("\\", "/").lstrip("./")
                    feature_by_key[key] = desc
                    feature_by_key[Path(key).name] = desc

            with h5py.File(path, "r", libver="latest") as fd:
                fd.visititems(collect)

            descs: List[np.ndarray] = []
            refs: List[Path] = []
            missing: List[str] = []
            for ref in self.reference_images:
                desc = None
                for key in self._candidate_feature_keys(ref):
                    desc = feature_by_key.get(key)
                    if desc is not None:
                        break
                if desc is None:
                    missing.append(ref.as_posix())
                    continue
                descs.append(desc.astype(np.float32, copy=False))
                refs.append(ref)

            if not descs:
                if self.require_precomputed_features:
                    raise RuntimeError(f"No reference descriptors matched required precomputed features: {path}")
                _warn(f"No reference descriptors matched precomputed features: {path}; fallback to image encoding.")
                return None
            dims = {int(d.shape[0]) for d in descs}
            if len(dims) != 1:
                if self.require_precomputed_features:
                    raise RuntimeError(f"Precomputed descriptor dimensions are inconsistent: {sorted(dims)}")
                _warn(f"Precomputed descriptor dimensions are inconsistent: {sorted(dims)}; fallback to image encoding.")
                return None
            if missing:
                _warn(
                    "Precomputed retrieval features missing some refs; using matched subset only: "
                    f"matched={len(refs)}, missing={len(missing)}, first_missing={missing[:3]}"
                )
            self.reference_images = refs
            self.score_type = "l2_sq_precomputed_h5"
            _info(f"Retrieval DB descriptors loaded from H5: {path}, matched_refs={len(refs)}, dim={descs[0].shape[0]}")
            return np.stack(descs, axis=0).astype(np.float32)
        except Exception as e:
            if self.require_precomputed_features:
                raise
            _warn(f"Failed to load precomputed retrieval features from {path}: {e}; fallback to image encoding.")
            return None

    def retrieve(
        self,
        query_path: Path,
        topk: int,
    ) -> Tuple[List[Path], List[float], Optional[str], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "encode_ms": 0.0,
            "search_ms": 0.0,
            "backend": "faiss" if self.faiss_index is not None else "numpy_l2",
            "db_size": int(len(self.reference_images)),
            "k": 0,
        }
        if not self.available:
            return [], [], f"retrieval model unavailable: {self.error_message}", stats
        if self.db_desc is None or len(self.db_desc) == 0:
            return [], [], "database descriptors unavailable", stats

        t_enc = time.perf_counter()
        q_desc = self._encode_images([query_path], batch_size=1)
        stats["encode_ms"] = (time.perf_counter() - t_enc) * 1000.0
        if q_desc.size == 0:
            return [], [], "failed to encode query descriptor", stats
        qv = q_desc.astype(np.float32)
        k = min(int(topk), len(self.reference_images))
        stats["k"] = int(k)
        if k <= 0:
            return [], [], "invalid topk", stats

        t_search = time.perf_counter()
        if self.faiss_index is not None:
            dists, idx = self.faiss_index.search(qv, k)
            idx_arr = idx[0].astype(np.int64)
            dist_arr = dists[0].astype(np.float32)
        else:
            q = qv[0]
            d2 = np.sum((self.db_desc.astype(np.float32) - q[None, :]) ** 2, axis=1)
            idx_arr = np.argpartition(d2, kth=k - 1)[:k]
            idx_arr = idx_arr[np.argsort(d2[idx_arr])]
            dist_arr = d2[idx_arr].astype(np.float32)
        stats["search_ms"] = (time.perf_counter() - t_search) * 1000.0

        refs = [self.reference_images[int(i)] for i in idx_arr]
        scores = [float(x) for x in dist_arr.tolist()]
        return refs, scores, None, stats
