from __future__ import annotations

import copy
import contextlib
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ricloc.helpers import _normalize_path_str
from ricloc.logging_utils import _info, _warn

TORCH_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    import torch.nn.functional as F
except Exception as _e:
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_IMPORT_ERROR = str(_e)

# $RICLOC_REPO_ROOT points at the tree providing thirdparty/vggt + models/; default = two levels up.
REPO_ROOT = Path(os.environ.get("RICLOC_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
MODELS_DIR = Path(os.environ.get("RICLOC_MODELS_DIR", str(REPO_ROOT / "models")))
USE_AMP = True
VGGT_MODEL_RESOLUTION = 518
VGGT_FALLBACK_LOAD_RESOLUTION = 1024
VGGT_TRACK_QUERY_GRID_COLS = 16
VGGT_TRACK_QUERY_GRID_ROWS = 16
SAME_SPACE_TRACK_VIS_THRESH = 0.20
SAME_SPACE_TRACK_CONF_THRESH = 0.10
SAME_SPACE_QUERY_TRACK_TOPK = 128
SAME_SPACE_QUERY_TRACK_MIN_SCORE = 0.0
SAME_SPACE_PRUNING_MIN_REF_OBS = 1
SAME_SPACE_PRUNING_REF_AWARE_BLEND = 0.5
SAME_SPACE_PRUNING_REF_AWARE_BASE = 0.25
def _maybe_cuda_synchronize(device: str) -> None:
    if torch is None:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def discover_local_vggt_checkpoint() -> Optional[Path]:
    candidates: List[Path] = [
        MODELS_DIR / "vggt_1B_commercial.pt",
        MODELS_DIR / "vggt_1B.pt",
        Path("vggt/model.pt"),
        Path("vggt/checkpoints/model.pt"),
    ]

    seen = set()
    for c in candidates:
        cp = c.expanduser()
        key = _normalize_path_str(cp)
        if key in seen:
            continue
        seen.add(key)
        if cp.is_file():
            return cp
    return None


def resolve_vggt_checkpoint_path(ckpt_value: str) -> str:
    raw = str(ckpt_value or "").strip()
    is_auto = (not raw) or (raw.upper() in {"AUTO", "AUTO_LOCAL"})
    if is_auto:
        local_ckpt = discover_local_vggt_checkpoint()
        if local_ckpt is not None:
            _info(f"VGGT checkpoint auto-discovered local file: {local_ckpt}")
            return _normalize_path_str(local_ckpt)
        raise FileNotFoundError("VGGT local checkpoint not found. expected: src/models/vggt_1B_commercial.pt")

    p = Path(raw).expanduser()
    if p.exists():
        return _normalize_path_str(p)

    raise FileNotFoundError(f"Requested VGGT checkpoint path not found: {raw}")


class VGGTPoseEngine:
    def __init__(self, device: str, checkpoint_path: str):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path_resolved = checkpoint_path
        self.available = False
        self.error_message = ""

        self.model = None
        self.dtype = torch.float32 if torch is not None else None
        self.autocast_enabled = False
        self.autocast_dtype = torch.float32 if torch is not None else None
        self._legacy_preproc_info_logged = False
        cache_disabled = str(os.environ.get("UNIFIED_VGGT_DISABLE_CACHE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.cache_enabled = not cache_disabled
        self.cache_dir = Path(os.environ.get("UNIFIED_VGGT_CACHE_DIR", str(REPO_ROOT / ".cache" / "vggt_pose"))).expanduser()
        self.cache_max_bytes = self._parse_cache_size_bytes(os.environ.get("UNIFIED_VGGT_CACHE_MAX_GB", "64"))
        try:
            self.cache_max_files = max(0, int(os.environ.get("UNIFIED_VGGT_CACHE_MAX_FILES", "0")))
        except Exception:
            self.cache_max_files = 0
        if self.cache_enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.cache_enabled = False
                _warn(f"VGGT cache disabled: could not create {self.cache_dir}: {e}")

        self._load_and_preload()

    def _load_and_preload(self) -> None:
        try:
            if torch is None or F is None:
                raise RuntimeError(f"torch unavailable: {TORCH_IMPORT_ERROR}")

            # locate the importable VGGT source tree (RICLOC_VGGT_SRC, then repo-relative paths)
            env_src = os.environ.get("RICLOC_VGGT_SRC", "").strip()
            candidates = []
            if env_src:
                candidates.append(Path(env_src).expanduser())
            candidates += [REPO_ROOT / "thirdparty" / "vggt", REPO_ROOT / "vggt", Path("vggt")]
            vggt_root = next((c for c in candidates if c.exists()), None)
            if vggt_root is None:
                raise FileNotFoundError(
                    "VGGT source tree not found. Set RICLOC_VGGT_SRC to the vggt/ source directory "
                    f"(searched: {[str(c) for c in candidates]})."
                )
            sys.path.insert(0, str(vggt_root.resolve()))

            from vggt.models.vggt import VGGT  # type: ignore
            from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
            from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore

            self._load_and_preprocess_images_square = load_and_preprocess_images_square
            self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
            self._unproject_depth_map_to_point_map = unproject_depth_map_to_point_map

            model = VGGT()
            # fall back to manual attention when SDPA is unavailable (torch < 2.0)
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                _n_unfused = 0
                for _m in model.modules():
                    if hasattr(_m, "fused_attn"):
                        _m.fused_attn = False
                        _n_unfused += 1
                _warn(f"SDPA unavailable (torch {torch.__version__}); disabled fused_attn on {_n_unfused} attention layers")
            ckpt_requested = self.checkpoint_path.strip()
            ckpt = resolve_vggt_checkpoint_path(ckpt_requested)
            self.checkpoint_path_resolved = ckpt

            state = torch.load(ckpt, map_location="cpu")

            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]

            if not isinstance(state, dict):
                raise RuntimeError("Invalid VGGT checkpoint format (state_dict not found)")

            state_clean = {}
            for k, v in state.items():
                nk = str(k)
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                state_clean[nk] = v

            try:
                model.load_state_dict(state_clean, strict=True)
            except Exception as e_strict:
                _warn(f"VGGT strict load failed; retry strict=False. reason={e_strict}")
                model.load_state_dict(state_clean, strict=False)

            model.eval()
            model = model.to(self.device)

            if self.device.startswith("cuda") and torch.cuda.is_available():
                major = torch.cuda.get_device_capability()[0]
                # Keep model params in fp32, reduce precision only inside autocast.
                self.dtype = torch.float32
                self.autocast_enabled = bool(USE_AMP)
                self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                self.dtype = torch.float32
                self.autocast_enabled = False
                self.autocast_dtype = torch.float32

            self.model = model
            self.available = True
            _info(
                "VGGT AMP config: "
                f"autocast_enabled={self.autocast_enabled}, "
                f"autocast_dtype={self.autocast_dtype}, "
                "model_param_dtype=float32"
            )
            _info(
                f"VGGT preload done: ckpt_requested={ckpt_requested}, ckpt_resolved={self.checkpoint_path_resolved}, "
                f"device={self.device}, dtype={self.dtype}"
            )
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            _warn(f"VGGT preload failed: {e}")

    @staticmethod
    def _parse_cache_size_bytes(raw: str) -> int:
        try:
            value = float(str(raw).strip())
        except Exception:
            return 0
        if not np.isfinite(value) or value <= 0.0:
            return 0
        return int(value * (1024**3))

    def _load_preprocessed_inputs(self, image_paths_str: List[str]) -> Tuple[Any, Any, bool]:
        if not self._legacy_preproc_info_logged:
            _info(
                "VGGT preprocess mode: fixed "
                f"target_size={VGGT_FALLBACK_LOAD_RESOLUTION} + interpolate->{VGGT_MODEL_RESOLUTION}"
            )
            self._legacy_preproc_info_logged = True
        images, original_coords = self._load_and_preprocess_images_square(
            image_paths_str,
            target_size=VGGT_FALLBACK_LOAD_RESOLUTION,
        )
        return images, original_coords, True

    def _cache_key_for_inputs(
        self,
        image_paths_str: List[str],
        compact_for_alignment: bool,
        include_reconstruction_outputs: bool,
    ) -> Optional[str]:
        try:
            def _file_sha256(path: Path) -> str:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()

            entries = []
            for raw in image_paths_str:
                p = Path(raw).expanduser()
                st = p.stat()
                entries.append(
                    {
                        # hash by content so identical inputs stay cacheable
                        "sha256": _file_sha256(p),
                        "size": int(st.st_size),
                    }
                )
            ckpt = Path(str(self.checkpoint_path_resolved)).expanduser()
            ckpt_stat = ckpt.stat() if ckpt.exists() else None
            payload = {
                "version": 3,
                "checkpoint": str(ckpt.resolve()) if ckpt.exists() else str(ckpt),
                "checkpoint_size": int(ckpt_stat.st_size) if ckpt_stat is not None else None,
                "checkpoint_mtime_ns": int(ckpt_stat.st_mtime_ns) if ckpt_stat is not None else None,
                "model_resolution": int(VGGT_MODEL_RESOLUTION),
                "fallback_load_resolution": int(VGGT_FALLBACK_LOAD_RESOLUTION),
                "query_grid_cols": int(VGGT_TRACK_QUERY_GRID_COLS),
                "query_grid_rows": int(VGGT_TRACK_QUERY_GRID_ROWS),
                "compact_for_alignment": bool(compact_for_alignment),
                "include_reconstruction_outputs": bool(include_reconstruction_outputs),
                "paths": entries,
            }
            blob = repr(payload).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()
        except Exception:
            return None

    def _cacheable_value(self, value: Any) -> Any:
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Path):
            return _normalize_path_str(value)
        if isinstance(value, dict):
            return {str(k): self._cacheable_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._cacheable_value(v) for v in value]
        return value

    def _load_cached_inference(self, cache_key: Optional[str], image_paths: List[Path]) -> Optional[Dict[str, Any]]:
        if not self.cache_enabled or not cache_key:
            return None
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if not cache_path.is_file():
            return None
        try:
            with open(cache_path, "rb") as f:
                out = pickle.load(f)
            if not isinstance(out, dict):
                return None
            if int(out.get("cache_schema_version", 3) or 3) != 3:
                return None
            poses = out.get("poses")
            if isinstance(poses, list):
                for pose in poses:
                    if isinstance(pose, dict) and bool(pose.get("compact", False)):
                        if not bool(pose.get("valid", False)) or "C" not in pose or "Rcw" not in pose or "tcw" not in pose:
                            return None
            out = copy.deepcopy(out)
            out["image_paths"] = list(image_paths)
            out["cache_hit"] = True
            out["cache_key"] = str(cache_key)
            out["cache_path"] = _normalize_path_str(cache_path)
            out["cached_model_infer_ms"] = float(out.get("model_infer_ms", 0.0) or 0.0)
            out["model_infer_ms"] = 0.0
            profile = out.get("profile_ms")
            if not isinstance(profile, dict):
                profile = {}
            profile["vggt_cache_hit"] = 1.0
            out["profile_ms"] = profile
            _info(f"VGGT cache hit: views={len(image_paths)}, key={cache_key[:12]}")
            return out
        except Exception as e:
            _warn(f"VGGT cache read failed: {cache_path}: {e}")
            return None

    def _save_cached_inference(self, cache_key: Optional[str], out: Dict[str, Any]) -> None:
        if not self.cache_enabled or not cache_key:
            return
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        tmp_path = self.cache_dir / f"{cache_key}.tmp"
        try:
            cache_out = self._cacheable_value(out)
            if not isinstance(cache_out, dict):
                return
            cache_out["cache_hit"] = False
            cache_out["cache_key"] = str(cache_key)
            cache_out["cache_path"] = _normalize_path_str(cache_path)
            profile = cache_out.get("profile_ms")
            if isinstance(profile, dict):
                profile["vggt_cache_hit"] = 0.0
            with open(tmp_path, "wb") as f:
                pickle.dump(cache_out, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            self._prune_cache_if_needed()
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            _warn(f"VGGT cache write failed: {cache_path}: {e}")

    def _prune_cache_if_needed(self) -> None:
        if not self.cache_enabled or (self.cache_max_bytes <= 0 and self.cache_max_files <= 0):
            return
        try:
            entries = []
            for path in self.cache_dir.glob("*.pkl"):
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                entries.append((float(st.st_mtime), int(st.st_size), path))
            if not entries:
                return
            total_bytes = int(sum(size for _mtime, size, _path in entries))
            entries.sort(key=lambda x: x[0])
            removed = 0
            while entries and (
                (self.cache_max_bytes > 0 and total_bytes > self.cache_max_bytes)
                or (self.cache_max_files > 0 and len(entries) > self.cache_max_files)
            ):
                _mtime, size, path = entries.pop(0)
                try:
                    path.unlink()
                    total_bytes -= int(size)
                    removed += 1
                except FileNotFoundError:
                    total_bytes -= int(size)
                except Exception as e:
                    _warn(f"VGGT cache prune skipped {path}: {e}")
                    break
            if removed:
                _info(
                    "VGGT cache pruned: "
                    f"removed={removed}, remaining_files={len(entries)}, remaining_gb={total_bytes / (1024**3):.2f}"
                )
        except Exception as e:
            _warn(f"VGGT cache prune failed: {e}")

    def _build_track_query_points(
        self,
        width: int,
        height: int,
        content_box_model_xyxy: Optional[Tuple[float, float, float, float]] = None,
    ) -> Any:
        if torch is None:
            raise RuntimeError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
        cols = max(1, int(VGGT_TRACK_QUERY_GRID_COLS))
        rows = max(1, int(VGGT_TRACK_QUERY_GRID_ROWS))

        x_lo = 0.5
        y_lo = 0.5
        x_hi = max(float(width) - 0.5, 0.5)
        y_hi = max(float(height) - 0.5, 0.5)
        if content_box_model_xyxy is not None:
            try:
                box = np.asarray(content_box_model_xyxy, dtype=np.float64).reshape(-1)
                if box.size >= 4 and np.all(np.isfinite(box[:4])):
                    bx1 = float(np.clip(box[0], 0.0, max(float(width) - 1.0, 0.0)))
                    by1 = float(np.clip(box[1], 0.0, max(float(height) - 1.0, 0.0)))
                    bx2 = float(np.clip(box[2], 0.0, max(float(width) - 1.0, 0.0)))
                    by2 = float(np.clip(box[3], 0.0, max(float(height) - 1.0, 0.0)))
                    if bx2 > bx1 + 1e-6 and by2 > by1 + 1e-6:
                        x_lo = max(0.5, bx1 + 0.5)
                        y_lo = max(0.5, by1 + 0.5)
                        x_hi = min(max(float(width) - 0.5, 0.5), bx2 - 0.5)
                        y_hi = min(max(float(height) - 0.5, 0.5), by2 - 0.5)
                        if x_hi < x_lo:
                            x_lo, x_hi = x_hi, x_lo
                        if y_hi < y_lo:
                            y_lo, y_hi = y_hi, y_lo
            except Exception:
                pass

        xs = np.linspace(float(x_lo), float(max(x_hi, x_lo)), num=cols, dtype=np.float32)
        ys = np.linspace(float(y_lo), float(max(y_hi, y_lo)), num=rows, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        qpts = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
        return torch.from_numpy(qpts.astype(np.float32))

    def infer_pose_from_preprocessed(
        self,
        images: Any,
        original_coords: Any,
        image_paths: List[Path],
        image_sizes: List[Tuple[int, int]],
        compact_for_alignment: bool = False,
        include_track_outputs: bool = True,
        include_reconstruction_outputs: bool = False,
    ) -> Dict[str, Any]:
        # run VGGT on already-preprocessed tensors (no path-based disk I/O)
        if not self.available or self.model is None:
            raise RuntimeError(f"VGGT unavailable: {self.error_message}")
        if torch is None or F is None:
            raise RuntimeError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
        if len(image_paths) == 0:
            raise ValueError("VGGT input image list is empty")
        if not isinstance(images, torch.Tensor) or not isinstance(original_coords, torch.Tensor):
            raise TypeError("images/original_coords must be torch.Tensor")
        if images.ndim != 4:
            raise ValueError(f"images must be 4D [N,C,H,W], got shape={tuple(images.shape)}")
        if original_coords.ndim != 2:
            raise ValueError(f"original_coords must be 2D [N,*], got shape={tuple(original_coords.shape)}")
        n = len(image_paths)
        if int(images.shape[0]) != n:
            raise ValueError(f"images batch mismatch: expected N={n}, got {int(images.shape[0])}")
        if int(original_coords.shape[0]) != n:
            raise ValueError(f"original_coords batch mismatch: expected N={n}, got {int(original_coords.shape[0])}")
        if len(image_sizes) != n:
            raise ValueError(f"image_sizes length mismatch: expected {n}, got {len(image_sizes)}")

        images = images.to(self.device, non_blocking=True)

        resized = images
        used_interpolate = (int(images.shape[-2]) != VGGT_MODEL_RESOLUTION) or (int(images.shape[-1]) != VGGT_MODEL_RESOLUTION)
        if used_interpolate:
            resized = F.interpolate(
                images,
                size=(VGGT_MODEL_RESOLUTION, VGGT_MODEL_RESOLUTION),
                mode="bilinear",
                align_corners=False,
            )
        autocast_ctx = (
            torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype if self.autocast_dtype is not None else torch.float16,
                enabled=self.autocast_enabled,
            )
            if self.autocast_enabled
            else contextlib.nullcontext()
        )

        profile_ms: Dict[str, float] = {
            "vggt_aggregator_ms": 0.0,
            "vggt_camera_head_ms": 0.0,
            "vggt_track_head_ms": 0.0,
            "track_cpu_copy_ms": 0.0,
            "track_postprocess_ms": 0.0,
        }
        t_model = time.perf_counter()
        track_xy_t = None
        track_vis_t = None
        track_conf_t = None
        depth_map_t = None
        depth_conf_t = None
        track_source = "none"
        track_warn = ""
        with torch.inference_mode():
            with autocast_ctx:
                resized_batched = resized[None]
                _maybe_cuda_synchronize(self.device)
                t_agg = time.perf_counter()
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(resized_batched)
                _maybe_cuda_synchronize(self.device)
                profile_ms["vggt_aggregator_ms"] = (time.perf_counter() - t_agg) * 1000.0

                _maybe_cuda_synchronize(self.device)
                t_cam = time.perf_counter()
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = self._pose_encoding_to_extri_intri(pose_enc, resized_batched.shape[-2:])
                _maybe_cuda_synchronize(self.device)
                profile_ms["vggt_camera_head_ms"] = (time.perf_counter() - t_cam) * 1000.0

                # Keep query seed points inside the valid content crop (exclude padded area).
                query_track_content_box_model: Optional[Tuple[float, float, float, float]] = None
                try:
                    if int(original_coords.shape[1]) >= 4:
                        load_h = float(images.shape[-2])
                        load_w = float(images.shape[-1])
                        model_h = float(resized.shape[-2])
                        model_w = float(resized.shape[-1])
                        x1_load = float(original_coords[0, 0].item())
                        y1_load = float(original_coords[0, 1].item())
                        x2_load = float(original_coords[0, 2].item())
                        y2_load = float(original_coords[0, 3].item())
                        sx = model_w / max(load_w, 1e-9)
                        sy = model_h / max(load_h, 1e-9)
                        x1_model = x1_load * sx
                        y1_model = y1_load * sy
                        x2_model = x2_load * sx
                        y2_model = y2_load * sy
                        if (
                            np.isfinite(x1_model)
                            and np.isfinite(y1_model)
                            and np.isfinite(x2_model)
                            and np.isfinite(y2_model)
                            and x2_model > x1_model + 1e-6
                            and y2_model > y1_model + 1e-6
                        ):
                            query_track_content_box_model = (
                                float(x1_model),
                                float(y1_model),
                                float(x2_model),
                                float(y2_model),
                            )
                except Exception:
                    query_track_content_box_model = None

                if include_reconstruction_outputs and getattr(
                    self.model, "depth_head", None
                ) is not None:
                    depth_map_t, depth_conf_t = self.model.depth_head(
                        aggregated_tokens_list,
                        resized_batched,
                        patch_start_idx,
                    )

                if include_track_outputs and getattr(self.model, "track_head", None) is not None:
                    try:
                        qpts = self._build_track_query_points(
                            width=int(resized_batched.shape[-1]),
                            height=int(resized_batched.shape[-2]),
                            content_box_model_xyxy=query_track_content_box_model,
                        )
                        qpts = qpts.to(self.device, non_blocking=True)
                        _maybe_cuda_synchronize(self.device)
                        t_trk = time.perf_counter()
                        track_list, vis_scores, conf_scores = self.model.track_head(
                            aggregated_tokens_list,
                            images=resized_batched,
                            patch_start_idx=patch_start_idx,
                            query_points=qpts[None],
                        )
                        _maybe_cuda_synchronize(self.device)
                        profile_ms["vggt_track_head_ms"] = (time.perf_counter() - t_trk) * 1000.0
                        if isinstance(track_list, (list, tuple)):
                            track_xy_t = track_list[-1]
                        else:
                            track_xy_t = track_list
                        track_vis_t = vis_scores
                        track_conf_t = conf_scores
                        track_source = "vggt_track_head_grid"
                    except Exception as e:
                        track_warn = f"track_head_failed: {e}"

                if include_track_outputs and track_xy_t is None and not track_warn:
                    track_warn = "track_output_unavailable"
        model_infer_ms = (time.perf_counter() - t_model) * 1000.0

        extr = extrinsic.squeeze(0)
        intr = intrinsic.squeeze(0)

        def _resolve_size(idx: int) -> Tuple[int, int]:
            w_sz, h_sz = image_sizes[idx]
            if w_sz <= 0 or h_sz <= 0:
                if int(original_coords.shape[1]) >= 6:
                    w_sz = int(round(float(original_coords[idx, 4].item())))
                    h_sz = int(round(float(original_coords[idx, 5].item())))
                else:
                    raise ValueError(f"invalid image size at idx={idx}: {(w_sz, h_sz)}")
            w = int(w_sz)
            h = int(h_sz)
            return w, h

        def _build_pose_entry(idx: int, path: Path) -> Dict[str, Any]:
            R_cw_t = extr[idx, :3, :3]
            t_cw_t = extr[idx, :3, 3]

            if compact_for_alignment and idx > 0:
                C_t = -(R_cw_t.transpose(0, 1) @ t_cw_t.reshape(3))
                return {
                    "path": path,
                    "name": path.name,
                    "C": C_t.detach().float().cpu().numpy().astype(np.float64),
                    "Rcw": R_cw_t.detach().float().cpu().numpy().astype(np.float64),
                    "tcw": t_cw_t.detach().float().cpu().numpy().astype(np.float64),
                    "valid": True,
                    "compact": True,
                }

            K_vggt_t = intr[idx]
            w, h = _resolve_size(idx)
            max_dim = max(float(w), float(h), 1.0)
            scale = max_dim / float(VGGT_MODEL_RESOLUTION)

            K_scaled = K_vggt_t.detach().float().cpu().numpy().astype(np.float64)
            K_scaled[0, 0] *= scale
            K_scaled[1, 1] *= scale
            K_scaled[0, 2] = w * 0.5
            K_scaled[1, 2] = h * 0.5

            return {
                "path": path,
                "name": path.name,
                "Rcw": R_cw_t.detach().float().cpu().numpy().astype(np.float64),
                "tcw": t_cw_t.detach().float().cpu().numpy().astype(np.float64),
                "K": K_scaled,
                "size": (w, h),
            }

        poses = [_build_pose_entry(i, p) for i, p in enumerate(image_paths)]

        def _build_track_frame_xy_to_original(idx: int) -> Dict[str, Any]:
            # Map track xy (model-518 coordinates) -> original image pixel coordinates.
            w, h = _resolve_size(idx)
            load_h = float(images.shape[-2])
            load_w = float(images.shape[-1])
            model_h = float(resized.shape[-2])
            model_w = float(resized.shape[-1])

            x1_load, y1_load = 0.0, 0.0
            x2_load, y2_load = load_w, load_h
            if int(original_coords.shape[1]) >= 4:
                try:
                    x1_load = float(original_coords[idx, 0].item())
                    y1_load = float(original_coords[idx, 1].item())
                    x2_load = float(original_coords[idx, 2].item())
                    y2_load = float(original_coords[idx, 3].item())
                except Exception:
                    x1_load, y1_load = 0.0, 0.0
                    x2_load, y2_load = load_w, load_h

            sx = model_w / max(load_w, 1e-9)
            sy = model_h / max(load_h, 1e-9)
            x1_model = x1_load * sx
            y1_model = y1_load * sy
            x2_model = x2_load * sx
            y2_model = y2_load * sy
            span_x = x2_model - x1_model
            span_y = y2_model - y1_model
            valid = bool(np.isfinite(span_x) and np.isfinite(span_y) and span_x > 1e-6 and span_y > 1e-6)
            if not valid:
                x1_model, y1_model = 0.0, 0.0
                span_x = max(model_w, 1.0)
                span_y = max(model_h, 1.0)

            return {
                "frame_idx": int(idx),
                "path": _normalize_path_str(image_paths[idx]),
                "model_size_wh": [float(model_w), float(model_h)],
                "crop_box_model_xyxy": [float(x1_model), float(y1_model), float(x1_model + span_x), float(y1_model + span_y)],
                "orig_size_wh": [float(w), float(h)],
                "scale_to_orig_xy": [float(float(w) / max(span_x, 1e-9)), float(float(h) / max(span_y, 1e-9))],
                "valid": bool(valid),
                "coord_space": "model_518",
            }

        track_coord_space = "model_518"
        frame_xy_to_original = [_build_track_frame_xy_to_original(i) for i in range(len(image_paths))]

        tracks_out: Dict[str, Any] = {}
        if track_xy_t is not None and track_vis_t is not None:
            try:
                _maybe_cuda_synchronize(self.device)
                t_post = time.perf_counter()
                track_xy_pack = torch.as_tensor(track_xy_t, dtype=torch.float32, device=self.device).detach()
                if track_xy_pack.ndim == 4 and int(track_xy_pack.shape[0]) == 1:
                    track_xy_pack = track_xy_pack.squeeze(0)
                if track_xy_pack.ndim != 3 or int(track_xy_pack.shape[-1]) != 2:
                    raise ValueError(f"track_xy_pack_shape_invalid:{tuple(track_xy_pack.shape)}")
                track_xy_pack = torch.nan_to_num(
                    track_xy_pack.contiguous(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                track_vis_pack = torch.as_tensor(track_vis_t, dtype=torch.float32, device=self.device).detach()
                if track_vis_pack.ndim == 3 and int(track_vis_pack.shape[0]) == 1:
                    track_vis_pack = track_vis_pack.squeeze(0)
                if track_vis_pack.ndim != 2:
                    raise ValueError(f"track_vis_pack_shape_invalid:{tuple(track_vis_pack.shape)}")
                if int(track_vis_pack.shape[0]) != int(track_xy_pack.shape[0]) or int(track_vis_pack.shape[1]) != int(
                    track_xy_pack.shape[1]
                ):
                    raise ValueError(
                        "track_vis_track_shape_mismatch:"
                        f"vis={tuple(track_vis_pack.shape)},tracks={tuple(track_xy_pack.shape)}"
                    )
                track_vis_pack = torch.nan_to_num(
                    track_vis_pack.contiguous(),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp_(0.0, 1.0)
                track_conf_pack = None
                if track_conf_t is not None:
                    track_conf_pack = torch.as_tensor(track_conf_t, dtype=torch.float32, device=self.device).detach()
                    if track_conf_pack.ndim == 3 and int(track_conf_pack.shape[0]) == 1:
                        track_conf_pack = track_conf_pack.squeeze(0)
                    if tuple(track_conf_pack.shape) != tuple(track_vis_pack.shape):
                        track_conf_pack = None
                    else:
                        track_conf_pack = torch.nan_to_num(
                            track_conf_pack.contiguous(),
                            nan=0.0,
                            posinf=1.0,
                            neginf=0.0,
                        ).clamp_(0.0, 1.0)
                _maybe_cuda_synchronize(self.device)
                profile_ms["track_postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0

                # keep track tensors on device; only scalar summaries move to CPU later
                t_copy = time.perf_counter()
                _maybe_cuda_synchronize(self.device)
                profile_ms["track_cpu_copy_ms"] = (time.perf_counter() - t_copy) * 1000.0
                query_grid_cols = int(VGGT_TRACK_QUERY_GRID_COLS)
                query_grid_rows = int(VGGT_TRACK_QUERY_GRID_ROWS)
                tracks_out = {
                    "track_xy_t": track_xy_pack,
                    "vis_t": track_vis_pack,
                    "conf_t": track_conf_pack,
                    "frame_xy_to_original": frame_xy_to_original,
                    "query_grid_cols": int(query_grid_cols),
                    "query_grid_rows": int(query_grid_rows),
                    "vis_thresh": float(SAME_SPACE_TRACK_VIS_THRESH),
                    "conf_thresh": float(SAME_SPACE_TRACK_CONF_THRESH),
                    "query_track_topk": int(SAME_SPACE_QUERY_TRACK_TOPK),
                    "query_track_min_score": float(SAME_SPACE_QUERY_TRACK_MIN_SCORE),
                    "pruning_min_ref_obs": int(SAME_SPACE_PRUNING_MIN_REF_OBS),
                    "pruning_ref_aware_blend": float(SAME_SPACE_PRUNING_REF_AWARE_BLEND),
                    "pruning_ref_aware_base": float(SAME_SPACE_PRUNING_REF_AWARE_BASE),
                    "track_source": str(track_source),
                }
                _info(
                    "Track generation: "
                    f"source={track_source}, "
                    f"frames={int(track_xy_pack.shape[0])}, "
                    f"tracks={int(track_xy_pack.shape[1])}, "
                    f"coord_space={track_coord_space}"
                )
            except Exception as e:
                track_warn = f"track_tensor_pack_failed: {e}"

        reconstruction_out: Dict[str, Any] = {}
        if include_reconstruction_outputs and depth_map_t is not None and depth_conf_t is not None:
            try:
                depth_map = depth_map_t.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
                depth_conf = depth_conf_t.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
                extr_np = extr.detach().float().cpu().numpy().astype(np.float64)
                intr_np = intr.detach().float().cpu().numpy().astype(np.float64)
                world_points = self._unproject_depth_map_to_point_map(depth_map, extr_np, intr_np).astype(np.float32)

                colors = resized.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                colors = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
                reconstruction_out = {
                    "points_vggt": world_points,
                    "conf": depth_conf,
                    "colors": colors,
                }
            except Exception as e:
                if track_warn:
                    track_warn += f"; reconstruction_export_failed: {e}"
                else:
                    track_warn = f"reconstruction_export_failed: {e}"

        out: Dict[str, Any] = {
            "poses": poses,
            "preprocess_mode": "preprocessed_resized_to_518" if used_interpolate else "preprocessed_native_518",
            "image_paths": image_paths,
            "model_infer_ms": float(model_infer_ms),
            "profile_ms": profile_ms,
        }
        if tracks_out:
            out["tracks"] = tracks_out
        if reconstruction_out:
            out["reconstruction"] = reconstruction_out
        if track_warn:
            out["track_warning"] = track_warn
        return out

    def infer_pose_only(
        self,
        image_paths: List[Path],
        compact_for_alignment: bool = False,
        include_reconstruction_outputs: bool = False,
    ) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError(f"torch unavailable: {TORCH_IMPORT_ERROR}")
        image_paths_str = [_normalize_path_str(p) for p in image_paths]
        cache_key = self._cache_key_for_inputs(
            image_paths_str=image_paths_str,
            compact_for_alignment=bool(compact_for_alignment),
            include_reconstruction_outputs=bool(include_reconstruction_outputs),
        )
        if not bool(include_reconstruction_outputs):
            cached = self._load_cached_inference(cache_key, image_paths=image_paths)
            if cached is not None:
                return cached
        images, original_coords, used_interpolate_fallback = self._load_preprocessed_inputs(image_paths_str)
        image_sizes: List[Tuple[int, int]] = []
        for i in range(len(image_paths)):
            w = int(round(float(original_coords[i, 4].item())))
            h = int(round(float(original_coords[i, 5].item())))
            image_sizes.append((w, h))
        out = self.infer_pose_from_preprocessed(
            images,
            original_coords,
            image_paths,
            image_sizes,
            compact_for_alignment=compact_for_alignment,
            include_reconstruction_outputs=include_reconstruction_outputs,
        )
        out["preprocess_mode"] = "fallback_1024_plus_interpolate" if used_interpolate_fallback else "single_stage_518"
        out["cache_schema_version"] = 3
        out["cache_hit"] = False
        out["cache_key"] = str(cache_key or "")
        profile = out.get("profile_ms")
        if isinstance(profile, dict):
            profile["vggt_cache_hit"] = 0.0
        if not bool(include_reconstruction_outputs):
            self._save_cached_inference(cache_key, out)
        return out
