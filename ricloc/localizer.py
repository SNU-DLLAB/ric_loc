"""RIC-Loc localization pipeline.

Per query:
    1. MegaLoc retrieval  (top-K global descriptors vs the prebuilt H5 DB)
    2. top-K reference selection  (first num_refs retrieved)
    3. one VGGT forward pass on [query + refs]  -> relative poses + tracks + pointmap
    4. Sim(3) RANSAC alignment of VGGT-local ref centers to COLMAP-world ref centers  -> scale s
    5. reference-induced consensus  -> R_cons + scalar C_cons
    6. map-point-free covariance consensus  -> C_cons (covariance) + posterior sigma_new / sigma_disp
    7. output pose = (R_cons, covariance-C_cons if available else scalar-C_cons)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import FROZEN, FrozenConfig
from .geometry import camera_center_from_world_to_camera, qvec_to_rotmat, rotmat_to_qvec_wxyz
from .helpers import _normalize_path_str, _sorted_image_list_from_path, _sorted_reference_images
from .colmap_sparse_model import ColmapSparseModel
from .retrieval_engine import RetrievalEngine
from .vggt_pose_engine import VGGTPoseEngine
from .pose_solvers import estimate_alignment_with_fallback
from .consensus import mapfree_consensus_weight, reference_consensus_pose
from .mapfree_adapter import mapfree_consensus_from_vggt

OUTPUT_POSE_SPEC = "colmap_world_quat_t"


def _is_valid_colmap_dir(path: Path) -> bool:
    has_bin = all((path / n).exists() for n in ("cameras.bin", "images.bin", "points3D.bin"))
    has_txt = all((path / n).exists() for n in ("cameras.txt", "images.txt", "points3D.txt"))
    return path.is_dir() and (has_bin or has_txt)


def find_reference_image_dir(reference_root: Path) -> Path:
    for cand in (reference_root / "dense" / "images", reference_root / "images",
                 reference_root / "image", reference_root / "reference_images", reference_root):
        if cand.is_dir():
            try:
                _sorted_image_list_from_path(cand)
                return cand
            except Exception:
                pass
    for cand in sorted((p for p in reference_root.rglob("*") if p.is_dir()),
                       key=lambda p: (len(p.parts), str(p))):
        if cand.name.lower() not in {"images", "image", "reference_images"}:
            continue
        try:
            _sorted_image_list_from_path(cand)
            return cand
        except Exception:
            continue
    raise FileNotFoundError(f"reference image folder not found under: {reference_root}")


def find_colmap_dir(reference_root: Path) -> Path:
    for cand in (reference_root / "dense" / "sparse" / "0", reference_root / "dense" / "sparse",
                 reference_root / "sparse" / "0", reference_root / "sparse"):
        if _is_valid_colmap_dir(cand):
            return cand
    for cand in sorted((p for p in reference_root.rglob("*") if p.is_dir()),
                       key=lambda p: (len(p.parts), str(p))):
        if _is_valid_colmap_dir(cand):
            return cand
    raise FileNotFoundError(f"COLMAP sparse model not found under: {reference_root}")


def find_retrieval_features(reference_root: Path) -> Path:
    expected = reference_root / "hloc_out" / "global" / "global-feats-megaloc.h5"
    if expected.is_file():
        return expected
    matches = sorted(reference_root.rglob("global-feats-megaloc.h5"), key=lambda p: (len(p.parts), str(p)))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"global-feats-megaloc.h5 not found under: {reference_root}")


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class FrozenLocalizer:
    def __init__(
        self,
        reference_root: str | Path,
        vggt_ckpt: str | Path,
        device: Optional[str] = None,
        config: FrozenConfig = FROZEN,
        colmap_dir: Optional[str | Path] = None,
        retrieval_features: Optional[str | Path] = None,
    ):
        self.cfg = config
        self.cfg.apply_env()
        self.device = _resolve_device(device)
        reference_root = Path(reference_root).expanduser().resolve()

        self.reference_dir = find_reference_image_dir(reference_root)
        self.colmap_dir = Path(colmap_dir).expanduser().resolve() if colmap_dir else find_colmap_dir(reference_root)
        self.retrieval_features = (Path(retrieval_features).expanduser().resolve() if retrieval_features
                                   else find_retrieval_features(reference_root))
        self.reference_images: List[Path] = _sorted_reference_images(self.reference_dir)

        self.retrieval = RetrievalEngine(
            self.reference_images, self.device, reference_dir=self.reference_dir,
            precomputed_features_path=self.retrieval_features, require_precomputed_features=True,
        )
        self.vggt = VGGTPoseEngine(device=self.device, checkpoint_path=str(vggt_ckpt))
        self.colmap = ColmapSparseModel(self.colmap_dir)

        # precompute reference -> COLMAP image-id map
        self.ref_to_colmap_image_id: Dict[str, int] = {}
        for ref_path in self.reference_images:
            iid = self.colmap.resolve_ref_image_id(ref_path, self.reference_dir)
            if iid is not None:
                self.ref_to_colmap_image_id[self._ref_key(ref_path)] = iid

    @staticmethod
    def _ref_key(p: Path) -> str:
        try:
            return str(Path(p).resolve())
        except Exception:
            return str(p)

    def _resolve_colmap_id(self, ref_path: Path, vggt_name: str) -> Optional[int]:
        key = self._ref_key(ref_path)
        iid = self.ref_to_colmap_image_id.get(key)
        if iid is None:
            iid = self.colmap.resolve_ref_image_id_with_vggt_name(
                ref_path=ref_path, reference_dir=self.reference_dir, vggt_name=vggt_name, camera_id_hint=None,
            )
            if iid is not None:
                self.ref_to_colmap_image_id[key] = iid
        if iid is None or int(iid) not in self.colmap.images:
            return None
        return int(iid)

    def _colmap_Rw_Cw(self, image_id: int):
        cimg = self.colmap.images[int(image_id)]
        Rw = qvec_to_rotmat(np.asarray(cimg.qvec, dtype=np.float64))
        Cw = self.colmap.image_centers.get(int(image_id))
        if Cw is None:
            Cw = camera_center_from_world_to_camera(Rw, np.asarray(cimg.tvec, dtype=np.float64))
        return Rw, np.asarray(Cw, dtype=np.float64).reshape(3)

    @staticmethod
    def _track_conf_means(vggt_out: Dict[str, Any], n_views: int) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * int(max(n_views, 0))
        tracks = vggt_out.get("tracks", {})
        if not isinstance(tracks, dict):
            return out
        conf = tracks.get("conf_t", tracks.get("conf"))
        vis = tracks.get("vis_t", tracks.get("vis"))
        if conf is None:
            return out
        try:
            conf = conf.detach().float().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf, np.float64)
            conf = np.asarray(conf, np.float64)
            if conf.ndim == 3 and conf.shape[0] == 1:
                conf = conf[0]
            if conf.ndim != 2:
                return out
            vis_arr = None
            if vis is not None:
                vis_arr = vis.detach().float().cpu().numpy() if hasattr(vis, "detach") else np.asarray(vis, np.float64)
                vis_arr = np.asarray(vis_arr, np.float64)
                if vis_arr.ndim == 3 and vis_arr.shape[0] == 1:
                    vis_arr = vis_arr[0]
                if vis_arr.shape != conf.shape:
                    vis_arr = None
            for v in range(min(len(out), conf.shape[0])):
                mask = np.isfinite(conf[v])
                if vis_arr is not None:
                    mask = mask & (vis_arr[v] >= 0.20)
                vals = conf[v][mask]
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    out[v] = float(np.clip(np.mean(vals), 0.0, 1.0))
        except Exception:
            return out
        return out

    @staticmethod
    def _query_center_v(pose: Dict[str, Any]) -> np.ndarray:
        if "C" in pose:
            return np.asarray(pose["C"], dtype=np.float64).reshape(3)
        Rcw = np.asarray(pose["Rcw"], dtype=np.float64).reshape(3, 3)
        tcw = np.asarray(pose["tcw"], dtype=np.float64).reshape(3)
        return camera_center_from_world_to_camera(Rcw, tcw)

    def localize(self, query_path: str | Path, dump_consensus: bool = False) -> Dict[str, Any]:
        query_path = Path(query_path)
        cfg = self.cfg
        result: Dict[str, Any] = {
            "query_path": _normalize_path_str(query_path),
            "query_name": query_path.name,
            "status": "ok",
            "output_pose_spec": OUTPUT_POSE_SPEC,
            "warnings": [],
            "timings_ms": {},
        }
        t_total = time.perf_counter()

        # 1) retrieval
        t = time.perf_counter()
        refs, scores, err, _stats = self.retrieval.retrieve(query_path, cfg.topk)
        result["timings_ms"]["retrieval"] = (time.perf_counter() - t) * 1000.0
        if err is not None or not refs:
            result["status"] = "fail_retrieval"
            result["warnings"].append(err or "empty retrieval result")
            return result
        result["retrieved_refs"] = [_normalize_path_str(p) for p in refs]
        result["retrieved_ref_scores"] = [float(s) for s in scores]

        # retrieval rank + score-uncertainty
        retrieval_count = len(refs)
        rank_by_key = {self._ref_key(r): i + 1 for i, r in enumerate(refs)}
        score_unc_by_key: Dict[str, float] = {}
        sarr = np.asarray(scores, dtype=np.float64)
        finite = sarr[np.isfinite(sarr)]
        if finite.size > 1:
            stype = str(self.retrieval.score_type).lower()
            lower_better = any(k in stype for k in ("l2", "dist", "distance", "error"))
            best = float(np.min(finite) if lower_better else np.max(finite))
            worst = float(np.max(finite) if lower_better else np.min(finite))
            denom = max(abs(worst - best), 1e-12)
            for r, s in zip(refs, scores):
                s = float(s)
                if not np.isfinite(s):
                    continue
                unc = (s - best) / denom if lower_better else (best - s) / denom
                score_unc_by_key[self._ref_key(r)] = float(np.clip(unc, 0.0, 1.0))

        # 2) top-K selection
        refs_subset = refs[: min(int(cfg.num_refs), len(refs))]

        # 3) VGGT forward
        t = time.perf_counter()
        try:
            vggt_out = self.vggt.infer_pose_only(
                [query_path] + list(refs_subset),
                compact_for_alignment=False,
                include_reconstruction_outputs=True,
            )
        except Exception as e:
            result["status"] = "fail_vggt"
            result["warnings"].append(str(e))
            return result
        result["timings_ms"]["vggt"] = (time.perf_counter() - t) * 1000.0

        poses = vggt_out.get("poses", [])
        n_inputs = 1 + len(refs_subset)
        if not isinstance(poses, list) or len(poses) != n_inputs:
            result["status"] = "fail_vggt"
            result["warnings"].append(f"VGGT pose count mismatch: expected {n_inputs}, got {len(poses) if isinstance(poses, list) else 'invalid'}")
            return result
        query_vggt_pose = poses[0]                       # input order: [query, ref0, ref1, ...]
        if "Rcw" not in query_vggt_pose:
            result["status"] = "fail_vggt"
            result["warnings"].append("query VGGT rotation missing")
            return result
        conf_means = self._track_conf_means(vggt_out, n_inputs)

        # resolve each ref once: COLMAP id, R_w, C_w, VGGT view index, VGGT (R_v, C_v), weight
        ref_records: List[Dict[str, Any]] = []
        for i, ref_path in enumerate(refs_subset):
            view = i + 1
            vp = poses[view]
            if "Rcw" not in vp:
                continue
            iid = self._resolve_colmap_id(ref_path, str(vp.get("name", ref_path.name)))
            if iid is None:
                continue
            Rw, Cw = self._colmap_Rw_Cw(iid)
            Rv = np.asarray(vp["Rcw"], dtype=np.float64).reshape(3, 3)
            Cv = self._query_center_v(vp)
            w = mapfree_consensus_weight(
                track_conf_mean=conf_means[view] if view < len(conf_means) else None,
                retrieval_rank=rank_by_key.get(self._ref_key(ref_path)),
                retrieval_count=retrieval_count,
                retrieval_score_uncertainty=score_unc_by_key.get(self._ref_key(ref_path)),
                sigma_base=cfg.weight_sigma_base,
            )
            ref_records.append({
                "image_id": iid, "view": view, "R_w": Rw, "C_w": Cw,
                "R_v": Rv, "C_v": Cv, "weight": w,
            })

        if len(ref_records) < 3:
            result["status"] = "fail_align"
            result["warnings"].append(f"too few COLMAP-resolved refs for alignment: {len(ref_records)}")
            return result

        # 4) Sim(3) alignment of VGGT-local -> COLMAP-world reference centers
        src = np.asarray([r["C_v"] for r in ref_records], dtype=np.float64)
        dst = np.asarray([r["C_w"] for r in ref_records], dtype=np.float64)
        wpair = np.asarray([r["weight"] for r in ref_records], dtype=np.float64)
        align_result, _attempts, fallback_used = estimate_alignment_with_fallback(
            src=src, dst=dst, transform_type=cfg.align_type, seed=cfg.align_seed,
            primary_iters=cfg.align_ransac_iters, primary_thresh=cfg.align_inlier_thresh_m,
            primary_min_inliers=cfg.align_min_inliers, fallback_iters=cfg.align_fallback_ransac_iters,
            fallback_thresh=cfg.align_fallback_inlier_thresh_m, fallback_min_inliers=cfg.align_fallback_min_inliers,
            weights=wpair,
        )
        if not align_result.ok:
            result["status"] = "fail_align"
            result["warnings"].append(f"alignment failed: {align_result.message}")
            return result
        scale_s = float(getattr(align_result, "scale", 1.0))
        result["align_scale"] = scale_s
        result["align_rmse"] = float(align_result.rmse) if np.isfinite(align_result.rmse) else None
        result["align_inliers"] = int(align_result.inliers)
        result["align_fallback_used"] = bool(fallback_used)
        # Sim(3) conditioning
        _cond = float(getattr(align_result, "condition_number", float("inf")))
        result["align_condition_number"] = _cond if np.isfinite(_cond) else None
        # Sim(3) pose: full alignment applied to the query
        try:
            _RS = np.asarray(align_result.R, dtype=np.float64).reshape(3, 3)
            _tS = np.asarray(align_result.t, dtype=np.float64).reshape(3)
            _Rqv = np.asarray(query_vggt_pose["Rcw"], dtype=np.float64).reshape(3, 3)
            _Cqv = self._query_center_v(query_vggt_pose)
            _C_align = (scale_s * (_RS @ _Cqv) + _tS).astype(np.float64)
            _R_align = _Rqv @ _RS.T
            result["ablation_C_align"] = [float(x) for x in _C_align]
            result["ablation_R_align_qvec"] = [float(x) for x in rotmat_to_qvec_wxyz(_R_align)]
        except Exception as _e:
            result["ablation_C_align"] = None

        # 5) reference-induced consensus -> R_cons + scalar C_cons
        consensus = reference_consensus_pose(
            ref_hyps=ref_records,
            query_R_v=np.asarray(query_vggt_pose["Rcw"], dtype=np.float64),
            query_C_v=self._query_center_v(query_vggt_pose),
            scale_s=scale_s,
            dump=True,   # dump per-ref lifted hypotheses (Cqi/w/keep)
        )
        if consensus is None:
            result["status"] = "fail_consensus"
            result["warnings"].append("reference consensus unavailable (too few refs / rotation mean failed)")
            return result
        result["consensus_resection"] = {
            "center_dispersion_m": consensus["center_dispersion_m"],
            "num_refs": consensus["num_refs"], "num_inliers": consensus["num_inliers"],
        }
        if "consensus_hyp" in consensus:
            result["consensus_hyp"] = consensus["consensus_hyp"]

        # 6) map-point-free covariance consensus -> C_cons (covariance) + posterior sigma
        mf_diag = self._run_mapfree(vggt_out, poses, ref_records, scale_s)
        result["mapfree_cov"] = mf_diag

        # 7) output pose = (R_cons, covariance-C_cons if available else scalar-C_cons)
        Rcw = np.asarray(consensus["Rcw"], dtype=np.float64).reshape(3, 3)
        if mf_diag.get("status") == "ok" and mf_diag.get("C_cons") is not None:
            C = np.asarray(mf_diag["C_cons"], dtype=np.float64).reshape(3)
            source = "mapfree_cov_consensus"
            mf_diag["used_for_pose"] = True
        else:
            C = np.asarray(consensus["C"], dtype=np.float64).reshape(3)
            source = "se3_consensus"
            mf_diag["used_for_pose"] = False
            mf_diag.setdefault("promote_skip_reason", f"mapfree_cov_status={mf_diag.get('status')}")
        tcw = (-Rcw @ C).astype(np.float64)
        qvec = rotmat_to_qvec_wxyz(Rcw)
        result["pose_world"] = {
            "format": "quat_t",
            "qvec_wxyz_world_to_cam": [float(x) for x in qvec],
            "t_world_to_cam": [float(x) for x in tcw],
            "camera_center_world": [float(x) for x in C],
            "source": source,
        }
        result["final_pose_source"] = source
        result["accepted"] = True
        result["timings_ms"]["total"] = (time.perf_counter() - t_total) * 1000.0
        return result

    def _run_mapfree(self, vggt_out: Dict[str, Any], poses: List[Dict[str, Any]],
                     ref_records: List[Dict[str, Any]], scale_s: float) -> Dict[str, Any]:
        """Map-point-free covariance consensus."""
        diag: Dict[str, Any] = {}
        recon = vggt_out.get("reconstruction") or {}
        tracks = vggt_out.get("tracks") or {}
        pts = recon.get("points_vggt")
        txy = tracks.get("track_xy_t")
        vis = tracks.get("vis_t")
        conf = tracks.get("conf_t")
        if pts is None or txy is None or vis is None:
            diag["status"] = "missing_tracks_or_pointmap"
            return diag

        def _np(x):
            return (x.detach().float().cpu().numpy().astype(np.float64)
                    if hasattr(x, "detach") else np.asarray(x, np.float64))

        pts = _np(pts); txy = _np(txy); vis = _np(vis)
        conf = _np(conf) if conf is not None else None
        fxo = tracks.get("frame_xy_to_original")

        ref_R_w = [r["R_w"] for r in ref_records]
        ref_C_w = [r["C_w"] for r in ref_records]
        ref_v = [int(r["view"]) for r in ref_records]
        if len(ref_v) < 2:
            diag["status"] = "too_few_refs"
            return diag
        try:
            res = mapfree_consensus_from_vggt(
                poses=poses, points_vggt=pts, track_xy=txy, track_vis=vis, track_conf=conf,
                ref_R_w=ref_R_w, ref_C_w=ref_C_w, ref_view_indices=ref_v, scale_s=scale_s,
                frame_xy_to_original=fxo,
                vis_thresh=self.cfg.mapfree_vis_thresh, conf_thresh=self.cfg.mapfree_conf_thresh,
                min_tracks=self.cfg.mapfree_min_tracks, nu=self.cfg.mapfree_nu,
                diag=diag,
            )
            diag["status"] = "ok" if res is not None else "no_consensus"
        except Exception as e:
            diag["status"] = f"error:{e}"
        return diag
