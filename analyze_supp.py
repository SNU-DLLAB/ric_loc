#!/usr/bin/env python3
"""Compute the supplementary RIC-Loc tables from saved results.json."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_tables as AT
from analyze_tables import (SEVEN, CAMB, TWELVE, load_dataset, load_naver, _eligible, _is_elig,
                            _gt_center, set_sigjoint_loso, set_sigjoint_fixed, auroc)
from ricloc import evaluation as ev
from ricloc.gate import calibrate_sigjoint

BOOT_N = 2000
BOOT_SEED = 12345


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------- reliability signals ----------
def _sc(r):   # sigma_cons (higher = worse); reliability score = -sigma
    v = (r.get("mapfree_cov") or {}).get("sigma_new")
    return float(v) if (v is not None and np.isfinite(float(v))) else None


def _sd(r):   # sigma_disp
    v = (r.get("mapfree_cov") or {}).get("sigma_disp_cov")
    return float(v) if (v is not None and np.isfinite(float(v))) else None


def _frank(vals):
    """Fractional ascending rank in [0,1] (ties -> average rank). Larger value -> larger rank."""
    a = np.asarray(vals, float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0
    ranks = avg[inv]
    return ranks / max(len(a) - 1, 1)


def _rank_fusion_joint(el):
    """Per-eligible rank-fusion joint reliability = -max(rank(sigcons), rank(sigdisp)); higher=better.
    Rows missing either signal get None."""
    sc = [_sc(r) for r in el]; sd = [_sd(r) for r in el]
    ok = [i for i in range(len(el)) if sc[i] is not None and sd[i] is not None]
    out = [None] * len(el)
    if len(ok) >= 2:
        rc = _frank([sc[i] for i in ok]); rd = _frank([sd[i] for i in ok])
        for j, i in enumerate(ok):
            out[i] = -max(rc[j], rd[j])
    return out


def _auc_from(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    if labels.size < 2 or len(set(labels.tolist())) < 2:
        return None
    return ev._rank_auc(labels, scores)


# ---------- bootstrap ----------
def _boot_ci(labels, scores, stat="auc", n=BOOT_N, seed=BOOT_SEED):
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    m = labels.size
    if m < 2:
        return (None, None)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        lb = labels[idx]
        if stat == "auc":
            if len(set(lb.tolist())) < 2:
                continue
            vals.append(ev._rank_auc(lb.astype(bool), scores[idx]))
        else:  # mean (strict success)
            vals.append(float(lb.mean()))
    if not vals:
        return (None, None)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _boot_paired_delta_ci(labels, score_a, score_b, n=BOOT_N, seed=BOOT_SEED):
    """95% CI of AUROC(a) - AUROC(b) under a paired query resample."""
    labels = np.asarray(labels, bool); a = np.asarray(score_a, float); b = np.asarray(score_b, float)
    m = labels.size
    if m < 2:
        return (None, None)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        lb = labels[idx]
        if len(set(lb.tolist())) < 2:
            continue
        d.append(ev._rank_auc(lb, a[idx]) - ev._rank_auc(lb, b[idx]))
    if not d:
        return (None, None)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def _sig(lo, hi):
    if lo is None:
        return "n/a"
    star = "*" if (lo > 0 or hi < 0) else "ns"
    return f"[{lo:+.3f},{hi:+.3f}] {star}"


# ---------- per-dataset supplementary block ----------
def report_supp(name, rows, dataset):
    print(f"\n##### {name} (supplementary) #####")
    el = _eligible(rows)
    n_all = len(rows)
    n_lab = len([r for r in rows if r.get("success_label") is not None])
    succ_el = np.asarray([bool(r["success_label"]) for r in el], bool)

    # ||e||/sigma_cons calibration
    stds = []
    for r in el:
        gc = _gt_center(r); C = (r.get("mapfree_cov") or {}).get("C_cons"); s = _sc(r)
        if gc is None or C is None or s in (None, 0):
            continue
        e = float(np.linalg.norm(np.asarray(C, float) - gc))
        stds.append((e, e / s))
    if stds:
        e_arr = np.asarray([x[0] for x in stds]); r_arr = np.asarray([x[1] for x in stds]); s_arr = e_arr / r_arr
        within = lambda k: float(np.mean(e_arr < k * s_arr)) * 100.0
        print(f"[tab:cov_calib] median||e||/sigcons={_f(np.median(r_arr),2)} "
              f"<1sig={within(1):.1f}% <2sig={within(2):.1f}% <3sig={within(3):.1f}% (n={len(stds)})")
    else:
        print("[tab:cov_calib] n/a (no eligible query with GT + C_cons)")

    # rank-fusion joint + LOSO joint + sigcons
    rfj = _rank_fusion_joint(el)
    lab_rf = [bool(r["success_label"]) for r, v in zip(el, rfj) if v is not None]
    sc_rf = [v for v in rfj if v is not None]
    auc_rf = _auc_from(lab_rf, sc_rf)
    auc_ho = auroc(el, lambda r: -(r.get("_sj")) if r.get("_sj") is not None else None)
    auc_sc = auroc(el, lambda r: -_sc(r) if _sc(r) is not None else None)
    auc_sd = auroc(el, lambda r: -_sd(r) if _sd(r) is not None else None)
    print(f"[tab:supp_openworld] sigjoint(heldout)={_f(auc_ho)} sigjoint(rankfusion)={_f(auc_rf)} "
          f"sigcons-alone={_f(auc_sc)}")

    # bootstrap CIs (rank-fusion joint, sigcons, strict success)
    ci_j = _boot_ci(np.asarray(lab_rf, bool), np.asarray(sc_rf, float), "auc")
    sc_lab = [bool(r["success_label"]) for r in el if _sc(r) is not None]
    sc_val = [-_sc(r) for r in el if _sc(r) is not None]
    ci_c = _boot_ci(np.asarray(sc_lab, bool), np.asarray(sc_val, float), "auc")
    # strict success bootstrap: eligible-gated full-set (ineligible=failure)
    succ_full = np.asarray([1.0 if (_is_elig(r) and r.get("success_label") is True) else 0.0
                            for r in rows if r.get("success_label") is not None], float)
    ci_s = _boot_ci(succ_full, succ_full, "mean")
    fmt = lambda ci: "n/a" if ci[0] is None else f"[{ci[0]:.3f},{ci[1]:.3f}]"
    print(f"[tab:supp_ci] jointAUROC={fmt(ci_j)} sigconsAUROC={fmt(ci_c)} strict={fmt(ci_s)}")

    # full-set signals (ineligible ranked last, counted as failure)
    def _fullset_auc(score_of):
        lb, sc = [], []
        for r in rows:
            if r.get("success_label") is None and r.get("status") == "ok":
                continue
            lb.append(bool(_is_elig(r) and r.get("success_label") is True))
            v = score_of(r); sc.append(v if v is not None else -np.inf)
        return _auc_from(lb, sc)
    # rank-fusion joint over eligible, ineligible -> -inf
    rf_map = {id(r): v for r, v in zip(el, rfj)}
    fs_j = _fullset_auc(lambda r: rf_map.get(id(r)))
    fs_c = _fullset_auc(lambda r: (-_sc(r) if _sc(r) is not None else None))
    fs_d = _fullset_auc(lambda r: (-_sd(r) if _sd(r) is not None else None))
    fs_gate = _fullset_auc(lambda r: (0.0 if _is_elig(r) else None))  # gate-only: eligible better than ineligible

    # full-set joint/sigcons + paired Delta(joint-sigcons) on eligible
    both = [(bool(r["success_label"]), v, -_sc(r)) for r, v in zip(el, rfj)
            if v is not None and _sc(r) is not None]
    if both:
        lb = np.asarray([x[0] for x in both], bool)
        d_ci = _boot_paired_delta_ci(lb, [x[1] for x in both], [x[2] for x in both])
    else:
        d_ci = (None, None)
    print(f"[tab:significance] elig%={len(el)/max(n_all,1)*100:.1f} fullset joint={_f(fs_j)}/sigcons={_f(fs_c)} "
          f"Delta(joint-sigcons)={_sig(*d_ci)}")

    # gate-only vs sigcons vs sigdisp (full-stream) + within-elig + Deltas
    win_c = auc_sc; win_d = auc_sd
    dc_gate = _boot_paired_delta_ci(*_paired_fullset(rows, rf_map, which=("sigcons", "gate")))
    dc_disp = None
    both2 = [(bool(r["success_label"]), -_sc(r), -_sd(r)) for r in el if _sc(r) is not None and _sd(r) is not None]
    if both2:
        lb2 = np.asarray([x[0] for x in both2], bool)
        dc_disp = _boot_paired_delta_ci(lb2, [x[1] for x in both2], [x[2] for x in both2])
    print(f"[tab:fullstream] gate-only={_f(fs_gate)} sigcons={_f(fs_c)} sigdisp={_f(fs_d)} | "
          f"within-elig sigcons/sigdisp={_f(win_c)}/{_f(win_d)} | "
          f"D(cons-gate)={_sig(*dc_gate) if dc_gate else 'n/a'} D(cons-disp)={_sig(*dc_disp) if dc_disp else 'n/a'}")

    # sigma_cons deciles strict-failure rate + Spearman
    pairs = [(_sc(r), 0.0 if r.get("success_label") is True else 1.0) for r in el if _sc(r) is not None]
    if len(pairs) >= 10:
        s_vals = np.asarray([p[0] for p in pairs]); fail = np.asarray([p[1] for p in pairs])
        order = np.argsort(s_vals, kind="stable")
        deciles = np.array_split(order, 10)
        drates = [float(fail[d].mean()) if len(d) else float("nan") for d in deciles]
        # Spearman = Pearson of ranks
        rs = _frank(s_vals); rf = _frank(fail)
        rs -= rs.mean(); rf -= rf.mean()
        den = float(np.linalg.norm(rs) * np.linalg.norm(rf))
        sp = float(np.dot(rs, rf) / den) if den > 1e-12 else None
        print("[tab:supp_calib] deciles(low->high)=" + " ".join(f"{x:.2f}" for x in drates) +
              f" Spearman={_f(sp,2)}")
    else:
        print("[tab:supp_calib] n/a (need >=10 eligible)")

    # consensus-inlier-count AUROC vs sigcons
    def _inlier(r):
        v = (r.get("consensus_resection") or {}).get("num_inliers")
        return float(v) if v is not None else None
    a_pos = auroc(el, lambda r: _inlier(r))
    a_neg = auroc(el, lambda r: (-_inlier(r) if _inlier(r) is not None else None))
    a_in = None if (a_pos is None or a_neg is None) else max(a_pos, a_neg)
    print(f"[tab:supp_inlier] inlier-count={_f(a_in)} sigcons={_f(auc_sc)}")

    # covariance-eligible% over all queries + cause breakdown
    n_elig = sum(1 for r in rows if _is_elig(r))          # covariance-eligible (sigma_cons finite)
    inelig = [r for r in rows if not _is_elig(r)]
    from collections import Counter
    causes = Counter((r.get("mapfree_cov") or {}).get("status", "unknown") for r in inelig)
    top = ", ".join(f"{k}:{v}" for k, v in causes.most_common(3)) if causes else "---"
    print(f"[tab:supp_elig] eligible={n_elig/max(n_all,1)*100:.1f}% ineligible={len(inelig)/max(n_all,1)*100:.1f}% "
          f"| dominant cause: {top}")

    # median per-stage timings (recorded: retrieval/vggt/total)
    def _stage(key):
        xs = [float((r.get("timings_ms") or {}).get(key)) for r in rows
              if (r.get("timings_ms") or {}).get(key) is not None]
        return float(np.median(xs)) if xs else None
    st = {k: _stage(k) for k in ["preproc", "retrieval", "selection", "vggt", "align", "refine", "total"]}
    print("[tab:supp_runtime] " + " ".join(f"{k}={'n/a' if st[k] is None else f'{st[k]:.0f}'}ms" for k in st) +
          "  (only retrieval/vggt/total are instrumented in this package)")


def _paired_fullset(rows, rf_map, which):
    """Full-stream paired arrays (label, score_cons, score_gate) for the cons-vs-gate delta."""
    lb, a, b = [], [], []
    for r in rows:
        if r.get("success_label") is None and r.get("status") == "ok":
            continue
        lb.append(bool(_is_elig(r) and r.get("success_label") is True))
        a.append((-_sc(r)) if _sc(r) is not None else -1e18)  # sigcons full-stream (ineligible worst)
        b.append(0.0 if _is_elig(r) else -1e18)               # gate-only
    return np.asarray(lb, bool), a, b


def report_12scenes_perscene(rows):
    """Per-scene 12-Scenes median t (cm) / strict@5cm/5deg."""
    print("\n##### 12-Scenes per-scene (tab:supp_12scenes, ours) #####")
    tm, rd = AT.FROZEN.success_threshold_for("12scenes")
    all_t = []
    n_succ_pool = 0; n_pool = 0
    for sc in TWELVE:
        rs = [r for r in rows if r.get("_scene") == sc and r.get("success_label") is not None]
        if not rs:
            print(f"  {sc:<18} n/a"); continue
        te = np.asarray([float(r["translation_error"]) for r in rs], float)
        strict = np.mean([1.0 if (_is_elig(r) and r.get("success_label") is True) else 0.0 for r in rs])
        all_t.append(te); n_succ_pool += int(sum(1 for r in rs if _is_elig(r) and r.get("success_label") is True))
        n_pool += len(rs)
        print(f"  {sc:<18} {np.median(te)*100:.2f}cm / {strict:.3f}")
    if all_t:
        pooled = np.concatenate(all_t)
        print(f"  {'POOLED':<18} {np.median(pooled)*100:.2f}cm / {n_succ_pool/max(n_pool,1):.3f} (n={n_pool})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result/probe_keepweak")
    ap.add_argument("--gt7", default="/media/dllab/HDD/data/7scenes/processed")
    ap.add_argument("--gtc", default="/media/dllab/HDD/data/cambridge/processed")
    ap.add_argument("--gt12", default="/media/dllab/HDD/data/12scenes/processed")
    ap.add_argument("--gtnb1", default=None)
    ap.add_argument("--gtnb2", default=None)
    ap.add_argument("--only", default="", help="7scenes|cambridge|12scenes|naver subset")
    args = ap.parse_args()

    if not args.only or args.only == "7scenes":
        r7 = load_dataset(args.root, "7scenes", SEVEN, args.gt7 + "/{scene}/queries_full/gt_poses.txt", "7scenes")
        if r7:
            set_sigjoint_loso(r7); report_supp("7-Scenes", r7, "7scenes")
    if not args.only or args.only == "cambridge":
        rc = load_dataset(args.root, "cambridge", CAMB, args.gtc + "/{scene}/queries_full/gt_poses.txt", "cambridge")
        if rc:
            set_sigjoint_loso(rc); report_supp("Cambridge", rc, "cambridge")
    if not args.only or args.only == "12scenes":
        r12 = load_dataset(args.root, "12scenes", TWELVE, args.gt12 + "/{scene}/queries/gt_poses.txt", "12scenes")
        if r12:
            set_sigjoint_loso(r12); report_supp("12-Scenes", r12, "12scenes"); report_12scenes_perscene(r12)
    if not args.only or args.only == "naver":
        try:
            b1 = load_naver(args.root, "b1", args.gtnb1); b2 = load_naver(args.root, "b2", args.gtnb2)
            for r in b1:
                r["_scene"] = "b1"
            for r in b2:
                r["_scene"] = "b2"
            set_sigjoint_fixed(b1, calibrate_sigjoint(b2)); set_sigjoint_fixed(b2, calibrate_sigjoint(b1))
            report_supp("NAVER-B2 (calib=B1)", b2, "naver"); report_supp("NAVER-B1 (calib=B2)", b1, "naver")
        except FileNotFoundError as e:
            print(f"[naver] skipped: {e}")

    print("\n[note] tab:backbone (DUSt3R/MASt3R) and tab:supp_ksweep (K-sweep) need external backbones / "
          "sweep runs -> not on this one-command path (see README).")


if __name__ == "__main__":
    main()
