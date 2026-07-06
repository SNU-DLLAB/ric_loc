#!/usr/bin/env python3
"""Compute RIC-Loc accuracy / reject / risk-coverage tables from saved results.json."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ricloc import evaluation as ev
from ricloc.gate import calibrate_sigjoint, decide_sigjoint_selective_policy
from ricloc.config import FROZEN

SEVEN = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
CAMB = ["greatcourt", "kingscollege", "oldhospital", "shopfacade", "stmaryschurch"]
TWELVE = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke",
          "office1_gates362", "office1_gates381", "office1_lounge", "office1_manolis",
          "office2_5a", "office2_5b"]


# ---------- loading + per-query error labelling ----------
def _label(rows, dataset, gt):
    tm, rd = FROZEN.success_threshold_for(dataset)
    thr = {"profile": f"{dataset}_strict", "trans_m": tm, "rot_deg": rd}
    for r in rows:
        if r.get("translation_error") is None and r.get("status") == "ok" and gt:
            pw = r.get("pose_world")
            g = ev.match_gt_pose(gt, r.get("query_name", "")) or ev.match_gt_pose(gt, r.get("query_path", ""))
            if pw and g:
                e = ev.compute_pose_errors(pw, g)
                if e:
                    r["translation_error"] = e["translation_error"]
                    r["rotation_error_deg"] = e["rotation_error_deg"]
        if r.get("translation_error") is not None and r.get("rotation_error_deg") is not None:
            ev.add_success_label(r, thr)
        r["_gt"] = (ev.match_gt_pose(gt, r.get("query_name", "")) or
                    ev.match_gt_pose(gt, r.get("query_path", ""))) if gt else None
    return thr


def load_dataset(root, name, scenes, gt_tmpl, dataset):
    rows = []
    for sc in scenes:
        fp = Path(root) / name / sc / "results.json"
        if not fp.exists():
            print(f"  [warn] missing {fp}"); continue
        d = json.loads(fp.read_text())
        gt = ev.load_gt_poses(Path(gt_tmpl.format(scene=sc))) if gt_tmpl else {}
        _label(d, dataset, gt)
        for r in d:
            r["_scene"] = sc
        rows += d
    return rows


def load_naver(root, b, gt_path=None):
    # accept both layouts: <root>/naver/<b>/results.json and <root>/naver_<b>/results.json
    cands = [Path(root) / "naver" / b / "results.json", Path(root) / f"naver_{b}" / "results.json"]
    fp = next((c for c in cands if c.exists()), cands[0])
    d = json.loads(fp.read_text())
    thr = {"profile": "naver_strict", "trans_m": 0.25, "rot_deg": 2.0}
    gt = ev.load_gt_poses(Path(gt_path)) if gt_path else {}
    for r in d:
        if r.get("translation_error") is not None and r.get("rotation_error_deg") is not None:
            ev.add_success_label(r, thr)
        # GT center for the estimator comparison
        r["_gt"] = (ev.match_gt_pose(gt, r.get("query_name", "")) or
                    ev.match_gt_pose(gt, r.get("query_path", ""))) if gt else None
    return d


# ---------- sigma_joint (leave-one-scene-out) ----------
def set_sigjoint_loso(rows):
    scenes = sorted({r["_scene"] for r in rows})
    for held in scenes:
        others = [r for r in rows if r["_scene"] != held]
        try:
            cal = calibrate_sigjoint(others)
        except ValueError:
            continue
        for r in [x for x in rows if x["_scene"] == held]:
            _apply_sj(r, cal)


def set_sigjoint_fixed(rows, cal):
    for r in rows:
        _apply_sj(r, cal)


def _apply_sj(r, cal):
    mc = r.get("mapfree_cov") or {}
    dd = decide_sigjoint_selective_policy(mc.get("sigma_new"), mc.get("sigma_disp_cov"),
                                          median_cons=cal["median_cons"], median_disp=cal["median_disp"],
                                          tau_accept=cal.get("tau_accept", float("inf")))
    sj = dd.get("sigma_joint")
    r["_sj"] = float(sj) if (sj is not None and np.isfinite(sj)) else None


# ---------- signal extraction ----------
def _retrieval_gap(r):
    # top-1 - top-2 gap over MegaLoc squared-L2 distances (larger => more confident)
    s = r.get("retrieved_ref_scores")
    if not s or len(s) < 2:
        return None
    s = sorted(float(x) for x in s)  # ascending squared-L2 distances; s[0]=d_1, s[1]=d_2
    return s[1] - s[0]


def _is_elig(r):
    return (r.get("mapfree_cov") or {}).get("sigma_new") is not None


def _eligible(rows):
    return [r for r in rows if _is_elig(r) and r.get("success_label") is not None]


def auroc(rows, score_fn):
    lab, sc = [], []
    for r in rows:
        v = score_fn(r)
        if v is None or not np.isfinite(float(v)):
            continue
        lab.append(bool(r["success_label"])); sc.append(float(v))
    lab = np.asarray(lab, bool); sc = np.asarray(sc, float)
    if lab.size < 2 or len(set(lab.tolist())) < 2:
        return None
    return ev._rank_auc(lab, sc)


# ---------- estimator comparison (R_cons fixed; only center varies) ----------
def _student_t_isotropic(Cqi, w, nu=5.0, iters=20):
    C = (w[:, None] * Cqi).sum(0) / max(w.sum(), 1e-9)
    for _ in range(iters):
        d2 = ((Cqi - C) ** 2).sum(1)
        scale = max(np.median(d2), 1e-9)
        om = (nu + 3.0) / (nu + d2 / scale)
        ww = om * w
        Cn = (ww[:, None] * Cqi).sum(0) / max(ww.sum(), 1e-9)
        if np.linalg.norm(Cn - C) < 1e-9:
            C = Cn; break
        C = Cn
    return C


def estimator_centers(r):
    """Return dict of center estimates for one query (or None if enriched fields absent)."""
    ch = r.get("consensus_hyp")
    if not ch or not ch.get("Cqi"):
        return None
    Cqi = np.asarray(ch["Cqi"], float)
    w = np.asarray(ch.get("w", [1.0] * len(Cqi)), float)
    keep = np.asarray(ch.get("keep_rot", [True] * len(Cqi)), bool)
    if keep.sum() < 1:
        keep = np.ones(len(Cqi), bool)
    Ck, wk = Cqi[keep], w[keep]
    out = {}
    ca = r.get("ablation_C_align")
    out["align"] = np.asarray(ca, float) if ca else None
    out["plain"] = Ck.mean(0)
    out["weighted"] = (wk[:, None] * Ck).sum(0) / max(wk.sum(), 1e-9)
    out["robust"] = _student_t_isotropic(Ck, wk)
    mc = (r.get("mapfree_cov") or {}).get("C_cons")
    out["covariance"] = np.asarray(mc, float) if mc is not None else out["weighted"]
    return out


def _gt_center(r):
    g = r.get("_gt")
    if not g:
        return None
    q = g.get("qvec_wxyz_world_to_cam", g.get("qwqxqyqz"))
    t = g.get("t_world_to_cam", g.get("t_xyz"))
    c = g.get("camera_center_world")
    if c is not None:
        return np.asarray(c, float)
    if q is None or t is None:
        return None
    R = ev._qvec_to_rotmat(q)
    return ev._camera_center(R, np.asarray(t, float))


def ablation_strict(rows, dataset):
    tm, rd = FROZEN.success_threshold_for(dataset)
    keys = ["align", "plain", "weighted", "robust", "covariance"]
    succ = {k: [] for k in keys}
    n = 0
    for r in rows:
        if r.get("success_label") is None:
            continue
        gc = _gt_center(r)
        cen = estimator_centers(r)
        if gc is None or cen is None:
            continue
        rot_ok = float(r["rotation_error_deg"]) <= rd  # R_cons fixed across rows
        n += 1
        for k in keys:
            c = cen[k]
            if c is None:
                succ[k].append(False); continue
            te = float(np.linalg.norm(c - gc))
            succ[k].append(bool(te <= tm and rot_ok))
    return {k: (float(np.mean(v)) if v else None) for k, v in succ.items()}, n


# ---------- risk-coverage ----------
def risk_coverage(rows, dataset):
    el = _eligible(rows)
    sj = np.asarray([r.get("_sj") if r.get("_sj") is not None else np.inf for r in el], float)
    succ = np.asarray([bool(r["success_label"]) for r in el], bool)
    te = np.asarray([float(r["translation_error"]) for r in el], float)
    re = np.asarray([float(r["rotation_error_deg"]) for r in el], float)
    ct, cr = ev.catastrophic_threshold_for(dataset + "_strict")
    cat = (te > ct) | (re > cr)
    order = np.argsort(sj)
    n = len(el)
    pts = {}
    for cov in [1.0, 0.8, 0.7, 0.5]:
        k = max(1, int(np.ceil(cov * n))); ch = order[:k]
        pts[cov] = (1.0 - succ[ch].mean(), cat[ch].mean())
    # AURC: joint (sigma order), random (= full risk constant), oracle (true-error order)
    def aurc(ordr):
        covs, risks = [], []
        for k in range(1, n + 1):
            covs.append(k / n); risks.append(1.0 - succ[ordr[:k]].mean())
        return float(np.trapz(risks, covs))
    # oracle: accept every strict-correct query first (rank by success label)
    oracle_order = np.argsort(np.logical_not(succ), kind="stable")
    return pts, aurc(order), float(1.0 - succ.mean()), aurc(oracle_order)


# ---------- additional diagnostics ----------
def _neg_sigma(key):
    """score_fn: reliability = -sigma (low sigma = reliable). None when the field is absent."""
    def fn(r):
        v = (r.get("mapfree_cov") or {}).get(key)
        return -float(v) if (v is not None and np.isfinite(float(v))) else None
    return fn


def insample_vs_heldout_sigjoint(rows):
    """LOSO vs in-sample sigma_joint failure-AUROC and the |delta| in pp."""
    el = _eligible(rows)
    a_ho = auroc(el, lambda r: -(r["_sj"]) if r.get("_sj") is not None else None)
    cal = calibrate_sigjoint(rows)                       # in-sample: one fit on the whole pool
    for r in rows:
        mc = r.get("mapfree_cov") or {}
        dd = decide_sigjoint_selective_policy(mc.get("sigma_new"), mc.get("sigma_disp_cov"),
                                              median_cons=cal["median_cons"], median_disp=cal["median_disp"],
                                              tau_accept=cal.get("tau_accept", float("inf")))
        sj = dd.get("sigma_joint")
        r["_sj_is"] = float(sj) if (sj is not None and np.isfinite(sj)) else None
    a_is = auroc(el, lambda r: -(r["_sj_is"]) if r.get("_sj_is") is not None else None)
    delta = (abs(a_is - a_ho) * 100.0) if (a_is is not None and a_ho is not None) else None
    return a_ho, a_is, delta


def per_scene_sigcons_vs_sigdisp(rows, scenes):
    """Per-scene sigma_cons vs sigma_disp failure-AUROC and count of scenes with cons>disp."""
    per = []
    for sc in scenes:
        el = [r for r in rows if r.get("_scene") == sc and _is_elig(r) and r.get("success_label") is not None]
        per.append((sc, auroc(el, _neg_sigma("sigma_new")), auroc(el, _neg_sigma("sigma_disp_cov"))))
    defined = [(s, c, d) for s, c, d in per if c is not None and d is not None]
    wins = sum(1 for _, c, d in defined if c > d)
    return per, wins, len(defined)


def risk_coverage_fullset(rows, dataset):
    """Full-set risk-coverage: rank all queries by sigma_joint (ineligible/non-localized worst)."""
    ct, cr = ev.catastrophic_threshold_for(dataset + "_strict")

    def sj_key(r):
        v = r.get("_sj")
        return float(v) if (v is not None and np.isfinite(v)) else np.inf

    def is_succ(r):
        return bool(_is_elig(r) and r.get("success_label") is True)

    def is_cat(r):
        te, re = r.get("translation_error"), r.get("rotation_error_deg")
        if te is None or re is None:
            return True                                  # non-localized -> catastrophic failure
        return (float(te) > ct) or (float(re) > cr)

    srt = sorted(rows, key=sj_key)
    n = len(srt)
    if n == 0:
        return {c: (None, None) for c in [1.0, 0.8, 0.7, 0.5]}, None
    succ = np.asarray([is_succ(r) for r in srt], bool)
    cat = np.asarray([is_cat(r) for r in srt], bool)
    pts = {}
    for cov in [1.0, 0.8, 0.7, 0.5]:
        k = max(1, int(np.ceil(cov * n)))
        pts[cov] = (float(1.0 - succ[:k].mean()), float(cat[:k].mean()))
    covs = [k / n for k in range(1, n + 1)]
    risks = [float(1.0 - succ[:k].mean()) for k in range(1, n + 1)]
    return pts, float(np.trapz(risks, covs))


# ---------- reporting ----------
def acc_stats(rows, dataset):
    lab = [r for r in rows if r.get("success_label") is not None]
    n_all = len(rows)
    succ = np.asarray([bool(r["success_label"]) for r in lab], bool)
    te = np.asarray([float(r["translation_error"]) for r in lab], float)
    re = np.asarray([float(r["rotation_error_deg"]) for r in lab], float)
    # full-set recall is eligible-gated (ineligible query counts as a failure)
    n_succ_elig = sum(1 for r in lab if _is_elig(r) and bool(r["success_label"]))
    return dict(n_all=n_all, n_loc=len(lab),
                strict_full=n_succ_elig / max(n_all, 1),      # eligible-gated full-set recall
                strict_loc=succ.sum() / max(n_all, 1),        # any-localized recall (diagnostic)
                strict_cond=succ.mean(), medT=np.median(te), medR=np.median(re))


def reject_table(rows):
    el = _eligible(rows)
    return {
        "sigjoint": auroc(el, lambda r: -(r.get("_sj")) if r.get("_sj") is not None else None),
        "sigcons": auroc(el, lambda r: -(r.get("mapfree_cov") or {}).get("sigma_new")),
        "sigdisp": auroc(el, lambda r: -(r.get("mapfree_cov") or {}).get("sigma_disp_cov")),
        "centre_disp_scalar": auroc(el, lambda r: -(r.get("consensus_resection") or {}).get("center_dispersion_m")
                                    if (r.get("consensus_resection") or {}).get("center_dispersion_m") is not None else None),
        "sim3_cond": auroc(el, lambda r: -(r.get("align_condition_number")) if r.get("align_condition_number") is not None else None),
        "retrieval_gap": auroc(el, _retrieval_gap),
        "random": 0.5,
        "coverage": len(el) / max(len([r for r in rows if r.get("success_label") is not None]), 1),
    }


def fullset_auroc(rows, key):
    lab, sc = [], []
    for r in rows:
        if r.get("success_label") is None and r.get("status") == "ok":
            continue
        s = bool(r.get("success_label")) if r.get("success_label") is not None else False
        mc = r.get("mapfree_cov") or {}
        if key == "sj":
            v = r.get("_sj"); score = -float(v) if (v is not None and np.isfinite(v)) else -np.inf
        else:
            x = mc.get(key); score = -float(x) if (x is not None and np.isfinite(float(x))) else -np.inf
        lab.append(s); sc.append(score)
    return ev._rank_auc(np.asarray(lab, bool), np.asarray(sc, float))


def recall_buckets(rows, n_all=None):
    # eligible-gated: only covariance-eligible localized queries can count as recall successes
    lab = [r for r in rows if r.get("success_label") is not None and _is_elig(r)]
    te = np.asarray([float(r["translation_error"]) for r in lab], float)
    re = np.asarray([float(r["rotation_error_deg"]) for r in lab], float)
    N = n_all if n_all else len(rows)
    return {f"{tm}/{rd}": float(np.sum((te <= tm) & (re <= rd)) / max(N, 1))
            for tm, rd in [(0.1, 1.0), (0.25, 2.0), (1.0, 5.0)]}


def report_public(name, rows, dataset, scenes=None):
    print(f"\n##### {name} #####")
    a = acc_stats(rows, dataset)
    print(f"[tab:public] n_all={a['n_all']} n_loc={a['n_loc']} | strict full(elig-gated)={a['strict_full']:.4f} "
          f"loc={a['strict_loc']:.4f} cond={a['strict_cond']:.4f} | median {a['medT']*100:.1f}cm / {a['medR']:.2f}deg")
    if abs(a['strict_full'] - a['strict_cond']) > 1e-4:
        print(f"             note: paper tab:public strict = cond {a['strict_cond']:.4f} "
              f"(full-set eligible-gated = {a['strict_full']:.4f})")
    rj = reject_table(rows)
    print(f"[tab:reject] sigjoint={_f(rj['sigjoint'])} sigcons={_f(rj['sigcons'])} sigdisp={_f(rj['sigdisp'])} "
          f"centre_disp={_f(rj['centre_disp_scalar'])} sim3cond={_f(rj['sim3_cond'])} retr_gap={_f(rj['retrieval_gap'])} "
          f"cov={rj['coverage']*100:.1f}%")
    pts, aj, arand, aor = risk_coverage(rows, dataset)
    print(f"[tab:riskcov] " + " ".join(f"cov{c}={pts[c][0]:.3f}/{pts[c][1]:.3f}" for c in [1.0, 0.8, 0.7, 0.5]) +
          f" | AURC joint={aj:.3f} random={arand:.3f} oracle={aor:.3f}")
    ab, nab = ablation_strict(rows, dataset)
    print(f"[tab:matched] (n={nab}) align={_f(ab['align'])} plain={_f(ab['plain'])} weighted={_f(ab['weighted'])} "
          f"robust={_f(ab['robust'])} covariance={_f(ab['covariance'])}")
    print(f"[full-set] sigcons={_f(fullset_auroc(rows,'sigma_new'))} sigjoint={_f(fullset_auroc(rows,'sj'))}")
    # LOSO vs in-sample sigma_joint AUROC
    a_ho, a_is, delta = insample_vs_heldout_sigjoint(rows)
    print(f"[heldout-delta] sigjoint AUROC held-out={_f(a_ho)} in-sample={_f(a_is)} "
          f"|delta|={'n/a' if delta is None else f'{delta:.2f}pp'}")
    # per-scene sigma_cons vs sigma_disp
    if scenes:
        per, wins, ndef = per_scene_sigcons_vs_sigdisp(rows, scenes)
        print(f"[per-scene] cons>disp on {wins}/{ndef} scenes (defined-AUROC only):")
        for sc, c, d in per:
            mark = "" if (c is None or d is None) else ("  cons>disp" if c > d else "  disp>=cons")
            print(f"            {sc:<14} sigcons={_f(c)} sigdisp={_f(d)}{mark}")


def report_naver(name, rows, dataset="naver"):
    print(f"\n##### {name} #####")
    a = acc_stats(rows, dataset)
    rb = recall_buckets(rows, a["n_all"])
    el = _eligible(rows)
    print(f"[tab:naver] n_all={a['n_all']} cov={len(el)/a['n_all']*100:.1f}% | recall " +
          " ".join(f"{k}={v*100:.1f}%" for k, v in rb.items()) + f" | medT={a['medT']:.3f}m medR={a['medR']:.2f}deg")
    rj = reject_table(rows)
    print(f"[tab:reject] sigjoint={_f(rj['sigjoint'])} sigcons={_f(rj['sigcons'])} sigdisp={_f(rj['sigdisp'])} "
          f"centre_disp={_f(rj['centre_disp_scalar'])} sim3cond={_f(rj['sim3_cond'])} retr_gap={_f(rj['retrieval_gap'])}")
    pts, aj, arand, aor = risk_coverage(rows, dataset)
    print(f"[tab:riskcov] " + " ".join(f"cov{c}={pts[c][0]:.3f}/{pts[c][1]:.3f}" for c in [1.0, 0.8, 0.7, 0.5]) +
          f" | AURC joint={aj:.3f} random={arand:.3f} oracle={aor:.3f}")
    ab, nab = ablation_strict(rows, dataset)
    print(f"[tab:matched] (n={nab}) align={_f(ab['align'])} plain={_f(ab['plain'])} weighted={_f(ab['weighted'])} "
          f"robust={_f(ab['robust'])} covariance={_f(ab['covariance'])}")
    print(f"[full-set] sigcons={_f(fullset_auroc(rows,'sigma_new'))} sigjoint={_f(fullset_auroc(rows,'sj'))}")
    # full-set risk-coverage (ineligible ranked worst)
    fpts, faurc = risk_coverage_fullset(rows, dataset)
    print(f"[fullset-riskcov] " +
          " ".join(f"cov{c}={fpts[c][0]:.3f}/{fpts[c][1]:.3f}" for c in [1.0, 0.8, 0.7, 0.5]) +
          f" | AURC full={'n/a' if faurc is None else f'{faurc:.3f}'}  (strict-risk/catastrophic)")


def _f(x):
    return "n/a" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result/probe_keepweak")
    ap.add_argument("--gt7", default="/media/dllab/HDD/data/7scenes/processed")
    ap.add_argument("--gtc", default="/media/dllab/HDD/data/cambridge/processed")
    ap.add_argument("--gt12", default="/media/dllab/HDD/data/12scenes/processed")
    ap.add_argument("--gtnb1", default=None, help="NAVER-B1 gt_poses.txt (enables tab:matched NAVER column)")
    ap.add_argument("--gtnb2", default=None, help="NAVER-B2 gt_poses.txt")
    ap.add_argument("--only", default="", help="7scenes|cambridge|12scenes|naver subset")
    args = ap.parse_args()

    if not args.only or args.only == "7scenes":
        r7 = load_dataset(args.root, "7scenes", SEVEN, args.gt7 + "/{scene}/queries_full/gt_poses.txt", "7scenes")
        if r7:
            set_sigjoint_loso(r7); report_public("7-Scenes", r7, "7scenes", scenes=SEVEN)
    if not args.only or args.only == "cambridge":
        rc = load_dataset(args.root, "cambridge", CAMB, args.gtc + "/{scene}/queries_full/gt_poses.txt", "cambridge")
        if rc:
            set_sigjoint_loso(rc); report_public("Cambridge", rc, "cambridge", scenes=CAMB)
    if not args.only or args.only == "12scenes":
        # 12-Scenes uses queries/ (stride=1, already full); strict 5cm/5deg (config).
        r12 = load_dataset(args.root, "12scenes", TWELVE, args.gt12 + "/{scene}/queries/gt_poses.txt", "12scenes")
        if r12:
            set_sigjoint_loso(r12); report_public("12-Scenes", r12, "12scenes", scenes=TWELVE)
    if not args.only or args.only == "naver":
        try:
            b1, b2 = load_naver(args.root, "b1", args.gtnb1), load_naver(args.root, "b2", args.gtnb2)
            for r in b1:
                r["_scene"] = "b1"
            for r in b2:
                r["_scene"] = "b2"
            set_sigjoint_fixed(b1, calibrate_sigjoint(b2))  # cross-building
            set_sigjoint_fixed(b2, calibrate_sigjoint(b1))
            report_naver("NAVER-B2 (calib=B1)", b2); report_naver("NAVER-B1 (calib=B2)", b1)
        except FileNotFoundError as e:
            print(f"[naver] skipped: {e}")


if __name__ == "__main__":
    main()
