"""Aggregate result_s*.json files into numbers with bootstrap-over-actors intervals (paired across methods).

  python aggregate.py --runs DIR [DIR ...] --raw DIR_RAW [--out summary.json] [--md table.md]

Definitions (also in results_notes/C.md):
  utility  U   = emotion UAR (6 classes, chance 1/6) of the best retrained probe (max over linear/MLP/kNN/transformer),
                 pooled over the 91 sealed test actors (each actor is a test actor in exactly one fold).
  privacy  ID  = identity top-1 (91-way, chance 1/91) of the worst-case attacker (max over the attacker suite) on T1 (unseen
                 sentences) for actors the encoder never saw ('unseen'); also reported on encoder-train actors ('seen') and T3.
  leakage  L   = (ID - base) / (ID_raw - base), base = emotion-label-only identity guess; ID_raw = same statistic on raw features.
  Interval = 95% percentile bootstrap over actors (B=1000, the same actor resamples for every method, so differences are paired);
  mean over seeds inside each resample.  Seed spread is reported separately (min..max of the per-seed point estimates).
"""
import argparse, glob, json, os
import numpy as np

B = 1000
ACTORS = list(range(1001, 1092))
RS = np.random.RandomState(12345)
MULT = np.stack([np.bincount(RS.randint(0, 91, 91), minlength=91) for _ in range(B)]).astype(float)  # [B,91]
EMO_KINDS = ["linear", "mlp", "knn", "lda", "lgbm", "transformer", "tdnn"]


def load(dirpath):
    out = []
    for f in sorted(glob.glob(os.path.join(dirpath, "result_s*.json"))):
        out.append(json.load(open(f)))
    return out


def _arr_counts(d):
    """dict {actor: [c, n]} -> (c[91], n[91])"""
    c = np.zeros(91); n = np.zeros(91)
    for a, (cc, nn) in d.items():
        i = int(a) - 1001
        c[i] += cc; n[i] += nn
    return c, n


def top1(res, kind, T="T1", grp="unseen"):
    c, n = _arr_counts(res["id"][kind][T][grp])
    return c, n


def boot_ratio(c, n):
    """accuracy for each bootstrap resample, plus point estimate"""
    pt = c.sum() / max(n.sum(), 1)
    return (MULT @ c) / np.maximum(MULT @ n, 1), pt


def emo_uar_curve(res, kind):
    C = np.zeros((91, 6, 6))
    for a, m in res["emo"][kind].items():
        C[int(a) - 1001] = np.array(m)
    tot = np.einsum("ba,acd->bcd", MULT, C)
    rec = np.diagonal(tot, axis1=1, axis2=2) / np.maximum(tot.sum(2), 1)
    pt = C.sum(0); ptr = np.diag(pt) / np.maximum(pt.sum(1), 1)
    return rec.mean(1), float(ptr.mean())


def emo_subset_curve(res, kind, classes):
    """mean per-class recall over `classes` (probe trained on all six emotions on the released z), bootstrap over actors"""
    C = np.zeros((91, 6, 6))
    for a, m in res["emo"][kind].items():
        C[int(a) - 1001] = np.array(m)
    tot = np.einsum("ba,acd->bcd", MULT, C)
    rec = np.diagonal(tot, axis1=1, axis2=2) / np.maximum(tot.sum(2), 1)
    pt = C.sum(0); ptr = np.diag(pt) / np.maximum(pt.sum(1), 1)
    return rec[:, classes].mean(1), float(ptr[classes].mean())


def emo_subset(runs, classes):
    best = None
    for k in kinds_present(runs[0], "emo"):
        b, p, _ = stat_over_seeds(runs, lambda r: emo_subset_curve(r, k, classes))
        if best is None or p > best[1]:
            best = (b, p, k)
    return dict(point=best[1], ci=ci(best[0]), kind=best[2])


def attr_ba(res, name, kind, ncls):
    C = np.zeros((91, ncls, ncls))
    for a, m in res["secrets"][name][kind].items():
        C[int(a) - 1001] = np.array(m)
    tot = np.einsum("ba,acd->bcd", MULT, C)
    rec = np.diagonal(tot, axis1=1, axis2=2) / np.maximum(tot.sum(2), 1)
    present = C.sum((0, 2)) > 0
    pt = C.sum(0)
    ptr = (np.diag(pt) / np.maximum(pt.sum(1), 1))[present].mean()
    return rec[:, present].mean(1), float(ptr)


def attr_age(res, kind):
    s = np.zeros(91); n = np.zeros(91)
    for a, (e, k) in res["secrets"]["age"][kind].items():
        s[int(a) - 1001] = e; n[int(a) - 1001] = k
    return (MULT @ s) / np.maximum(MULT @ n, 1), s.sum() / max(n.sum(), 1)


def stat_over_seeds(runs, fn):
    """fn(res) -> (boot[B], point). Returns mean boot [B], point mean, per-seed points."""
    bs, pts = [], []
    for r in runs:
        b, p = fn(r)
        bs.append(b); pts.append(p)
    return np.mean(bs, 0), float(np.mean(pts)), [float(x) for x in pts]


def kinds_present(res, root):
    return [k for k in EMO_KINDS if k in res[root]]


def worst_id(res, T="T1", grp="unseen"):
    bs, pts = [], []
    for k in kinds_present(res, "id"):
        b, p = boot_ratio(*top1(res, k, T, grp))
        bs.append(b); pts.append(p)
    i = int(np.argmax(pts))
    return np.max(bs, 0), float(max(pts)), kinds_present(res, "id")[i]


def best_emo(res):
    bs, pts = [], []
    ks = kinds_present(res, "emo")
    for k in ks:
        b, p = emo_uar_curve(res, k)
        bs.append(b); pts.append(p)
    i = int(np.argmax(pts))
    return np.max(bs, 0), float(max(pts)), ks[i]


def label_only(res, T="T1", grp="unseen"):
    return boot_ratio(*top1(res, "label_only", T, grp))


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarize(runs, raw_runs):
    """runs/raw_runs: lists of result dicts (one per seed)."""
    S = dict(name=runs[0]["name"], method=runs[0]["method"], cfg=runs[0]["cfg"], seeds=[r["seed"] for r in runs],
             shuffle=runs[0].get("shuffle"), commit=runs[0].get("commit"), backbone=runs[0].get("backbone"))
    # utility
    b, p, _ = stat_over_seeds(runs, lambda r: best_emo(r)[:2])
    S["emo_uar"] = dict(point=p, ci=ci(b), per_seed=[best_emo(r)[1] for r in runs], best_kind=[best_emo(r)[2] for r in runs])
    S["emo_recall_trained_classes_ANG_HAP_NEU_SAD"] = emo_subset(runs, [0, 3, 4, 5])
    S["emo_recall_DIS_FEA"] = emo_subset(runs, [1, 2])
    S["emo_uar_by_kind"] = {k: float(np.mean([emo_uar_curve(r, k)[1] for r in runs])) for k in kinds_present(runs[0], "emo")}
    if "head" in runs[0]["emo"]:
        S["emo_uar_head"] = float(np.mean([emo_uar_curve(r, "head")[1] for r in runs]))
    # identity (worst-case attacker), unseen/seen, T1/T2/T3
    S["id"] = {}
    for T in ("T1", "T2", "T3"):
        for grp in ("unseen", "seen"):
            if top1(runs[0], kinds_present(runs[0], "id")[0], T, grp)[1].sum() == 0:
                S["id"][f"{T}_{grp}"] = dict(point=float("nan"), ci=[float("nan")] * 2, per_seed=[], worst_kind=[])
                continue
            b, p, _ = stat_over_seeds(runs, lambda r: worst_id(r, T, grp)[:2])
            S["id"][f"{T}_{grp}"] = dict(point=p, ci=ci(b), per_seed=[worst_id(r, T, grp)[1] for r in runs],
                                         worst_kind=[worst_id(r, T, grp)[2] for r in runs])
    S["id_by_kind_T1_unseen"] = {k: float(np.mean([boot_ratio(*top1(r, k))[1] for r in runs])) for k in kinds_present(runs[0], "id")}
    lo_b, lo_p, _ = stat_over_seeds(runs, lambda r: label_only(r))
    S["id_label_only_T1_unseen"] = dict(point=lo_p, ci=ci(lo_b))
    # normalised leakage against raw reference (paired resamples)
    if raw_runs:
        rb, rp, _ = stat_over_seeds(raw_runs, lambda r: worst_id(r, "T1", "unseen")[:2])
        S["raw_ref_id_T1"] = dict(point=rp, ci=ci(rb))
        b, p, _ = stat_over_seeds(runs, lambda r: worst_id(r, "T1", "unseen")[:2])
        base_b, base_p = lo_b, lo_p
        L = (b - base_b) / np.maximum(rb - base_b, 1e-9)
        Lp = (p - base_p) / max(rp - base_p, 1e-9)
        S["leakage_T1_unseen"] = dict(point=Lp, ci=ci(L))
        # per-seed leakage points
        S["leakage_T1_unseen"]["per_seed"] = [(worst_id(r, "T1", "unseen")[1] - base_p) / max(rp - base_p, 1e-9) for r in runs]
        # utility retention vs raw
        eb, ep, _ = stat_over_seeds(raw_runs, lambda r: best_emo(r)[:2])
        ub, up, _ = stat_over_seeds(runs, lambda r: best_emo(r)[:2])
        S["raw_ref_emo_uar"] = dict(point=ep, ci=ci(eb))
        S["utility_retention"] = dict(point=(up - 1 / 6) / (ep - 1 / 6), ci=ci((ub - 1 / 6) / (eb - 1 / 6)))
    # secondary secrets
    if "secrets" in runs[0] and runs[0]["secrets"]:
        sec = {}
        for name, ncls in (("sex", 2), ("race", 3)):
            best = None
            for k in ("linear", "mlp"):
                b, p, _ = stat_over_seeds(runs, lambda r: attr_ba(r, name, k, ncls))
                if best is None or p > best[1]:
                    best = (b, p, k)
            e = dict(point=best[1], ci=ci(best[0]), kind=best[2])
            if raw_runs and "secrets" in raw_runs[0] and raw_runs[0]["secrets"]:
                rbest = max((stat_over_seeds(raw_runs, lambda r: attr_ba(r, name, k, ncls)) for k in ("linear", "mlp")), key=lambda t: t[1])
                chance = 1.0 / ncls
                e["leakage"] = dict(point=(best[1] - chance) / max(rbest[1] - chance, 1e-9),
                                    ci=ci((best[0] - chance) / np.maximum(rbest[0] - chance, 1e-9)), raw=rbest[1])
            sec[name] = e
        best = None
        for k in ("ridge", "mlp"):
            b, p, _ = stat_over_seeds(runs, lambda r: attr_age(r, k))
            if best is None or p < best[1]:
                best = (b, p, k)
        cb, cp, _ = stat_over_seeds(runs, lambda r: attr_age(r, "const_mean"))
        e = dict(mae=best[1], ci=ci(best[0]), kind=best[2], const_mean_mae=cp)
        if raw_runs and "secrets" in raw_runs[0] and raw_runs[0]["secrets"]:
            rbest = min((stat_over_seeds(raw_runs, lambda r: attr_age(r, k)) for k in ("ridge", "mlp")), key=lambda t: t[1])
            e["leakage"] = dict(point=(cp - best[1]) / max(cp - rbest[1], 1e-9), ci=ci((cb - best[0]) / np.maximum(cb - rbest[0], 1e-9)), raw_mae=rbest[1])
        sec["age"] = e
        S["secrets"] = sec
    # rate and in-pipeline diagnostics
    if "folds" in runs[0] and runs[0]["folds"] and "rate" in runs[0]["folds"][0]:
        S["rate"] = {k: float(np.mean([f["rate"][k] for r in runs for f in r["folds"] if f["rate"].get(k) is not None]))
                     for k in runs[0]["folds"][0]["rate"] if runs[0]["folds"][0]["rate"][k] is not None}
        S["val_uar_head"] = [f["val_uar_head"] for f in runs[0]["folds"]]
    return S


def fmt(x, d=3):
    return f"{x:.{d}f}"


def row(S):
    f = lambda e: f"{e['point']:.3f} [{e['ci'][0]:.3f},{e['ci'][1]:.3f}]"
    L = f(S["leakage_T1_unseen"]) if "leakage_T1_unseen" in S else "-"
    r = S.get("rate", {})
    cfg = S.get("cfg", {})
    if cfg.get("release") in ("posterior", "label"):
        rate = "6 post." if cfg["release"] == "posterior" else f"{np.log2(6):.1f} b"
    elif cfg.get("bott") == "vq":
        rate = f"{cfg.get('vq_groups', 1) * np.log2(cfg.get('vq_codes', 64)):.0f} b"
    elif cfg.get("bott") == "vib" and cfg.get("beta", 0) > 0:
        rate = f"{r['kl_nats'] / np.log(2):.0f} b KL, d={cfg.get('dz')}"
    elif cfg.get("dz"):
        rate = f"d={cfg['dz']}" + (f"/seg{cfg['pool']}" if cfg.get("pool", "utt") != "utt" else "")  # width only, no rate penalty
    else:
        rate = "-"
    idp = S["id"]["T1_unseen"]["point"]
    return (f"| {S['name']} | {len(S['seeds'])} | {f(S['emo_uar'])} | {f(S['id']['T1_unseen'])} | {idp * 91:.1f}x | {L} | "
            f"{S['id']['T1_seen']['point']:.3f} | {S['id']['T3_unseen']['point']:.3f} | {rate} |")


HEAD = ("| run | seeds | emotion UAR [95% CI] | identity top-1, worst attacker, T1 unseen [95% CI] | x chance | leakage L [95% CI] | "
        "id T1 seen | id T3 unseen | release size |\n|---|---:|---|---|---:|---|---:|---:|---:|")


def detail(runs):
    """per-attacker table: identity top-1 on T1/T2/T3 (x chance, bootstrap CI over actors) and emotion UAR per probe."""
    lines = ["| run | attacker | id T1 | x chance | id T2 | id T3 | emotion UAR |", "|---|---|---|---:|---|---|---|"]
    ks = [k for k in EMO_KINDS if k in runs[0]["id"]]
    for k in ks + ["label_only"]:
        cells = []
        for T in ("T1", "T2", "T3"):
            b, p, _ = stat_over_seeds(runs, lambda r: boot_ratio(*top1(r, k, T, "unseen")))
            cells.append((p, ci(b)))
        emo = ""
        if k in runs[0]["emo"]:
            b, p, _ = stat_over_seeds(runs, lambda r: emo_uar_curve(r, k))
            emo = f"{p:.3f} [{ci(b)[0]:.3f},{ci(b)[1]:.3f}]"
        f = lambda c: f"{c[0]:.3f} [{c[1][0]:.3f},{c[1][1]:.3f}]"
        lines.append(f"| {runs[0]['name']} | {k} | {f(cells[0])} | {cells[0][0] * 91:.1f}x | {f(cells[1])} | {f(cells[2])} | {emo} |")
    return "\n".join(lines)


def compare(runs_a, runs_b):
    """Paired difference B - A (same actor resamples): worst-case identity T1 unseen and best emotion UAR."""
    out = {}
    ba, pa, _ = stat_over_seeds(runs_a, lambda r: worst_id(r, "T1", "unseen")[:2])
    bb, pb, _ = stat_over_seeds(runs_b, lambda r: worst_id(r, "T1", "unseen")[:2])
    out["d_identity_T1"] = dict(point=pb - pa, ci=ci(bb - ba))
    ea, qa, _ = stat_over_seeds(runs_a, lambda r: best_emo(r)[:2])
    eb, qb, _ = stat_over_seeds(runs_b, lambda r: best_emo(r)[:2])
    out["d_emotion_uar"] = dict(point=qb - qa, ci=ci(eb - ea))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None, help="A B : paired difference B - A")
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(load(a.compare[0]), load(a.compare[1])), indent=1))
        return
    raw = load(a.raw) if a.raw else None
    sums = []
    for d in a.runs:
        runs = load(d)
        if runs:
            sums.append(summarize(runs, raw))
    md = HEAD + "\n" + "\n".join(row(s) for s in sums)
    print(md)
    if a.detail:
        for d in a.runs:
            runs = load(d)
            if runs and "id" in runs[0]:
                print("\n" + detail(runs))
    if a.out:
        json.dump(sums, open(a.out, "w"), indent=1)
    if a.md:
        open(a.md, "w").write(md + "\n")


if __name__ == "__main__":
    main()
