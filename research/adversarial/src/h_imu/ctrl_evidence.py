"""H3-016 (H.md 7.1): does independent per-window dither compose with evidence?  Account-level / K-window pooling of the stored fold-O posteriors.
python ctrl_evidence.py --dirs out/stage0,out/def,runs/h_nn_aae,... --ns1 out/ctrl/ns1 --out OUT.json
No model is retrained here; posteriors come from the frozen first-pass suite (seed-averaged); evidence rule = sum of log((p + 1e-3)/(1 + 24e-3)) over the bag."""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import datasets as D

EPS = 1e-3; NS = 24; CH = 1.0 / NS
MAIN_TAB = ["linear", "mlp", "knn", "lgbm", "rf"]; MAIN_NN = ["cnn1d", "mlp_flat"]
KS = [1, 2, 4, 8, 16, 32]; NDRAW = 50; ACC_SUB = [70, 140, 280]
ARMS = ["raw", "predlabel", "predlabel_pd", "act8", "act8_r1", "act60", "act60_r1", "lda2", "lda3", "lda3_r0.5", "lda3_r1", "msda", "rep", "pd_pub_s0", "ctrl_const", "shuf_raw", "oracle_act",
        "act8_r1_ns1", "act60_r1_ns1", "lda3_r1_ns1", "lda3_r0.5_ns1"]
HIGH_U = ["act8", "act8_r1", "act60", "act60_r1", "lda3", "predlabel", "predlabel_pd"]


def load_post(arm, dirs):
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(os.path.expanduser(d), f"preds__{arm}__*.npz"))
    post, te = {}, None
    for fn in files:
        z = np.load(fn)
        if "fO_te_idx" not in z.files: continue
        te = z["fO_te_idx"].astype(int)
        for k in z.files:
            if k.startswith("fO_") and k.endswith("__id"):
                att = k[3:-4]
                if att in MAIN_TAB + MAIN_NN:
                    post[att] = z[k].astype(np.float32).mean(0)  # seed average -> (n_te, 24)
    return post, te


def build_draws(S, te):
    """windows of each fold-O test trial (ordered by start), shared random / contiguous draw indices (same for every arm and attacker)"""
    tk = C.trial_key(S)[te]; st = S["start"][te]; sub = S["sub"][te]
    order = np.lexsort((st, tk)); tks = tk[order]
    bounds = np.r_[0, np.where(np.diff(tks) != 0)[0] + 1, len(tks)]
    trials = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]; trials.append(dict(idx=idx, sub=int(sub[idx[0]]), key=int(tks[a])))
    rng = np.random.default_rng(2026)
    for t in trials:
        n = len(t["idx"]); t["rand"] = {}; t["contig"] = {}
        for K in KS:
            if K >= n: t["rand"][K] = t["contig"][K] = None; continue
            t["rand"][K] = np.stack([rng.choice(n, K, replace=False) for _ in range(NDRAW)])
            starts = rng.integers(0, n - K + 1, size=NDRAW); t["contig"][K] = starts[:, None] + np.arange(K)[None, :]
    accounts = []
    for s in range(NS):
        idx = np.concatenate([t["idx"] for t in trials if t["sub"] == s]); n = len(idx)
        acc = dict(sub=s, idx=idx, subsets={})
        for K in ACC_SUB:
            acc["subsets"][K] = None if K >= n else np.stack([rng.choice(n, K, replace=False) for _ in range(NDRAW)])
        accounts.append(acc)
    return trials, accounts


def bag_correct(LP, idx_sets, sub):
    """LP (n,24) log-posteriors, idx_sets (D,K) or None(all)->(correct fraction, n)"""
    if idx_sets is None:
        return float(LP.sum(0).argmax() == sub), 1
    sc = LP[idx_sets].sum(1)
    return float((sc.argmax(1) == sub).mean()), 1


def eval_arm(post, trials, accounts, te_local):
    """returns {setting: {att: (corr[24], n[24])}}"""
    out = {}
    for att, P in post.items():
        LP = np.log((P + EPS) / (1 + NS * EPS))
        def add(setting, s, v, w=1.0):
            rec = out.setdefault(setting, {}).setdefault(att, [np.zeros(NS), np.zeros(NS)]); rec[0][s] += v * w; rec[1][s] += w
        for t in trials:
            LPt = LP[t["idx"]]
            for K in KS:
                for typ in ("rand", "contig"):
                    v, _ = bag_correct(LPt, t[typ][K], t["sub"]); add(f"trial_{typ}_K{K}", t["sub"], v)
            v, _ = bag_correct(LPt, None, t["sub"]); add("trial_all", t["sub"], v)
        for ac in accounts:
            LPa = LP[ac["idx"]]
            v, _ = bag_correct(LPa, None, ac["sub"]); add("account_all", ac["sub"], v)
            for K in ACC_SUB:
                v, _ = bag_correct(LPa, ac["subsets"][K], ac["sub"]); add(f"account_rand_{K}", ac["sub"], v)
    return out


def boot_counts(B, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, NS, size=(B, NS))
    return np.stack([np.bincount(r, minlength=NS) for r in idx]).astype(np.float64)


def summarise(rec, Wt):
    res = {}
    for setting, atts in rec.items():
        pts = {a: float(c.sum() / n.sum()) for a, (c, n) in atts.items()}
        bs = np.stack([(Wt @ c) / np.maximum(Wt @ n, 1e-9) for a, (c, n) in atts.items()])
        res[setting] = dict(L=max(pts.values()), best=max(pts, key=pts.get), per_attacker=pts, boot=bs.max(0))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", required=True); ap.add_argument("--ns1", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--first_pass", default=None, help="results/h_imu/result.json (utilities U of the first pass)")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    dirs = a.dirs.split(",") + ([a.ns1] if a.ns1 else [])
    S = D.get_set("astar"); Wt = boot_counts(a.B)
    summ, te0, meta = {}, None, {}
    for arm in ARMS:
        post, te = load_post(arm, dirs)
        if not post: meta[arm] = "missing"; continue
        if te0 is None:
            te0 = te; trials, accounts = build_draws(S, te0)
        assert np.array_equal(te, te0), arm
        summ[arm] = summarise(eval_arm(post, trials, accounts, te0), Wt); meta[arm] = sorted(post)
    n_per_trial = float(np.mean([len(t["idx"]) for t in trials])); n_per_acct = float(np.mean([len(x["idx"]) for x in accounts]))
    U = {}
    if a.first_pass:
        fp = json.load(open(a.first_pass)); U = {k: v["U"] for k, v in fp["arms"].items()}
    SET_SEQ = [f"trial_rand_K{K}" for K in KS] + ["trial_all", "account_all"]
    tab = {arm: {s: dict(L=round(v["L"], 4), ci=[round(float(x), 4) for x in np.percentile(v["boot"], [2.5, 97.5])], best=v["best"]) for s, v in sets.items()} for arm, sets in summ.items()}
    claims = {}
    def L(arm, s): return summ[arm][s]["L"]
    def lo(arm, s): return float(np.percentile(summ[arm][s]["boot"], 2.5))
    def hi(arm, s): return float(np.percentile(summ[arm][s]["boot"], 97.5))
    # (a)
    if all(x in summ for x in ("act60_r1", "lda3_r1")):
        a60, l31 = "act60_r1", "lda3_r1"
        pa = dict(act60_r1=dict(account=L(a60, "account_all"), trial_all=L(a60, "trial_all"), lo=lo(a60, "account_all"), hi=hi(a60, "account_all")),
                  lda3_r1=dict(account=L(l31, "account_all"), trial_all=L(l31, "trial_all"), lo=lo(l31, "account_all"), hi=hi(l31, "account_all")))
        conf_o = pa["act60_r1"]["account"] >= 0.19 and pa["lda3_r1"]["account"] >= 0.20
        fals_o = pa["act60_r1"]["account"] <= 0.12 and pa["lda3_r1"]["account"] <= 0.15
        conf_t = conf_o and pa["act60_r1"]["lo"] > pa["act60_r1"]["trial_all"] and pa["lda3_r1"]["lo"] > pa["lda3_r1"]["trial_all"] and pa["act60_r1"]["lo"] >= 0.14 and pa["lda3_r1"]["lo"] >= 0.15
        fals_t = fals_o and pa["act60_r1"]["hi"] < 0.19 and pa["lda3_r1"]["hi"] < 0.20
        claims["a_dither_composes"] = dict(values=pa, original="CONFIRMED" if conf_o else ("FALSIFIED" if fals_o else "UNRESOLVED"), tightened="CONFIRMED" if conf_t else ("FALSIFIED" if fals_t else "UNRESOLVED"))
    # (b)
    b = {}
    for arm in ("predlabel", "predlabel_pd"):
        if arm in summ: b[arm] = {s: round(v["L"], 4) for s, v in summ[arm].items()}
    if b:
        mx = max(max(v.values()) for v in b.values()); acc = b.get("predlabel", {}).get("account_all", None)
        claims["b_predlabel_flat"] = dict(max_over_settings=mx, account=acc, original="PASS" if mx <= 0.08 else ("FALSIFIED" if (acc or 0) > 0.12 else "FAIL"), tightened="PASS" if all(hi(a_, s) <= 0.08 + 0.05 for a_ in b for s in summ[a_]) and mx <= 0.08 else ("FALSIFIED" if (acc or 0) > 0.12 and lo("predlabel", "account_all") > 0.12 else "UNRESOLVED/FAIL"))
    # (c)
    c = {}
    for r1, r0 in (("act60_r1", "act60"), ("lda3_r1", "lda3")):
        if r1 in summ and r0 in summ:
            ratios = [L(r1, s) / max(L(r0, s), 1e-9) for s in SET_SEQ]
            mono = all(ratios[i + 1] >= ratios[i] - 0.03 for i in range(len(ratios) - 1))
            c[r1] = dict(ratios=[round(x, 3) for x in ratios], monotone_within_0p03=bool(mono), account_r1_le_r0=bool(L(r1, "account_all") <= L(r0, "account_all")))
    if c: claims["c_ratio_monotone_and_bounded"] = dict(values=c, original="PASS" if all(v["monotone_within_0p03"] and v["account_r1_le_r0"] for v in c.values()) else "FAIL")
    # (d)
    if "msda" in summ and "rep" in summ:
        viol = []
        for s in SET_SEQ + [f"account_rand_{K}" for K in ACC_SUB]:
            for arm in HIGH_U:
                if arm in summ and s in summ[arm] and (L(arm, s) >= L("msda", s) or L(arm, s) >= L("rep", s)): viol.append((arm, s, round(L(arm, s), 3), round(L("msda", s), 3), round(L("rep", s), 3)))
        claims["d_ordering_floors_below_aaes"] = dict(violations=viol, account=dict(msda=L("msda", "account_all"), rep=L("rep", "account_all")), original="PASS" if not viol else "FAIL")
    # (e) extrapolation
    e = {}
    for arm in ("act8_r1", "act60_r1", "lda3_r0.5", "lda3_r1"):
        if arm in summ:
            ks = np.array([1, 2, 4, 8, 16], float); ls = np.array([L(arm, f"trial_rand_K{int(k)}") for k in ks]) - CH
            if (ls > 0).sum() >= 3:
                m = ls > 0; b1, a1 = np.polyfit(np.log(ks[m]), np.log(ls[m]), 1)
                pred = lambda K: CH + float(np.exp(a1) * K ** b1)
                e[arm] = dict(slope=round(float(b1), 3), pred_trial_all=round(pred(n_per_trial), 3), obs_trial_all=round(L(arm, "trial_all"), 3), pred_account=round(pred(n_per_acct), 3), obs_account=round(L(arm, "account_all"), 3),
                              within_0p05=bool(abs(pred(n_per_trial) - L(arm, "trial_all")) <= 0.05 and abs(pred(n_per_acct) - L(arm, "account_all")) <= 0.05))
            else: e[arm] = dict(note="L(K<=16) not above chance in >= 3 points: fit not possible")
    claims["e_extrapolation"] = e
    # controls
    ctl = {}
    for arm in ("ctrl_const", "shuf_raw", "oracle_act"):
        if arm in summ:
            ctl[arm] = {s: dict(L=round(v["L"], 4), tol=round(CH + max(0.06, 1.96 * float(np.std(v["boot"]))), 4), ok=bool(v["L"] <= CH + max(0.06, 1.96 * float(np.std(v["boot"]))))) for s, v in summ[arm].items()}
    claims["controls"] = dict(all_ok=all(x["ok"] for d in ctl.values() for x in d.values()), detail={k: [s for s, x in d.items() if not x["ok"]] for k, d in ctl.items()})
    # seed replication of the dithered arms
    rep = {}
    for arm in ("act8_r1", "act60_r1", "lda3_r0.5", "lda3_r1"):
        if arm in summ and f"{arm}_ns1" in summ:
            rep[arm] = {s: [round(L(arm, s), 3), round(L(f"{arm}_ns1", s), 3)] for s in ("trial_rand_K1", "trial_rand_K16", "trial_all", "account_all")}
    claims["seed_replication_seed0_vs_seed1"] = rep
    a_ = claims.get("a_dither_composes", {})
    out = dict(control="H3-016", fold="O (96 clean trials, 24 subjects; dev)", evidence_rule="sum log((p+1e-3)/(1+24e-3)); worst case = max over main-suite attackers; B=2000 subject bootstrap",
               windows_per_trial_mean=n_per_trial, windows_per_account_mean=n_per_acct, arms_loaded=meta, table=tab, claims=claims,
               verdict_original=a_.get("original", "NOT_RUN"), verdict_tightened=a_.get("tightened", "NOT_RUN"),
               note="verdicts are for claim (a) (dither composes); (b)-(e) are in claims; UNRESOLVED is allowed; cannot change a floor-vs-defender label")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("verdict_original", "verdict_tightened", "windows_per_trial_mean", "windows_per_account_mean")}, indent=1))
    for arm in ("predlabel", "act8_r1", "act60", "act60_r1", "lda3", "lda3_r1", "msda", "rep", "pd_pub_s0", "raw"):
        if arm in tab: print(arm, {s: tab[arm][s]["L"] for s in SET_SEQ + ["account_rand_70", "account_rand_140"] if s in tab[arm]})
    print(json.dumps({k: (v if k != "values" else "...") for k, v in claims.items() if k in ("a_dither_composes", "b_predlabel_flat", "controls")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
