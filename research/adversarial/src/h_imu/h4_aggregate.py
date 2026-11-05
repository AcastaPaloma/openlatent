"""H4 aggregation (H.md 9.3): ladder, rules R1-R3, verdict_original / verdict_tightened.
python h4_aggregate.py --strong DIR1,DIR2,... --frozen DIR_A,DIR_B,... --out h4_result.json
--strong: job output dirs holding strong__<arm>.json ; --frozen: arm-B/B2 prediction dirs (out/sealed, runs/h_b_nn_floors, runs/h_b_nn_aae, out/sealed_b2, runs/h_b2_c2s0)."""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import aggregate as A
import datasets as D

NS = 24; CH = 1 / 24
M_L, M_U = 0.15, 0.0285
AAE = ["aae_re", "aae_re_c2s0"]
RE_ATTACKED = ["raw", "aae_re", "aae_re_c2s0", "predlabel", "act60_r1", "lda3", "lda3_r1", "lda3_r2"]


def load_strong(dirs):
    S = {}
    for d in dirs:
        for fn in glob.glob(os.path.join(os.path.expanduser(d), "strong__*.json")):
            j = json.load(open(fn)); arm = j["arm"]
            for pk, r in j["rungs"].items():
                S.setdefault(arm, {})[float(pk[1:])] = dict(c=np.array(r["per_subject_corr"]), n=np.array(r["per_subject_n"]), seg=r["seg_acc"], L=r["L"])
    return S


def stat(c, n, Wt): return float(c.sum() / n.sum()), (Wt @ c) / np.maximum(Wt @ n, 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--strong", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Wt = A.boot_counts(2000); A.NSUB = NS
    SS = load_strong(a.strong.split(","))
    res = dict(table={}, ladder={}, gates={}, rules={})
    LB = {}
    for arm, rungs in SS.items():
        for p, r in sorted(rungs.items()):
            L, Lb = stat(r["c"], r["n"], Wt); LB[(arm, p)] = (L, Lb)
            res["table"].setdefault(arm, {})[str(p)] = dict(L=L, ci=[float(x) for x in np.percentile(Lb, [2.5, 97.5])], seg_acc=r["seg"], n_trials=int(r["n"].sum()))
    # ladder
    for arm in SS:
        ps = sorted(SS[arm]); Ls = [LB[(arm, p)][0] for p in ps]
        if len(ps) >= 3:
            nondec = all(Ls[i + 1] >= Ls[i] - 0.03 for i in range(len(Ls) - 1)); gain = Ls[-1] - Ls[-2]
            res["ladder"][arm] = dict(rungs=ps, L=[round(x, 4) for x in Ls], non_decreasing_within_0p03=bool(nondec), gain_1_vs_0p5=round(gain, 4), reading="LOWER BOUND (still rising)" if (nondec and gain > 0.03) else ("SATURATED for this capacity" if gain <= 0.03 else "non-monotone: unresolved"))
    g = lambda arm, p=1.0: LB.get((arm, p))
    # gates
    if g("ctrl_const"):
        se = float(g("ctrl_const")[1].std()); res["gates"]["G-S1"] = dict(L=g("ctrl_const")[0], tol=CH + max(0.06, 1.96 * se), ok=bool(g("ctrl_const")[0] <= CH + max(0.06, 1.96 * se)))
    if g("shuf_raw"):
        se = float(g("shuf_raw")[1].std()); res["gates"]["G-S2"] = dict(L=g("shuf_raw")[0], tol=CH + max(0.06, 1.96 * se), ok=bool(g("shuf_raw")[0] <= CH + max(0.06, 1.96 * se)))
    if g("raw"): res["gates"]["G-S3"] = dict(L=g("raw")[0], need=0.943, ok=bool(g("raw")[0] >= 0.943))
    gates_ok = all(v["ok"] for v in res["gates"].values())
    # R1, R2
    r1 = {}
    if g("predlabel"):
        Lp, Lpb = g("predlabel")
        r1["predlabel"] = dict(L=Lp, hi95=float(np.percentile(Lpb, 97.5)), limit=CH + 0.05)
        orig_ok = Lp <= CH + 0.05; tight_ok = r1["predlabel"]["hi95"] <= CH + 0.05
        for aae in AAE:
            if g(aae):
                d = g(aae)[1] - Lpb; r1[aae] = dict(L=g(aae)[0], diff=float(g(aae)[0] - Lp), diff_lo95=float(np.percentile(d, 2.5)))
                orig_ok &= (g(aae)[0] - Lp >= 0.15); tight_ok &= (r1[aae]["diff_lo95"] >= 0.15)
        res["rules"]["R1_ordering"] = dict(values=r1, original="ROBUST" if orig_ok else "NOT ROBUST", tightened="ROBUST" if tight_ok else "NOT ROBUST")
    r2 = {}
    for arm in ("lda3_r1", "act60_r1", "lda3_r2"):
        if g(arm): r2[arm] = dict(L=g(arm)[0], hi95=float(np.percentile(g(arm)[1], 97.5)))
    if "lda3_r1" in r2 and "act60_r1" in r2:
        res["rules"]["R2_dither"] = dict(values=r2, original="ROBUST" if (r2["lda3_r1"]["L"] <= 0.20 and r2["act60_r1"]["L"] <= 0.20) else "NOT ROBUST",
                                         tightened="ROBUST" if (r2["lda3_r1"]["hi95"] <= 0.20 and r2["act60_r1"]["hi95"] <= 0.20) else "NOT ROBUST")
    # R3: re-derive arm-B outcomes with L_worst
    S = D.get_set("sealed"); st = A.Store(S); files = {}
    for d in a.frozen.split(","):
        for fn in glob.glob(os.path.join(os.path.expanduser(d), "preds__*__*.npz")):
            _, arm, fam = os.path.basename(fn)[:-4].split("__"); files.setdefault(arm, []).append(fn)
    R = {}
    for arm, fl in files.items():
        if arm == "aae_re_c1s1": continue
        rec = st.arm_arrays(fl, ["S0", "S1", "S2"]); has_nn = any(f.endswith("__nn.npz") for f in fl)
        names = A.MAIN_TAB + (A.MAIN_NN if has_nn else [])
        L, Lb, _ = A.stat_from(rec, names, "id", "t", Wt); U, Ub, _ = A.stat_from(rec, names, "act", "w", Wt)
        if L is None or U is None: continue
        R[arm] = dict(L=L, Lb=Lb, U=U, Ub=Ub)
    Rw = {k: dict(v) for k, v in R.items()}
    for arm in RE_ATTACKED:
        if arm in Rw and g(arm):
            Rw[arm]["L"] = max(R[arm]["L"], g(arm)[0]); Rw[arm]["Lb"] = np.maximum(R[arm]["Lb"], g(arm)[1])
    def outcome(Rx, arm, floor_set):
        Ls_b = np.stack([Rx[f]["Lb"] for f in floor_set], 1); Us_b = np.stack([Rx[f]["Ub"] for f in floor_set], 1)
        La_b, Ua_b = Rx[arm]["Lb"], Rx[arm]["Ub"]
        dL = np.array([A.frontier_L_at_U(Ls_b[b], Us_b[b], Ua_b[b]) - La_b[b] for b in range(len(La_b))])
        dU = np.array([Ua_b[b] - A.frontier_U_at_L(Ls_b[b], Us_b[b], La_b[b]) for b in range(len(La_b))])
        Ls0 = np.array([Rx[f]["L"] for f in floor_set]); Us0 = np.array([Rx[f]["U"] for f in floor_set])
        pL = A.frontier_L_at_U(Ls0, Us0, Rx[arm]["U"]) - Rx[arm]["L"]; pU = Rx[arm]["U"] - A.frontier_U_at_L(Ls0, Us0, Rx[arm]["L"])
        lab = {}
        for nm, p_, bs, m in (("L", pL, dL, M_L), ("U", pU, dU, M_U)):
            lo, hi = np.percentile(bs, [5, 95]); lab[nm] = dict(delta=float(p_), lo5=float(lo), hi95=float(hi), margin=m, label="WIN" if (p_ >= m and lo > 0) else ("MATCH" if hi < m else "UNRESOLVED"))
        ls_ = [lab["L"]["label"], lab["U"]["label"]]
        lab["defence_label"] = "WIN" if "WIN" in ls_ else ("MATCH" if ls_ == ["MATCH", "MATCH"] else "UNRESOLVED")
        return lab
    learned = [x for x in AAE if x in Rw]
    excl_learn = set(AAE) | {"aae_re_c1s1", "msda", "rep"}
    base = [x for x in Rw if x not in A.CONTROLS | A.ORACLES | excl_learn and not x.startswith(("pd_", "msda", "rep", "aae_re")) and x != "predlabel"]
    r3 = {}
    for name, fl in (("original_floorlist", base), ("tightened_plus_predlabel", base + ["predlabel"])):
        r3[name] = {}
        for arm in learned:
            r3[name][arm] = dict(frozen_suite_only=outcome(R, arm, [x for x in fl if x in R]), L_worst=outcome(Rw, arm, [x for x in fl if x in Rw]))
    res["rules"]["R3_rederived_outcome"] = r3
    def family(name, key):
        labs = [r3[name][x][key]["defence_label"] for x in learned]
        return "WIN" if "WIN" in labs else ("MATCH" if all(l == "MATCH" for l in labs) else "UNRESOLVED")
    res["arm_B_reading"] = dict(frozen_suite=dict(original=family("original_floorlist", "frozen_suite_only"), tightened=family("tightened_plus_predlabel", "frozen_suite_only")),
                                with_strong_attacker=dict(original=family("original_floorlist", "L_worst"), tightened=family("tightened_plus_predlabel", "L_worst")))
    changed_o = res["arm_B_reading"]["with_strong_attacker"]["original"] != "MATCH"; changed_t = res["arm_B_reading"]["with_strong_attacker"]["tightened"] != "MATCH"
    r1o = res["rules"].get("R1_ordering", {}); r2o = res["rules"].get("R2_dither", {})
    res["verdict_original"] = dict(R1=r1o.get("original"), R2=r2o.get("original"), arm_B_reading_changed=bool(changed_o), family_label=res["arm_B_reading"]["with_strong_attacker"]["original"])
    res["verdict_tightened"] = dict(R1=r1o.get("tightened"), R2=r2o.get("tightened"), arm_B_reading_changed=bool(changed_t), family_label=res["arm_B_reading"]["with_strong_attacker"]["tightened"])
    res["gates_ok"] = bool(gates_ok)
    res["frozen_vs_strong_L"] = {x: dict(frozen=R[x]["L"], strong=g(x)[0]) for x in RE_ATTACKED if x in R and g(x)}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("gates_ok", "verdict_original", "verdict_tightened", "arm_B_reading", "ladder")}, indent=1, default=float))
    for arm in sorted(res["table"]):
        print(arm, {p: (round(v["L"], 3), round(v["seg_acc"], 3)) for p, v in res["table"][arm].items()})
    print(json.dumps(res["frozen_vs_strong_L"], default=float))


if __name__ == "__main__":
    main()
