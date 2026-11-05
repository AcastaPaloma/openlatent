"""Aggregator for 20B, the learned source-removal mask (results_notes/F.md section 20).  CPU only.  Reuses aggregate_hlca_rep (Fast, assemble, helpers) and E's Data.

  python aggregate_hlca_mask.py --runs <dirs...> [--engage_only] [--B 1000] [--Bseed 300] [--json out.json] [--md out.md]

--engage_only : the pre-registered stop-rule check on the available seeds (pooled over folds): is the removal term ENGAGED, i.e. at some lambda
                L(adp_f{b}_r0) - L(lm_f{b}_l{lambda}_r0) >= 0.05 at budget b?  Prints and exits.
Headline: dU_match = U(assembled mask) - U(assembled comparator floor) at matched leakage (tolerance 0.05), lambda and r and the comparator chosen on the
validation folds (leave-one-fold-out) by the headline criterion itself; selection fixed in the donor / study bootstrap.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if p and Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "e_singlecell"))

import numpy as np  # noqa: E402

import aggregate as AG  # noqa: E402  (E)

spec = importlib.util.spec_from_file_location("agg_rep", HERE / "aggregate_hlca_rep.py")
AR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AR)

LAMS = ["0.3", "1", "3"]
BUD = [10, 100]
TOL = 0.05
M_ORIG, M_TIGHT = 0.02, 0.035
ENGAGE = 0.05


def MC(b):
    return [f"lm_f{b}_l{l}_r{r}" for l in LAMS for r in ("0", "0.5")]


def LF(b):
    return [AR.S_of(b), AR.X_of(b)] + AR.CE_of(b) + AR.EX_of(b)


def verdict(res, m):
    cats = {}
    for b in BUD:
        if b not in res:
            continue
        r = res[b]
        sup = bool(r["dU_match"] >= m and r["dU_match_donor"][0] > 0 and r["dU_match_study"][0] > 0 and sum(v >= m for v in r["dU_match_folds"]) >= 2
                   and sum(1 for s in r["seed"].values() if s["dU_match"] >= m) >= 2)
        notsup = bool(r["dU_match_donor"][1] < m)
        cats[b] = "SUPPORTED" if sup else ("NOT SUPPORTED" if notsup else "UNRESOLVED")
    v = list(cats.values())
    if v and all(c == "SUPPORTED" for c in v):
        return "LEARNED MASK SUPPORTED", cats
    if v and all(c == "NOT SUPPORTED" for c in v):
        return "LEARNED MASK NOT SUPPORTED", cats
    if v and all(c == "UNRESOLVED" for c in v):
        return "INCONCLUSIVE", cats
    return "PARTIAL", cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--Bseed", type=int, default=300)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--engage_only", action="store_true")
    a = ap.parse_args()
    AR.EXTRA[:] = ["cenn", "cenp"]
    D = AG.Data(a.dataset, a.runs)
    folds, seeds = list(D.fall), sorted(D.S)
    present = set()
    for s in seeds:
        present |= {x for x, _ in D.S[s]["pr"]}
    universe = [x for x in AR.IND + ["ctrl_noise16"] if x in present]
    for b in BUD:
        universe += [x for x in MC(b) + AR.AD_of(b) + LF(b) if x in present]
    universe = list(dict.fromkeys(universe))
    ones = np.ones(D.n)
    FP = AR.Fast(D, universe, folds)
    Up, Lp, ch = FP.UL([x for x in universe if FP.has(x)], ones)
    # ---- engagement (stop rule 1)
    eng = {}
    for b in BUD:
        ad0 = f"adp_f{b}_r0"
        if ad0 not in Lp:
            continue
        gain = {l: Lp[ad0] - Lp.get(f"lm_f{b}_l{l}_r0", np.nan) for l in LAMS}
        du = {l: Up.get(f"lm_f{b}_l{l}_r0", np.nan) - Up[ad0] for l in LAMS}
        eng[b] = dict(dL_vs_noremoval=gain, dU_vs_noremoval=du, engaged=bool(np.nanmax(list(gain.values())) >= ENGAGE))
    print("engagement (L(adp r0) - L(lm r0), U(lm r0) - U(adp r0); engaged iff max dL >= 0.05):", json.dumps(eng, default=float), flush=True)
    if a.engage_only:
        return
    # ---- trained gates
    gates = {}
    for r in a.runs:
        for p in Path(r).glob("meta_s*.json"):
            s = int(p.stem.split("_s")[1])
            for k, v in json.loads(p.read_text()).get("adapters", {}).items():
                gates[(s, k)] = v
    pool_all = AR.IND + [x for bb in BUD for x in LF(bb) + AR.AD_of(bb)]
    pool_all = [x for x in pool_all if x in present]

    def pick(U, L, b, avail, same_budget=False):
        pl = pool_all if not same_budget else [x for x in AR.IND + LF(b) + AR.AD_of(b) if x in present]
        pl = [x for x in pl if x in avail]
        best = None
        for c in [x for x in MC(b) if x in avail and np.isfinite(L.get(x, np.nan))]:
            okf = [x for x in pl if np.isfinite(L.get(x, np.nan)) and L[x] <= L[c] + TOL]
            ref = max(U[x] for x in okf) if okf else 0.0
            gap = U[c] - ref
            if best is None or gap > best[0]:
                best = (gap, c, max(okf, key=lambda x: U[x]) if okf else None)
        return best

    # ---- leave-one-fold-out selection
    sel = {b: {"mask": {}, "floor": {}, "gap_val": {}} for b in BUD}
    for k in folds:
        Fk = AR.Fast(D, universe, [f for f in folds if f != k])
        arms_k = [x for x in universe if Fk.has(x)]
        Uk, Lk, _ = Fk.UL(arms_k, ones)
        for b in BUD:
            g, c, fl = pick(Uk, Lk, b, set(arms_k))
            sel[b]["mask"][k], sel[b]["floor"][k], sel[b]["gap_val"][k] = c, fl, g
    virt = {}
    for b in BUD:
        for role in ("mask", "floor"):
            if any(v is None for v in sel[b][role].values()):
                continue
            nm = f"V_{role}_f{b}"
            AR.assemble(D, nm, sel[b][role], folds)
            virt[(role, b)] = nm
    allarms = list(dict.fromkeys(universe + list(virt.values())))
    FA = AR.Fast(D, allarms, folds)
    Upt, Lpt, _ = FA.UL([x for x in allarms if FA.has(x)], ones)

    # ---- bootstrap
    boots = {}
    for label, cluster in (("donor", False), ("study", True)):
        Ws = AG.resamples(D, a.B, cluster=cluster)
        acc = {b: [] for b in BUD}
        lam_tab = {x: ([], []) for b in BUD for x in MC(b) + AR.AD_of(b) if x in present}
        for w in Ws:
            den, chw = FA.den_chance(w)
            raw = FA.worst("raw50", w, den)
            for b in BUD:
                vm, vf = virt.get(("mask", b)), virt.get(("floor", b))
                if vm and vf:
                    acc[b].append((FA.U(vm, w) - FA.U(vf, w)))
            if label == "donor":
                for x, (lu, ll) in lam_tab.items():
                    lu.append(FA.U(x, w))
                    wa = FA.worst(x, w, den)
                    ll.append((wa - chw) / (raw - chw) if raw - chw >= 0.10 else np.nan)
        boots[label] = dict(acc=acc, lam=lam_tab, Ws=Ws)
        print(f"bootstrap {label}: done", flush=True)

    # ---- seed and fold level
    Wd = boots["donor"]["Ws"][: a.Bseed]
    res = {}
    for b in BUD:
        vm, vf = virt.get(("mask", b)), virt.get(("floor", b))
        if not (vm and vf):
            continue
        seed_e = {}
        for s in seeds:
            Fs = AR.Fast(D, allarms, folds, seeds=[s])
            if Fs.has(vm) and Fs.has(vf):
                d_ = Fs.U(vm, ones) - Fs.U(vf, ones)
                dd = [Fs.U(vm, w) - Fs.U(vf, w) for w in Wd]
                seed_e[s] = dict(dU_match=d_, ci=AG.ci(dd))
        fold_e = []
        for k in folds:
            Fk = AR.Fast(D, allarms, [k])
            fold_e.append(Fk.U(vm, ones) - Fk.U(vf, ones))
        # gates of the selected lambdas
        gate_rows = []
        for k in folds:
            c = sel[b]["mask"][k]
            lam = c.split("_l")[1].split("_r")[0]
            for s in seeds:
                g = gates.get((s, f"{k}|lm_f{b}_l{lam}"))
                gate_rows.append(dict(seed=s, fold=k, lam=lam, ratio=(g["loss_last200"] / g["loss_first20"]) if g else None, rank=g["eff_rank_held"] if g else None,
                                      adv_acc=g.get("adv_acc_last200") if g else None))
        gate_ok = all(r["ratio"] is not None and r["ratio"] <= AR.GATE_LOSS and r["rank"] >= AR.GATE_RANK for r in gate_rows)
        # same-budget-only pool, test-selected (descriptive)
        gsb = pick(Up, Lp, b, set(universe), same_budget=True)
        res[b] = dict(sel_mask=sel[b]["mask"], sel_floor=sel[b]["floor"], gap_val=sel[b]["gap_val"], U_mask=Upt[vm], L_mask=Lpt[vm], U_floor=Upt[vf], L_floor=Lpt[vf],
                      dU_match=Upt[vm] - Upt[vf], dU_match_donor=AG.ci(boots["donor"]["acc"][b]), dU_match_study=AG.ci(boots["study"]["acc"][b]),
                      dU_match_folds=fold_e, dL=Lpt[vm] - Lpt[vf], seed=seed_e, gate_ok=bool(gate_ok), gate_rows=gate_rows,
                      same_budget_pool_test_selected=dict(mask=gsb[1], floor=gsb[2], dU=(Up[gsb[1]] - Up[gsb[2]]) if gsb[2] else None) if gsb else None,
                      n_seeds=FA.n_seeds(vm))
    ctrl = {}
    for s in seeds:
        Fs = AR.Fast(D, ["ctrl_noise16", "raw50"], folds, seeds=[s])
        den, chs = Fs.den_chance(ones)
        ctrl[s] = dict(U=Fs.U("ctrl_noise16", ones), worst=Fs.worst("ctrl_noise16", ones, den), chance=chs)
    u0 = AG.f1_chance(D, ones, folds)
    ctrl_ok = all(abs(v["U"] - u0) <= 0.01 and v["worst"] <= v["chance"] + 0.03 for v in ctrl.values())
    valid = bool(ctrl_ok and all(r["gate_ok"] for r in res.values()))
    engaged_any = any(e["engaged"] for e in eng.values())
    V = {}
    for nm_, m in (("original_0.02", M_ORIG), ("tightened_0.035", M_TIGHT)):
        v, cats = verdict(res, m)
        if not engaged_any:
            v = "INERT TERM (claim untestable with this term and grid)"
        elif not valid:
            v = "INVALID (" + v + ")"
        V[nm_] = dict(verdict=v, by_budget=cats)
    tab = {x: dict(U=Upt[x], U_ci=AG.ci(boots["donor"]["lam"][x][0]), L=Lpt[x], L_ci=AG.ci(boots["donor"]["lam"][x][1])) for x in boots["donor"]["lam"] if x in Upt}
    js = dict(verdicts=V, valid=valid, ctrl_ok=ctrl_ok, engaged=eng, budgets=res, lambda_table=tab, seeds=seeds, folds=folds, B=a.B,
              original_verdicts="19.2: Q1 REPLICATES, Q2 SUPCON GAIN SURVIVES (0.02, 0.035) vs CE floors; 18.2 K-RULE REFUTED; JUMP NO-GO at S4; E SC-0 NO-GO")
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1, default=float))
    L = [f"# 20B learned source-removal mask (GRL source head): original {V['original_0.02']['verdict']} | tightened {V['tightened_0.035']['verdict']}; valid={valid}; seeds {seeds}",
         f"engagement: {json.dumps({b: dict(engaged=e['engaged'], dL=e['dL_vs_noremoval']) for b, e in eng.items()}, default=float)}", "",
         "| f | n seeds | mask (per fold) | comparator (per fold) | U mask | L mask | U floor | L floor | dU_match [donor] [study] | folds | seeds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in res.items():
        smask = "/".join(r["sel_mask"][k] for k in folds)
        sfloor = "/".join(r["sel_floor"][k] for k in folds)
        sfold = ", ".join("%+.3f" % v for v in r["dU_match_folds"])
        sseed = ", ".join("s%d:%+.3f" % (s_, v_["dU_match"]) for s_, v_ in sorted(r["seed"].items()))
        d0, d1 = r["dU_match_donor"]
        t0, t1 = r["dU_match_study"]
        L.append("| %d%% | %d | %s | %s | %.3f | %.2f | %.3f | %.2f | %+.3f [%+.3f,%+.3f] [%+.3f,%+.3f] | %s | %s |"
                 % (b, r["n_seeds"], smask, sfloor, r["U_mask"], r["L_mask"], r["U_floor"], r["L_floor"], r["dU_match"], d0, d1, t0, t1, sfold, sseed))
    L += ["", "| arm | U [95%] | L [95%] |", "|---|---|---|"] + ["| %s | %.3f [%.3f,%.3f] | %.2f [%.2f,%.2f] |" % (x, v["U"], v["U_ci"][0], v["U_ci"][1], v["L"], v["L_ci"][0], v["L_ci"][1]) for x, v in tab.items()]
    L += ["", "'lowers leakage below the floors at equal utility' is the claim tested; it is not 'removes batch'."]
    if a.md:
        Path(a.md).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
