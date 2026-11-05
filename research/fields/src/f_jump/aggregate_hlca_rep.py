"""Aggregator for the controlled replication (results_notes/F.md section 19).  CPU only.  Reuses E's Data / resamples / power_audit.

  python aggregate_hlca_rep.py --runs <dirs...> --budgets 1,3,10,30,100 [--B 1000] [--Bseed 300] [--json out.json] [--md out.md] [--dataset hlca] [--no_power]

Primary selection is leave-one-fold-out (validation folds): for each held-out fold k the adapter's dither r, the best label floor B* and the full
envelope F* are chosen from the other two folds only, assembled over folds into one prediction set per role, and every statistic is computed on the
assembled sets with selection fixed.  The test-selected (all folds pooled) numbers are printed as a sensitivity.
Fast kernels (multi-seed, E's definitions) are checked against E's all_metrics at the point estimate before any interval is computed.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if p and Path(p).resolve() != HERE]
sys.path.insert(0, str(HERE.parent / "e_singlecell"))

import numpy as np  # noqa: E402

import aggregate as AG  # noqa: E402  (E)

KINDS, PROBES = AG.KINDS, AG.PROBES
IND = ["raw50", "pca16", "rp16", "noise_r0.5", "noise_r1", "dith16_r0.5", "dith16_r1", "ldaC_pca16"]
TOL_L = 0.05
M_ORIG, M_TIGHT = 0.02, 0.035
GATE_LOSS, GATE_RANK, GATE_PERM = 0.8, 4.0, 0.02
PRIMARY = (10, 100)


def S_of(b):
    return "ldaF16" if b == 100 else f"ldaM_f{b}"


def AD_of(b):
    return [f"adp_f{b}_r0", f"adp_f{b}_r0.5"]


def CE_of(b):
    return [f"ce_f{b}_r0", f"ce_f{b}_r0.5"]


def X_of(b):
    return f"ldaX_f{b}"


class Fast:
    """multi-seed kernels with E's definitions: U = max over probes of the seed-averaged macro-F1; worst = max over attackers of the seed-averaged
    (fold-mean) balanced accuracy; L = (worst - chance) / (raw50 worst - chance)."""

    def __init__(self, D, arms, fset, seeds=None, probes=PROBES):
        self.D, self.fset, self.K = D, list(fset), len(D.fine_names)
        self.seeds = sorted(D.S) if seeds is None else list(seeds)
        self.probes = probes
        self.m = D.evalm & np.isin(D.fold_of, self.fset)
        self.t = D.fine[self.m]
        self.pidx, self.ok = {}, {}
        for a in arms:
            for kind in probes:
                L = []
                for s in self.seeds:
                    key = (a, kind)
                    if key in D.S[s]["pr"]:
                        p = D.S[s]["pr"][key][self.m]
                        if (p < 0).any():
                            continue                         # arm not complete on these folds in this seed: not used
                        L.append((self.t * self.K + p.astype(np.int64)).astype(np.int32))
                if L:
                    self.pidx[(a, kind)] = L
            for kind in KINDS:
                L = []
                for s in self.seeds:
                    key = (a, kind)
                    if key in D.S[s]["att"]:
                        pa = D.S[s]["att"][key]
                        L.append((pa == D.yb).astype(np.float32))
                if L:
                    self.ok[(a, kind)] = L

    def has(self, arm):
        return any((arm, k) in self.pidx for k in self.probes) and any((arm, k) in self.ok for k in KINDS)

    def n_seeds(self, arm):
        return max((len(self.pidx.get((arm, k), [])) for k in self.probes), default=0)

    def U(self, arm, w, kinds=None):
        wm = w[self.m]
        best = np.nan
        for kind in (kinds or self.probes):
            if (arm, kind) not in self.pidx:
                continue
            vals = []
            for idx in self.pidx[(arm, kind)]:
                C = np.bincount(idx, weights=wm, minlength=self.K * self.K).reshape(self.K, self.K)
                tp = np.diag(C)
                fn, fp = C.sum(1) - tp, C.sum(0) - tp
                sup = C.sum(1) > 0
                f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
                vals.append(float(f1[sup].mean()))
            v = float(np.mean(vals))
            best = v if np.isnan(best) else max(best, v)
        return best

    def worst(self, arm, w, den):
        D = self.D
        best = np.nan
        for kind in KINDS:
            if (arm, kind) not in self.ok:
                continue
            vals = []
            for ok in self.ok[(arm, kind)]:
                num = np.bincount(D.cls, weights=w * ok, minlength=D.nf * 16)
                accs = []
                for f in self.fset:
                    sl = slice(f * 16, f * 16 + D.Bf[f])
                    o = den[sl] > 0
                    if o.any():
                        accs.append(np.mean(num[sl][o] / den[sl][o]))
                vals.append(float(np.mean(accs)))
            v = float(np.mean(vals))
            best = v if np.isnan(best) else max(best, v)
        return best

    def den_chance(self, w):
        D = self.D
        den = np.bincount(D.cls, weights=w, minlength=D.nf * 16)
        ch = float(np.mean([1.0 / max(int((den[f * 16: f * 16 + D.Bf[f]] > 0).sum()), 1) for f in self.fset]))
        return den, ch

    def UL(self, arms, w, need_L=True):
        den, ch = self.den_chance(w)
        raw = self.worst("raw50", w, den) if need_L else np.nan
        U, L = {}, {}
        for a in arms:
            U[a] = self.U(a, w)
            if need_L:
                wa = self.worst(a, w, den)
                L[a] = (wa - ch) / (raw - ch) if (raw - ch >= 0.10 and not np.isnan(wa)) else np.nan
        return U, L, ch


def assemble(D, name, choice, folds):
    """virtual arm: on the cells of fold k use the predictions of arm choice[k]; defined for the seeds where every component exists."""
    made = []
    for s, sd in D.S.items():
        comps = set(choice.values())
        if not all(all((c, k) in sd["pr"] for k in PROBES) and all((c, k) in sd["att"] for k in KINDS) for c in comps):
            continue
        for kind in PROBES:
            arr = np.full(D.n, -1, np.int16)
            for k in folds:
                cells = D.fold_of == k
                arr[cells] = sd["pr"][(choice[k], kind)][cells]
            sd["pr"][(name, kind)] = arr
        for kind in KINDS:
            arr = np.full(D.n, -1, np.int8)
            for k in folds:
                cells = D.fold_of == k
                arr[cells] = sd["att"][(choice[k], kind)][cells]
            sd["att"][(name, kind)] = arr
        made.append(s)
    return made


def choose(U, L, b, arms_present):
    """returns (adapter, bstar, fstar, flags) for budget b from metrics U, L (dicts)."""
    ad_c = [a for a in AD_of(b) if a in arms_present]
    S = S_of(b)
    ok = [c for c in ad_c if np.isfinite(L.get(c, np.nan)) and np.isfinite(L.get(S, np.nan)) and L[c] <= L[S] + TOL_L]
    flag = []
    if ok:
        ad = max(ok, key=lambda c: U[c])
    else:
        ad = min(ad_c, key=lambda c: L[c])
        flag.append("adapter_no_arm_within_tol")
    lab = [x for x in [S, X_of(b)] + CE_of(b) if x in arms_present]
    okb = [c for c in lab if np.isfinite(L.get(c, np.nan)) and L[c] <= L[ad] + TOL_L]
    if okb:
        bs = max(okb, key=lambda c: U[c])
    else:
        bs = min([c for c in lab if np.isfinite(L.get(c, np.nan))], key=lambda c: L[c])
        flag.append("bstar_no_arm_within_tol")
    pool = IND + [x for bb in BUDGETS for x in [S_of(bb), X_of(bb)] + CE_of(bb)]
    pool = [x for x in pool if x in arms_present]
    okf = [c for c in pool if np.isfinite(L.get(c, np.nan)) and L[c] <= L[ad] + TOL_L]
    fs = max(okf, key=lambda c: U[c]) if okf else None
    return ad, bs, fs, flag


def ci(v):
    return AG.ci(v)


BUDGETS = [1, 3, 10, 30, 100]


def verdicts(res, margin, ce_adequate):
    out = {}
    q1_b, q2_b = {}, {}
    for b in PRIMARY:
        if b not in res:
            continue
        r = res[b]
        q1_b[b] = bool(r["dU_same"] >= margin and r["dU_same_donor"][0] > 0 and r["dU_same_study"][0] > 0 and sum(v >= margin for v in r["dU_same_folds"]) >= 2
                       and all((r["seed"].get(s, {}).get("dU_same") is not None and r["seed"][s]["dU_same"] >= margin and r["seed"][s]["dU_same_lb"] > 0) for s in (1, 2) if s in r["seed"]))
        surv = bool(r["dU_label"] >= margin and r["dU_label_donor"][0] > 0 and r["dU_label_study"][0] > 0 and sum(v >= margin for v in r["dU_label_folds"]) >= 2
                    and sum((r["seed"][s]["dU_label"] >= margin) for s in r["seed"] if r["seed"][s].get("dU_label") is not None) >= 2)
        gen = bool(r["dU_label_donor"][1] < margin)
        unconf = bool(r["dU_label"] >= margin and r["dU_label_donor"][0] > 0 and not surv)
        q2_b[b] = "SURV" if surv else ("GEN" if gen else ("EFFECT_UNCONFIRMED" if unconf else "STRADDLE"))
    n1 = sum(q1_b.values())
    out["Q1"] = "REPLICATES" if n1 == len(q1_b) and q1_b else ("PARTIAL" if n1 > 0 else "NOT REPLICATED")
    cats = list(q2_b.values())
    if cats and all(c == "SURV" for c in cats):
        q2 = "SUPCON GAIN SURVIVES"
    elif cats and all(c == "GEN" for c in cats):
        q2 = "GAIN IS GENERIC LABEL SUPERVISION"
    elif cats and all(c == "STRADDLE" for c in cats):
        q2 = "INCONCLUSIVE"
    else:
        q2 = "PARTIAL"
    if not ce_adequate:
        if q2 == "GAIN IS GENERIC LABEL SUPERVISION":
            q2 = "INCONCLUSIVE (CE control inadequate: generic verdict unavailable)"
        elif q2 == "SUPCON GAIN SURVIVES":
            q2 = "SURVIVES AGAINST AN INADEQUATE CONTROL (shortcut question INCONCLUSIVE)"
    out["Q2"] = q2
    out["Q1_by_budget"], out["Q2_by_budget"] = q1_b, q2_b
    return out


def main():
    global BUDGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--budgets", default="1,3,10,30,100")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--Bseed", type=int, default=300)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--no_power", action="store_true")
    a = ap.parse_args()
    BUDGETS = [int(x) for x in a.budgets.split(",")]
    D = AG.Data(a.dataset, a.runs)
    folds = list(D.fall)
    seeds = sorted(D.S)
    print(f"seeds {seeds}, folds {folds}", flush=True)
    # ---- trained gates from the per-seed meta files
    gates = {}
    for r in a.runs:
        for p in Path(r).glob("meta_s*.json"):
            s = int(p.stem.split("_s")[1])
            for k, v in json.loads(p.read_text()).get("adapters", {}).items():
                gates[(s, k)] = v
    raw_present = set()
    for s in seeds:
        raw_present |= {x for x, _ in D.S[s]["pr"]}
    universe = [x for x in IND + ["ctrl_noise16"] if x in raw_present]
    for b in BUDGETS:
        universe += [x for x in [S_of(b), X_of(b)] + AD_of(b) + CE_of(b) + [f"perm_f{b}_r0"] if x in raw_present]
    universe = list(dict.fromkeys(universe))
    ones = np.ones(D.n)

    # ---- fidelity check of the multi-seed kernels against E's own all_metrics
    FP = Fast(D, universe, folds)
    chk = [x for x in universe if FP.has(x) and not x.startswith(("perm", "ctrl"))]
    U0, L0, ch = FP.UL(chk, ones)
    ref = AG.all_metrics(D, D.S, chk, ones, folds)
    for arm in chk:
        assert abs(ref[arm]["U"] - U0[arm]) < 1e-9, (arm, ref[arm]["U"], U0[arm])
        if np.isfinite(ref[arm]["L"]):
            assert abs(ref[arm]["L"] - L0[arm]) < 1e-9, (arm, ref[arm]["L"], L0[arm])
    print("multi-seed fast kernels == E all_metrics at the point estimate (all arms): OK", flush=True)
    u0 = ref["_u0"]
    present = {x for x in universe if FP.has(x) or x.startswith("perm")}
    bud_present = [b for b in BUDGETS if S_of(b) in present and any(c in present for c in AD_of(b))]

    # ---- test-selected sensitivity
    Uall, Lall, _ = FP.UL([x for x in universe if FP.has(x)], ones)
    test_sel = {}
    for b in bud_present:
        ad, bs, fs, fl = choose(Uall, Lall, b, present)
        test_sel[b] = dict(adapter=ad, bstar=bs, fstar=fs, flags=fl, dU_same=Uall[ad] - Uall[S_of(b)], dU_label=Uall[ad] - Uall[bs],
                           dU_env=(Uall[ad] - Uall[fs]) if fs else None)

    # ---- leave-one-fold-out (validation-fold) selection and assembly
    lofo = {b: {"adapter": {}, "bstar": {}, "fstar": {}, "flags": {}} for b in bud_present}
    for k in folds:
        others = [f for f in folds if f != k]
        Fk = Fast(D, universe, others)
        arms_k = [x for x in universe if Fk.has(x)]
        Uk, Lk, _ = Fk.UL(arms_k, ones)
        pk = set(arms_k)
        for b in bud_present:
            ad, bs, fs, fl = choose(Uk, Lk, b, pk)
            lofo[b]["adapter"][k], lofo[b]["bstar"][k], lofo[b]["fstar"][k] = ad, bs, fs
            lofo[b]["flags"][k] = fl
    virt = {}
    for b in bud_present:
        for role in ("adapter", "bstar", "fstar"):
            ch_ = lofo[b][role]
            if any(v is None for v in ch_.values()):
                continue
            nm = f"V_{role}_f{b}"
            made = assemble(D, nm, ch_, folds)
            virt[(role, b)] = (nm, made)
    varms = [nm for nm, _ in virt.values()]
    allarms = list(dict.fromkeys(universe + varms))
    FA = Fast(D, allarms, folds)
    present2 = {x for x in allarms if FA.has(x)}
    Upt, Lpt, ch_pt = FA.UL([x for x in allarms if FA.has(x)], ones)
    nm = lambda role, b: virt[(role, b)][0] if (role, b) in virt else None  # noqa: E731

    # ---- bootstrap (donor, study cluster) with the selection fixed
    tab_arms = {b: [x for x in AD_of(b) + CE_of(b) + [X_of(b), S_of(b)] if x in present2] for b in bud_present}
    res_boot = {}
    for label, cluster in (("donor", False), ("study", True)):
        Ws = AG.resamples(D, a.B, cluster=cluster)
        acc = {b: {k: [] for k in ("dS", "dL", "dE", "Uad", "Lad", "Ubs", "Lbs", "Ufs", "Us")} for b in bud_present}
        tabU = {x: [] for b in bud_present for x in tab_arms[b] if b in PRIMARY}
        tabL = {x: [] for x in tabU}
        indU = {x: [] for x in IND if x in present2}
        indL = {x: [] for x in indU}
        for w in Ws:
            den, chw = FA.den_chance(w)
            raw = FA.worst("raw50", w, den)

            def Lof(arm):
                wa = FA.worst(arm, w, den)
                return (wa - chw) / (raw - chw) if raw - chw >= 0.10 else np.nan
            for b in bud_present:
                S = S_of(b)
                va, vb, vf = nm("adapter", b), nm("bstar", b), nm("fstar", b)
                ua, us = FA.U(va, w), FA.U(S, w)
                ub = FA.U(vb, w) if vb else np.nan
                uf = FA.U(vf, w) if vf else np.nan
                acc[b]["dS"].append(ua - us)
                acc[b]["dL"].append(ua - ub)
                acc[b]["dE"].append(ua - uf)
                acc[b]["Uad"].append(ua)
                acc[b]["Ubs"].append(ub)
                acc[b]["Us"].append(us)
                if b in PRIMARY:
                    acc[b]["Lad"].append(Lof(va))
                    acc[b]["Lbs"].append(Lof(vb) if vb else np.nan)
                    for x in tab_arms[b]:
                        tabU[x].append(FA.U(x, w))
                        tabL[x].append(Lof(x))
            for x in indU:
                indU[x].append(FA.U(x, w))
                indL[x].append(Lof(x))
        res_boot[label] = dict(acc=acc, tabU=tabU, tabL=tabL, indU=indU, indL=indL, Ws=Ws)
        print(f"bootstrap {label}: done", flush=True)

    # ---- seed-level and fold-level estimates
    seed_est = {b: {} for b in bud_present}
    Wd = res_boot["donor"]["Ws"][: a.Bseed]
    for s in seeds:
        Fs = Fast(D, allarms, folds, seeds=[s])
        for b in bud_present:
            va, vb, S = nm("adapter", b), nm("bstar", b), S_of(b)
            if not (Fs.has(va) and Fs.has(S) and (vb is None or Fs.has(vb))):
                continue
            e = dict(dU_same=Fs.U(va, ones) - Fs.U(S, ones), dU_label=(Fs.U(va, ones) - Fs.U(vb, ones)) if vb and Fs.has(vb) else None)
            if b in (1, 10, 100):
                ds = [Fs.U(va, w) - Fs.U(S, w) for w in Wd]
                e["dU_same_lb"] = ci(ds)[0]
                e["dU_same_ci"] = ci(ds)
                if e["dU_label"] is not None:
                    dl = [Fs.U(va, w) - Fs.U(vb, w) for w in Wd]
                    e["dU_label_lb"] = ci(dl)[0]
                    e["dU_label_ci"] = ci(dl)
            seed_est[b][s] = e
    fold_est = {b: {} for b in bud_present}
    for k in folds:
        Fk = Fast(D, allarms, [k])
        for b in bud_present:
            va, vb, vf, S = nm("adapter", b), nm("bstar", b), nm("fstar", b), S_of(b)
            fold_est[b][k] = dict(dU_same=Fk.U(va, ones) - Fk.U(S, ones), dU_label=(Fk.U(va, ones) - Fk.U(vb, ones)) if vb else None,
                                  dU_env=(Fk.U(va, ones) - Fk.U(vf, ones)) if vf else None)

    # ---- assemble results per budget
    res = {}
    for b in bud_present:
        ac_d, ac_s = res_boot["donor"]["acc"][b], res_boot["study"]["acc"][b]
        va, vb, vf, S = nm("adapter", b), nm("bstar", b), nm("fstar", b), S_of(b)
        res[b] = dict(
            lofo_choice={role: lofo[b][role] for role in ("adapter", "bstar", "fstar")}, flags=lofo[b]["flags"],
            n_seeds=FA.n_seeds(va),
            U_adapter=Upt[va], L_adapter=Lpt[va], U_same=Upt[S], L_same=Lpt[S],
            U_bstar=Upt.get(vb), L_bstar=Lpt.get(vb), U_fstar=Upt.get(vf), L_fstar=Lpt.get(vf),
            dU_same=Upt[va] - Upt[S], dU_same_donor=ci(ac_d["dS"]), dU_same_study=ci(ac_s["dS"]), dU_same_folds=[fold_est[b][k]["dU_same"] for k in folds],
            dU_label=(Upt[va] - Upt[vb]) if vb else None, dU_label_donor=ci(ac_d["dL"]), dU_label_study=ci(ac_s["dL"]),
            dU_label_folds=[fold_est[b][k]["dU_label"] for k in folds],
            dU_env=(Upt[va] - Upt[vf]) if vf else None, dU_env_donor=ci(ac_d["dE"]), dU_env_study=ci(ac_s["dE"]),
            dU_env_folds=[fold_est[b][k]["dU_env"] for k in folds],
            seed=seed_est[b], test_selected=test_sel[b])
        if "noise_r1" in present2:
            res[b]["dU_vs_noise_r1"] = Upt[va] - Upt["noise_r1"]
            res[b]["L_noise_r1"] = Lpt["noise_r1"]
        if X_of(b) in present2 and FA.has(X_of(b)):
            res[b]["ldaX_mlp_U"] = FA.U(X_of(b), ones, kinds=("mlp",)) if any(((X_of(b), "mlp") in D.S[s]["pr"]) for s in seeds) else None
    # mlp probe of ldaX needs its own Fast with the extra probe kind
    Fm = Fast(D, [X_of(b) for b in bud_present if X_of(b) in present2], folds, probes=("lin", "knn", "mlp"))
    for b in bud_present:
        if X_of(b) in present2:
            res[b]["ldaX_mlp_U"] = Fm.U(X_of(b), ones, kinds=("mlp",))
            res[b]["ldaX_U"] = Upt[X_of(b)]

    # ---- table arms (U, L with donor CIs) for the primary budgets and inductive floors
    table = {}
    for b in PRIMARY:
        if b not in bud_present:
            continue
        for x in tab_arms[b]:
            table[x] = dict(U=Upt[x], U_ci=ci(res_boot["donor"]["tabU"][x]), L=Lpt[x], L_ci=ci(res_boot["donor"]["tabL"][x]), n_seeds=FA.n_seeds(x))
    for x in IND:
        if x in present2:
            table[x] = dict(U=Upt[x], U_ci=ci(res_boot["donor"]["indU"][x]), L=Lpt[x], L_ci=ci(res_boot["donor"]["indL"][x]), n_seeds=FA.n_seeds(x))

    # ---- validity, gates, adequacy
    ctrl = {}
    for s in seeds:
        Fs = Fast(D, ["ctrl_noise16", "raw50"], folds, seeds=[s])
        den, chs = Fs.den_chance(ones)
        ctrl[s] = dict(U=Fs.U("ctrl_noise16", ones), u0=u0, worst=Fs.worst("ctrl_noise16", ones, den), chance=chs)
    ctrl_ok = all(abs(v["U"] - v["u0"]) <= 0.01 and v["worst"] <= v["chance"] + 0.03 for v in ctrl.values())
    trained = {}
    for b in PRIMARY:
        rows = []
        for s in seeds:
            for k in folds:
                g = gates.get((s, f"{k}|adp_f{b}"))
                gc = gates.get((s, f"{k}|ce_f{b}"))
                rows.append(dict(seed=s, fold=k, supcon_ratio=(g["loss_last200"] / g["loss_first20"]) if g else None, supcon_rank=g["eff_rank_held"] if g else None,
                                 ce_ratio=(gc["loss_last200"] / gc["loss_first20"]) if gc else None, ce_rank=gc["eff_rank_held"] if gc else None))
        sup_ok = all(r["supcon_ratio"] is None or (r["supcon_ratio"] <= GATE_LOSS and r["supcon_rank"] >= GATE_RANK) for r in rows)
        ce_ok = all(r["ce_ratio"] is not None and r["ce_ratio"] <= GATE_LOSS and r["ce_rank"] >= GATE_RANK for r in rows)
        perm = Upt.get(f"perm_f{b}_r0")
        dperm = (Upt.get(f"adp_f{b}_r0") - perm) if perm is not None and Upt.get(f"adp_f{b}_r0") is not None else None
        trained[b] = dict(rows=rows, supcon_ok=bool(sup_ok), ce_ok=bool(ce_ok), dU_vs_perm=dperm, perm_ok=(dperm is None or dperm >= GATE_PERM))
    sup_valid = all(t["supcon_ok"] and t["perm_ok"] for t in trained.values())
    ce_ad = bool(all(t["ce_ok"] for t in trained.values()) and (100 in bud_present) and Upt.get("ce_f100_r0") is not None and Upt["ce_f100_r0"] >= Upt["ldaF16"] - 0.02)
    valid = bool(ctrl_ok and sup_valid)
    V = {"original_0.02": verdicts(res, M_ORIG, ce_ad), "tightened_0.035": verdicts(res, M_TIGHT, ce_ad)}
    if not valid:
        V["INVALID"] = True

    mde = None
    if not a.no_power and 10 in bud_present and nm("adapter", 10) and nm("bstar", 10):
        try:
            rows = AG.power_audit(D, [nm("adapter", 10), nm("bstar", 10)], nperm=2000, B=1000)
            mde = rows[0]
            print("MDE80 f=10 (seed 0 only; adapter vs best label floor):", mde, flush=True)
        except Exception as e:                                   # never lose the bootstrap results to the power audit
            mde = dict(error=repr(e))
            print("power audit failed:", repr(e), flush=True)

    js = dict(valid=valid, ctrl=ctrl, ctrl_ok=ctrl_ok, supcon_valid=sup_valid, ce_adequate=ce_ad, verdicts=V, budgets=res, table=table, trained=trained,
              mde80_f10_seed0=mde, seeds=seeds, folds=folds, B=a.B, Bseed=a.Bseed, tol_L=TOL_L, selection="leave-one-fold-out (validation folds)",
              original_verdicts="section 18.2: K-RULE REFUTED at margin 0.02 (seed 0); JUMP NO-GO at S4 gate; E SC-0 NO-GO")
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1, default=float))
    L = [f"# Controlled replication (HLCA, d16, seeds {seeds}, folds {folds}); valid={valid}; CE control adequate={ce_ad}",
         f"Q1 (replicates 18.2): original {V['original_0.02']['Q1']} | tightened {V['tightened_0.035']['Q1']}",
         f"Q2 (shortcut?): original {V['original_0.02']['Q2']} | tightened {V['tightened_0.035']['Q2']}  (by budget: {V['original_0.02']['Q2_by_budget']})", "",
         "Selection: leave-one-fold-out. dU in macro-F1; intervals 95% donor bootstrap [study-cluster]; folds F1/F2/F3; seeds s0/s1/s2.", "",
         "| f | n seeds | U adp | L adp | U same | dU_same | dU_label | B* (per fold) | dU_env | vs noise_r1 | folds dU_label | seeds dU_label | test-selected dU_label |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in bud_present:
        r = res[b]
        sl = ", ".join(f"s{s}:{(v['dU_label'] if v['dU_label'] is not None else float('nan')):+.3f}" for s, v in sorted(r["seed"].items()))
        fl = ", ".join(f"{v:+.3f}" if v is not None else "nan" for v in r["dU_label_folds"])
        L.append(f"| {b}% | {r['n_seeds']} | {r['U_adapter']:.3f} | {r['L_adapter']:.2f} | {r['U_same']:.3f} | {r['dU_same']:+.3f} [{r['dU_same_donor'][0]:+.3f},{r['dU_same_donor'][1]:+.3f}] [{r['dU_same_study'][0]:+.3f},{r['dU_same_study'][1]:+.3f}] | "
                 f"{r['dU_label']:+.3f} [{r['dU_label_donor'][0]:+.3f},{r['dU_label_donor'][1]:+.3f}] [{r['dU_label_study'][0]:+.3f},{r['dU_label_study'][1]:+.3f}] | "
                 f"{'/'.join(r['lofo_choice']['bstar'][k] for k in folds)} | {r['dU_env']:+.3f} [{r['dU_env_donor'][0]:+.3f},{r['dU_env_donor'][1]:+.3f}] | {r.get('dU_vs_noise_r1', float('nan')):+.3f} | {fl} | {sl} | {r['test_selected']['dU_label']:+.3f} |")
    L += ["", "Leakage axis (U, L; raw50 = 1, chance = 0), primary budgets:", "", "| arm | n seeds | U [95%] | L [95%] |", "|---|---|---|---|"]
    for x, v in table.items():
        L.append(f"| {x} | {v['n_seeds']} | {v['U']:.3f} [{v['U_ci'][0]:.3f},{v['U_ci'][1]:.3f}] | {v['L']:.2f} [{v['L_ci'][0]:.2f},{v['L_ci'][1]:.2f}] |")
    L += ["", f"controls: {ctrl}", f"MDE80 f=10 (seed 0): {mde.get('mde80', mde) if mde else 'not run'}",
          "'beats label floors' is not 'removes batch': L stays high at r = 0 for the adapter and the CE floor."]
    if a.md:
        Path(a.md).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
