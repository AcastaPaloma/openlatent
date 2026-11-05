"""Aggregator for the reduced H3-003 (results_notes/F.md section 18).  CPU only.  Reuses E's Data / resamples / acc_by_kind / f1_macro / power_audit.

  python aggregate_hlca_k.py --runs <dir_f0> <dir_f1> <dir_f2> [--B 1000] [--json out.json] [--md out.md]

Fast kernels (precomputed prediction indices) are checked against E's own all_metrics at the point estimate before any interval is computed.
Selection (declared in the pre-registration): adapter arm per budget = higher pooled U of r in {0, 0.5} among L <= L(same-label LDA) + 0.05;
full envelope F*_f = highest pooled U among floors and same-label LDAs (any budget) with L <= L(adapter) + 0.05.  Selection is fixed at the
point-estimate choice in every resample.
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
FLOORS = ["raw50", "pca16", "rp16", "noise_r0.5", "noise_r1", "dith16_r0.5", "dith16_r1", "ldaC_pca16", "ldaF16"]
BUDGETS = [1, 3, 10, 30, 100]
TOL_L = 0.05
MARGIN_SAME = 0.02            # K-rule refutation / consistency threshold on dU_same
MARGIN_FULL = 0.035           # break-even threshold vs the full envelope (E's floor-vs-floor MDE80 median)
PRED_SAME = (-0.03, 0.01)
PRED_FULL_MAX = 0.015
GATE_LOSS, GATE_RANK, GATE_PERM = 0.8, 4.0, 0.02


def lda_name(b):
    return "ldaF16" if b == 100 else f"ldaM_f{b}"


class Fast:
    def __init__(self, D, arms, fset):
        self.D, self.fset, self.K = D, list(fset), len(D.fine_names)
        s = D.S[min(D.S)]
        self.s = s
        self.m = D.evalm & np.isin(D.fold_of, self.fset)
        self.t = D.fine[self.m]
        self.pidx, self.ok = {}, {}
        for a in arms:
            for kind in PROBES:
                if (a, kind) in s["pr"]:
                    p = s["pr"][(a, kind)][self.m]
                    assert (p >= 0).all(), (a, kind)
                    self.pidx[(a, kind)] = self.t * self.K + p.astype(np.int64)
            for kind in KINDS:
                if (a, kind) in s["att"]:
                    self.ok[(a, kind)] = (s["att"][(a, kind)] == D.yb).astype(float)

    def U(self, arm, w):
        wm = w[self.m]
        best = -1.0
        for kind in PROBES:
            if (arm, kind) not in self.pidx:
                continue
            C = np.bincount(self.pidx[(arm, kind)], weights=wm, minlength=self.K * self.K).reshape(self.K, self.K)
            tp = np.diag(C)
            fn, fp = C.sum(1) - tp, C.sum(0) - tp
            sup = C.sum(1) > 0
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
            best = max(best, float(f1[sup].mean()))
        return best

    def worst(self, arm, w, den):
        D = self.D
        best = np.nan
        for kind in KINDS:
            if (arm, kind) not in self.ok:
                continue
            num = np.bincount(D.cls, weights=w * self.ok[(arm, kind)], minlength=D.nf * 16)
            accs = []
            for f in self.fset:
                sl = slice(f * 16, f * 16 + D.Bf[f])
                o = den[sl] > 0
                if o.any():
                    accs.append(np.mean(num[sl][o] / den[sl][o]))
            v = float(np.mean(accs))
            best = v if np.isnan(best) else max(best, v)
        return best

    def chance(self, w, den):
        D = self.D
        return float(np.mean([1.0 / max(int((den[f * 16: f * 16 + D.Bf[f]] > 0).sum()), 1) for f in self.fset]))

    def UL(self, arms, w):
        den = np.bincount(self.D.cls, weights=w, minlength=self.D.nf * 16)
        ch = self.chance(w, den)
        raw = self.worst("raw50", w, den)
        U, L = {}, {}
        for a in arms:
            U[a] = self.U(a, w)
            wa = self.worst(a, w, den)
            L[a] = (wa - ch) / (raw - ch) if (raw - ch >= 0.10 and not np.isnan(wa)) else np.nan
        return U, L, ch


def pick_adapter(U, L, b):
    ref = lda_name(b)
    cands = [f"adp_f{b}_r0", f"adp_f{b}_r0.5"]
    ok = [c for c in cands if L[c] <= L[ref] + TOL_L]
    if ok:
        return max(ok, key=lambda c: U[c]), False
    return min(cands, key=lambda c: L[c]), True


def pick_envelope(U, L, Lad, pool):
    ok = [c for c in pool if L[c] <= Lad + TOL_L]
    return max(ok, key=lambda c: U[c]) if ok else None


def ci(v):
    return AG.ci(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--dataset", default="hlca")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--no_power", action="store_true")
    a = ap.parse_args()
    D = AG.Data(a.dataset, a.runs)
    fall = list(D.fall)
    assert fall == [0, 1, 2], f"folds done {fall}"
    meta = {}
    for r in a.runs:
        for p in Path(r).glob("meta_s*.json"):
            meta.update({k: v for k, v in json.loads(p.read_text()).get("adapters", {}).items()})
    lda_pool = [lda_name(b) for b in BUDGETS]
    adp = [f"adp_f{b}_r{r}" for b in BUDGETS for r in ("0", "0.5")]
    perm = [f"perm_f{b}_r0" for b in BUDGETS]
    boot_arms = FLOORS + [x for x in lda_pool if x != "ldaF16"] + adp
    all_arms = boot_arms + perm + ["ctrl_noise16"]
    ones = np.ones(D.n)

    # ---- fidelity check against E's own kernels
    FP = Fast(D, all_arms, fall)
    U, L, ch = FP.UL(boot_arms, ones)
    ref = AG.all_metrics(D, D.S, boot_arms, ones, fall)
    for arm in boot_arms:
        assert abs(ref[arm]["U"] - U[arm]) < 1e-9, (arm, ref[arm]["U"], U[arm])
        if np.isfinite(ref[arm]["L"]):
            assert abs(ref[arm]["L"] - L[arm]) < 1e-9, (arm, ref[arm]["L"], L[arm])
    print("fast kernels == E all_metrics at the point estimate (all arms): OK", flush=True)
    u0 = ref["_u0"]
    Up = {x: FP.U(x, ones) for x in perm}
    U["ctrl_noise16"] = FP.U("ctrl_noise16", ones)
    den1 = np.bincount(D.cls, weights=ones, minlength=D.nf * 16)
    ctrl_att = FP.worst("ctrl_noise16", ones, den1)

    # ---- point selection
    sel = {}
    for b in BUDGETS:
        ad, flag = pick_adapter(U, L, b)
        env = pick_envelope(U, L, L[ad], FLOORS + [x for x in lda_pool])
        sel[b] = dict(adapter=ad, no_arm_within_tol=flag, envelope=env, same=lda_name(b))

    # ---- trained gate
    gate = {}
    for b in BUDGETS:
        rows = []
        for f in fall:
            i, j = meta.get(f"{f}|adp_f{b}"), meta.get(f"{f}|perm_f{b}")
            ratio = i["loss_last200"] / i["loss_first20"] if i and not i["nan"] else float("nan")
            rows.append(dict(fold=f, loss_ratio=ratio, eff_rank=i["eff_rank_held"] if i else float("nan"),
                             perm_loss_ratio=(j["loss_last200"] / j["loss_first20"]) if j else float("nan"), n_labelled=i["n_labelled"] if i else 0))
        dperm = U[f"adp_f{b}_r0"] - Up[f"perm_f{b}_r0"]
        ok = all(r["loss_ratio"] <= GATE_LOSS and r["eff_rank"] >= GATE_RANK for r in rows) and dperm >= GATE_PERM
        gate[b] = dict(per_fold=rows, dU_vs_perm=dperm, trained=bool(ok))

    # ---- bootstrap (donor, study cluster) with the selection fixed
    def need(b):
        return [sel[b]["adapter"], sel[b]["same"]] + ([sel[b]["envelope"]] if sel[b]["envelope"] else [])
    res = {}
    for label, cluster in (("donor", False), ("study", True)):
        Ws = AG.resamples(D, a.B, cluster=cluster)
        dS = {b: [] for b in BUDGETS}
        dF = {b: [] for b in BUDGETS}
        Ub = {x: [] for x in boot_arms}
        Lb = {x: [] for x in boot_arms}
        for w in Ws:
            Uw, Lw, _ = FP.UL(boot_arms, w)
            for x in boot_arms:
                Ub[x].append(Uw[x])
                Lb[x].append(Lw[x])
            for b in BUDGETS:
                dS[b].append(Uw[sel[b]["adapter"]] - Uw[sel[b]["same"]])
                dF[b].append(Uw[sel[b]["adapter"]] - Uw[sel[b]["envelope"]] if sel[b]["envelope"] else np.nan)
        res[label] = dict(dS=dS, dF=dF, U=Ub, L=Lb)
        print(f"bootstrap {label}: done", flush=True)

    # ---- fold-level dU_same (selection fixed)
    FF = {f: Fast(D, all_arms, [f]) for f in fall}
    fold_dS = {b: [FF[f].U(sel[b]["adapter"], ones) - FF[f].U(sel[b]["same"], ones) for f in fall] for b in BUDGETS}
    fold_dF = {b: [(FF[f].U(sel[b]["adapter"], ones) - FF[f].U(sel[b]["envelope"], ones)) if sel[b]["envelope"] else np.nan for f in fall] for b in BUDGETS}

    # ---- verdict
    out_b = {}
    for b in BUDGETS:
        s = sel[b]
        dS = U[s["adapter"]] - U[s["same"]]
        dF = (U[s["adapter"]] - U[s["envelope"]]) if s["envelope"] else float("nan")
        out_b[b] = dict(sel=s, U_adapter=U[s["adapter"]], L_adapter=L[s["adapter"]], U_same=U[s["same"]], L_same=L[s["same"]],
                        U_env=U[s["envelope"]] if s["envelope"] else None, L_env=L[s["envelope"]] if s["envelope"] else None,
                        dU_same=dS, dU_same_donor=ci(res["donor"]["dS"][b]), dU_same_study=ci(res["study"]["dS"][b]), dU_same_folds=fold_dS[b],
                        dU_full=dF, dU_full_donor=ci(res["donor"]["dF"][b]), dU_full_study=ci(res["study"]["dF"][b]), dU_full_folds=fold_dF[b],
                        trained=gate[b]["trained"], in_pred_same=bool(PRED_SAME[0] <= dS <= PRED_SAME[1]), in_pred_full=bool(dF <= PRED_FULL_MAX) if np.isfinite(dF) else None)
    n_trained = sum(gate[b]["trained"] for b in BUDGETS)
    refuted_at = [b for b in BUDGETS if b <= 30 and gate[b]["trained"] and out_b[b]["dU_same"] >= MARGIN_SAME
                  and out_b[b]["dU_same_donor"][0] > 0 and out_b[b]["dU_same_study"][0] > 0 and sum(v >= MARGIN_SAME for v in fold_dS[b]) >= 2]
    ub_all = [out_b[b]["dU_same_donor"][1] for b in BUDGETS if gate[b]["trained"]]
    ctrl_ok = bool(abs(U["ctrl_noise16"] - u0) <= 0.01 and ctrl_att <= ch + 0.03)
    if n_trained < 3 or not ctrl_ok:
        verdict = "INVALID"
    elif refuted_at:
        verdict = "K-RULE REFUTED"
    elif all(x < MARGIN_SAME for x in ub_all):
        verdict = "K-RULE CONSISTENT"
    else:
        verdict = "INCONCLUSIVE"
    breakeven = [b for b in BUDGETS if gate[b]["trained"] and np.isfinite(out_b[b]["dU_full"]) and out_b[b]["dU_full"] >= MARGIN_FULL and out_b[b]["dU_full_donor"][0] > 0]

    # ---- MDE80 of the headline difference (adapter f=10 vs same-label LDA f=10), E's paired within-donor sign-flip method
    mde = None
    if not a.no_power:
        rows = AG.power_audit(D, [sel[10]["adapter"], sel[10]["same"]], nperm=2000, B=1000)
        mde = rows[0]
        print("MDE80 f=10:", mde, flush=True)

    js = dict(verdict=verdict, n_trained=n_trained, control_ok=ctrl_ok, ctrl_noise16=dict(U=U["ctrl_noise16"], u0=u0, worst_att=ctrl_att, chance=ch),
              refuted_at=refuted_at, breakeven_vs_full=breakeven, mde80_f10=mde, budgets=out_b, gate=gate,
              pooled={x: dict(U=U[x], U_ci=ci(res["donor"]["U"][x]), L=L[x], L_ci=ci(res["donor"]["L"][x])) for x in boot_arms},
              perm_U=Up, raw_L_note="L normalised so that raw50 = 1, chance = 0", prediction=dict(dU_same_range=PRED_SAME, dU_full_max=PRED_FULL_MAX, k_rule_point=-0.012),
              verdict_original=dict(jump="NO-GO at the S4 gate; stage 1 partial survival; F4 PARTIAL; K-law falsified by the wells-per-class control",
                                    single_cell="SC-0 NO-GO for stage 2 (E); no learned arm was run"),
              B=a.B, seeds=sorted(D.S), folds=fall)
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1, default=float))
    lines = [f"# Reduced H3-003 (HLCA, R2, d16, 1 seed, 3 folds): verdict {verdict}", "",
             f"control ctrl_noise16: U {U['ctrl_noise16']:.3f} vs chance {u0:.3f}, worst attacker {ctrl_att:.3f} vs chance {ch:.3f} -> {'ok' if ctrl_ok else 'FAIL'}; trained budgets {n_trained}/5",
             "", "| f | trained | adapter arm | U_ad | L_ad | U_LDA(same) | L_LDA | dU_same [donor] [study] | folds | envelope | dU_full [donor] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in BUDGETS:
        o = out_b[b]
        lines.append(f"| {b}% | {o['trained']} (dU vs perm {gate[b]['dU_vs_perm']:+.3f}) | {o['sel']['adapter']}{'*' if o['sel']['no_arm_within_tol'] else ''} | {o['U_adapter']:.3f} | {o['L_adapter']:.2f} | {o['U_same']:.3f} | {o['L_same']:.2f} | "
                     f"{o['dU_same']:+.3f} [{o['dU_same_donor'][0]:+.3f},{o['dU_same_donor'][1]:+.3f}] [{o['dU_same_study'][0]:+.3f},{o['dU_same_study'][1]:+.3f}] | "
                     f"{', '.join(f'{v:+.3f}' for v in o['dU_same_folds'])} | {o['sel']['envelope']} | {o['dU_full']:+.3f} [{o['dU_full_donor'][0]:+.3f},{o['dU_full_donor'][1]:+.3f}] |")
    lines += ["", f"refuted_at: {refuted_at}; break-even vs full envelope at: {breakeven}; MDE80 (f=10 adapter vs same-label LDA): {mde['mde80'] if mde else 'not run'}"]
    if a.md:
        Path(a.md).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
