"""E2 aggregation (E.md section 6): inductive learned encoders vs the inductive floor envelope at the leakage of the scVI / Harmony points, HLCA.  CPU only.

  python aggregate_e2.py --runs <floors s0-2> <tscvi s0-2> <dense s0-2> <e2 s0-2> --B 1000 --cluster 300 --json out.json --md out.md
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as AG  # noqa: E402
import inductive as IN  # noqa: E402
from aggregate import ci, fmt  # noqa: E402

TARGETS = {"scvi": dict(U=0.816, L=0.80), "harmony": dict(U=0.668, L=0.37)}      # fixed from SC-0 (HLCA, 3-seed means), E.md 6.6
M = 0.05
LO_RE = re.compile(r"^(pca\d+|rp\d+|ldaC|ldaC_pca\d+|ldaF16|leace_pca16)$")


def select(D, pool, norm_arm, LT, fall, ones):
    """cross-fold selection: for every outer fold the pool member with the highest U on the other folds among those with L <= LT + 0.05 there"""
    sel = {}
    for f in fall:
        oth = [x for x in fall if x != f]
        Mo = AG.all_metrics(D, D.S, list(pool) + [norm_arm], ones, oth)
        cand = [a for a in pool if Mo[a]["L"] <= LT + 0.05 and np.isfinite(Mo[a]["U"])]
        sel[f] = max(cand, key=lambda a: Mo[a]["U"]) if cand else None
    return sel


def virtual(D, name, sel):
    for s, sd in D.S.items():
        for kind in AG.KINDS:
            v = np.full(D.n, -1, np.int8)
            for f, a in sel.items():
                m = D.fold_of == f
                v[m] = sd["att"][(a, kind)][m]
            sd["att"][(name, kind)] = v
        for kind in AG.PROBES:
            v = np.full(D.n, -1, np.int16)
            for f, a in sel.items():
                m = D.fold_of == f
                v[m] = sd["pr"][(a, kind)][m]
            sd["pr"][(name, kind)] = v


def load_checks(run_dirs):
    chk = {}
    for r in run_dirs:
        for p in sorted(Path(r).glob("meta_s*.json")):
            j = json.loads(p.read_text())
            if not j.get("e2"):
                continue
            for k, v in j["checks"].items():
                chk.setdefault(k, []).append(v)
    return chk


def eligibility(chk, learned, nf):
    out = {}
    for arm in learned:
        cfg = IN.parse(arm[3:])
        why, ok = [], True
        for f in range(nf):
            vs = chk.get(f"{f}|{arm}", [])
            if not vs:
                ok = False
                why.append(f"fold {f}: no check")
                continue
            kl = np.mean([v["kl"] for v in vs])
            r2, r2p = np.mean([v["r2"] for v in vs]), np.mean([v["r2_pca16"] for v in vs])
            excess = max(np.mean([v["acc_linear"] for v in vs]), np.mean([v["acc_mlp"] for v in vs])) - vs[0]["chance_train"]
            if kl < 1.0 or r2 < 0.5 * r2p:
                ok = False
                why.append(f"fold {f}: collapsed (KL {kl:.1f}, R2 {r2:.2f} vs pca16 {r2p:.2f})")
            if cfg["adv"] and excess > 0.05:
                ok = False
                why.append(f"fold {f}: adversary not converged (donor-disjoint excess {excess:.3f})")
            ins = max(np.mean([v["in_linear"] for v in vs]), np.mean([v["in_mlp"] for v in vs]))
            fl = np.mean([v["floor_in_noise"] for v in vs])
            if cfg["adv"] and ins > fl + 0.05:
                ok = False
                why.append(f"fold {f}: in-sample adversary {ins:.3f} above memorisation floor {fl:.3f} + 0.05")
        out[arm] = dict(eligible=ok, why=why, adversarial=cfg["adv"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    D = AG.Data("hlca", a.runs)
    ones = np.ones(D.n)
    fall = list(D.fall)
    arms = D.arms()
    learned_all = sorted(x for x in arms if x.startswith("e2_"))
    floors = [x for x in arms if AG._IND_RE.match(x)]
    lo = [x for x in floors if LO_RE.match(x)]
    refs = [x for x in ("T_scvi", "T_harmony") if x in arms]
    chk = load_checks(a.runs)
    elig = eligibility(chk, learned_all, D.nf)
    learned = [x for x in learned_all if elig[x]["eligible"]]
    L, P = [], None
    P = L.append
    P(f"## E2 (HLCA): seeds {sorted(D.S)}, folds {fall}, {D.n} cells, {len(D.units)} donors, B={a.B}; floor pool {len(floors)} arms, learned configs {len(learned_all)} ({len(learned)} eligible)\n")
    pool_all = sorted(set(floors + learned_all + refs + ["raw50"]))
    pt = AG.all_metrics(D, D.S, pool_all, ones, fall)
    Ws = AG.resamples(D, min(a.B, 300))
    boot = {x: {k: [] for k in ("U", "Uh", "L")} for x in learned_all + refs + ["raw50"]}
    for w in Ws:
        m = AG.all_metrics(D, D.S, learned_all + refs + ["raw50"], w, fall)
        for x in boot:
            for k in boot[x]:
                boot[x][k].append(m[x][k])
    P("| learned config | eligible | in-sample adv excess (lin/mlp, fold-mean) | KL | R2 / pca16 R2 | U_fine [CI] | L [CI] | U_hold |\n|---|---|---|---|---|---|---|---|")
    js = dict(elig=elig, rows={})
    for x in learned_all:
        vs = [v for f in range(D.nf) for v in chk.get(f"{f}|{x}", [])]
        el = elig[x]
        P(f"| {x} | {el['eligible']}{'' if el['eligible'] else ' (' + '; '.join(el['why'][:2]) + ')'} | "
          f"{np.mean([v['acc_linear'] for v in vs]) - vs[0]['chance_train']:.3f} / {np.mean([v['acc_mlp'] for v in vs]) - vs[0]['chance_train']:.3f} | {np.mean([v['kl'] for v in vs]):.1f} | "
          f"{np.mean([v['r2'] for v in vs]):.2f} / {np.mean([v['r2_pca16'] for v in vs]):.2f} | {fmt(pt[x]['U'], boot[x]['U'])} | {fmt(pt[x]['L'], boot[x]['L'], 2)} | {pt[x]['Uh']:.3f} |")
        js["rows"][x] = dict(U=pt[x]["U"], L=pt[x]["L"], Uh=pt[x]["Uh"], eligible=el["eligible"], U_ci=ci(boot[x]["U"]), L_ci=ci(boot[x]["L"]))
    for r in refs + ["raw50"]:
        P(f"| (reference) {r} | | | | | {fmt(pt[r]['U'], boot[r]['U'])} | {fmt(pt[r]['L'], boot[r]['L'], 2)} | {pt[r]['Uh']:.3f} |")

    # --- K1' recomputed from the SC-0 raw50 / control arms
    raw = pt["raw50"]
    ch = pt["_chance"]
    ctrl = {c: pt[c]["worst"] - ch for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_batch") if c in pt}
    k1 = bool(raw["worst"] - ch >= 0.15 and all(v <= 0.03 for v in ctrl.values()))
    P(f"\n**E2-K1'**: raw50 worst attacker {raw['worst']:.3f} vs chance {ch:.3f} (excess {raw['worst'] - ch:.3f} >= 0.15); controls {({k: round(v, 3) for k, v in ctrl.items()})}: {'PASS' if k1 else 'FAIL'}")
    js["K1"] = dict(passed=k1, raw_excess=raw["worst"] - ch)

    res = {}
    for tname, T in TARGETS.items():
        LT = T["L"]
        out = dict(target=T)
        # label-only / PCA family at the target's leakage (K2')
        sel_lo = select(D, lo, "raw50", LT, fall, ones)
        if all(v is not None for v in sel_lo.values()):
            virtual(D, f"sel_lo_{tname}", sel_lo)
            m = AG.all_metrics(D, D.S, [f"sel_lo_{tname}", "raw50"], ones, fall)
            U_lo, L_lo = m[f"sel_lo_{tname}"]["U"], m[f"sel_lo_{tname}"]["L"]
            out["K2p"] = dict(sel=sel_lo, U=U_lo, L=L_lo, gap=T["U"] - U_lo, fires=bool(T["U"] - U_lo <= M),
                              retained_fraction=(U_lo - pt["_u0"]) / (T["U"] - pt["_u0"]))
        else:
            out["K2p"] = dict(sel=sel_lo, U=None, fires=False, note="no label-only/PCA arm feasible at this leakage in at least one fold")
        # learned vs floor envelope
        sel_f = select(D, floors, "raw50", LT, fall, ones)
        sel_l = select(D, learned, "raw50", LT, fall, ones) if learned else {f: None for f in fall}
        out["sel_floor"], out["sel_learned"] = sel_f, sel_l
        feas_l = all(v is not None for v in sel_l.values())
        feas_f = all(v is not None for v in sel_f.values())
        if feas_l:
            virtual(D, f"sel_learned_{tname}", sel_l)
        if feas_f:
            virtual(D, f"sel_floor_{tname}", sel_f)
        if feas_l and feas_f:
            va, vf = f"sel_learned_{tname}", f"sel_floor_{tname}"
            m0 = AG.all_metrics(D, D.S, [va, vf, "raw50"], ones, fall)
            bd, bl = [], []
            for w in AG.resamples(D, a.B):
                m = AG.all_metrics(D, D.S, [va, vf, "raw50"], w, fall)
                bd.append(m[va]["U"] - m[vf]["U"])
                bl.append(m[va]["L"])
            seeds = {}
            for s in sorted(D.S):
                ms = AG.all_metrics(D, {s: D.S[s]}, [va, vf, "raw50"], ones, fall)
                seeds[s] = ms[va]["U"] - ms[vf]["U"]
            pa = AG.power_audit(D, [va, vf])
            delta = m0[va]["U"] - m0[vf]["U"]
            out.update(U_learned=m0[va]["U"], L_learned=m0[va]["L"], U_floor=m0[vf]["U"], L_floor=m0[vf]["L"], delta=delta, delta_ci=ci(bd), L_learned_ci=ci(bl),
                       seed_delta=seeds, mde80=pa[0]["mde80"], mde_se=pa[0]["se"], gap_closed=delta / max(T["U"] - m0[vf]["U"], 1e-6))
            if a.cluster:
                bc = []
                for w in AG.resamples(D, a.cluster, cluster=True):
                    m = AG.all_metrics(D, D.S, [va, vf, "raw50"], w, fall)
                    bc.append(m[va]["U"] - m[vf]["U"])
                out["delta_cluster_ci"] = ci(bc)
            conds = dict(L_ok=bool(out["L_learned"] <= LT + 0.05), delta_ok=bool(delta >= M and ci(bd)[0] > 0),
                         seeds_ok=bool(sum(v >= M for v in seeds.values()) >= 2), power_ok=bool(out["mde80"] <= M))
            out["conds"] = conds
            out["resolved_null"] = bool(ci(bd)[1] < M)
        else:
            out["conds"] = None
            out["note"] = f"infeasible: learned {feas_l}, floor {feas_f}, eligible learned arms {len(learned)}"
        res[tname] = out
        P(f"\n### Target {tname} (U_T {T['U']}, L_T {LT}; feasible = pooled L <= {LT + 0.05:.2f})")
        k2 = out["K2p"]
        if k2.get("U") is not None:
            P(f"E2-K2' (label-only/PCA family at the target leakage): selected {k2['sel']} -> U {k2['U']:.3f}, L {k2['L']:.2f}; gap to target {k2['gap']:.3f} (fires if <= {M}): **{k2['fires']}**; retained fraction vs target {k2['retained_fraction']:.2f} (old 0.70 threshold, sensitivity)")
        else:
            P(f"E2-K2': {k2.get('note')}")
        if out["conds"] is None:
            P(out["note"])
            continue
        P(f"selected floor per fold {sel_f}: U {out['U_floor']:.3f}, L {out['L_floor']:.2f}")
        P(f"selected learned per fold {sel_l}: U {out['U_learned']:.3f}, L {out['L_learned']:.2f} {out['L_learned_ci']}")
        P(f"**Delta = U_learned - U_floor = {out['delta']:.3f} {tuple(round(x, 3) for x in out['delta_ci'])}** (cluster {tuple(round(x, 3) for x in out.get('delta_cluster_ci', (float('nan'),) * 2))}); "
          f"per-seed {({s: round(v, 3) for s, v in out['seed_delta'].items()})}; gap closed {out['gap_closed']:.2f}; MDE80 {out['mde80']:.3f} (SE {out['mde_se']:.3f}); conditions {out['conds']}; resolved null {out['resolved_null']}")

    # --- test-selected best config per family at each target (optimistic bound, attribution)
    P("\n### Best configuration per family at each target leakage (test-selected, eligible or not; optimistic, for attribution)")
    P("| target | family | config | eligible | U | L |\n|---|---|---|---|---|---|")
    for tname, T in TARGETS.items():
        for fam in ("vae", "cvae", "vaeadv", "cvaeadv", "vaeplc"):
            c = [x for x in learned_all if IN.parse(x[3:])["fam"] == fam and pt[x]["L"] <= T["L"] + 0.05]
            if c:
                b = max(c, key=lambda x: pt[x]["U"])
                P(f"| {tname} | {fam} | {b} | {elig[b]['eligible']} | {pt[b]['U']:.3f} | {pt[b]['L']:.2f} |")
            else:
                P(f"| {tname} | {fam} | none feasible | | | |")

    # --- verdict
    pr = res["scvi"]
    v = None
    if not k1:
        v = "E2-NO-GO (E2-K1' fails)"
    elif pr["K2p"].get("fires"):
        v = "E2-NO-GO (E2-K2' fires: the label-only/PCA family fills the target at its leakage)"
    elif pr["conds"] is None:
        v = f"E2-NO-GO (no feasible/eligible learned arm at the scVI leakage: {pr.get('note')})"
    else:
        c = pr["conds"]
        if not c["power_ok"]:
            v = f"E2-INCONCLUSIVE (UNDERPOWERED: MDE80 {pr['mde80']:.3f} > {M})"
        elif all(c.values()):
            v = "E2-GO (nominates Focus E for locking; the lock decision stays with the main agent)"
        elif pr["resolved_null"] or not c["L_ok"]:
            v = f"E2-NO-GO (resolved: Delta {pr['delta']:.3f}, upper bound {pr['delta_ci'][1]:.3f} < {M}, or leakage constraint failed: {c})"
        else:
            v = f"E2-INCONCLUSIVE (Delta {pr['delta']:.3f} {tuple(round(x, 3) for x in pr['delta_ci'])}, conditions {c})"
    P(f"\n**E2 VERDICT: {v}**  (verdict_original: SC-0 NO-GO, unchanged)")
    js["verdict"] = v
    js["res"] = res
    md = "\n".join(L)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
