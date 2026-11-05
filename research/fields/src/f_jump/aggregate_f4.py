"""F4 aggregator (results_notes/F.md section 10): does the label-budget break-even generalise?  Leave-source-out folds, a second feature view, a 5-point budget grid.

  python aggregate_f4.py --runs lso2=dir,lso3=dir,lso6=dir,lso8=dir,lso10=dir,pcA=dir,pcB=dir --salt s0 --scratch $SCR --out $SCR/f4 [--orig d --f2 d] [--B 1000]
Writes <out>.json and <out>.md (use a scratch path on the VM, then copy into the repo).  Deterministic (fixed seeds).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import aggregate as AG
import aggregate_f2 as F2

DELTA_U = 0.010
POWER_MAX = DELTA_U / 2
POWER_KEY = "b30"        # budget whose comparison must have MDE80 <= POWER_MAX (the 5-source F4 rule); the seen-source reference of H3-001 uses b10
ATT_MAX = 0.23           # repaired-control attacker bound: chance + 0.03 for the five-source table (H3-001 sets 0.13 for ten sources)
LF = re.compile(r"^(raw|pca\d+|rp\d+|dpca\d+_r[\d.]+|noise_r[\d.]+)$")
BUD = [("b1", 0.01), ("b3", 0.03), ("b10", 0.10), ("b30", 0.30), ("b100", 1.0)]
ALLR = ["a"] + [b for b, _ in BUD]


def crossing(y, delta=DELTA_U):
    """first crossing of y(ln f) = delta by log-linear interpolation; returns (value in %, flag) with flag 'ok' | '<=1' | '>100'."""
    x = np.log([f for _, f in BUD])
    if y[0] >= delta:
        return 1.0, "<=1"
    for k in range(1, len(y)):
        if y[k] >= delta:
            t = (delta - y[k - 1]) / (y[k] - y[k - 1])
            return float(100 * np.exp(x[k - 1] + t * (x[k] - x[k - 1]))), "ok"
    return float("inf"), ">100"


def analyse(dirp, salt, B, seed, extra_floors=None, extra_label=""):
    d = Path(dirp)
    P = {k: v for k, v in np.load(d / f"preds_{salt}.npz").items()}
    meta = json.loads((d / f"meta_{salt}.json").read_text())
    sp = AG.Split(d, salt, B, seed, True, "ldaT2")
    regime = meta["regime"]
    floors = [x for x in sp.names(AG.FAM_IND) if sp.has_att(x) and sp.U("V", x) is not None]
    matched = [x for x in sp.arms if sp.fam[x] == "matched" and sp.has_att(x)]
    E_lf = [x for x in floors if LF.match(x)]
    E_full = floors + matched
    out = dict(hold_source=meta.get("f4_hold_source"), view=meta.get("f4_view"), dim=meta.get("f4_dim"), n_fit=meta.get("f4_n_fit"), labels=meta["f4_labels"],
               n_compounds=meta["n_compounds"], train=meta["f4_train"])
    # controls
    un, us = sp.U("H", "ctrl_noise16"), float(np.mean(P["apcnull|H|ctrl_noise16|xs"]))
    att = sp.worst("H", "ctrl_noise16")
    adapters = [x for x in sp.arms if sp.fam[x] == "adapter"]
    nulls = {x: float(np.mean(P[f"apcnull|H|{x}|xs"])) for x in adapters + matched if f"apcnull|H|{x}|xs" in P}
    ctrl_ok = bool(abs(un - us) <= 0.01 and att <= ATT_MAX and max(nulls.values()) <= un + 0.01)
    out["controls"] = dict(U_noise16=un, U_noise16_shuffled=us, attacker_noise16=att, null_max=max(nulls.values()), ok=ctrl_ok)
    # trained (memorisation-calibrated)
    trained, tinfo = {}, {}
    for x in adapters:
        R = regime[x]
        info = meta["f4_train"][f"{R}_d8"]
        c1 = info["loss_last200"] <= 0.8 * info["loss_first20"]
        c2 = info["eff_rank"] >= 8 / 4
        up = sp.U("V", f"perm_{R}_d64")
        c3 = bool(sp.U("V", x) - up >= DELTA_U)
        trained[x] = bool(c1 and c2 and c3)
        tinfo[x] = dict(loss_ratio=info["loss_last200"] / info["loss_first20"], eff_rank=info["eff_rank"], U_V=sp.U("V", x), U_V_perm=up, U_H_perm=sp.U("H", f"perm_{R}_d64"), trained=trained[x],
                        insample_probe=info["insample_source_probe_bacc"])
    out["trained"] = tinfo
    out["memorisation_floor"] = {R: dict(U_V=sp.U("V", f"perm_{R}_d64"), U_H=sp.U("H", f"perm_{R}_d64"), loss_first=meta["f4_train"][f"perm_{R}_d64"]["loss_first20"],
                                         loss_last=meta["f4_train"][f"perm_{R}_d64"]["loss_last200"]) for R in ALLR}
    # headroom gate
    b100 = [x for x in adapters if regime[x] == "b100" and trained[x]]
    reg = {}
    if b100:
        ceil = max(b100, key=lambda x: sp.U("V", x))
        Lc = sp.L("V", ceil)
        cand = [x for x in E_full if sp.L("V", x) <= Lc + F2.MATCH and x != ceil]
        fb = max(cand, key=lambda x: sp.U("V", x))
        out["headroom"] = dict(ceiling_arm=ceil, ceiling_U_H=sp.U("H", ceil), best_floor=fb, floor_U_H=sp.U("H", fb), headroom=sp.U("H", ceil) - sp.U("H", fb))
        gate = bool(out["headroom"]["headroom"] >= 0.03)
    else:
        out["headroom"] = None
        gate = False
    out["gate_H4"] = gate
    # per-budget comparisons
    curve_full, curve_match, boot, ys = {}, {}, {}, []
    for R in ALLR:
        arms_R = [x for x in adapters if regime[x] == R and trained[x]]
        E_match = E_lf + [x for x in matched if regime.get(x) == R]
        entry = dict(trained_arms=arms_R, all_arms=[x for x in adapters if regime[x] == R])
        if arms_R:
            tf = {x: F2.compare(sp, P, x, E_full) for x in arms_R}
            tm = {x: F2.compare(sp, P, x, E_match) for x in arms_R}
            af = max(arms_R, key=lambda x: tf[x]["dU_V"])
            am = max(arms_R, key=lambda x: tm[x]["dU_V"])
            entry.update(a_full=af, sel_full=tf[af], a_match=am, sel_match=tm[am])
            lg = [x for x in matched if regime.get(x) == R]
            if lg:
                g = lg[0]
                entry["vs_same_label_lda"] = dict(arm=g, U_H=sp.U("H", g), L_H=sp.L("H", g), adapter_U_H=sp.U("H", af), adapter_L_H=sp.L("H", af), gap=sp.U("H", af) - sp.U("H", g))
            if tf[af].get("F_star"):
                entry["dboot"] = sp.dU_boot("H", af, tf[af]["F_star"])
        reg[R] = entry
    # break-even
    wins = [bool(reg[R].get("sel_full", {}).get("win")) for R, _ in BUD]
    first = next((R for (R, _), w in zip(BUD, wins) if w), None)
    y_pt = np.array([reg[R]["sel_full"]["dU_H"] if "sel_full" in reg[R] and reg[R]["sel_full"].get("F_star") else -1.0 for R, _ in BUD])
    be_pt = crossing(y_pt)
    Y = np.stack([reg[R].get("dboot", np.full(B, -1.0)) for R, _ in BUD], 1)
    bes = [crossing(Y[b]) for b in range(B)]
    vals = np.array([v for v, _ in bes])
    fin = vals[np.isfinite(vals)]
    out["break_even"] = dict(first_winning_budget=first, wins=dict(zip([R for R, _ in BUD], wins)), dU_H_curve=dict(zip([R for R, _ in BUD], map(float, y_pt))), interpolated_percent=be_pt[0], flag=be_pt[1],
                             boot_percentiles_percent=[float(np.percentile(vals, q)) for q in (2.5, 50, 97.5)], boot_share_above_100=float((~np.isfinite(vals)).mean()),
                             boot_share_at_or_below_1=float((vals <= 1.0).mean()))
    # power for the rule's budget (30%) and the ceiling
    pw = {}
    for R in ("b10", "b30", "b100"):
        if R in reg and reg[R].get("sel_full", {}).get("F_star"):
            dv = sp.apc("H", reg[R]["a_full"]) - sp.apc("H", reg[R]["sel_full"]["F_star"])
            pw[R] = AG.mde80(dv)
    out["power"] = pw
    powered = bool(pw.get(POWER_KEY, {}).get("mde80", 1.0) <= POWER_MAX)
    out["powered"] = powered
    valid = bool(ctrl_ok and b100)
    out["valid"] = valid
    be_le30 = bool(first in ("b1", "b3", "b10", "b30"))
    out["break_even_le_30"] = be_le30
    out["break_even_le_10"] = bool(first in ("b1", "b3", "b10"))
    out["supports"] = bool(valid and powered and gate and be_le30)
    for R in reg:
        reg[R].pop("dboot", None)
    out["regimes"] = reg
    return sp, P, meta, out


def table(out):
    rows = ["| budget | a* (V-sel., E_full) | U_H [L_H] | dU_H vs E_full [95% CI] (plate-cluster CI) | F* | win_full | a* (E_match) | dU_H vs E_match [CI] | win_match | gap to same-label LDA-8 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for R in ALLR:
        e = out["regimes"][R]
        if not e["trained_arms"]:
            rows.append(f"| {R} | none trained ({e['all_arms']}) | | | | | | | | |")
            continue
        sf, sm = e["sel_full"], e["sel_match"]
        g = e.get("vs_same_label_lda")
        rows.append(f"| {R} | {e['a_full']} | {AG.fmt(sf['U_H'])} [{AG.fmt(sf['L_H'], 2)}] | {AG.fci(sf.get('dU_H'), sf.get('dU_H_ci', [float('nan')] * 2), 4)} ({AG.fmt(sf.get('dU_H_plate_ci', [float('nan')] * 2)[0], 4)}, {AG.fmt(sf.get('dU_H_plate_ci', [float('nan')] * 2)[1], 4)}) | {sf.get('F_star')} | {sf['win']} | "
                    f"{e['a_match']} | {AG.fci(sm.get('dU_H'), sm.get('dU_H_ci', [float('nan')] * 2), 4)} | {sm['win']} | {'-' if g is None else AG.fmt(g['gap'], 3)} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="name=dir,... names: lso2 lso3 lso6 lso8 lso10 pcA pcB")
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--orig", default="", help="S5 floors dir (f_s5_eval_s0) for the complete-envelope sensitivity of pcA")
    ap.add_argument("--f2", default="", help="F2 dir (f2_eval_s0)")
    ap.add_argument("--scratch", default="/tmp/f4scratch")
    a = ap.parse_args()
    runs = dict(x.split("=") for x in a.runs.split(","))
    res = dict(margin=dict(delta_U=DELTA_U, power_max=POWER_MAX, match=F2.MATCH),
               verdict_original="NO-GO at the S4 gate: JK1 v0 literal control-AP clause fires in 3 of 3 splits (F.md 5.1)",
               verdict_F2_literal="NO-GO: gate G-H (linear in-sample oracle headroom < 0.03; flagged design flaw) (F.md 7.1)",
               verdict_F3_tightened="GO as pre-registered, rated fragile (label-free-ish arm c gained +0.004-0.006 over same-label LDA); supported finding: break-even 10% vs the full envelope (F.md 9)", runs={})
    md = [f"# F4 aggregate (pre-registration: results_notes/F.md section 10); units: H compounds (U), plates (L), held-out sources; B={a.B}\n"]
    sps = {}
    for i, (nm, dp) in enumerate(runs.items()):
        sp, P, meta, out = analyse(dp, a.salt, a.B, 500 + i)
        res["runs"][nm] = out
        sps[nm] = (sp, P)
        h = out["headroom"]
        be = out["break_even"]
        md.append(f"## run {nm}: hold_source={out['hold_source'] or '-'}, view={out['view']} (d={out['dim']}), n_fit_wells={out['n_fit']}, H/V compounds {out['n_compounds']}\n")
        md.append(f"**Headroom (G-H4)**: " + (f"ceiling {h['ceiling_arm']} U_H {AG.fmt(h['ceiling_U_H'])} vs best floor {h['best_floor']} U_H {AG.fmt(h['floor_U_H'])} -> {AG.fmt(h['headroom'])}; gate {out['gate_H4']}" if h else "no trained b100 adapter") + f"; controls ok={out['controls']['ok']} (U(noise16) {AG.fmt(out['controls']['U_noise16'])} vs shuffled {AG.fmt(out['controls']['U_noise16_shuffled'])}, attacker {AG.fmt(out['controls']['attacker_noise16'], 3)}, max null {AG.fmt(out['controls']['null_max'])}); powered={out['powered']} (MDE80 b30 {AG.fmt(out['power'].get('b30', {}).get('mde80'), 4)}, b100 {AG.fmt(out['power'].get('b100', {}).get('mde80'), 4)}); valid={out['valid']}\n")
        md.append(table(out) + "\n")
        md.append(f"**Break-even vs the full envelope**: first winning budget {be['first_winning_budget']}; interpolated {AG.fmt(be['interpolated_percent'], 1)}% ({be['flag']}), bootstrap percentiles 2.5/50/97.5 = {[round(v, 1) for v in be['boot_percentiles_percent']]}% (share > 100%: {be['boot_share_above_100']:.3f}, share <= 1%: {be['boot_share_at_or_below_1']:.3f}); dU_H(f) = {json.dumps({k: round(v, 4) for k, v in be['dU_H_curve'].items()})}; supports(<=30%)={out['supports']}\n")
        tr = ["| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |", "|---|---|---|---|---|---|---|"]
        for x, t in out["trained"].items():
            tr.append(f"| {x} | {t['loss_ratio']:.2f} | {t['eff_rank']:.1f} | {AG.fmt(t['U_V'])} | {AG.fmt(t['U_V_perm'])} | {AG.fmt(t['U_H_perm'])} | {t['trained']} |")
        md.append("\n".join(tr) + "\n")
    # complete-envelope sensitivity for pcA
    if a.orig and a.f2 and "pcA" in runs:
        zo = np.load(Path(a.orig) / f"preds_{a.salt}.npz")
        z2 = np.load(Path(a.f2) / f"preds_{a.salt}.npz")
        z4 = np.load(Path(runs["pcA"]) / f"preds_{a.salt}.npz")
        for k in ("idx|H", "idx|V", "half|H"):
            assert np.array_equal(zo[k], z4[k]) and np.array_equal(z2[k], z4[k])
        mo, m2, m4 = (json.loads((Path(p) / f"meta_{a.salt}.json").read_text()) for p in (a.orig, a.f2, runs["pcA"]))
        Pm = {}
        for z in (zo, z2, z4):
            Pm.update({k: z[k] for k in z.files})
        arms = list(mo["arms"]) + [x for x in m2["arms"] if x not in mo["arms"]]
        arms += [x for x in m4["arms"] if x not in arms]
        fam = {**mo["families"], **m2["families"], **m4["families"]}
        dims = {**mo["dims"], **m2["dims"], **m4["dims"]}
        meta = dict(m4)
        meta.update(arms=arms, families=fam, dims=dims)
        dd = Path(a.scratch) / "merged_pcA"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez(dd / f"preds_{a.salt}.npz", **Pm)
        (dd / f"meta_{a.salt}.json").write_text(json.dumps(meta))
        sp = AG.Split(dd, a.salt, a.B, 600, True, "ldaT2")
        floors = [x for x in sp.names(AG.FAM_IND) if sp.has_att(x) and sp.U("V", x) is not None]
        matched = [x for x in sp.arms if sp.fam[x] == "matched" and sp.has_att(x)]
        env = floors + matched
        adapters = [x for x in m4["regime"] if x.startswith("mlp_")]
        sens = {}
        for R, _ in BUD:
            arms_R = [x for x in adapters if m4["regime"][x] == R and res["runs"]["pcA"]["trained"].get(x)]
            if arms_R:
                tf = {x: F2.compare(sp, Pm, x, env) for x in arms_R}
                af = max(arms_R, key=lambda x: tf[x]["dU_V"])
                sens[R] = dict(arm=af, F_star=tf[af].get("F_star"), dU_H=tf[af].get("dU_H"), ci=tf[af].get("dU_H_ci"), plate=tf[af].get("dU_H_plate_ci"), win=tf[af]["win"])
        res["runs"]["pcA"]["sensitivity_complete_S5_envelope"] = sens
        first = next((R for R, _ in BUD if sens.get(R, {}).get("win")), None)
        res["runs"]["pcA"]["sensitivity_complete_S5_envelope_first_win"] = first
        md.append("## pcA against the COMPLETE S5 floor envelope (all S5 floors + F2 small-d LDA + F4 floors)\n\n" + "\n".join(f"- {R}: {v['arm']} vs {v['F_star']}: dU_H {AG.fci(v['dU_H'], v['ci'], 4)}, plate {v['plate']}, win {v['win']}" for R, v in sens.items()) + f"\n\nfirst winning budget: {first}\n")
    # cross-run verdict
    lso = [n for n in runs if n.startswith("lso")]
    R = res["runs"]
    pow_valid = [n for n in lso if R[n]["valid"] and R[n]["powered"]]
    sup = [n for n in lso if R[n]["supports"]]
    sup10 = [n for n in lso if R[n]["valid"] and R[n]["powered"] and R[n]["gate_H4"] and R[n]["break_even_le_10"]]
    P1 = None if len(pow_valid) < 4 else bool(len(sup) >= 4)
    invalid_lso = [n for n in lso if not R[n]["valid"]]
    P2 = None
    if "pcB" in R:
        P2 = bool(R["pcB"]["supports"]) if R["pcB"]["valid"] else None
    repro = bool(R["pcA"]["valid"] and R["pcA"]["break_even_le_10"]) if "pcA" in R else None
    if len(invalid_lso) > 1 or P1 is None or P2 is None:
        v = "INVALID"
    elif P1 and P2:
        v = "GENERALISES"
    elif P1 or P2:
        v = "PARTIAL (leave-source-out holds, second view does not)" if P1 else "PARTIAL (second view holds, leave-source-out does not)"
    else:
        v = "DOES-NOT-GENERALISE"
    res["summary"] = dict(break_even_by_run={n: dict(first=R[n]["break_even"]["first_winning_budget"], interpolated_percent=R[n]["break_even"]["interpolated_percent"], boot_pct=R[n]["break_even"]["boot_percentiles_percent"],
                                                      valid=R[n]["valid"], powered=R[n]["powered"], gate=R[n]["gate_H4"]) for n in R},
                          P1_lso_supports=sup, P1_lso_supports_le10=sup10, powered_valid_folds=pow_valid, P1=P1, P2_viewB=P2, reproduces_F3=repro)
    res["verdict_tightened"] = v
    md.append("## Verdict fields\n\n```\n" + json.dumps(dict(verdict_original=res["verdict_original"], verdict_F2_literal=res["verdict_F2_literal"], verdict_F3_tightened=res["verdict_F3_tightened"], verdict_tightened=v, summary=res["summary"], margin=res["margin"]), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
