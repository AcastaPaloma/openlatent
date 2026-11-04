"""F2 aggregator (results_notes/F.md section 6): merges the existing S5 outputs (f_s5_eval_*) with the F2 jobs (f2_eval_*), then
  oracle headroom (first), controls, adapter-trained check, extended JK2/JK3, adapter-vs-envelope comparison at matched leakage, verdict fields.

  python aggregate_f2.py --orig d0,d1,d2 --f2 e0,e1,e2 --salts s0,s1,s2 --scratch /tmp/f2merge --out research/fields/results/f_jump/f2 [--B 1000]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregate as AG

DELTA_U = 0.010
GAIN_L = 0.15
MATCH = 0.05
NEW_SMALL_D = [f"{l}_{d}" for l in ("ldaT2", "ldaP") for d in (2, 4, 8)]
STEPS = 8000


def merge(orig, f2, salt, scratch):
    zo, z2 = np.load(Path(orig) / f"preds_{salt}.npz"), np.load(Path(f2) / f"preds_{salt}.npz")
    mo = json.loads((Path(orig) / f"meta_{salt}.json").read_text())
    m2 = json.loads((Path(f2) / f"meta_{salt}.json").read_text())
    for k in ("idx|H", "idx|V", "half|H", "half|V", "plate|H"):
        assert np.array_equal(zo[k], z2[k]), f"{salt}: {k} differs between the original and the F2 run"
    P = {k: zo[k] for k in zo.files}
    P.update({k: z2[k] for k in z2.files})
    arms = list(mo["arms"]) + [a for a in m2["arms"] if a not in mo["arms"]]
    fam = dict(mo["families"])
    fam.update(m2["families"])
    meta = dict(mo)
    meta.update(arms=arms, families=fam, dims={**mo["dims"], **m2["dims"]}, f2_train=m2["f2_train"], f2_labels=m2.get("f2_labels"))
    d = Path(scratch) / f"merged_{salt}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"preds_{salt}.npz", **P)
    (d / f"meta_{salt}.json").write_text(json.dumps(meta))
    return d, meta, P


def plate_cluster(P, a, b, B=1000, seed=11):
    """query-weighted mean AP difference U(a)-U(b) on H, bootstrap over plates."""
    ax, ay = P[f"apq|H|{a}"].astype(np.float64), P[f"apq|H|{b}"].astype(np.float64)
    ok = ~np.isnan(ax) & ~np.isnan(ay)
    ids, inv = np.unique(P["plate|H"], return_inverse=True)
    ds = np.bincount(inv[ok], weights=(ax - ay)[ok], minlength=len(ids))
    n = np.bincount(inv[ok], minlength=len(ids)).astype(np.float64)
    rng = np.random.default_rng(seed)
    bs = np.array([(w @ ds) / (w @ n) for w in (np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(B))])
    return float(ds.sum() / n.sum()), AG.ci(bs)


def envelope(sp, exclude=()):
    return [a for a in sp.names(AG.FAM_IND) if sp.has_att(a) and sp.U("V", a) is not None and a not in exclude]


def compare(sp, P, a, env):
    """adapter arm a against the envelope arms `env` at matched leakage; V-selection of the envelope arm, H read."""
    La = sp.L("V", a)
    cand = [x for x in env if sp.L("V", x) <= La + MATCH]
    out = dict(arm=a, U_V=sp.U("V", a), L_V=La, U_H=sp.U("H", a), L_H=sp.L("H", a))
    if cand:
        F = max(cand, key=lambda x: sp.U("V", x))
        pt, pci = plate_cluster(P, a, F)
        out.update(F_star=F, dU_V=sp.U("V", a) - sp.U("V", F), U_H_F=sp.U("H", F), L_H_F=sp.L("H", F), dU_H=sp.U("H", a) - sp.U("H", F),
                   dU_H_ci=AG.ci(sp.dU_boot("H", a, F)), dU_H_plate=pt, dU_H_plate_ci=pci)
        out["u_win"] = bool(out["dU_H"] >= DELTA_U and out["dU_H_ci"][0] > 0 and pci[0] > 0)
    else:
        out.update(F_star=None, dU_V=float("inf"), u_win=False, note_U="no envelope arm reaches L_V(a)+0.05")
    cand2 = [x for x in env if sp.U("V", x) >= sp.U("V", a) - 0.01]
    if cand2:
        FL = min(cand2, key=lambda x: sp.L("V", x))
        g = sp.L("H", FL) - sp.L("H", a)
        gb = sp.dL_boot("H", FL, a)
        out.update(F_L=FL, U_H_FL=sp.U("H", FL), L_H_FL=sp.L("H", FL), gain_L=g, gain_L_ci=AG.ci(gb), l_win=bool(g >= GAIN_L and AG.ci(gb)[0] > 0))
    else:
        out.update(F_L=None, gain_L=float("nan"), l_win=False, note_L="no envelope arm has U_V >= U_V(a)-0.01 (utility above the whole envelope)")
    out["win"] = bool(out["u_win"] or out["l_win"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--f2", required=True)
    ap.add_argument("--salts", default="s0,s1,s2")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    res = dict(margin=dict(delta_U=DELTA_U, gain_L=GAIN_L, match=MATCH, mde80_printed=0.0010, headroom_gate=3 * DELTA_U),
               verdict_original="NO-GO at the S4 gate: JK1 v0 literal control-AP clause fires in 3 of 3 splits (F.md 5.1)", splits={})
    md = ["# F2 aggregate (pre-registration: results_notes/F.md section 6); units: compounds (U), plates (L); B=%d\n" % a.B]
    inv = []
    wins, trained_all, gh, ctrl_ok, jk2f, jk3f_ext, jk3f_v0, jk2f_v0, wins_v0 = [], [], [], [], [], [], [], [], []
    for i, (o, f, salt) in enumerate(zip(a.orig.split(","), a.f2.split(","), a.salts.split(","))):
        d, meta, P = merge(o, f, salt, a.scratch)
        sp = AG.Split(d, salt, a.B, 200 + i, True, "ldaT2")
        r = dict(f2_labels=meta.get("f2_labels"), train=meta["f2_train"])
        # --- training check
        steps_ok = all(v.get("steps") == STEPS for v in meta["f2_train"].values())
        r["steps_ok"] = bool(steps_ok)
        trained = {k: bool(v["trained"]) for k, v in meta["f2_train"].items()}
        # --- controls
        un, us = sp.U("H", "ctrl_noise16"), float(np.mean(P["apcnull|H|ctrl_noise16|xs"]))
        att = {c: sp.worst("H", c) for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_source") if sp.has_att(c)}
        f2arms = [x for x in sp.arms if sp.fam[x] in ("adapter", "labelonly") and f"apcnull|H|{x}|xs" in P]
        nulls = {x: float(np.mean(P[f"apcnull|H|{x}|xs"])) for x in f2arms}
        r["controls"] = dict(U_noise16=un, U_noise16_shuffled=us, same_embedding_ok=bool(abs(un - us) <= 0.01), attackers=att, attackers_ok=bool(all(v <= 0.23 for v in att.values())),
                             null_max=max(nulls.values()), null_ok=bool(max(nulls.values()) <= un + 0.01))
        c_ok = r["controls"]["same_embedding_ok"] and r["controls"]["attackers_ok"] and r["controls"]["null_ok"]
        ctrl_ok.append(c_ok)
        # --- oracle headroom (FIRST)
        env_ext = envelope(sp)
        env_v0 = envelope(sp, exclude=NEW_SMALL_D)
        orc = {}
        for x in [y for y in sp.arms if sp.fam[y] == "oracle"]:
            cand = [e for e in env_ext if sp.L("H", e) <= sp.L("H", x) + MATCH]
            Fb = max(cand, key=lambda e: sp.U("H", e)) if cand else None
            orc[x] = dict(U_H=sp.U("H", x), L_H=sp.L("H", x), envelope_arm=Fb, U_H_env=None if Fb is None else sp.U("H", Fb), headroom=None if Fb is None else sp.U("H", x) - sp.U("H", Fb))
        best_h = max([v["headroom"] for v in orc.values() if v["headroom"] is not None] or [float("nan")])
        r["oracle"] = orc
        r["oracle_headroom"] = best_h
        gh.append(bool(best_h >= 3 * DELTA_U))
        # --- JK2 / JK3 extended vs v0 grid
        R = sp.pick_ref("U")
        r["R"] = R
        k2e, k3e = sp.jk2(R), sp.jk3(R)
        old = {x: sp.fam[x] for x in NEW_SMALL_D}
        for x in NEW_SMALL_D:
            sp.fam[x] = "excluded"
        k2o, k3o = sp.jk2(R), sp.jk3(R)
        for x, v in old.items():
            sp.fam[x] = v
        r["jk2_extended"], r["jk2_v0_grid"] = k2e, k2o
        r["jk3_extended"], r["jk3_v0_grid"] = k3e, k3o
        jk2f.append(k2e["fires"])
        jk2f_v0.append(k2o["fires"])
        jk3f_ext.append(k3e["fires"])
        jk3f_v0.append(k3o["fires"])
        # --- adapter vs envelope
        adapters = [x for x in sp.arms if sp.fam[x] == "adapter"]
        tr = {x: trained[x.split("_")[1]] for x in adapters}
        r["adapter_trained"] = tr
        valid = [x for x in adapters if tr[x]]
        table = {x: compare(sp, P, x, env_ext) for x in adapters}
        table_v0 = {x: compare(sp, P, x, env_v0) for x in adapters}
        r["table"], r["table_v0_envelope"] = table, table_v0
        if valid:
            astar = max(valid, key=lambda x: table[x]["dU_V"])
            r["a_star"] = astar
            r["selected"] = table[astar]
            astar0 = max(valid, key=lambda x: table_v0[x]["dU_V"])
            r["a_star_v0_envelope"] = astar0
            r["selected_v0_envelope"] = table_v0[astar0]
            wins.append(bool(table[astar]["win"]))
            wins_v0.append(bool(table_v0[astar0]["win"]))
        else:
            wins.append(False)
            wins_v0.append(False)
        trained_all.append(bool(valid) and bool(steps_ok))
        res["splits"][salt] = r
        # --- markdown
        md.append(f"## split {salt}\n")
        md.append(f"**Oracle headroom (first)**: " + "; ".join(f"{k}: U_H {AG.fmt(v['U_H'])}, L_H {AG.fmt(v['L_H'], 2)}, envelope {v['envelope_arm']} U_H {AG.fmt(v['U_H_env'])} -> headroom {AG.fmt(v['headroom'])}" for k, v in orc.items()) + f"; gate (>= {3 * DELTA_U:.2f}): {gh[-1]}\n")
        md.append(f"**Controls**: U(noise16) {AG.fmt(un)} vs shuffled-on-noise16 {AG.fmt(us)} (|diff| <= 0.01: {r['controls']['same_embedding_ok']}); attacker controls {json.dumps({k: round(v, 3) for k, v in att.items()})} ok={r['controls']['attackers_ok']}; max same-embedding null over F2 arms {AG.fmt(r['controls']['null_max'])} <= U(noise16)+0.01: {r['controls']['null_ok']}\n")
        md.append("**Adapter training**: " + "; ".join(f"{k}: loss {v['loss_first20']:.3f} -> {v['loss_last200']:.3f}, eff.rank {v['eff_rank']:.1f}, in-sample source probe {v['insample_source_probe_bacc']:.3f}, trained={v['trained']}, steps {v['steps']}" for k, v in meta["f2_train"].items()) + "\n")
        rows = ["| arm | family | d | U_V | U_H xs [95% CI] | U_null_H | L_V | L_H [CI] | worst att H | lin / mlp / knn / lgbm |", "|---|---|---|---|---|---|---|---|---|---|"]
        for x in [y for y in sp.arms if sp.fam[y] in ("adapter", "labelonly", "oracle", "ref", "ctrl") and (y in f2arms or sp.fam[y] in ("oracle",) or y in ("ctrl_noise16", "ldaP_16", "ldaT2_16", "T_harmony", "raw") or y.startswith("ldaP_") or y.startswith("ldaT2_"))]:
            if not sp.has_att(x):
                continue
            uH = sp.U("H", x)
            accs = " / ".join(AG.fmt(sp.leak["H"].acc(x, k), 2) for k in sp.kinds)
            rows.append(f"| {x} | {sp.fam[x]} | {meta['dims'][x]} | {AG.fmt(sp.U('V', x))} | {AG.fci(uH, AG.ci(sp.U_boot('H', x))) if uH is not None else 'n/a'} | {AG.fmt(nulls.get(x))} | {AG.fmt(sp.L('V', x), 2)} | {AG.fci(sp.L('H', x), AG.ci(sp.L_boot('H', x)), 2)} | {AG.fmt(sp.worst('H', x), 2)} | {accs} |")
        md.append("\n".join(rows) + "\n")
        md.append(f"**JK2** extended grid: fires={k2e['fires']}, best={k2e.get('best')} (U_H {AG.fmt(k2e.get('U_H'))}, L_H {AG.fmt(k2e.get('L_H'), 2)}, candidates with L_V <= 0.25: {k2e['constrained_candidates']}); v0 grid: fires={k2o['fires']}; literal reading: {k2e['literal']}\n")
        md.append(f"**JK3** extended: F* {k3e.get('F_star')} (U_H {AG.fmt(k3e.get('U_H_F'))}, L_H {AG.fmt(k3e.get('L_H_F'), 2)}) vs R={R} (U_H {AG.fmt(k3e['U_H_R'])}, L_H {AG.fmt(k3e['L_H_R'], 2)}) fires={k3e['fires']}; v0 grid F* {k3o.get('F_star')} fires={k3o['fires']}\n")
        if valid:
            s = r["selected"]
            md.append(f"**Adapter comparison (V-selected a\\* = {astar})**: U_H {AG.fmt(s['U_H'])}, L_H {AG.fmt(s['L_H'], 2)}; envelope F\\* = {s.get('F_star')} (U_H {AG.fmt(s.get('U_H_F'))}, L_H {AG.fmt(s.get('L_H_F'), 2)}); "
                      f"dU_H = {AG.fci(s.get('dU_H'), s.get('dU_H_ci', [float('nan')] * 2), 4)}, plate-cluster {AG.fci(s.get('dU_H_plate'), s.get('dU_H_plate_ci', [float('nan')] * 2), 4)} -> u_win={s['u_win']}; "
                      f"L-axis F_L {s.get('F_L')} gain {AG.fmt(s.get('gain_L'), 3)} {s.get('gain_L_ci')} -> l_win={s['l_win']}; WIN={s['win']}  (v0 envelope: a\\*={astar0} dU_H {AG.fmt(r['selected_v0_envelope'].get('dU_H'), 4)}, win={r['selected_v0_envelope']['win']})\n")
            rows = ["| adapter arm | trained | U_H | L_H | envelope F*(V) | dU_H [CI] | plate-cluster CI | gain_L | win |", "|---|---|---|---|---|---|---|---|---|"]
            for x in adapters:
                t = table[x]
                rows.append(f"| {x} | {tr[x]} | {AG.fmt(t['U_H'])} | {AG.fmt(t['L_H'], 2)} | {t.get('F_star')} | {AG.fci(t.get('dU_H'), t.get('dU_H_ci', [float('nan')] * 2), 4)} | {AG.fmt(t.get('dU_H_plate_ci', [float('nan')] * 2)[0], 4)}..{AG.fmt(t.get('dU_H_plate_ci', [float('nan')] * 2)[1], 4)} | {AG.fmt(t.get('gain_L'), 2)} | {t['win']} |")
            md.append("Sensitivity, every adapter arm (no adapter selection; envelope arm V-selected per arm):\n\n" + "\n".join(rows) + "\n")
        else:
            md.append("**No trained adapter in this split: INVALID.**\n")
    ns = len(wins)
    go = bool(sum(gh) >= 2 and all(trained_all) and all(ctrl_ok) and sum(wins) >= 2)
    invalid = bool((not all(trained_all)) or (not all(ctrl_ok)))
    res["verdict_tightened"] = "INVALID" if invalid else ("GO (for a label-free or semi-supervised stage 2; a pre-registered separate attempt, not a claim)" if go else "NO-GO")
    go_no_gh = bool(all(trained_all) and all(ctrl_ok) and sum(wins) >= 2)
    reasons = []
    if sum(gh) < 2:
        reasons.append(f"G-H: linear in-sample oracle headroom < {3 * DELTA_U:.2f} in {ns - sum(gh)} of {ns} splits")
    if not all(trained_all):
        reasons.append("adapter not trained / steps != 8000 in some split")
    if not all(ctrl_ok):
        reasons.append("a control failed")
    if sum(wins) < 2:
        reasons.append("adapter wins in < 2 of 3 splits")
    res["verdict_tightened_reasons"] = reasons
    res["verdict_if_G_H_dropped (post-hoc reading, NOT the pre-registered verdict)"] = "GO-equivalent: adapter trained, controls pass, wins in %d of %d splits" % (sum(wins), ns) if go_no_gh else "NO-GO"
    res["summary"] = dict(oracle_gate_splits=int(sum(gh)), trained_all_splits=bool(all(trained_all)), controls_ok_all=bool(all(ctrl_ok)), wins_splits=int(sum(wins)),
                          wins_splits_v0_envelope=int(sum(wins_v0)), jk2_extended_fires_splits=int(sum(jk2f)), jk2_v0_grid_fires_splits=int(sum(jk2f_v0)),
                          jk3_extended_fires_splits=int(sum(jk3f_ext)), jk3_v0_grid_fires_splits=int(sum(jk3f_v0)))
    res["jk2_extended"] = bool(sum(jk2f) >= 2)
    res["jk3_extended"] = bool(sum(jk3f_ext) >= 2)
    res["oracle_headroom"] = {s: r["oracle_headroom"] for s, r in res["splits"].items()}
    md.append("## Verdict fields\n\n```\n" + json.dumps(dict(verdict_original=res["verdict_original"], verdict_tightened=res["verdict_tightened"], verdict_tightened_reasons=res["verdict_tightened_reasons"], verdict_if_G_H_dropped=res["verdict_if_G_H_dropped (post-hoc reading, NOT the pre-registered verdict)"], margin=res["margin"], summary=res["summary"],
                                                             jk2_extended=res["jk2_extended"], jk3_extended=res["jk3_extended"], oracle_headroom=res["oracle_headroom"]), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print("\n".join(md[-1:]))


if __name__ == "__main__":
    main()
