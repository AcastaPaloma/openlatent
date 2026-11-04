"""F3 aggregator (results_notes/F.md section 8): label-budget curve.  Merges the S5 outputs (f_s5_eval_*), the F2 jobs (b100 adapter arms) and the F3 jobs.

  python aggregate_f3.py --orig d0,d1,d2 --f2 e0,e1,e2 --f3 g0,g1,g2 --salts s0,s1,s2 --scratch /tmp/f3merge --out research/fields/results/f_jump/f3 [--B 1000]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import aggregate as AG
import aggregate_f2 as F2

DELTA_U = 0.010
LF = re.compile(r"^(raw|pca\d+|rp\d+|dpca\d+_r[\d.]+|noise_r[\d.]+)$")
REGIMES = ["a", "b1", "b10", "b100", "c"]
B100_ARMS = {"mlp_d8": 8, "mlp_d64": 64, "mlp_d8_r0.5": 8, "mlp_d64_r0.5": 64}


def merge3(orig, f2, f3, salt, scratch):
    zs = [np.load(Path(p) / f"preds_{salt}.npz") for p in (orig, f2, f3)]
    ms = [json.loads((Path(p) / f"meta_{salt}.json").read_text()) for p in (orig, f2, f3)]
    for z in zs[1:]:
        for k in ("idx|H", "idx|V", "half|H", "half|V", "plate|H"):
            assert np.array_equal(zs[0][k], z[k]), f"{salt}: {k} differs"
    P = {}
    for z in zs:
        P.update({k: z[k] for k in z.files})
    arms = list(ms[0]["arms"])
    fam = dict(ms[0]["families"])
    dims = dict(ms[0]["dims"])
    for m in ms[1:]:
        for a in m["arms"]:
            if a not in arms:
                arms.append(a)
        fam.update(m["families"])
        dims.update(m["dims"])
    meta = dict(ms[0])
    meta.update(arms=arms, families=fam, dims=dims, f2_train=ms[1]["f2_train"], f3_train=ms[2]["f3_train"], regime=ms[2]["regime"], f3_labels=ms[2]["f3_labels"])
    d = Path(scratch) / f"merged3_{salt}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"preds_{salt}.npz", **P)
    (d / f"meta_{salt}.json").write_text(json.dumps(meta))
    return d, meta, P


def dim_of(name):
    return int(re.search(r"_d(\d+)", name).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--f2", required=True)
    ap.add_argument("--f3", required=True)
    ap.add_argument("--salts", default="s0,s1,s2")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    res = dict(margin=dict(delta_U=DELTA_U, gain_L=F2.GAIN_L, match=F2.MATCH, mde80_printed=0.0010),
               verdict_original="NO-GO at the S4 gate: JK1 v0 literal control-AP clause fires in 3 of 3 splits (F.md 5.1)",
               verdict_F2_literal="NO-GO: gate G-H (linear in-sample oracle headroom 0.0014-0.0018 < 0.03; a design flaw, not dropped) (F.md 7.1)",
               label_free_ish_arm="c: replicate pairs from Metadata_Well position on Target2 plates, no ids, no production labels (F.md 8.1)", splits={})
    md = [f"# F3 aggregate (pre-registration: results_notes/F.md section 8); units: compounds (U), plates (L); B={a.B}\n"]
    per = {R: dict(match=[], full=[], trained=[]) for R in REGIMES + ["ae"]}
    ctrl_ok, gh_ok, ae_wins = [], [], []
    summ = {R: [] for R in REGIMES + ["ae"]}
    for i, (o, f2d, f3d, salt) in enumerate(zip(a.orig.split(","), a.f2.split(","), a.f3.split(","), a.salts.split(","))):
        d, meta, P = merge3(o, f2d, f3d, salt, a.scratch)
        sp = AG.Split(d, salt, a.B, 400 + i, True, "ldaT2")
        r = dict(labels=meta["f3_labels"], train=meta["f3_train"])
        regime = dict(meta["regime"])
        for nm in B100_ARMS:
            regime[nm] = "b100"
        # ---- controls
        un, us = sp.U("H", "ctrl_noise16"), float(np.mean(P["apcnull|H|ctrl_noise16|xs"]))
        att = {c: sp.worst("H", c) for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_source") if sp.has_att(c)}
        f3arms = [x for x in meta["arms"] if x in regime and f"apcnull|H|{x}|xs" in P]
        nulls = {x: float(np.mean(P[f"apcnull|H|{x}|xs"])) for x in f3arms}
        c_ok = bool(abs(un - us) <= 0.01 and all(v <= 0.23 for v in att.values()) and max(nulls.values()) <= un + 0.01)
        r["controls"] = dict(U_noise16=un, U_noise16_shuffled=us, attackers=att, null_max=max(nulls.values()), ok=c_ok)
        ctrl_ok.append(c_ok)
        # ---- comparators
        allfloors = [x for x in sp.names(AG.FAM_IND) if sp.has_att(x) and sp.U("V", x) is not None]
        matched = [x for x in sp.arms if sp.fam[x] == "matched" and sp.has_att(x)]
        E_lf = [x for x in allfloors if LF.match(x)]
        E_full = allfloors + matched
        # ---- headroom (G-H3)
        b100 = [x for x in B100_ARMS if x in sp.arms and not x.endswith("_r0.5")]
        ceil = max(b100, key=lambda x: sp.U("V", x))
        Lc = sp.L("V", ceil)

        def best(pool):
            c = [x for x in pool if sp.L("V", x) <= Lc + F2.MATCH]
            b = max(c, key=lambda x: sp.U("V", x))
            return b, sp.U("H", b)
        flf, ffull = best(E_lf), best(E_full)
        r["headroom"] = dict(ceiling_arm=ceil, ceiling_U_H=sp.U("H", ceil), best_label_free_floor=flf[0], lf_U_H=flf[1], best_envelope_floor=ffull[0], env_U_H=ffull[1],
                             vs_label_free=sp.U("H", ceil) - flf[1], vs_envelope=sp.U("H", ceil) - ffull[1])
        gh_ok.append(bool(r["headroom"]["vs_label_free"] >= 0.03 and r["headroom"]["vs_envelope"] >= 0.03))
        # ---- trained flags (memorisation-floor calibrated)
        trained, tinfo = {}, {}
        for x in [y for y in sp.arms if y in regime and sp.fam[y] in ("adapter", "adapter_lf")]:
            R = regime[x]
            dd = dim_of(x)
            if R == "b100":
                info = meta["f2_train"][f"d{dd}"]
            else:
                info = meta["f3_train"][f"{R}_d{dd}" if R != "ae" else f"ae_d{dd}"]
            c1 = info["loss_last200"] <= 0.8 * info["loss_first20"]
            c2 = info["eff_rank"] >= dd / 4
            if R == "ae":
                c3, uperm = True, None
            else:
                pn = f"perm_{R}_d64"
                uperm = sp.U("V", pn)
                c3 = bool(sp.U("V", x) - uperm >= DELTA_U)
            trained[x] = bool(c1 and c2 and c3)
            tinfo[x] = dict(loss_ratio=info["loss_last200"] / info["loss_first20"], eff_rank=info["eff_rank"], U_V=sp.U("V", x), U_V_perm=uperm, calib=c3, trained=trained[x],
                            insample_probe=info.get("insample_source_probe_bacc"))
        r["trained"] = tinfo
        perm = {R: dict(U_V=sp.U("V", f"perm_{R}_d64"), U_H=sp.U("H", f"perm_{R}_d64"), loss=(meta["f3_train"].get(f"perm_{R}_d64") or {}).get("loss_last200"),
                        loss_first=(meta["f3_train"].get(f"perm_{R}_d64") or {}).get("loss_first20")) for R in REGIMES if f"perm_{R}_d64" in sp.arms}
        r["memorisation_floor"] = perm
        # ---- per-regime comparisons
        r["regimes"] = {}
        for R in REGIMES + ["ae"]:
            arms_R = [x for x in sp.arms if regime.get(x) == R and sp.fam[x] in ("adapter", "adapter_lf") and trained.get(x)]
            allR = [x for x in sp.arms if regime.get(x) == R and sp.fam[x] in ("adapter", "adapter_lf")]
            E_match = E_lf if R == "ae" else E_lf + [x for x in matched if regime.get(x) == R]
            out = dict(trained_arms=arms_R, all_arms=allR)
            if arms_R and R in ("a", "c"):
                E_plus = E_match + [x for x in allfloors if x.startswith("ldaT2_")]
                tp = {x: F2.compare(sp, P, x, E_plus) for x in arms_R}
                ap_ = max(arms_R, key=lambda x: tp[x]["dU_V"])
                out.update(a_star_match_plus=ap_, sel_match_plus=tp[ap_])
            if arms_R:
                tm = {x: F2.compare(sp, P, x, E_match) for x in arms_R}
                tf = {x: F2.compare(sp, P, x, E_full) for x in arms_R}
                am = max(arms_R, key=lambda x: tm[x]["dU_V"])
                af = max(arms_R, key=lambda x: tf[x]["dU_V"])
                out.update(a_star_match=am, sel_match=tm[am], a_star_full=af, sel_full=tf[af], table_match=tm, table_full=tf)
                lfU = out["sel_match"].get("U_H_F") if R == "ae" else None
                # recovery fraction against the best label-free floor at the selected arm's leakage
                cl = [x for x in E_lf if sp.L("V", x) <= sp.L("V", am) + F2.MATCH]
                bl = max(cl, key=lambda x: sp.U("V", x)) if cl else None
                if bl is not None:
                    out["rec"] = (sp.U("H", am) - sp.U("H", bl)) / (sp.U("H", ceil) - sp.U("H", bl))
                    out["lf_floor"] = bl
                per[R]["match"].append(bool(tm[am]["win"]))
                per[R]["full"].append(bool(tf[af]["win"]))
                per[R]["trained"].append(True)
            else:
                per[R]["match"].append(False)
                per[R]["full"].append(False)
                per[R]["trained"].append(False)
            r["regimes"][R] = out
            summ[R].append(out)
        res["splits"][salt] = r
        # ---- markdown
        md.append(f"## split {salt}\n")
        md.append(f"**Headroom (G-H3)**: ceiling {ceil} U_H {AG.fmt(r['headroom']['ceiling_U_H'])}; best label-free floor {flf[0]} U_H {AG.fmt(flf[1])} -> {AG.fmt(r['headroom']['vs_label_free'])}; best envelope floor {ffull[0]} U_H {AG.fmt(ffull[1])} -> {AG.fmt(r['headroom']['vs_envelope'])}; gate (>= 0.03 both): {gh_ok[-1]}\n")
        md.append(f"**Controls**: U(noise16) {AG.fmt(un)} vs shuffled {AG.fmt(us)}; attackers {json.dumps({k: round(v, 3) for k, v in att.items()})}; max same-embedding null {AG.fmt(max(nulls.values()))} -> ok={c_ok}\n")
        md.append("**Memorisation floors (permuted labels, d=64, r=0)**: " + "; ".join(f"{R}: U_V {AG.fmt(v['U_V'])}, U_H {AG.fmt(v['U_H'])}, loss {AG.fmt(v['loss_first'])} -> {AG.fmt(v['loss'])}" for R, v in perm.items()) + "\n")
        md.append("**Labelled sets**: " + "; ".join(f"{R}: {v['n_classes']} classes ({v['n_prod_classes']} production), {v['n_labeled_wells']} wells" for R, v in meta["f3_labels"].items()) + "\n")
        rows = ["| arm | regime | d | U_V | U_H xs [95% CI] | U_null_H | L_V | L_H [CI] | worst att H | loss ratio | eff.rank | U_V - U_V(perm) | trained |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in [y for y in sp.arms if (y in regime and sp.fam[y] in ("adapter", "adapter_lf", "matched")) or y in ("ctrl_noise16", "ldaP_8", "ldaP_16", "ldaT2_16", "dpca16_r1")]:
            if not sp.has_att(x):
                continue
            t = tinfo.get(x, {})
            rows.append(f"| {x} | {regime.get(x, '-')} | {meta['dims'][x]} | {AG.fmt(sp.U('V', x))} | {AG.fci(sp.U('H', x), AG.ci(sp.U_boot('H', x)))} | {AG.fmt(nulls.get(x))} | {AG.fmt(sp.L('V', x), 2)} | {AG.fci(sp.L('H', x), AG.ci(sp.L_boot('H', x)), 2)} | {AG.fmt(sp.worst('H', x), 2)} | "
                        f"{AG.fmt(t.get('loss_ratio'), 2)} | {AG.fmt(t.get('eff_rank'), 1)} | {AG.fmt(None if t.get('U_V_perm') is None else t['U_V'] - t['U_V_perm'])} | {t.get('trained', '-')} |")
        md.append("\n".join(rows) + "\n")
        rows = ["| regime | trained arms | a* (V-sel., E_match) | U_H / L_H | F* in E_match | dU_H vs E_match [CI] | plate-cluster CI | win_match | a* (E_full) | F* in E_full | dU_H vs E_full [CI] | plate-cluster CI | win_full | REC |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for R in REGIMES + ["ae"]:
            o = r["regimes"][R]
            if not o["trained_arms"]:
                rows.append(f"| {R} | none trained (all: {o['all_arms']}) | | | | | | | | | | | | |")
                continue
            sm, sf = o["sel_match"], o["sel_full"]
            rows.append(f"| {R} | {len(o['trained_arms'])}/{len(o['all_arms'])} | {o['a_star_match']} | {AG.fmt(sm['U_H'])} / {AG.fmt(sm['L_H'], 2)} | {sm.get('F_star')} | {AG.fci(sm.get('dU_H'), sm.get('dU_H_ci', [float('nan')] * 2), 4)} | "
                        f"[{AG.fmt(sm.get('dU_H_plate_ci', [float('nan')] * 2)[0], 4)}, {AG.fmt(sm.get('dU_H_plate_ci', [float('nan')] * 2)[1], 4)}] | {sm['win']} | {o['a_star_full']} | {sf.get('F_star')} | "
                        f"{AG.fci(sf.get('dU_H'), sf.get('dU_H_ci', [float('nan')] * 2), 4)} | [{AG.fmt(sf.get('dU_H_plate_ci', [float('nan')] * 2)[0], 4)}, {AG.fmt(sf.get('dU_H_plate_ci', [float('nan')] * 2)[1], 4)}] | {sf['win']} | {AG.fmt(o.get('rec'), 2)} |")
        md.append("\n".join(rows) + "\n")
        rows = ["| regime | a* | U_H / L_H | F* in E_match+ (adds ldaT2_*: raw-space LDA on the SAME Target2 labels) | U_H(F*) / L_H(F*) | dU_H [CI] | plate-cluster CI | win |", "|---|---|---|---|---|---|---|---|"]
        for R in ("a", "c"):
            o = r["regimes"][R]
            if "sel_match_plus" in o:
                sm = o["sel_match_plus"]
                rows.append(f"| {R} | {o['a_star_match_plus']} | {AG.fmt(sm['U_H'])} / {AG.fmt(sm['L_H'], 2)} | {sm.get('F_star')} | {AG.fmt(sm.get('U_H_F'))} / {AG.fmt(sm.get('L_H_F'), 2)} | {AG.fci(sm.get('dU_H'), sm.get('dU_H_ci', [float('nan')] * 2), 4)} | [{AG.fmt(sm.get('dU_H_plate_ci', [float('nan')] * 2)[0], 4)}, {AG.fmt(sm.get('dU_H_plate_ci', [float('nan')] * 2)[1], 4)}] | {sm['win']} |")
        md.append("Sensitivity (NOT the pre-registered comparator): E_match plus `ldaT2_*`\n\n" + "\n".join(rows) + "\n")
    ns = len(ctrl_ok)
    passes = {R: dict(match=int(sum(per[R]["match"])), full=int(sum(per[R]["full"])), trained_splits=int(sum(per[R]["trained"])),
                      passes_match=bool(sum(per[R]["match"]) >= 2), passes_full=bool(sum(per[R]["full"]) >= 2)) for R in REGIMES + ["ae"]}
    c_trained = bool(all(per["c"]["trained"]))
    controls = bool(all(ctrl_ok))
    ghp = bool(sum(gh_ok) >= 2)
    if not controls or not c_trained:
        v = "INVALID"
    elif ghp and passes["c"]["passes_match"]:
        v = "GO-strong (label-free-ish stage warranted; c also beats the full envelope)" if passes["c"]["passes_full"] else "GO (label-free-ish stage warranted; c beats the budget-matched envelope but not the full one)"
    else:
        v = "NO-GO"
    budget = next((f for f, R in (("1%", "b1"), ("10%", "b10"), ("100%", "b100")) if passes[R]["passes_full"]), None)
    rec = {R: [s.get("rec") for s in summ[R]] for R in REGIMES + ["ae"]}
    plus = {R: int(sum(bool(res["splits"][k]["regimes"][R].get("sel_match_plus", {}).get("win")) for k in res["splits"])) for R in ("a", "c")}
    res["sensitivity_E_match_plus_ldaT2_wins_splits"] = plus
    res["passes"] = passes
    res["recovery_fraction"] = rec
    res["break_even_label_budget_vs_full_envelope"] = budget
    res["verdict_tightened"] = v
    res["gates"] = dict(G_H3_splits=int(sum(gh_ok)), controls_ok_all=controls, c_trained_all_splits=c_trained)
    md.append("## Verdict fields\n\n```\n" + json.dumps(dict(verdict_original=res["verdict_original"], verdict_F2_literal=res["verdict_F2_literal"], verdict_tightened=v, label_free_ish_arm=res["label_free_ish_arm"],
                                                             gates=res["gates"], passes=passes, sensitivity_E_match_plus_ldaT2_wins_splits=plus, break_even_label_budget_vs_full_envelope=budget,
                                                             recovery_fraction={R: [None if x is None else round(x, 3) for x in xs] for R, xs in rec.items()}, margin=res["margin"]), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
