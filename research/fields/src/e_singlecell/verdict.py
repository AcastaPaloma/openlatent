"""Tables and the SC-K0..K3 evaluation (E.md section 2.7) from the bootstrap output of aggregate.py; `python verdict.py combine a.json b.json c.json`."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as AG  # noqa: E402
from aggregate import REFS, INDUCTIVE, POWER_ARMS, ci, fmt  # noqa: E402


def label_only_floor(D, seed):
    """predict the batch from the true fine type (frequency table of the attacker-training half), cross-fit, mean over folds: context for 'chance'"""
    sd = D.S[seed]
    accs = []
    for f in D.fall:
        held = np.where(D.fold_of == f)[0]
        z = np.load(sd["paths"][0])
        hf = z[f"half|{f}"]
        fine = D.fine[held]
        y = D.yb[held]
        B = D.Bf[f]
        r = []
        for d in (0, 1):
            tr, te = hf == d, hf != d
            tab = np.zeros((len(D.fine_names) + 1, B))
            np.add.at(tab, (np.where(fine[tr] >= 0, fine[tr], len(D.fine_names)), y[tr]), 1)
            tab = tab / np.maximum(tab.sum(0, keepdims=True), 1)         # P(type | batch): class-balanced Bayes
            pred = tab[np.where(fine[te] >= 0, fine[te], len(D.fine_names))].argmax(1)
            r.append(np.mean([(pred[y[te] == c] == c).mean() for c in np.unique(y[te])]))
        accs.append(np.mean(r))
    return float(np.mean(accs))


def verdict_tables(D, R, args):
    arms, pt, boot = R["arms"], R["pt"], R["boot"]
    B = len(R["Ws"])
    js = dict(dataset=D.ds, seeds=sorted(D.S), cells=D.n, units=len(D.units), n_studies=int(len(set(D.study_of_unit))), B=B)
    for s in D.S:
        D.S[s]["paths"] = D.seeds[s]
    L = []
    P = L.append
    chance = pt["_chance"]
    P(f"## {D.ds}: seeds {sorted(D.S)}, {D.n} cells, {len(D.units)} units ({'donors' if D.meta['has_donor'] else 'batches: DESCRIPTIVE intervals'}), "
      f"{js['n_studies']} studies, folds {[list(map(int, f)) for f in D.folds]}, B={B}\n")
    P(f"chance (mean over folds 1/B_f) = {chance:.3f}; label-only floor (batch from true fine type) = {label_only_floor(D, min(D.S)):.3f}\n")
    # --- main table
    P("| arm | d | U_fine [95% CI] | U_hold [95% CI] | worst attacker acc [95% CI] | L [95% CI] | lin / mlp / knn / lgbm | retention |")
    P("|---|---|---|---|---|---|---|---|")
    order = [a for a in AG.INDUCTIVE + ["ldaF16"] + [x + "_H" for x in ("ldaC", "ldaC_pca16", "ldaC_pca32", "ldaF16")] + AG.REFS +
             ["onehot_oracle", "ctrl_const", "ctrl_noise16", "ctrl_shuffle_batch", "ctrl_shuffle_labels"] if a in arms]
    order += sorted(a for a in arms if a not in order)
    rows = {}
    for a in order:
        m = pt[a]
        ret = [np.mean(list(D.S[s]["ret"][a].values())) for s in D.S if a in D.S[s]["ret"]]
        rtxt = f"{np.mean(ret):.2f}" if ret else ""
        P(f"| {a} | {D.dims.get(a, '')} | {fmt(m['U'], boot[a]['U'])} | {fmt(m['Uh'], boot[a]['Uh'])} | {fmt(m['worst'], boot[a]['worst'])} | {fmt(m['L'], boot[a]['L'], 2)} | "
          f"{m.get('acc_linear', float('nan')):.3f} / {m.get('acc_mlp', float('nan')):.3f} / {m.get('acc_knn', float('nan')):.3f} / {m.get('acc_lgbm', float('nan')):.3f} | {rtxt} |")
        rows[a] = dict(U=m["U"], Uh=m["Uh"], worst=m["worst"], L=m["L"], U_ci=ci(boot[a]["U"]), Uh_ci=ci(boot[a]["Uh"]), L_ci=ci(boot[a]["L"]),
                       worst_ci=ci(boot[a]["worst"]), ret=float(np.mean(ret)) if ret else None)
    js["rows"] = rows
    js["chance"] = chance
    ar = lambda a, k: np.asarray(boot[a][k])                   # noqa: E731

    # --- K1
    raw = pt["raw50"]
    k1 = dict(raw_acc=raw["worst"], raw_excess=raw["worst"] - chance, raw_acc_ci=ci(boot["raw50"]["worst"]))
    k1["raw_ok"] = bool(raw["worst"] - chance >= 0.15 and raw["worst"] >= min(3 * chance, 0.85))
    ctrl = {}
    for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_batch"):
        if c in pt:
            ctrl[c] = pt[c]["worst"] - chance
    u0 = pt["_u0"]
    uctrl = {c: pt[c]["U"] - u0 for c in ("ctrl_const", "ctrl_noise16", "ctrl_shuffle_labels") if c in pt}
    k1["controls_excess_att"] = ctrl
    k1["controls_excess_U"] = uctrl
    k1["controls_ok"] = bool(all(v <= 0.03 for v in ctrl.values()) and all(v <= 0.03 for v in uctrl.values()))
    k1["pass"] = bool(k1["raw_ok"] and k1["controls_ok"])
    js["K1"] = k1
    P(f"\n**K1 measurability**: raw50 worst-attacker accuracy {raw['worst']:.3f} {ci(boot['raw50']['worst'])} vs chance {chance:.3f} (excess {k1['raw_acc'] - chance:.3f}, "
      f"needs >= 0.15 and >= min(3 x chance, 0.85) = {min(3 * chance, 0.85):.3f}): {'ok' if k1['raw_ok'] else 'FAIL'}. "
      f"Controls (attacker excess over chance; probe U minus the analytic random-guess macro-F1 u0={u0:.3f}): {ctrl} / {uctrl}; allowed <= 0.03: {'ok' if k1['controls_ok'] else 'FAIL'}. => K1 {'PASS' if k1['pass'] else 'FAIL'}")

    # --- K2
    def ratio(num_arm, key, den_arm):
        u0v = np.asarray(R["boot_u0" if key == "U" else "boot_u0h"])
        u0p = pt["_u0" if key == "U" else "_u0h"]
        v = (ar(num_arm, key) - u0v) / np.maximum(ar(den_arm, key) - u0v, 1e-6)
        p = (pt[num_arm][key] - u0p) / max(pt[den_arm][key] - u0p, 1e-6)
        return p, v

    r_fine, v1 = ratio("ldaC_pca16", "U", "raw50")
    r_hold, v2 = ratio("ldaC_pca16", "Uh", "raw50")
    # U_hold of ldaC_pca16 already uses the _H probes (hold_name); r_fine uses the full-fit probes
    k2 = dict(r_fine=r_fine, r_fine_ci=ci(v1), r_hold=r_hold, r_hold_ci=ci(v2))
    k2["holds_both"] = bool(r_fine > 0.70 and r_hold > 0.70)
    k2["holds_either"] = bool(r_fine > 0.70 or r_hold > 0.70)
    js["K2"] = k2
    P(f"\n**K2 under-determination**: ldaC_pca16 retains r_fine = {r_fine:.2f} {ci(v1)} of raw50's above-chance fine-label macro-F1 and r_hold = {r_hold:.2f} {ci(v2)} "
      f"of its held-type macro-F1 (threshold 0.70). Label-only floor holds: both = {k2['holds_both']}, either = {k2['holds_either']} (pre-registered reading: both).")

    # --- K3
    fall = list(D.fall)
    refs = [r for r in REFS if r in arms]
    ind = [a for a in arms if AG._IND_RE.match(a)]
    k3 = dict(refs={r: dict(U=pt[r]["U"], L=pt[r]["L"]) for r in refs})
    if refs:
        ref = max(refs, key=lambda r: pt[r]["U"])
        k3["ref"] = ref
        sel, feas = {}, True
        for f in fall:
            oth = [x for x in fall if x != f]
            Mo = AG.all_metrics(D, D.S, ind + [ref, "raw50"], np.ones(D.n), oth)
            lref = Mo[ref]["L"]
            cand = [a for a in ind if Mo[a]["L"] <= lref + 0.05]
            if not cand:
                feas = False
                sel[f] = None
                continue
            sel[f] = max(cand, key=lambda a: Mo[a]["U"])
        k3["selected_per_fold"] = sel
        k3["all_folds_feasible"] = feas
        if feas:
            # virtual arm: held-out predictions of fold f come from the arm selected for fold f
            for s, sd in D.S.items():
                for kind in AG.KINDS:
                    v = np.full(D.n, -1, np.int8)
                    for f in fall:
                        m = D.fold_of == f
                        v[m] = sd["att"][(sel[f], kind)][m]
                    sd["att"][("sel_floor", kind)] = v
                for kind in AG.PROBES:
                    v = np.full(D.n, -1, np.int16)
                    for f in fall:
                        m = D.fold_of == f
                        v[m] = sd["pr"][(sel[f], kind)][m]
                    sd["pr"][("sel_floor", kind)] = v
            bs_U, bs_L = [], []
            for w in R["Ws"]:
                m = AG.all_metrics(D, D.S, ["sel_floor", ref, "raw50"], w, fall)
                bs_U.append(m[ref]["U"] - m["sel_floor"]["U"])
                bs_L.append(m["sel_floor"]["L"] - m[ref]["L"])
            m0 = AG.all_metrics(D, D.S, ["sel_floor", ref, "raw50"], np.ones(D.n), fall)
            k3["U_ref_minus_floor"] = m0[ref]["U"] - m0["sel_floor"]["U"]
            k3["U_gap_ci"] = ci(bs_U)
            k3["L_floor_minus_ref"] = m0["sel_floor"]["L"] - m0[ref]["L"]
            k3["L_gap_ci"] = ci(bs_L)
            k3["U_floor"], k3["L_floor"] = m0["sel_floor"]["U"], m0["sel_floor"]["L"]
            k3["fires"] = bool(k3["U_ref_minus_floor"] <= 0.05 and k3["L_floor_minus_ref"] <= 0.05)
        else:
            k3["fires"] = False
        # optimistic bound: test-selected best floor within L_ref + 0.05
        cand = [a for a in ind if pt[a]["L"] <= pt[ref]["L"] + 0.05]
        if cand:
            best = max(cand, key=lambda a: pt[a]["U"])
            k3["test_selected"] = dict(arm=best, U=pt[best]["U"], L=pt[best]["L"], gap=pt[ref]["U"] - pt[best]["U"], fires=bool(pt[ref]["U"] - pt[best]["U"] <= 0.05))
        else:
            k3["test_selected"] = dict(arm=None, fires=False)
        # hold-type sensitivity (test-selected, _H versions for label-supervised arms)
        refh = max(refs, key=lambda r: pt[r]["Uh"])
        candh = [a for a in ind if pt[hold(a)]["L"] <= pt[refh]["L"] + 0.05]
        if candh:
            bh = max(candh, key=lambda a: pt[a]["Uh"])
            k3["hold_sens"] = dict(ref=refh, arm=bh, Uh_ref=pt[refh]["Uh"], Uh=pt[bh]["Uh"], gap=pt[refh]["Uh"] - pt[bh]["Uh"],
                                   fires=bool(pt[refh]["Uh"] - pt[bh]["Uh"] <= 0.05))
        P(f"\n**K3 headroom**: best transductive reference = {ref} (U {pt[ref]['U']:.3f}, L {pt[ref]['L']:.2f}); references: "
          + ", ".join(f"{r} U {pt[r]['U']:.3f} L {pt[r]['L']:.2f}" for r in refs))
        P(f"V-selected inductive floor per fold (selection on the other folds, L <= L_ref + 0.05): {sel}")
        if feas:
            P(f"selected floor pooled: U {k3['U_floor']:.3f}, L {k3['L_floor']:.2f}; U_ref - U_floor = {k3['U_ref_minus_floor']:.3f} {k3['U_gap_ci']}; "
              f"L_floor - L_ref = {k3['L_floor_minus_ref']:.3f} {k3['L_gap_ci']}; K3 fires on this dataset: {k3['fires']}")
        else:
            P("no inductive configuration reaches L <= L_ref + 0.05 in the selection folds of at least one fold: headroom exists on the leakage axis (K3 does not fire)")
        P(f"sensitivity: test-selected floor {k3['test_selected']}; hold-type version {k3.get('hold_sens')}")
    js["K3"] = k3

    # --- power audit (K0)
    pa = [a for a in POWER_ARMS if a in arms]
    if len(pa) >= 2:
        rows_pa = AG.power_audit(D, pa)
        med = float(np.median([r["mde80"] for r in rows_pa]))
        js["K0"] = dict(n_donors=len(D.units) if D.meta["has_donor"] else None, n_studies=js["n_studies"], mde80_median=med, pairs=rows_pa,
                        units_are_donors=bool(D.meta["has_donor"]))
        P("\n**K0 power audit** (seed %d predictions; units = %s): n_units %d, n_studies %d" % (min(D.S), "donors" if D.meta["has_donor"] else "batches", len(D.units), js["n_studies"]))
        P("| pair | dU | SE (boot) | crit (perm 5%) | MDE80 |\n|---|---|---|---|---|")
        for r in rows_pa:
            P(f"| {r['a']} vs {r['b']} | {r['dU']:.3f} | {r['se']:.3f} | {r['crit']:.3f} | {r['mde80']:.3f} |")
        P(f"median MDE80 over pairs = {med:.3f} (needs <= 0.05 on HLCA)")
    return "\n".join(L), js


def hold(a):
    return AG.hold_name(a)


def combine(paths):
    J = {Path(p).stem.split("_")[-1] if False else json.loads(Path(p).read_text())["dataset"]: json.loads(Path(p).read_text()) for p in paths}
    out = []
    P = out.append
    h = J.get("hlca")
    k0 = bool(h and h.get("K0") and (h["n_studies"] < 8 or (h["K0"]["n_donors"] or 0) < 40 or h["K0"]["mde80_median"] > 0.05))
    P(f"K0 (HLCA): studies {h['n_studies']}, donors {h['K0']['n_donors']}, median MDE80 {h['K0']['mde80_median']:.3f} -> fires: {k0}")
    k1_pass = {d: j["K1"]["pass"] for d, j in J.items()}
    k1 = (not k1_pass.get("hlca", False)) or sum(k1_pass.values()) < 2
    P(f"K1 pass per dataset {k1_pass} -> fires: {k1}")
    ok = [d for d in J if k1_pass[d]]
    k2h = {d: J[d]["K2"]["holds_both"] for d in ok}
    k2 = sum(k2h.values()) >= 2 and k2h.get("hlca", False)
    P(f"K2 label-only floor holds (both) per dataset {k2h}; either-reading {({d: J[d]['K2']['holds_either'] for d in ok})} -> fires: {k2}")
    k3h = {d: J[d]["K3"].get("fires", False) for d in ok}
    k3 = sum(k3h.values()) >= 2 and k3h.get("hlca", False)
    P(f"K3 no headroom per dataset {k3h}; test-selected-floor sensitivity {({d: J[d]['K3'].get('test_selected', {}).get('fires') for d in ok})} -> fires: {k3}")
    fired = [n for n, v in (("SC-K0", k0), ("SC-K1", k1), ("SC-K2", k2), ("SC-K3", k3)) if v]
    P("\nVERDICT: " + ("GO for stage 2 (no kill condition fired)" if not fired else "NO-GO, fired: " + ", ".join(fired) + (" (instrument)" if "SC-K0" in fired else "")))
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "combine":
        print(combine(sys.argv[2:]))
