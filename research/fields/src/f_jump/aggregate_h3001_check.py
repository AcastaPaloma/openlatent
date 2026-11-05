"""H3-001 closing check aggregator (results_notes/F.md section 24).

  python aggregate_h3001_check.py --staged blk=dir,s1=dir,s9=dir,s11=dir --nulls blk=dir,s1=dir,s9=dir,s11=dir --ref dir --salt s0 --scratch S --out S/h3001_check [--B 1000]

(1) Pabc shuffled null of ctrl_noise16 per staged fold (reproducibility vs the stored Pabc AP, |U - U_null| <= 0.01) and the Pabc verdict re-derived with the corrected control;
(2) the seen-source reference (nothing held out; b10 and b100) analysed with the unchanged F4 machinery under P0 and Pabc, rules R-A / R-B / R-C.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aggregate as AG
import aggregate_f2 as F2
import aggregate_f4 as F4
import aggregate_h3001 as H


def part1(a, md, res):
    staged = dict(x.split("=") for x in a.staged.split(","))
    nulls = dict(x.split("=") for x in a.nulls.split(","))
    chk = {}
    md.append("## (1) Pabc shuffled null of ctrl_noise16, per staged fold\n\n| fold | U Pabc (stored) | U Pabc (recomputed) | U shuffled-label null (Pabc) | |dU| | control holds (<= 0.01) | V: U / null |\n|---|---|---|---|---|---|---|")
    for nm in staged:
        zo = np.load(Path(staged[nm]) / f"preds_{a.salt}.npz")
        zn = np.load(Path(nulls[nm]) / f"preds_{a.salt}.npz")
        uo, un_ = float(np.mean(zo["apc|H|ctrl_noise16|Pabc"])), float(np.mean(zn["apc|H|ctrl_noise16|Pabc"]))
        nul = float(np.mean(zn["apcnull|H|ctrl_noise16|Pabc"]))
        vu, vn = float(np.mean(zn["apc|V|ctrl_noise16|Pabc"])), float(np.mean(zn["apcnull|V|ctrl_noise16|Pabc"]))
        chk[nm] = dict(U_stored=uo, U_recomputed=un_, reproducible=bool(abs(uo - un_) <= 1e-4), U_null=nul, delta=un_ - nul, holds=bool(abs(un_ - nul) <= 0.01), V_U=vu, V_null=vn)
        md.append(f"| {nm} | {uo:.4f} | {un_:.4f} | {nul:.4f} | {abs(un_ - nul):.4f} | {chk[nm]['holds']} (reproducible {chk[nm]['reproducible']}) | {vu:.4f} / {vn:.4f} |")
    res["pabc_null"] = chk
    # re-derived tightened verdict with the corrected control
    rows = {}
    for i, (nm, dp) in enumerate(staged.items()):
        d = H.alias(dp, a.salt, Path(a.scratch) / f"alias_chk_{nm}", null_dir=nulls[nm])
        sp, P, meta, out = F4.analyse(d, a.salt, a.B, 700 + i)
        rows[nm] = H.fold_row(out)
        md.append(f"\nPabc fold {nm} with the corrected control: controls ok={out['controls']['ok']} (U(noise16) {AG.fmt(out['controls']['U_noise16'])} vs Pabc null {AG.fmt(out['controls']['U_noise16_shuffled'])}), valid={out['valid']}, powered={out['powered']}, trained b100 adapter={'yes' if any(out['trained'][x]['trained'] for x in out['trained'] if x.startswith('mlp_b100')) else 'no'}")
    v = H.verdict(rows)
    res["staged_pabc_corrected"] = dict(rows=rows, verdict=v)
    md.append(f"\n**verdict_tightened (Pabc) with the corrected control: {v['verdict']}** (valid and powered folds {v['folds_valid_powered']})\n")


def part2(a, md, res):
    F4.BUD = [("b10", 0.10), ("b100", 1.0)]
    F4.ALLR = [b for b, _ in F4.BUD]
    F4.POWER_KEY = "b10"
    out_p = {}
    for proto in ("P0", "Pabc"):
        d = a.ref if proto == "P0" else H.alias(a.ref, a.salt, Path(a.scratch) / "alias_ref")
        sp, P, meta, out = F4.analyse(d, a.salt, a.B, 800)
        reg = out["regimes"]
        tr = {R: [x for x in out["trained"] if x.startswith(f"mlp_{R}_") and out["trained"][x]["trained"]] for R in ("b10", "b100")}
        win = {R: bool(reg[R].get("sel_full", {}).get("win")) for R in ("b10", "b100")}
        valid = bool(out["valid"] and out["powered"] and tr["b10"] and tr["b100"])
        if valid and win["b10"] and win["b100"]:
            rule = "R-A SEEN-SOURCE GAIN, UNSEEN-SOURCE FAILURE"
        elif valid and win["b100"]:
            rule = "R-C PARTIAL (gain at b100 only)"
        else:
            rule = "R-B TABLE/ADAPTER CANNOT SUPPORT THE CLAIM"
        out_p[proto] = dict(rule=rule, valid=valid, win=win, trained_arms=tr, controls=out["controls"], powered=out["powered"], power=out["power"], gate_H4=out["gate_H4"], headroom=out["headroom"],
                            dU_H={R: reg[R].get("sel_full", {}).get("dU_H") for R in ("b10", "b100")}, dU_H_ci={R: reg[R].get("sel_full", {}).get("dU_H_ci") for R in ("b10", "b100")},
                            dU_H_plate_ci={R: reg[R].get("sel_full", {}).get("dU_H_plate_ci") for R in ("b10", "b100")}, F_star={R: reg[R].get("sel_full", {}).get("F_star") for R in ("b10", "b100")},
                            U_H={R: reg[R].get("sel_full", {}).get("U_H") for R in ("b10", "b100")}, L_H={R: reg[R].get("sel_full", {}).get("L_H") for R in ("b10", "b100")},
                            trained_table=out["trained"], memorisation_floor=out["memorisation_floor"], n_fit=out["n_fit"], n_compounds=out["n_compounds"])
        h = out["headroom"]
        md.append(f"## (2) seen-source reference, {proto}: **{rule}**\n\nvalid={valid} (controls {out['controls']['ok']}: U(noise16) {AG.fmt(out['controls']['U_noise16'])} vs null {AG.fmt(out['controls']['U_noise16_shuffled'])}, attacker {AG.fmt(out['controls']['attacker_noise16'], 3)}; powered={out['powered']} MDE80 b10 {AG.fmt(out['power'].get('b10', {}).get('mde80'), 4)}, b100 {AG.fmt(out['power'].get('b100', {}).get('mde80'), 4)}; "
                  f"trained b10 {tr['b10']}, b100 {tr['b100']}); headroom " + (f"{AG.fmt(h['headroom'])} (ceiling {h['ceiling_arm']} U_H {AG.fmt(h['ceiling_U_H'])}, best floor {h['best_floor']} U_H {AG.fmt(h['floor_U_H'])})" if h else "n/a") + "\n")
        md.append(F4.table(out) + "\n")
        t = ["| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |", "|---|---|---|---|---|---|---|"]
        for x, v in out["trained"].items():
            t.append(f"| {x} | {v['loss_ratio']:.2f} | {v['eff_rank']:.1f} | {AG.fmt(v['U_V'])} | {AG.fmt(v['U_V_perm'])} | {AG.fmt(v['U_H_perm'])} | {v['trained']} |")
        md.append("\n".join(t) + "\n")
    res["reference"] = out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--nulls", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--skip_ref", action="store_true")
    a = ap.parse_args()
    AG.K, AG.CHANCE, F4.ATT_MAX = 10, 0.1, 0.13
    md, res = [f"# H3-001 closing check (F.md section 24); B={a.B}\n"], {}
    F4.BUD = [("b3", 0.03), ("b10", 0.10), ("b30", 0.30), ("b100", 1.0)]
    F4.ALLR = [b for b, _ in F4.BUD]
    part1(a, md, res)
    if a.ref and not a.skip_ref:
        part2(a, md, res)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print("\n".join(md[-1:]))


if __name__ == "__main__":
    main()
