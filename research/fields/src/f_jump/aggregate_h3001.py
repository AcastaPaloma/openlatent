"""H3-001 staged aggregator (results_notes/F.md section 23): leave-source-out break-even on the assembled 10-source table, F4's definitions on a 4-point grid.

  python aggregate_h3001.py --runs blk=dir,s1=dir,s9=dir,s11=dir --salt s0 --scratch $SCR --out $SCR/h3001 [--B 1000]

Each run is analysed twice with aggregate_f4.analyse (F4's break-even, trained check, headroom gate, power, controls): under P0 (same-plate negatives; the F3 / F4 protocol)
and under Pabc (positives at a different well, position-matched negatives, near-duplicate-pruned H; the stage-1 controls).  The Pabc view is made by aliasing the Pabc
arrays onto the P0 keys of a scratch copy so that the unchanged F4 machinery computes everything.  verdict_original = P0, verdict_tightened = Pabc.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import aggregate as AG
import aggregate_f4 as F4
import aggregate_f2 as F2

AG.K = 10
AG.CHANCE = 0.1
F4.ATT_MAX = 0.13
F4.BUD = [("b3", 0.03), ("b10", 0.10), ("b30", 0.30), ("b100", 1.0)]
F4.ALLR = [b for b, _ in F4.BUD]
LE30 = ("b3", "b10", "b30")
BLOCK_B100_MIN = 0.05


def alias(src_dir, salt, dst):
    """scratch copy whose P0 keys hold the Pabc arrays."""
    z = np.load(Path(src_dir) / f"preds_{salt}.npz")
    P = {k: z[k] for k in z.files}
    out = {k: v for k, v in P.items()}
    for k in list(P):
        parts = k.split("|")
        if len(parts) == 4 and parts[0] == "apc" and parts[3] == "Pabc":
            out[f"apc|{parts[1]}|{parts[2]}|xs"] = P[k]
        elif len(parts) == 3 and parts[0] == "cid" and parts[2] == "Pabc":
            out[f"cid|{parts[1]}|xs"] = P[k]
        elif len(parts) == 4 and parts[0] == "apq" and parts[3] == "Pabc":
            out[f"apq|{parts[1]}|{parts[2]}"] = P[k]
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"preds_{salt}.npz", **out)
    shutil.copy(Path(src_dir) / f"meta_{salt}.json", dst / f"meta_{salt}.json")
    return dst


def fold_row(out):
    reg = out["regimes"]
    be = out["break_even"]
    first = be["first_winning_budget"]
    row = dict(valid=out["valid"], powered=out["powered"], gate_H4=out["gate_H4"], first_winning=first, f_le30=bool(first in LE30), censored_or_gt30=bool(first not in LE30),
               interpolated_percent=be["interpolated_percent"], boot_pct=be["boot_percentiles_percent"], dU_H_curve=be["dU_H_curve"], mde80=out["power"],
               dU_b100=reg["b100"].get("sel_full", {}).get("dU_H") if "b100" in reg else None, dU_b10=reg["b10"].get("sel_full", {}).get("dU_H") if "b10" in reg else None,
               supports=out["supports"], headroom=out["headroom"], controls=out["controls"])
    return row


def verdict(rows, block="blk"):
    ok = {n: r for n, r in rows.items() if r["valid"] and r["powered"]}
    n_ok = len(ok)
    n_le30 = sum(r["f_le30"] for r in ok.values())
    n_gt30 = n_ok - n_le30
    blk = ok.get(block)
    blk_ok = bool(blk is not None and blk["f_le30"] and blk["dU_b100"] is not None and blk["dU_b100"] >= BLOCK_B100_MIN)
    if n_ok < 3:
        v = "INVALID"
    elif n_le30 >= 3 and blk_ok:
        v = "GENERALISES"
    elif n_gt30 >= 2:
        v = "DOES-NOT-GENERALISE"
    else:
        v = "PARTIAL"
    return dict(verdict=v, folds_valid_powered=sorted(ok), n_valid_powered=n_ok, n_f_le30=n_le30, n_f_gt30_or_censored=n_gt30, block_valid_powered=blk is not None,
                block_f_le30=bool(blk and blk["f_le30"]), block_dU_b100=None if blk is None else blk["dU_b100"], block_ok=blk_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="blk=dir,s1=dir,s9=dir,s11=dir")
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    runs = dict(x.split("=") for x in a.runs.split(","))
    res = dict(margin=dict(delta_U=F4.DELTA_U, power_max=F4.POWER_MAX, match=F2.MATCH, block_b100_min=BLOCK_B100_MIN), grid=[b for b, _ in F4.BUD], protocols={})
    md = [f"# H3-001 staged aggregate (pre-registration: results_notes/F.md section 23); B={a.B}; held-out units: folds {list(runs)}\n"]
    for proto in ("P0", "Pabc"):
        rows, outs = {}, {}
        for i, (nm, dp) in enumerate(runs.items()):
            d = dp if proto == "P0" else alias(dp, a.salt, Path(a.scratch) / f"alias_{nm}")
            sp, P, meta, out = F4.analyse(d, a.salt, a.B, 700 + i)
            outs[nm], rows[nm] = out, fold_row(out)
            md.append(f"## {proto} fold {nm}: hold_source={out['hold_source']}, n_fit_wells={out['n_fit']}, H/V compounds {out['n_compounds']}\n")
            h = out["headroom"]
            md.append(f"**Headroom (G-H4)**: " + (f"ceiling {h['ceiling_arm']} U_H {AG.fmt(h['ceiling_U_H'])} vs best floor {h['best_floor']} U_H {AG.fmt(h['floor_U_H'])} -> {AG.fmt(h['headroom'])}; gate {out['gate_H4']}" if h else "no trained b100 adapter") +
                      f"; controls ok={out['controls']['ok']} (U(noise16) {AG.fmt(out['controls']['U_noise16'])} vs shuffled {AG.fmt(out['controls']['U_noise16_shuffled'])}, attacker {AG.fmt(out['controls']['attacker_noise16'], 3)}, max null {AG.fmt(out['controls']['null_max'])}); "
                      f"powered={out['powered']} (MDE80 b30 {AG.fmt(out['power'].get('b30', {}).get('mde80'), 4)}, b100 {AG.fmt(out['power'].get('b100', {}).get('mde80'), 4)}); valid={out['valid']}\n")
            md.append(F4.table(out) + "\n")
            be = out["break_even"]
            md.append(f"**Break-even vs the full envelope**: first winning budget {be['first_winning_budget']}; interpolated {AG.fmt(be['interpolated_percent'], 1)}% ({be['flag']}), bootstrap percentiles 2.5/50/97.5 = {[round(v, 1) for v in be['boot_percentiles_percent']]}%; dU_H(f) = {json.dumps({k: round(v, 4) for k, v in be['dU_H_curve'].items()})}\n")
            tr = ["| arm | loss ratio | eff.rank | U_V | U_V(perm) | U_H(perm) | trained |", "|---|---|---|---|---|---|---|"]
            for x, t in out["trained"].items():
                tr.append(f"| {x} | {t['loss_ratio']:.2f} | {t['eff_rank']:.1f} | {AG.fmt(t['U_V'])} | {AG.fmt(t['U_V_perm'])} | {AG.fmt(t['U_H_perm'])} | {t['trained']} |")
            md.append("\n".join(tr) + "\n")
        v = verdict(rows)
        res["protocols"][proto] = dict(rows=rows, verdict=v, runs=outs)
        md.append(f"## {proto} cross-fold: **{v['verdict']}**; valid and powered folds {v['folds_valid_powered']}; f* <= 30% in {v['n_f_le30']}, f* > 30% or censored in {v['n_f_gt30_or_censored']}; block dU_H(b100) = {v['block_dU_b100']}\n")
    res["verdict_original"] = res["protocols"]["P0"]["verdict"]["verdict"]
    res["verdict_tightened"] = res["protocols"]["Pabc"]["verdict"]["verdict"]
    nsrc = {"blk": 4, "s1": 1, "s9": 1, "s11": 1}
    for proto in ("P0", "Pabc"):
        rows = res["protocols"][proto]["rows"]
        res["protocols"][proto]["sources_le30"] = int(sum(nsrc.get(n, 1) for n, r in rows.items() if r["valid"] and r["powered"] and r["f_le30"]))
        res["protocols"][proto]["sources_total"] = int(sum(nsrc.get(n, 1) for n in rows))
    md.append("## Verdict fields\n\n```\n" + json.dumps(dict(verdict_original_P0=res["verdict_original"], verdict_tightened_Pabc=res["verdict_tightened"],
                                                           P0=res["protocols"]["P0"]["verdict"], Pabc=res["protocols"]["Pabc"]["verdict"],
                                                           sources_le30_P0=res["protocols"]["P0"]["sources_le30"], sources_le30_Pabc=res["protocols"]["Pabc"]["sources_le30"],
                                                           sources_total=res["protocols"]["P0"]["sources_total"]), indent=1, default=float) + "\n```\n")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(md))
    print(md[-1])


if __name__ == "__main__":
    main()
