"""Engagement report for the final bounded learned-mask attempt (results_notes/F.md section 21).  Reads the meta files only (training-side quantities).

  python aggregate_hlca_eng.py --runs <dir_f0> <dir_f1> <dir_f2> [--json out.json] [--md out.md]

Per candidate, budget and fold: in-loop monitor accuracy (ENG-1 <= 0.20), fresh-adversary check (ENG-2: a_ref - a_noise >= 0.05 and
a_arm - a_noise <= 0.5 (a_ref - a_noise)), non-collapse (SupCon loss ratio <= 0.8, effective rank of the validation vectors >= 4).
A candidate PASSES a fold iff all three hold; ENGAGED at a budget iff some candidate passes in >= 2 of 3 folds.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    eng = {}
    for r in a.runs:
        for p in sorted(Path(r).glob("meta_s*.json")):
            seed = int(p.stem.split("_s")[1])
            for k, v in json.loads(p.read_text()).get("engage", {}).items():
                f, b = k.split("|")
                eng[(seed, int(b), int(f))] = v
    seeds = sorted({k[0] for k in eng})
    out, lines = {}, ["# Engagement report (section 21): candidate x budget; per fold: monitor acc / fresh a_arm vs noise a_noise, ref a_ref / rho / pass", ""]
    for s in seeds:
        for b in sorted({k[1] for k in eng if k[0] == s}):
            folds = sorted(k[2] for k in eng if k[0] == s and k[1] == b)
            cands = list(eng[(s, b, folds[0])]["cands"])
            row = {}
            lines.append(f"## seed {s}, f = {b}%  (chance {eng[(s, b, folds[0])]['chance']:.3f}; a_noise / a_ref per fold: " +
                         ", ".join(f"F{f}: {eng[(s, b, f)]['a_noise']:.3f}/{eng[(s, b, f)]['a_ref']:.3f}" for f in folds) + ")")
            lines += ["", "| candidate | per fold: mon acc, a_arm, rho, ENG1, ENG2, no-collapse (ratio, rank) | passes |", "|---|---|---|"]
            for c in cands:
                cells, npass = [], 0
                for f in folds:
                    e = eng[(s, b, f)]
                    x = e["cands"][c]
                    rho = (x["a_arm"] - e["a_noise"]) / (e["a_ref"] - e["a_noise"]) if abs(e["a_ref"] - e["a_noise"]) > 1e-9 else float("nan")
                    npass += int(x["passed"])
                    cells.append(f"F{f}: {x['adv_acc_last200']:.2f}, {x['a_arm']:.3f}, {rho:.2f}, {'Y' if x['eng1'] else 'n'}{'Y' if x['eng2'] else 'n'}{'Y' if x['nocollapse'] else 'n'} ({x['supcon_ratio']:.2f}, {x['eff_rank_cells']:.1f})")
                row[c] = dict(passes=npass, folds=len(folds))
                lines.append(f"| {c} | " + "; ".join(cells) + f" | {npass}/{len(folds)} |")
            engaged = any(v["passes"] >= 2 for v in row.values())
            out[f"s{s}_f{b}"] = dict(candidates=row, engaged=engaged)
            lines += ["", f"**ENGAGED at f={b}% (seed {s}): {engaged}**", ""]
    any_eng = any(v["engaged"] for v in out.values())
    lines.append("**VERDICT: " + ("ENGAGED at >= 1 budget (continue to seeds 1-2 and the headline comparison)" if any_eng else "engagement never achieved (HLCA learned-mask line CLOSED)") + "**")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(engaged_any=any_eng, detail=out), indent=1))
    if a.md:
        Path(a.md).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
