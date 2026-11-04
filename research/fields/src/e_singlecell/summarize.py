"""Compact SC-0 summary table (K0..K3 per dataset) from the three <ds>_final.json files.  python summarize.py immune.json pancreas.json hlca.json"""
import json
import sys
from pathlib import Path


def f(x, n=3):
    return f"{x:.{n}f}"


def ci(c, n=3):
    return f"[{c[0]:.{n}f}, {c[1]:.{n}f}]"


def main(paths):
    J = {json.loads(Path(p).read_text())["dataset"]: json.loads(Path(p).read_text()) for p in paths}
    order = [d for d in ("hlca", "immune", "pancreas") if d in J]
    print("| quantity | " + " | ".join(order) + " |\n|---|" + "---|" * len(order))

    def row(name, fn):
        print(f"| {name} | " + " | ".join(fn(J[d]) for d in order) + " |")
    row("cells / units / studies", lambda j: f"{j['cells']} / {j['units']} / {j['n_studies']}")
    row("chance (mean 1/B_f)", lambda j: f(j["chance"]))
    row("K0 median MDE80 (floor vs floor, macro-F1)", lambda j: f(j["K0"]["mde80_median"]))
    row("K1 raw50 worst attacker acc [CI]", lambda j: f"{f(j['K1']['raw_acc'])} {ci(j['K1']['raw_acc_ci'])}")
    row("K1 threshold min(3 x chance, 0.85)", lambda j: f(min(3 * j["chance"], 0.85)))
    row("K1 controls ok / pass", lambda j: f"{j['K1']['controls_ok']} / {j['K1']['pass']}")
    row("K2 r_fine [CI]", lambda j: f"{f(j['K2']['r_fine'], 2)} {ci(j['K2']['r_fine_ci'], 2)}")
    row("K2 r_hold [CI]", lambda j: f"{f(j['K2']['r_hold'], 2)} {ci(j['K2']['r_hold_ci'], 2)}")
    row("K2 label-only floor holds (both / either)", lambda j: f"{j['K2']['holds_both']} / {j['K2']['holds_either']}")
    row("K3 best reference by U (U, L)", lambda j: f"{j['K3'].get('ref')} ({f(j['K3']['refs'][j['K3']['ref']]['U'])}, {f(j['K3']['refs'][j['K3']['ref']]['L'], 2)})")
    row("K3 selected floor (U, L) / gap U_ref - U_floor [CI]",
        lambda j: (f"({f(j['K3']['U_floor'])}, {f(j['K3']['L_floor'], 2)}) / {f(j['K3']['U_ref_minus_floor'])} {ci(j['K3']['U_gap_ci'])}" if "U_floor" in j["K3"] else "no feasible floor"))
    row("K3 fires (registered / test-selected floor)", lambda j: f"{j['K3']['fires']} / {j['K3']['test_selected'].get('fires')}")
    for r in ("T_harmony", "T_scvi"):
        row(f"per-reference: {r} U, L; best floor U; gap [CI]",
            lambda j, r=r: (lambda x: (f"{f(x['U'])}, {f(x['L'], 2)}; {x['floor']} {f(x['floor_U'])}; {f(x['gap'])} {ci(x['gap_ci'])}" if x.get("floor") else f"{f(x['U'])}, {f(x['L'], 2)}; no floor reaches L"))(j["K3"]["per_reference"][r]))


if __name__ == "__main__":
    main(sys.argv[1:])
