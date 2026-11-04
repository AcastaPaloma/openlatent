"""Turns the raw result JSONs into the endpoint tables of results_notes/S.md (E1-E3 from the harness runs, E4-E5 from the planted engine run).
Definitions are those of S.md section 4; thresholds are copied from there. Prints markdown; writes a JSON next to the inputs.

python -m s_suite.summarize real    --within DIR --budget DIR --select DIR --hmdb DIR
python -m s_suite.summarize planted --jsonl FILE
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os

import numpy as np

from . import provenance as prov
from . import selection_bias as sb

SAT_LO, SAT_HI = 0.06, 0.95


def thr(P):
    """Eligibility floor for evidence-budget points (amendment v1.1): at least max(10, 40% of persons), at most all persons."""
    return min(P, max(10, math.ceil(0.4 * P)))


def e1(path_list):
    res = {}
    for p in path_list:
        res.update({k: v for k, v in json.load(open(p)).items() if not k.startswith("_")})
    out = {}
    for k, v in res.items():
        rel = v["releases"]
        ns = {r: x for r, x in rel.items() if SAT_LO < x["a2"] < SAT_HI and x["reproduces"]}
        adj = v["adjacent_pairs"]
        o = dict(units=v["n_units"], releases=len(rel), reproduced=sum(x["reproduces"] for x in rel.values()), dropped=v.get("dropped_releases"),
                 nonsat_reproduced=len(ns), moves=[r for r, x in ns.items() if x["moves"]],
                 d_ci_excludes_0=[r for r, x in ns.items() if x["optimism_a2_minus_h"]["resolved"]],
                 adjacent=len(adj), adj_resolved_a2=sum(q["a2"]["resolved"] for q in adj), adj_resolved_honest=sum(q["honest"]["resolved"] for q in adj),
                 median_mde80_adjacent_honest=float(np.median([q["honest"]["mde80"] for q in adj])) if adj else None,
                 median_mde80_D=float(np.median([x["optimism_a2_minus_h"]["mde80"] for x in ns.values()])) if ns else None,
                 max_abs_D=float(max(abs(x["optimism_a2_minus_h"]["diff"]) for x in ns.values())) if ns else None,
                 not_reproduced=[r for r, x in rel.items() if not x["reproduces"]])
        sa = {}
        for a in ("retrained|frame_r50", "retrained|r3d"):
            vals = [x["single_attackers"][a]["h_minus_single"]["exceeds_mde"] for x in ns.values() if x.get("single_attackers")]
            sa[a] = dict(within_mde=int(sum(not t for t in vals)), n=len(vals))
        o["k2r_single_within_mde_of_H"] = sa
        o["headline_pairs"] = v.get("headline_pairs")
        out[k] = o
    return out


def e2(path):
    r = json.load(open(path))
    rows = []
    for k, v in r["cells"].items():
        top = max(v["acc_full"])
        x = v["K"].get("2")
        if x and SAT_LO < top < SAT_HI:
            rows.append((k, top, x["optimism"], x["inflation"], x["optimism_boot_ci95"]))
    op = np.array([q[2] for q in rows]) * 100
    infl = np.array([q[3] for q in rows]) * 100
    return dict(cells_k2=len(rows), median_optimism=float(np.median(op)), iqr=[float(np.percentile(op, 25)), float(np.percentile(op, 75))], max_optimism=float(op.max()),
                median_inflation=float(np.median(infl)), n_ge_1pt=int((op >= 1).sum()),
                by_dataset={ds: dict(n=len([q for q in rows if q[0].startswith(ds + "/")]), median=float(np.median([q[2] * 100 for q in rows if q[0].startswith(ds + "/")])),
                                     max=float(max(q[2] * 100 for q in rows if q[0].startswith(ds + "/")))) for ds in {q[0].split("/")[0] for q in rows}},
                top5=[(q[0], round(100 * q[2], 2), [round(100 * c, 2) for c in q[4]]) for q in sorted(rows, key=lambda q: -q[2])[:5]],
                families={ds: dict(P=v["P"], units=v["n_units"], K={K: dict(optimism=x["optimism"], optimism_ci=x["optimism_boot_ci95"], inflation=x["inflation"], inflation_ci=x["inflation_boot_ci95"])
                                                              for K, x in v["K"].items()}) for ds, v in r["families"].items()},
                matters=bool(np.median(op) >= 1 or max(x["optimism"] for v in r["families"].values() for x in v["K"].values()) * 100 >= 2))


def e3(path):
    r = {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}
    K = r[next(iter(r))]["K"]
    i10 = K.index(10)
    out = dict(controls={}, gain={}, independence={}, nstar={})
    # controls: original rule (chance 1/C, all points) and tightened (eligibility floor), per scheme
    orig, tight = collections.defaultdict(list), collections.defaultdict(list)
    for key, x in r.items():
        if "/black/" not in key:
            continue
        P = x["n_persons"]
        for sc, v in x["schemes"].items():
            for i, k in enumerate(K):
                a, m, n, hi = v["acc"][i], v["mde80"][i], v["n_persons_eligible"][i], v["ci95"][1][i]
                if a is None or (isinstance(a, float) and math.isnan(a)):
                    continue
                if hi >= 1.0 / P + m:
                    orig[(key.split("/")[0], sc)].append((key, k, round(a, 3), n))
                    if n >= thr(P):
                        tight[(key.split("/")[0], sc)].append((key, k, round(a, 3), n))
    out["controls"] = dict(original_failures={f"{a}|{b}": v for (a, b), v in orig.items()}, tightened_failures={f"{a}|{b}": v for (a, b), v in tight.items()})
    rows = []
    for key, x in r.items():
        if not (0.06 < x["acc1"] < 0.90):
            continue
        P = x["n_persons"]
        v = x["schemes"]["diverse"]
        g = [q for q in v["gain_vs_k1"] if q["K"] == 10][0]
        if g["available"] and v["n_persons_eligible"][i10] >= thr(P):
            rows.append((key, x["acc1"], g["diff"], g["mde80"], g["inside_mde"]))
    by = collections.defaultdict(list)
    for q in rows:
        by[q[0].split("/")[0]].append(q)
    out["gain"] = dict(cells=len(rows), inside_mde=int(sum(q[4] for q in rows)),
                       by_dataset={ds: dict(n=len(qs), median_acc1=float(np.median([q[1] for q in qs])), median_gain=float(np.median([q[2] for q in qs])),
                                            median_mde80=float(np.median([q[3] for q in qs])), inside_mde=int(sum(q[4] for q in qs))) for ds, qs in by.items()})
    ind = collections.defaultdict(lambda: dict(n=0, detected=0, positive=0))
    for key, x in r.items():
        if x["sessions_per_person"] <= 1.0 or not (0.06 < x["acc1"] < 0.90) or x["independence"] is None:
            continue
        best = None
        for q in x["independence"]:
            if q.get("available") and q["n_persons"] >= thr(x["n_persons"]):
                best = q
        if best is None:
            continue
        d = ind[key.split("/")[0]]
        d["n"] += 1
        d["detected"] += int(best["dependence_detected"])
        d["positive"] += int(best["diff"] > 0)
    out["independence"] = dict(ind)
    rat = []
    for key, x in r.items():
        if not (0.06 < x["acc1"] < 0.90):
            continue
        n90, pred = x["schemes"]["diverse"]["n_star"]["0.9"], x["pred"]["n_star_pred"]
        if n90 is not None and pred:
            rat.append((key.split("/")[0], n90 / pred))
    rr = np.array([q[1] for q in rat])
    out["nstar"] = dict(cells=len(rat), median_ratio=float(np.median(rr)) if len(rr) else None, within_0p5_2=int(((rr >= 0.5) & (rr <= 2)).sum()) if len(rr) else 0)
    return out


def e45(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["channel"], r["dose"])].append(r)
    leak = [r for r in rows if r["channel"] != "null"]
    nul = [r for r in rows if r["channel"] == "null"]
    out = dict(n_rows=len(rows), n_leak=len(leak), n_null=len(nul), E4={}, E5={}, exploratory_circular={})
    for n in ("150", "600", "1425", "3000"):
        out["E4"][n] = dict(
            cover_bonf=float(np.mean([r["cert"][n]["cover_bonf"] for r in leak])), cover_maxt=float(np.mean([r["cert"][n]["cover_maxt"] for r in leak])),
            cover_ci_selected=float(np.mean([r["cert"][n]["cover_ci"] for r in leak])), cover_each_bonf=float(np.mean([r["cert"][n]["cover_each_bonf"] for r in leak])),
            min_cell_cover_bonf=float(min(np.mean([r["cert"][n]["cover_bonf"] for r in c]) for (ch, d), c in cells.items() if ch != "null")),
            null_false_leak=float(np.mean([r["cert"][n]["verdict"] == "LEAK" for r in nul])),
            null_coverage_bonf=float(np.mean([r["cert"][n]["cover_bonf"] for r in nul])),
            bias_selected_leak=float(np.mean([r["cert"][n]["sel_test"] - r["cert"][n]["true_sel"] for r in leak])),
            bias_selected_null=float(np.mean([r["cert"][n]["sel_test"] - r["cert"][n]["true_sel"] for r in nul])),
            floor_k1=float(np.mean([r["cert"][n]["floor_k1"] for r in nul])), floor_suite=float(np.mean([r["cert"][n]["floor_suite"] for r in nul])),
            mean_mde80_selected_null=float(np.mean([r["cert"][n]["sel_mde"] for r in nul])))
        if n in ("150", "1425"):
            sh = [r["shuffled"][n]["verdict"] == "LEAK" for r in leak if n in r["shuffled"]]
            out["E4"][n]["shuffled_false_leak"] = float(np.mean(sh)) if sh else None
            out["E4"][n]["shuffled_n"] = len(sh)
    for n in ("150", "1425"):
        for name, sel in (("leak", leak), ("null", nul)):
            agg = sb.aggregate_truth([{int(K): v for K, v in r["truth_infl"][n].items()} for r in sel], n_boot=500, seed=0)
            out["E5"][f"{name}|{n}"] = {str(K): {q: agg[K][q] for q in ("bias_naive", "bias_valsel", "inflation", "selection_loss", "true_best")} for K in agg}
    for (ch, d), c in sorted(cells.items()):
        out["exploratory_circular"][f"{ch}|{d}"] = {n: dict(
            detect=float(np.mean([r["cert"][n]["verdict"] == "LEAK" for r in c])), true_best=float(np.mean([r["cert"][n]["true_best12"] for r in c])),
            sel_test=float(np.mean([r["cert"][n]["sel_test"] for r in c])), knn5_raw_acc=float(np.mean([r["oneliner"][n]["knn5_raw"]["acc"] for r in c])),
            lda_raw_acc=float(np.mean([r["oneliner"][n]["lda_raw_auto"]["acc"] for r in c])),
            suite_minus_knn5=float(np.mean([r["oneliner"][n]["knn5_raw"]["diff"] for r in c])), suite_minus_lda=float(np.mean([r["oneliner"][n]["lda_raw_auto"]["diff"] for r in c])))
            for n in ("150", "1425")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["real", "planted"])
    ap.add_argument("--within"), ap.add_argument("--budget"), ap.add_argument("--select"), ap.add_argument("--hmdb"), ap.add_argument("--jsonl")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.kind == "real":
        res = dict(E1=e1([os.path.join(a.within, "rescore_programme1.json")] + ([os.path.join(a.hmdb, "rescore_programme1.json")] if a.hmdb else [])),
                   E2=e2(os.path.join(a.select, "harness_selection_bias.json")), E3=e3(os.path.join(a.budget, "harness_evidence_budget.json")))
        json.dump(prov.add(res, __file__), open(os.path.join(a.out, "summary_real.json"), "w"), indent=1, sort_keys=True, default=float)
    else:
        res = e45(a.jsonl)
        json.dump(prov.add(res, __file__), open(os.path.join(a.out, "summary_planted.json"), "w"), indent=1, sort_keys=True, default=float)
    print(json.dumps(res, indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
