#!/usr/bin/env python
"""H2-015 selection null for the best-of-N `ae` configurations (pre-registered in results_notes/A.md, 'AUDIT FOLLOW-UP: PRE-REGISTRATION').

  python selection_null.py --runs DIR [DIR ...] --out null.md [--B 1000] [--draws 5000]

Statistic T_obs = max over the N `ae` configurations of  [metric(ae) - metric(best INDUCTIVE floor with L <= L_ae + 0.05)]  (held-class accuracy and, separately, kNN-retention-xc).
Null A (frontier permutation): each `ae` configuration is replaced by an inductive-floor point with |L - L_ae| <= 0.1 (nearest if none), compared with the REMAINING floors exactly
  as the AE was, max over the N replacements, `draws` times; p = P(T_null >= T_obs).
Null B (max-T bootstrap): T^b = max_c [(boot_c - boot_floor(c)) - (pt_c - pt_floor(c))], 1000 unit bootstraps; threshold = 95th percentile.
A best-of-N gain is 'above selection noise' only if T_obs exceeds the 95th percentile of null A AND of null B.
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_p2 import Cell, load, family, ci, KC, FLOORS   # noqa


def build(cell):
    floor = cell.floor()
    raw_c = cell.centre("raw", KC)
    held = "labelheld/linear/C" in cell.p(list(cell.by["raw"].values())[0]).files
    tab = {}
    for cfg in sorted(cell.by):
        if cfg.startswith("ctrl_"):
            continue
        if cell.pooled and cell.seeds(cfg) != cell.fold_order:
            continue
        wc = cell.centre(cfg, KC)
        L = (wc[0] - floor) / max(raw_c[0] - floor, 1e-6)
        ut = cell.label(cfg, "held", "linear", "C") if held else cell.label(cfg, "", "linear", "C")
        tab[cfg] = dict(L=L, ut=ut, ret=cell.ret(cfg, "knn_xc"))
    return tab


def stat(tab, cfgs, floors, key, tol=0.05, skip=None):
    """list of (cfg, delta point, floor name) for configs that have a floor comparator"""
    out = []
    for c in cfgs:
        refs = [f for f in floors if f != c and tab[f]["L"] <= tab[c]["L"] + tol]
        if not refs:
            continue
        b = max(refs, key=lambda f: tab[f][key][0])
        out.append((c, tab[c][key][0] - tab[b][key][0], b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--undithered", action="store_true", help="post hoc variant: drop the dithered codes from the floor pool of BOTH the observed statistic and null A (their retention is low by construction)")
    a = ap.parse_args()
    recs = load(a.runs)
    cells = sorted({(r["encoder"], r["dataset"], r.get("variant", "crossed")) for r in recs if r["cfg"] == "raw"})
    L = ["| cell | metric | N ae configs | T_obs (best-of-N Δ) | best config | null A: 95th pct / p | null B (max-T bootstrap): 95th pct | above selection noise (exceeds both)? |", "|---|---|---|---|---|---|---|---|"]
    rng = np.random.RandomState(7)
    for enc, ds, var in cells:
        cell = Cell(enc, ds, var, recs, a.runs, a.B)
        if "raw" not in cell.by or not any(c.startswith("ae__") for c in cell.by):
            continue
        tab = build(cell)
        aes = [c for c in tab if family(c) == "ae"]
        floors = [c for c in tab if family(c) in FLOORS and not (a.undithered and family(c) == "dither")]
        for key, name in (("ut", "held-class acc" if "holdout" in var else "label acc"), ("ret", "retention xc")):
            obs = stat(tab, aes, floors, key)
            if not obs:
                continue
            T_obs, c_best, _ = max((d, c, f) for c, d, f in obs)
            # null A
            Ls = np.array([tab[f]["L"] for f in floors])
            Tn = []
            for _ in range(a.draws):
                best = -9
                for c in aes:
                    near = [f for f, l in zip(floors, Ls) if abs(l - tab[c]["L"]) <= 0.1]
                    if not near:
                        near = [floors[int(np.argmin(np.abs(Ls - tab[c]["L"])))]]
                    f = near[rng.randint(len(near))]
                    refs = [g for g in floors if g != f and tab[g]["L"] <= tab[f]["L"] + 0.05]
                    if not refs:
                        continue
                    b = max(refs, key=lambda g: tab[g][key][0])
                    best = max(best, tab[f][key][0] - tab[b][key][0])
                Tn.append(best)
            Tn = np.array(Tn)
            pA = float((Tn >= T_obs).mean()); qA = float(np.percentile(Tn, 95))
            # null B
            Tb = np.full(a.B, -9.0)
            for c, d, f in obs:
                dist = (tab[c][key][1] - tab[f][key][1]) - d
                Tb = np.maximum(Tb, dist)
            qB = float(np.nanpercentile(Tb, 95))
            above = T_obs > qA and T_obs > qB
            L.append(f"| {enc} / {ds} / {var} | {name} | {len(obs)} | {T_obs:+.3f} | {c_best} | {qA:+.3f} / p={pA:.3f} | {qB:+.3f} | {'YES' if above else 'no (selection noise)'} |")
    text = "\n".join(L)
    Path(a.out).write_text(text) if a.out != "-" else print(text)


if __name__ == "__main__":
    main()
