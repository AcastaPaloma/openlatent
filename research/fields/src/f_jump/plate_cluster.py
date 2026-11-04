"""Plate-cluster sensitivity for the headline JK3 difference U(R) - U(F*) (and U(F*) - U(raw)): query-weighted mean AP difference,
bootstrap over PLATES of the H set (the compound bootstrap of aggregate.py ignores that plates share negatives).  S5 only (per-query AP is stored for S5).
  python plate_cluster.py --json results/f_jump/s5.json --dirs d0,d1,d2 --out results/f_jump/s5_plate_cluster.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--dirs", required=True)
ap.add_argument("--salts", default="s0,s1,s2")
ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=1000)
a = ap.parse_args()
J = json.load(open(a.json))
rng = np.random.default_rng(7)
out = {}
for d, s in zip(a.dirs.split(","), a.salts.split(",")):
    z = np.load(Path(d) / f"preds_{s}.npz")
    r = J["splits"][s]
    R, F = r["JK3"]["R"], r["JK3"]["F_star"]
    pl = z["plate|H"]
    ids, inv = np.unique(pl, return_inverse=True)
    res = {}
    for name, (x, y) in {f"U({R})-U({F})": (R, F), f"U({F})-U(raw)": (F, "raw"), f"U({R})-U(raw)": (R, "raw")}.items():
        ax, ay = z[f"apq|H|{x}"].astype(np.float64), z[f"apq|H|{y}"].astype(np.float64)
        ok = ~np.isnan(ax) & ~np.isnan(ay)
        dsum = np.bincount(inv[ok], weights=(ax - ay)[ok], minlength=len(ids))
        n = np.bincount(inv[ok], minlength=len(ids)).astype(np.float64)
        pt = dsum.sum() / n.sum()
        bs = []
        for _ in range(a.B):
            w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
            bs.append((w @ dsum) / (w @ n))
        res[name] = dict(point=float(pt), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_plates=int(len(ids)), n_queries=int(n.sum()))
    out[s] = res
    print(s, json.dumps(res))
Path(a.out).write_text(json.dumps(out, indent=1))
