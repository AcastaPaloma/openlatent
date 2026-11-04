#!/usr/bin/env python
"""Compare Robustness Index results (ours, from official code on our embeddings) with the reference values shipped in the
PathoROB repo (phikonv2_clsmean, uni2h_clsmean) and with the README leaderboard."""
import json, sys
from pathlib import Path

LEADER = {  # README leaderboard (RI): tcga 2x2, camelyon, tolkach
    "phikon_clsmean": (0.623, 0.011, 0.795), "phikonv2_clsmean": (0.619, 0.019, 0.768), "kaiko_vitb8": (0.763, 0.147, 0.896)}
ours = Path(sys.argv[1])
ref = Path(sys.argv[2])
rows = []
for model in sorted(p.name for p in ours.iterdir() if p.is_dir() and p.name != "fig"):
    for ds, col in (("tcga", 0), ("camelyon", 1), ("tolkach_esca", 2)):
        f = next(iter((ours / model / ds).glob("*/results_summary.json")), None)
        if f is None:
            continue
        ri = json.loads(f.read_text())["robustness_index"]
        r = ref / model / ds / "-1_0" / "results_summary.json"
        ri_ref = json.loads(r.read_text())["robustness_index"] if r.exists() else None
        lb = LEADER.get(model, (None,) * 3)[col]
        rows.append((model, ds, ri, ri_ref, lb))
print("| encoder | dataset | RI (ours, official code) | RI (reference JSON in repo) | RI (README leaderboard) | |diff| vs ref |")
print("|---|---|---|---|---|---|")
for m, d, a, b, c in rows:
    print(f"| {m} | {d} | {a:.3f} | {'' if b is None else f'{b:.3f}'} | {'' if c is None else f'{c:.3f}'} | {'' if b is None else f'{abs(a-b):.3f}'} |")
