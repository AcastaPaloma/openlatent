#!/usr/bin/env python
"""A3 'adapter trained' criterion: held-out-slide adversary accuracy vs the noise-latent floor on the same held-out slides.

  python a3_report.py --runs DIR [DIR ...] [--tol 0.05] [--last 6]
trained  iff  mean over folds of (mean of the last `last` rounds of the held-out balanced accuracy)  <=  mean over folds of the noise floor + tol.
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--last", type=int, default=6)
a = ap.parse_args()
rows = {}
for d in a.runs:
    for jl in Path(d).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "adv_heldout_acc_per_round" in r["extra"]:
                rows.setdefault((r["encoder"], r["dataset"], r["cfg"]), {})[r["seed"]] = r["extra"]
print("| cell | config | folds | slides A1/A2 | noise floor (held-out): mean +- sd | in-sample adv acc (A1), last-6 mean | held-out adv acc (A2), first round | held-out adv acc (A2), last-6 mean [min, max over folds] | trained (<= floor + %.2f)? |" % a.tol)
print("|---|---|---|---|---|---|---|---|---|")
for (enc, ds, cfg), v in sorted(rows.items()):
    fl = np.mean([x["noise_floor_heldout"] for x in v.values()])
    fsd = np.mean([x["noise_floor_heldout_sd"] for x in v.values()])
    ho = [np.mean(x["adv_heldout_acc_per_round"][-a.last:]) for x in v.values()]
    ins = [np.mean(x["adv_train_acc_per_round"][-a.last:]) for x in v.values()]
    first = [x["adv_heldout_acc_per_round"][0] for x in v.values()]
    n = next(iter(v.values()))
    ok = np.mean(ho) <= fl + a.tol
    print(f"| {enc} / {ds} | {cfg} | {len(v)} | {n['n_units_A1']}/{n['n_units_A2']} | {fl:.3f} +- {fsd:.3f} | {np.mean(ins):.3f} | {np.mean(first):.3f} | {np.mean(ho):.3f} [{min(ho):.3f}, {max(ho):.3f}] | {'YES' if ok else 'no (adversary did not converge)'} |")
