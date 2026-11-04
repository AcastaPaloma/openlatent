#!/usr/bin/env python
"""F-02 adapter-trained check: in-sample accuracy of the retrained adversary after the last round, per `ae` configuration and fold.

  python f02_report.py --runs DIR [DIR ...] [--chance 0.5] [--tol 0.05]
'Adapter trained' = mean over folds of the LAST-round adversary accuracy <= chance + tol.  (The adversary is re-initialised every round and trained on the current
released sample of the adapter-train units A, i.e. this is the in-sample accuracy the adversary reaches against the final adapter.)
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--chance", type=float, default=0.5)
ap.add_argument("--tol", type=float, default=0.05)
a = ap.parse_args()
rows = {}
for d in a.runs:
    for jl in Path(d).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["method"] != "ae" or "adv_train_acc_per_round" not in r["extra"] or not r["extra"]["adv_train_acc_per_round"]:
                continue
            rows.setdefault(r["cfg"], {})[r["seed"]] = r["extra"]["adv_train_acc_per_round"]
print("| config | folds | rounds | first-round adv acc (mean) | last-round adv acc: mean [min, max over folds] | max over rounds | adapter trained (last-round mean <= chance + %.2f)? |" % a.tol)
print("|---|---|---|---|---|---|---|")
for cfg in sorted(rows, key=lambda c: (float(c.split("lam")[1].split("_")[0]), c)):
    v = rows[cfg]
    last = [x[-1] for x in v.values()]
    first = [x[0] for x in v.values()]
    mx = [max(x) for x in v.values()]
    ok = np.mean(last) <= a.chance + a.tol
    print(f"| {cfg} | {len(v)} | {len(next(iter(v.values())))} | {np.mean(first):.3f} | {np.mean(last):.3f} [{min(last):.3f}, {max(last):.3f}] | {np.mean(mx):.3f} | {'YES' if ok else 'no (adversary did not converge)'} |")
