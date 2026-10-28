"""Replace the prevalence-based action-conditional floor in collected results with the random-score floor.

Runs started before ppv/train.py computed the floor by Monte Carlo report the mean within-class prevalence (57.5), which
sits below the expected AP of random scores on ~30-clip cells. The random-score floor depends only on the test labels:
VP-UCF101 61.3, VP-HMDB51 61.9 (200 draws per cell, seed 0). The old value is kept as privacy_cond_floor_prevalence.
"""

import glob
import json
import os
import sys

MC = {"ucf101_vp": 61.3, "hmdb51_vp": 61.9}

for p in glob.glob(os.path.join(sys.argv[1], "*", "result.json")):
    r = json.load(open(p))
    if r.get("dataset") in MC and "privacy_cond_floor" in r and r["privacy_cond_floor"] < 58 and "privacy_cond_floor_prevalence" not in r:
        r["privacy_cond_floor_prevalence"], r["privacy_cond_floor"] = r["privacy_cond_floor"], MC[r["dataset"]]
        with open(p, "w") as f:
            json.dump(r, f, indent=1)
