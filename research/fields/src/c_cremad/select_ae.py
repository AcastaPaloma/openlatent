"""Pre-registered selection of the AE configuration on VALIDATION actors only (results_notes/C.md section 2c).
  python select_ae.py RUNS_ROOT PREFIX      e.g. PREFIX=c_ae_  (dirs c_ae_<cfg>_s<seed>)
U_HE_val = recall of DIS+FEA (best six-way probe fitted on train actors); ID_val = worst-attacker identity T1 on the validation actors.
Eligible if ID_val <= 0.10; pick the eligible config with the highest U_HE_val, else the one with the lowest ID_val."""
import os, re, sys
from collections import defaultdict
import numpy as np
import aggregate as ag

root, prefix = os.path.expanduser(sys.argv[1]), sys.argv[2]
groups = defaultdict(list)
for d in sorted(os.listdir(root)):
    if d.startswith(prefix):
        groups[re.sub(r"_s\d+$", "", d)].append(os.path.join(root, d))
rows = []
for base, dirs in groups.items():
    runs = [r for d in dirs for r in ag.load(d)]
    if not runs or "emo_val" not in runs[0]:
        continue
    he, idv = [], []
    for r in runs:
        best = -1
        for k, conf in r["emo_val"].items():
            C = np.zeros((6, 6))
            for a, m in conf.items():
                C += np.array(m)
            rec = np.diag(C) / np.maximum(C.sum(1), 1)
            best = max(best, rec[[1, 2]].mean())
        he.append(best)
        ids = []
        for k in ag.kinds_present(r, "id"):
            c = n = 0
            for a, (cc, nn) in r["id"][k]["T1"]["val"].items():
                c += cc; n += nn
            ids.append(c / max(n, 1))
        idv.append(max(ids))
    rows.append((base, len(runs), float(np.mean(he)), float(np.mean(idv))))
elig = [r for r in rows if r[3] <= 0.10]
pick = max(elig, key=lambda r: (r[2], -r[3])) if elig else min(rows, key=lambda r: r[3])
print("config                      seeds  U_HE_val  ID_val(T1, worst attacker)  eligible")
for r in sorted(rows):
    print("%-27s %5d  %.3f     %.3f  %s%s" % (r[0], r[1], r[2], r[3], r[3] <= 0.10, "   <== selected" if r is pick else ""))
