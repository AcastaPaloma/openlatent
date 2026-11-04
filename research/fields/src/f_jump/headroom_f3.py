"""F3 headroom statement (computed from the existing S5 and F2 outputs only; no F3 number).  For each split: U_H of the F2 V-selected adapter (the supervised ceiling),
and of the best label-free floor / best floor of the full envelope at the adapter's leakage (V-selected at L_V <= L_V(a*) + 0.05).
  python headroom_f3.py --merged /tmp/f2merge --salts s0,s1,s2
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as AG  # noqa: E402

LF = re.compile(r"^(raw|pca\d+|rp\d+|dpca\d+_r[\d.]+|noise_r[\d.]+)$")
ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True)
ap.add_argument("--salts", default="s0,s1,s2")
a = ap.parse_args()
out = {}
for i, s in enumerate(a.salts.split(",")):
    sp = AG.Split(Path(a.merged) / f"merged_{s}", s, 50, 300 + i, True, "ldaT2")
    adapters = [x for x in sp.arms if sp.fam[x] == "adapter" and not x.endswith(("_r0.5", "_r1"))]
    astar = max(adapters, key=lambda x: sp.U("V", x))
    env = [x for x in sp.names(AG.FAM_IND) if sp.has_att(x)]
    lf = [x for x in env if LF.match(x)]
    La = sp.L("V", astar)
    def best(pool):
        c = [x for x in pool if sp.L("V", x) <= La + 0.05]
        b = max(c, key=lambda x: sp.U("V", x))
        return b, sp.U("H", b), sp.L("H", b)
    bl, bf = best(lf), best(env)
    out[s] = dict(ceiling_arm=astar, ceiling_U_H=sp.U("H", astar), ceiling_L_H=sp.L("H", astar), best_label_free_floor=bl[0], lf_U_H=bl[1], lf_L_H=bl[2],
                  best_envelope_floor=bf[0], env_U_H=bf[1], env_L_H=bf[2], headroom_vs_label_free=sp.U("H", astar) - bl[1], headroom_vs_envelope=sp.U("H", astar) - bf[1])
    print(s, json.dumps(out[s]))
print(json.dumps(out))
