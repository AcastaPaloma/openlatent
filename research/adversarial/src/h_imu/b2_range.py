"""Arm B2 variation statistic (H.md 8): range of trial-level L over the AAE runs with a subject-bootstrap interval of the range.
python b2_range.py --dirs d1,d2,... --arms aae_re,aae_re_c1s1,aae_re_c2s0 --out OUT.json"""
import os, sys, glob, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import aggregate as A, datasets as D
ap = argparse.ArgumentParser(); ap.add_argument("--dirs", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
S = D.get_set("sealed"); A.NSUB = 24; st = A.Store(S); Wt = A.boot_counts(2000)
files = {}
for d in a.dirs.split(","):
    for fn in glob.glob(os.path.join(os.path.expanduser(d), "preds__*__*.npz")):
        _, arm, fam = os.path.basename(fn)[:-4].split("__"); files.setdefault(arm, []).append(fn)
res, boots = {}, []
for arm in a.arms.split(","):
    rec = st.arm_arrays(files[arm], ["S0", "S1", "S2"])
    names = A.MAIN_TAB + (A.MAIN_NN if any(f.endswith("__nn.npz") for f in files[arm]) else [])
    L, Lb, pts = A.stat_from(rec, names, "id", "t", Wt); U, Ub, _ = A.stat_from(rec, names, "act", "w", Wt)
    res[arm] = dict(L=L, ci=[float(x) for x in np.percentile(Lb, [2.5, 97.5])], U=U, U_ci=[float(x) for x in np.percentile(Ub, [2.5, 97.5])], per_attacker=pts); boots.append(Lb)
B = np.stack(boots); rng_b = B.max(0) - B.min(0); pt = [res[x]["L"] for x in res]
out = dict(arms=res, L_range=float(max(pt) - min(pt)), L_range_ci=[float(x) for x in np.percentile(rng_b, [2.5, 97.5])], stable_range_le_0p10=bool(max(pt) - min(pt) <= 0.10))
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: out[k] for k in ("L_range", "L_range_ci", "stable_range_le_0p10")}), {a_: (round(v["L"], 3), round(v["U"], 3)) for a_, v in res.items()})
