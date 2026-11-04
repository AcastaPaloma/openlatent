"""Reference leakage: attackers that see only the TRUE action (z = a_t) predict the distractor targets.  For action-correlated
kinds this is the leakage any perfect latent action carries 'for free', so it must be subtracted before calling z leaky.
python leak_ref.py --out $OUT   (B_LEVEL=strong)"""
import os, json, argparse, torch
import lam as L, data as Dt, distractors as D

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./out"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
res = {}
for kind in ["indep", "corr_det", "corr_noisy", "corr_glued"]:
    sets = {s: L.GData(s, kind, None) for s in ("train", "val", "test")}
    Z, T, E = {}, {}, {}
    for s, ds in sets.items():
        e, t = ds.pairs(); T[s] = ds.targets(e, t); Z[s] = T[s]["a"]; E[s] = e
    r = {}
    for tn in ("n1", "n2", "n4"):
        Ytr, Yva, Yte = T["train"][tn], T["val"][tn], T["test"][tn]
        if Ytr.std(0).max() < 1e-6:
            continue
        (pl,), _ = L.linear_probe(Z["train"], Ytr, Z["val"], Yva, [Z["test"]])
        (pm,) = L.mlp_probe(Z["train"], Ytr, Z["val"], Yva, [Z["test"]])
        for an, pp in (("lin", pl), ("mlp", pm)):
            rd = 1 - ((pp - Yte) ** 2).sum(0) / ((Yte - Ytr.mean(0)) ** 2).sum(0).clamp_min(1e-12)
            r[f"{tn}_{an}_maxdim"] = float(max(0.0, rd.max().item()))
            r[f"{tn}_{an}_pooled"] = float(max(0.0, 1 - ((pp - Yte) ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum()))
    res[kind] = r
    print(kind, r, flush=True)
json.dump(res, open(os.path.join(a.out, "leak_ref.json"), "w"), indent=1)
