"""Validity gate G0 (T.md 3.3), METADATA ONLY (no image): gradient-boosted metadata attackers trained on A-role N2 pairs and scored on T and V pairs pooled,
patient-cluster bootstrap upper bound; all fields, device-only, age+sex+view only; plus the N0 / N1 reference lines.  Pass iff point <= 0.56 and upper <= 0.58 for the three
subsets (the border-only attacker is checked in stage 1).  Fallback "N2-strict" (tier 0-1 positives only) is reported alongside.
   python -m t_cxr.g0 --data DIR --pri DIR --salt-file FILE --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import meta, pairs, stats as S
from .meta import PRI, hid


def gate(d, tier_max=3, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pA, _ = pairs.build(d, "A", "N2", 2, "tr0", col="role"); pV, _ = pairs.build(d, "V", "N2", 1, "v0", col="role"); pT, rT = pairs.build(d, "T", "N2", 1, "te0", col="role")
    pA, pV, pT = (p[p.tier <= tier_max].reset_index(drop=True) for p in (pA, pV, pT))
    pVT = __import__("pandas").concat([pV, pT], ignore_index=True)
    out = {}
    Xtr, Xte = pairs.pair_features(pA, d), pairs.pair_features(pVT, d)
    cl = pVT.anchor_pid.map(lambda p: hid(p, "g0")).to_numpy()
    pb = S.PatientBootstrap(cl, B=1000, seed=0)
    for fs in ("all", "device", "demo"):
        cols = pairs.FEATURE_SETS[fs] or list(Xtr.columns)
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=seed).fit(Xtr[cols], pA.label)
        sc = m.predict_proba(Xte[cols])[:, 1]
        r = S.summarize(S.auc(pVT.label, sc), S.auc_boot(pVT.label.to_numpy(), sc, pb.codes(cl), pb))
        out[fs] = dict(auc=r["point"], lo=r["lo"], hi=r["hi"], n_pos=int((pVT.label == 1).sum()), passes=bool(r["point"] <= 0.56 and r["hi"] <= 0.58))
    out["n_triples"] = dict(A=int(len(pA) // 2), V=int(len(pV) // 2), T=int(len(pT) // 2)); out["T_report"] = rT
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--salt-file", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d0 = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt")
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in PRI], d0)
    salt = Path(a.salt_file).read_text().strip()
    res = {}
    for rot in (0, 1, 2):
        d = meta.assign_roles(d0, seen, salt, rot)
        res[f"rot{rot}"] = dict(roles=meta.role_counts(d), N2=gate(d, 3), N2_strict=gate(d, 1))
        g = res[f"rot{rot}"]["N2"]
        print(rot, {k: (round(v["auc"], 3), round(v["hi"], 3), v["passes"]) for k, v in g.items() if k in ("all", "device", "demo")}, res[f"rot{rot}"]["roles"]["T"], flush=True)
    res["passes_all_rotations"] = bool(all(res[f"rot{r}"]["N2"][k]["passes"] for r in (0, 1, 2) for k in ("all", "device", "demo")))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "g0.json").write_text(json.dumps(res, indent=1, default=float))
    print("G0 passes on all rotations:", res["passes_all_rotations"])


if __name__ == "__main__":
    main()
