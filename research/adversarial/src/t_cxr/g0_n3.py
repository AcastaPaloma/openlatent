"""T4b controls first (T.md section 11), CPU, metadata and TRUE labels only, no image: on the T3 partition (rotations 0, 1)
 (i) metadata GBM on N3 pairs (A-trained, scored on T+V pooled); (ii) LABELS-ONLY GBM (Hamming distance of the two true 14-finding vectors, shared findings, both-No-Finding,
 numbers of findings) on N2 and on N3: must be <= 0.56 on N3 for N3 to count as closing the pathology channel; (iii) share of positives dropped by the N3 rule.
   python -m t_cxr.g0_n3 --data D --pri D --salt-file F --exclude F --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import meta, pairs, stats as S


def popcount(x):
    x = x.astype(np.int64); c = np.zeros(len(x), dtype=np.int64)
    for b in range(14):
        c += (x >> b) & 1
    return c


def label_features(p, code):
    a, b = code[p.a.to_numpy()], code[p.b.to_numpy()]
    return pd.DataFrame(dict(ham=popcount(a ^ b), shared=popcount(a & b), same=(a == b).astype(float), nf_both=((a == 0) & (b == 0)).astype(float), n_a=popcount(a), n_b=popcount(b)))


def gbm_auc(Xtr, ytr, Xte, yte, cl, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
    sc = m.predict_proba(Xte)[:, 1]
    pb = S.PatientBootstrap(cl, B=1000, seed=0)
    r = S.summarize(S.auc(yte, sc), S.auc_boot(np.asarray(yte), sc, pb.codes(cl), pb))
    return dict(auc=r["point"], lo=r["lo"], hi=r["hi"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rots", default="0,1")
    a = ap.parse_args()
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    d0 = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d0); salt = Path(a.salt_file).read_text().strip()
    Lm = meta.findings_matrix(d0, csv).astype(np.int64); code = (Lm * (1 << np.arange(14))).sum(1)
    res = {}
    for r in [int(x) for x in a.rots.split(",")]:
        d = meta.assign_roles(d0, seen, salt, r)
        out = {}
        for design in ("N2", "N3", "N3s"):
            kw = dict(labcode=code) if design in ("N3", "N3s") else {}
            pA, rA = pairs.build(d, "A", design, 2, "tr0", col="role", **kw); pV, rV = pairs.build(d, "V", design, 1, "v0", col="role", **kw); pT, rT = pairs.build(d, "T", design, 1, "te0", col="role", **kw)
            pVT = pd.concat([pV, pT], ignore_index=True)
            cl = pVT.anchor_pid.map(lambda p: meta.hid(p, "g0n3")).to_numpy()
            XA, XV = pairs.pair_features(pA, d), pairs.pair_features(pVT, d)
            out[design] = dict(metadata_gbm=gbm_auc(XA, pA.label, XV, pVT.label.to_numpy(), cl),
                               labels_only_gbm=gbm_auc(label_features(pA, code), pA.label, label_features(pVT, code), pVT.label.to_numpy(), cl),
                               T=dict(kept=rT["kept"], dropped=rT["dropped"], requested=rT["positives_requested"], tier_hist=rT["tier_hist"]), A=dict(kept=rA["kept"], dropped=rA["dropped"], requested=rA["positives_requested"]),
                               V=dict(kept=rV["kept"], dropped=rV["dropped"], requested=rV["positives_requested"]))
        for dz in ("N3", "N3s"):
            n3 = out[dz]
            out[dz + "_valid"] = bool(n3["metadata_gbm"]["auc"] <= 0.56 and n3["metadata_gbm"]["hi"] <= 0.58 and n3["labels_only_gbm"]["auc"] <= 0.56)
            out[dz + "_T_dropped_share"] = n3["T"]["dropped"] / max(n3["T"]["requested"], 1)
        res[f"rot{r}"] = out
        print(r, {dz: (round(out[dz]["metadata_gbm"]["auc"], 3), round(out[dz]["labels_only_gbm"]["auc"], 3), out[dz]["T"]["kept"], out[dz]["T"]["dropped"]) for dz in ("N2", "N3", "N3s")}, "valid N3/N3s:", out["N3_valid"], out["N3s_valid"], flush=True)
    res["N3_valid_all_rotations"] = bool(all(res[k]["N3_valid"] for k in res if k.startswith("rot")))
    res["N3s_valid_all_rotations"] = bool(all(res[k]["N3s_valid"] for k in res if k.startswith("rot")))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "g0_n3.json").write_text(json.dumps(res, indent=1, default=float))
    print("N3 valid on all rotations:", res["N3_valid_all_rotations"], "| N3s:", res["N3s_valid_all_rotations"])


if __name__ == "__main__":
    main()
