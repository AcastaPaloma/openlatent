"""H3-013 arm A (T.md section 9): is the linkage partly findings-based?  CPU, saved T2 predictions only.
   python -m t_cxr.findings_control --root ~/work/t_agg_t2 --rots 0,1,2 --data D --pri D --salt-file F --exclude F --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate as AG, meta, pairs, stats as S
from .run_utility import findings_matrix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rots", default="0,1,2"); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--releases", default="raw,labels14,pcxr_mu0.01,down_f4")
    a = ap.parse_args()
    rots = [int(x) for x in a.rots.split(",")]
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    d0 = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d0); salt = Path(a.salt_file).read_text().strip()
    Lab = findings_matrix(d0, csv).astype(np.int64)
    code = (Lab * (1 << np.arange(14))).sum(1); NF = Lab.sum(1) == 0
    ys, nf_pair, lm_pair = [], [], []
    for r in rots:
        d = meta.assign_roles(d0, seen, salt, r)
        pT, _ = pairs.build(d, "T", "N2", 1, "te0", col="role")
        lm = {}
        for t, g in pT.groupby("triple"):
            pos = g[g.label == 1].iloc[0]; neg = g[g.label == 0].iloc[0]
            P, N = {int(pos.a), int(pos.b)}, {int(neg.a), int(neg.b)}
            i = (P & N).pop(); j = (P - {i}).pop(); k = (N - {i}).pop()
            lm[t] = bool(code[k] == code[j])
        ys.append(pT.label.to_numpy()); nf_pair.append(NF[pT.a.to_numpy()] & NF[pT.b.to_numpy()]); lm_pair.append(pT.triple.map(lm).to_numpy())
    y = np.concatenate(ys); nfp = np.concatenate(nf_pair); lmp = np.concatenate(lm_pair)
    res = dict(rots=rots, n_pairs=int(len(y)), n_triples=int(len(y) // 2), share_triples_label_matched=float(lmp[y == 1].mean()), n_pos_label_matched=int((lmp & (y == 1)).sum()), n_pos_label_mismatched=int((~lmp & (y == 1)).sum()),
               n_pairs_NF=int(nfp.sum()), n_pos_NF=int((nfp & (y == 1)).sum()), n_neg_NF=int((nfp & (y == 0)).sum()), releases={})
    pb = None
    for rel in a.releases.split(","):
        att = AG.load_attackers(a.root, rel, rots)
        if att is None:
            continue
        ent = AG.suite_entries(att) if rel != "labels14" else {"f3_best": att[max([k for k in AG.F3 if k in att], key=lambda k: att[k]["val_auc"])]}
        keys = ["f3_best"] + (["f1_match_ens"] if rel == "raw" else [])
        out = {}
        for key in keys:
            e = ent[key]
            assert (e["y"] == y).all(), "pair order mismatch"
            if pb is None:
                pb = S.PatientBootstrap(np.unique(e["cluster"]), B=a.B, seed=0)
            codes = pb.codes(e["cluster"])
            def sub(mask):
                b = S.auc_boot(y[mask], e["score"][mask], codes[mask], pb)
                return dict(auc=S.auc(y[mask], e["score"][mask]), ci=AG.ci(b), ci_v2=AG.ci(b, AG.CI_KAPPA), boot=b, n_pos=int((y[mask] == 1).sum()), n_neg=int((y[mask] == 0).sum()))
            q = {"all": sub(np.ones(len(y), bool)), "NF": sub(nfp), "LM": sub(lmp), "LX": sub(~lmp)}
            s = (q["all"]["auc"] - q["LM"]["auc"]) / max(q["all"]["auc"] - 0.5, 1e-9)
            dif = S.paired(q["LX"]["boot"], q["LM"]["boot"], q["LX"]["auc"], q["LM"]["auc"])
            res_q = {k: {a_: b_ for a_, b_ in v.items() if a_ != "boot"} for k, v in q.items()}
            res_q.update(share_findings_based=float(s), LX_minus_LM=dif)
            if rel == "labels14":
                lm_auc = q["LM"]["auc"]
                res_q["reading"] = "findings-only linkage" if lm_auc <= 0.55 else ("partly findings" if lm_auc <= 0.65 else "carries more than findings")
            elif rel == "raw":
                A, nf, lm_ = q["all"]["auc"], q["NF"]["auc"], q["LM"]["auc"]
                if s > 0.30 or nf < A - 0.08:
                    rd = "LARGELY findings-based"
                elif s > 0.10 or nf < A - 0.03:
                    rd = "PARTLY findings-based"
                else:
                    rd = "NOT findings-based"
                res_q["reading"] = rd
            out[key] = res_q
        res["releases"][rel] = out
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "findings_control.json").write_text(json.dumps(res, indent=1, default=float))
    print({k: res[k] for k in res if k != "releases"})
    for rel, o in res["releases"].items():
        for key, q in o.items():
            print(rel, key, {k: round(q[k]["auc"], 3) for k in ("all", "NF", "LM", "LX")}, "n_pos", {k: q[k]["n_pos"] for k in ("NF", "LM", "LX")}, "s=%.3f" % q["share_findings_based"], "LX-LM %.3f (MDE80 %.3f)" % (q["LX_minus_LM"]["diff"], q["LX_minus_LM"]["mde80"]), q.get("reading", ""))


if __name__ == "__main__":
    main()
