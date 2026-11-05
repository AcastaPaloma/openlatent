"""POST-HOC diagnostic for T5 (T.md 12.x, labelled post-hoc; not pre-registered; CPU only, reads saved scores, builds no image, makes no sealed query).
Is the F1+ test AUC carried by identity or by the way the pairs were built?  Re-builds the T3 test pairs of a rotation (deterministic) and reports the test AUC of the saved F1 (R4) and F1+ (R5) scores inside strata:
negative-match tier of the triple (tier 0 = exact age + same spacing, 1, >=2), long follow-up gap of the positive (>= 1 y), view of the partner image (PA vs AP) and the 14-finding status of the anchor (No Finding vs any finding).
   python -m t_cxr.r5_strata --release down_f4 --rot 0 --root ~/work/runs/t_w_f1p_down_r0 --r4root ~/work/runs/t_z_lad_down_f4_all ... --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import meta, pairs, stats as S
from .meta import PRI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--rot", type=int, required=True)
    ap.add_argument("--r5dir", required=True, help="dir with preds_f1_match_ens.json of the R5 run"); ap.add_argument("--r4dir", required=True, help="same for the R4 run")
    ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True); ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    d = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in PRI], d)
    salt = Path(a.salt_file).read_text().strip()
    d = meta.assign_roles(d, seen, salt, a.rot)
    pT, _ = pairs.build(d, "T", "N2", 1, "te0", col="role")
    Lm = meta.findings_matrix(d, csv)
    nofind = (Lm.sum(1) == 0)
    p4 = json.loads((Path(a.r4dir) / "preds_f1_match_ens.json").read_text()); p5 = json.loads((Path(a.r5dir) / "preds_f1_match_ens.json").read_text())
    y = np.asarray(p5["y"]); assert (np.asarray(p4["y"]) == y).all() and len(y) == len(pT) and (pT.label.to_numpy() == y).all(), "pair order differs from the saved scores"
    s4, s5, cl = np.asarray(p4["score_test"]), np.asarray(p5["score_test"]), np.asarray(p5["cluster"])
    pb = S.PatientBootstrap(np.unique(cl), B=a.B, seed=1)
    tri = pT.triple.to_numpy(); tier_by_tri = pT.groupby("triple").tier.max()
    tier = tier_by_tri.loc[tri].to_numpy()
    lg_by_tri = pT[pT.label == 1].groupby("triple").long_gap.max()
    long_gap = lg_by_tri.reindex(tri).fillna(0).to_numpy()
    pa_b = d["pa"].to_numpy()[pT.b.to_numpy()]
    nof = nofind[pT.a.to_numpy()]
    strata = {"all": np.ones(len(y), bool), "tier0": tier == 0, "tier1": tier == 1, "tier>=2": tier >= 2, "long_gap(>=1y)": long_gap == 1, "short_gap(<1y)": long_gap == 0,
              "partner PA": pa_b == 1, "partner AP": pa_b == 0, "anchor No Finding": nof, "anchor has finding": ~nof}
    res = dict(release=a.release, rot=a.rot, note="POST-HOC diagnostic; saved scores only", strata={})
    for nm, m in strata.items():
        if m.sum() < 40 or len(np.unique(y[m])) < 2:
            res["strata"][nm] = dict(n=int(m.sum()), skipped=True); continue
        row = dict(n=int(m.sum()), n_pos=int(y[m].sum()))
        for tag, s in (("R4_F1", s4), ("R5_F1plus", s5)):
            b = S.auc_boot(y[m], s[m], pb.codes(cl[m]), pb)
            row[tag] = dict(auc=S.auc(y[m], s[m]), lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5)))
        res["strata"][nm] = row
        print(f"{nm:22s} n={row['n']:5d}  F1(R4) {row['R4_F1']['auc']:.3f}  F1+(R5) {row['R5_F1plus']['auc']:.3f} [{row['R5_F1plus']['lo']:.3f},{row['R5_F1plus']['hi']:.3f}]", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"r5_strata_{a.release}_rot{a.rot}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
