"""Stage-1 diagnostics from the SAVED predictions only (no training, no image): (a) the pre-registered controls as written in T.md 3.5 (shuffled-label F1, F3a, F5),
(b) the rim_w24 border-only attacker under the G0 fallback N2-strict (tier 0-1 positives only), (c) exploratory strata of the sealed rot-0 pairs (age gap, follow-up gap, view mix,
dimension mix) for raw / core_w24 / rim_w24, each with a patient-cluster bootstrap.  Output stage1_diag.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import meta, pairs, stats as S


def load_preds(root, rel, name, suffix=""):
    j = json.loads((Path(root) / f"{rel}_rot0{suffix}" / f"preds_{name}.json").read_text())
    return np.array(j["y"]), np.array(j["score_test"]), np.array(j["cluster"])


def ci_auc(y, s, cl, B=2000):
    pb = S.PatientBootstrap(cl, B=B, seed=0)
    r = S.summarize(S.auc(y, s), S.auc_boot(y, s, pb.codes(cl), pb))
    return dict(auc=r["point"], lo=r["lo"], hi=r["hi"], n_pos=int(y.sum()), n_neg=int((1 - y).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d)
    salt = Path(a.salt_file).read_text().strip()
    d = meta.assign_roles(d, seen, salt, 0)
    pT, rT = pairs.build(d, "T", "N2", 1, "te0", col="role")
    out = dict(T_report=rT)
    # sanity: saved pair order == rebuilt pair order
    y0, s0, c0 = load_preds(a.root, "raw", "f1_match_ens")
    assert len(y0) == len(pT) and (y0 == pT.label.to_numpy()).all(), "pair order mismatch"
    cl_ok = np.array([meta.hid(p, salt) for p in pT.anchor_pid]) == c0
    out["pair_order_check"] = bool(cl_ok.all())
    # (a) shuffled-label controls exactly as written in T.md 3.5
    ctrl = {}
    for nm in ("f1_match_ens", "f3a_logreg", "f3b_mlp", "f5a", "f5b", "f3c_cosine", "f3c_l2", "f3c_canberra"):
        try:
            y, s, c = load_preds(a.root, "raw", nm, "_shuf")
        except FileNotFoundError:
            continue
        ctrl[nm] = ci_auc(y, s, c, 1000)
    out["raw_shuffled_controls"] = ctrl
    # (b) rim_w24 under N2-strict
    tier = pT.tier.to_numpy()
    strict = {}
    for nm in ("f1_match_ens", "f3a_logreg", "f3b_mlp", "f3c_cosine", "f3c_l2", "f3c_canberra", "f4_match", "f4_anon2"):
        try:
            y, s, c = load_preds(a.root, "rim_w24", nm)
        except FileNotFoundError:
            continue
        r = {"all": ci_auc(y, s, c, 1000)}
        for lab, m in (("tier<=1", tier <= 1), ("tier==0", tier == 0)):
            r[lab] = ci_auc(y[m], s[m], c[m], 1000)
        strict[nm] = r
    out["rim_w24_by_attacker_and_tier"] = strict
    # (c) exploratory strata for raw / core / rim (worst attackers)
    X = pairs.pair_features(pT, d)
    strata = {"long_gap(age gap>=1)": pT.long_gap.to_numpy() == 1, "age gap 0": pT.long_gap.to_numpy() == 0, "fu_gap==1": pT.fu_gap.to_numpy() == 1, "fu_gap 2-3": (pT.fu_gap.to_numpy() >= 2) & (pT.fu_gap.to_numpy() <= 3),
              "fu_gap>=4": pT.fu_gap.to_numpy() >= 4, "same view": X.same_view.to_numpy() == 1, "view mix PA/AP": X.same_view.to_numpy() == 0, "dims equal (W,H)": (X.eqW.to_numpy() == 1) & (X.eqH.to_numpy() == 1), "dims differ": ~((X.eqW.to_numpy() == 1) & (X.eqH.to_numpy() == 1))}
    st = {}
    # strata are defined by TRIPLE (positive and its negative share the anchor and the stratum of the positive)
    trip_stratum = {k: np.isin(pT.triple.to_numpy(), pT.triple.to_numpy()[(pT.label.to_numpy() == 1) & m]) for k, m in strata.items()}
    for rel, nm in (("raw", "f3c_canberra"), ("raw", "f1_match_ens"), ("core_w24", "f1_match_ens"), ("rim_w24", "f1_match_ens")):
        y, s, c = load_preds(a.root, rel, nm)
        st[f"{rel}:{nm}"] = {k: ci_auc(y[m], s[m], c[m], 500) for k, m in trip_stratum.items() if m.sum() > 100}
    out["strata_exploratory"] = st
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "stage1_diag.json").write_text(json.dumps(out, indent=1, default=float))
    print("pair order check:", out["pair_order_check"])
    print("raw shuffled controls:", {k: round(v["auc"], 3) for k, v in ctrl.items()})
    print("rim by attacker/tier:", {k: {t: round(v[t]["auc"], 3) for t in v} for k, v in strict.items()})
    for k, v in st.items():
        print(k, {s_: (round(x["auc"], 3), x["n_pos"]) for s_, x in v.items()})


if __name__ == "__main__":
    main()
