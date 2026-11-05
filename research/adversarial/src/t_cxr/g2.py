"""Repaired validity design G2 (T.md 7.1), evaluated on the saved predictions of rotations 1 and 2 (pooled).  No training here.
 (1) raw worst-case >= 0.65 and >= metadata floor + 0.03   (2) raw - core_w24 <= 0.05 (point; paired CI and MDE80 printed)
 (3) stratum DIFF (positive pair with different pixel spacing sx): raw worst-case >= 0.65 and its 95% lower bound > the DIFF-stratum metadata AUC + 0.03
 (4) const and shuffled-label controls (F1, F3a, F5 as written in 3.5) within max(1.96 SE, MDE of a single AUC at chance).   rim_w24 is descriptive.
   python -m t_cxr.g2 --root DIR --rots 1,2 --data D --pri D --salt-file F --exclude F --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, meta, pairs, stats as S


def subset(att, mask):
    out = {}
    for k, e in att.items():
        out[k] = dict(y=e["y"][mask], cluster=e["cluster"][mask], score=e["score"][mask], val_auc=e["val_auc"])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rots", default="1,2"); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--f1-shuf-tol", type=float, default=0.0, help="T3 (D4): tolerance for the shuffled-label family-F1 (Siamese) control, calibrated from the T2 design offset")
    a = ap.parse_args()
    rots = [int(x) for x in a.rots.split(",")]
    d0 = meta.load(f"{a.data}/Data_Entry_2017_v2020.csv", f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d0); salt = Path(a.salt_file).read_text().strip()
    from sklearn.ensemble import HistGradientBoostingClassifier
    diff_mask, meta_scores, ys = [], [], []
    for r in rots:
        d = meta.assign_roles(d0, seen, salt, r)
        pA, _ = pairs.build(d, "A", "N2", 2, "tr0", col="role"); pT, _ = pairs.build(d, "T", "N2", 1, "te0", col="role")
        X, XA = pairs.pair_features(pT, d), pairs.pair_features(pA, d)
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=0).fit(XA, pA.label)
        meta_scores.append(m.predict_proba(X)[:, 1])
        ys.append(pT.label.to_numpy())
        pos = (pT.label.to_numpy() == 1) & (X["eqsx"].to_numpy() == 0)                     # positive with DIFFERENT sx
        diff_mask.append(np.isin(pT.triple.to_numpy(), pT.triple.to_numpy()[pos]))
    diff = np.concatenate(diff_mask); meta_s = np.concatenate(meta_scores)
    A_raw = AG.load_attackers(a.root, "raw", rots); A_core = AG.load_attackers(a.root, "core_w24", rots); A_rim = AG.load_attackers(a.root, "rim_w24", rots)   # rim is optional (descriptive; not run in T3)
    ycat = A_raw[next(iter(A_raw))]["y"]; assert len(ycat) == len(diff) and (ycat == np.concatenate(ys)).all(), "pair order / label mismatch between saved predictions and rebuilt pairs"
    cl = A_raw[next(iter(A_raw))]["cluster"]
    pb = S.PatientBootstrap(np.unique(cl), B=a.B, seed=0)
    wc_raw, wc_core = (AG.worst_case(AG.suite_entries(x), pb) for x in (A_raw, A_core))
    wc_rim = AG.worst_case(AG.suite_entries(A_rim), pb) if A_rim else dict(L=float("nan"), L_boot=np.full(pb.B, np.nan))
    meta_auc = S.auc(ycat, meta_s); meta_ci = S.summarize(meta_auc, S.auc_boot(ycat, meta_s, pb.codes(cl), pb))
    dd = S.paired(wc_raw["L_boot"], wc_core["L_boot"], wc_raw["L"], wc_core["L"])
    # stratified
    res = dict(rots=rots, n_triples=int(len(ycat) // 2), raw=dict(L=wc_raw["L"], ci=AG.ci(wc_raw["L_boot"]), ci_v2=AG.ci(wc_raw["L_boot"], AG.CI_KAPPA), worst=wc_raw["argmax"], points=wc_raw["points"]),
               core_w24=dict(L=wc_core["L"], ci=AG.ci(wc_core["L_boot"])), rim_w24_descriptive=dict(L=wc_rim["L"], ci=AG.ci(wc_rim["L_boot"])), metadata_gbm=dict(auc=meta_auc, ci=[meta_ci["lo"], meta_ci["hi"]]),
               raw_minus_core=dd)
    strata = {}
    for nm, mk in (("DIFF_sx", diff), ("SAME_sx", ~diff)):
        e_raw = AG.suite_entries(subset(A_raw, mk))
        w = AG.worst_case(e_raw, pb)
        ms = S.summarize(S.auc(ycat[mk], meta_s[mk]), S.auc_boot(ycat[mk], meta_s[mk], pb.codes(cl[mk]), pb))
        lo = AG.ci(w["L_boot"])[0]
        strata[nm] = dict(n_pos=int((ycat[mk] == 1).sum()), L=w["L"], ci=AG.ci(w["L_boot"]), ci_v2=AG.ci(w["L_boot"], AG.CI_KAPPA), worst=w["argmax"], metadata_auc=ms["point"], metadata_ci=[ms["lo"], ms["hi"]], pass_lower_bound=bool(lo > ms["point"] + 0.03))
    res["strata"] = strata
    # controls as written: const (all attackers), shuffled-label F1, F3a, F5
    ctrl = []
    for nm, suffix, names in (("const", "", None), ("raw", "_shuf", ("f1_match_ens", "f3a_logreg", "f5a", "f5b")), ("raw", "_shuf2", ("f1_match_ens", "f3a_logreg", "f5a", "f5b"))):
        att = AG.load_attackers(a.root, nm, rots, suffix)
        if att is None and nm == "const":                       # T3 lite plan: const on rotation 0 only -> its pairs are a subset; evaluated on their own
            att = AG.load_attackers(a.root, nm, rots[:1], suffix)
        if att is None:
            continue
        ent = AG.suite_entries(att) if names is None else {k: att[k] for k in names if k in att}
        for k, e in ent.items():
            b = S.auc_boot(e["y"], e["score"], pb.codes(e["cluster"]), pb)
            auc = S.auc(e["y"], e["score"]); se = float(np.nanstd(b, ddof=1)); tol = S.control_tolerance(se, int(len(e["y"]) // 2), int(len(e["y"]) // 2))
            if a.f1_shuf_tol and suffix in ("_shuf", "_shuf2") and k.startswith("f1"):
                tol = max(tol, a.f1_shuf_tol)
            ctrl.append(dict(name=f"{nm}{suffix}:{k}", auc=auc, se=se, tol=tol, ok=bool(abs(auc - 0.5) <= tol)))
    res["controls"] = ctrl
    c1 = bool(wc_raw["L"] >= 0.65 and wc_raw["L"] >= meta_auc + 0.03)
    c2 = bool(dd["diff"] <= 0.05)
    c3 = bool(strata["DIFF_sx"]["L"] >= 0.65 and strata["DIFF_sx"]["pass_lower_bound"])
    c4 = bool(ctrl and all(c["ok"] for c in ctrl))
    res["checks"] = dict(c1_raw_over_metadata=c1, c2_interior_holds_signal=c2, c3_diff_device_stratum=c3, c4_controls=c4)
    res["G2_passes"] = bool(c1 and c2 and c3 and c4)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "g2.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("n_triples", "checks", "G2_passes")}, default=float))
    print("raw", round(res["raw"]["L"], 3), res["raw"]["ci"], "core", round(res["core_w24"]["L"], 3), "diff", {k: round(v, 3) for k, v in dd.items()}, "meta", round(meta_auc, 3))
    print({k: (round(v["L"], 3), v["n_pos"], round(v["metadata_auc"], 3)) for k, v in strata.items()})
    print([(c["name"], round(c["auc"], 3), c["ok"]) for c in ctrl])


if __name__ == "__main__":
    main()
