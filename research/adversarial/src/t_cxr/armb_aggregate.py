"""T4b (T.md section 11 + A12): H3-013 arm B on the T3 partition - label-vector-matched negatives (design N3s).  Saved predictions + rebuilt pair tables only.
   python -m t_cxr.armb_aggregate --root ROOT --rots 0,1 --data D --pri D --salt-file F --exclude F --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, meta, pairs, stats as S

IMG = ["raw", "down_f4", "pcxr_mu0.01_s1_C2"]


def entries(root, rel, rots, suffix, with_f5=True):
    att = AG.load_attackers(root, rel, rots, suffix)
    if att is None:
        return None
    ent = {}
    if "f1_match_ens" in att:
        ent["f1"] = att["f1_match_ens"]
    f3 = [k for k in AG.F3 if k in att]
    ent["f3"] = att[max(f3, key=lambda k: att[k]["val_auc"])]
    if with_f5:
        for k in AG.F5:
            if k in att:
                ent[k] = att[k]
    return ent


def sub(ent, mask):
    return {k: dict(y=e["y"][mask], cluster=e["cluster"][mask], score=e["score"][mask], val_auc=e["val_auc"]) for k, e in ent.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rots", default="0,1"); ap.add_argument("--data", required=True); ap.add_argument("--pri", required=True)
    ap.add_argument("--salt-file", required=True); ap.add_argument("--exclude", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    rots = [int(x) for x in a.rots.split(",")]
    csv = f"{a.data}/Data_Entry_2017_v2020.csv"
    d0 = meta.load(csv, f"{a.data}/test_list.txt", f"{a.data}/train_val_list.txt", exclude_json=a.exclude)
    seen = meta.seen_patients([f"{a.pri}/{f}" for f in meta.PRI], d0); salt = Path(a.salt_file).read_text().strip()
    Lm = meta.findings_matrix(d0, csv).astype(np.int64); code = (Lm * (1 << np.arange(14))).sum(1)
    retained, n_pos_n2, n_pos_n3 = [], 0, 0
    for r in rots:
        d = meta.assign_roles(d0, seen, salt, r)
        p2, _ = pairs.build(d, "T", "N2", 1, "te0", col="role"); p3, _ = pairs.build(d, "T", "N3s", 1, "te0", col="role", labcode=code)
        pos3 = {frozenset((int(x), int(y))) for x, y in zip(p3[p3.label == 1].a, p3[p3.label == 1].b)}
        keep_trip = {t for t, g in p2.groupby("triple") if frozenset((int(g[g.label == 1].a.iloc[0]), int(g[g.label == 1].b.iloc[0]))) in pos3}
        retained.append(p2.triple.isin(keep_trip).to_numpy()); n_pos_n2 += int((p2.label == 1).sum()); n_pos_n3 += int((p3.label == 1).sum())
    ret = np.concatenate(retained)
    res = dict(rots=rots, n_pos_N2=n_pos_n2, n_pos_N3s=n_pos_n3, share_dropped=1 - n_pos_n3 / n_pos_n2, n2_retained_positive_pairs=int((ret.sum()) // 2), releases={})
    pb = None; boots = {}
    for rel in IMG + ["labels14"]:
        e3 = entries(a.root, rel, rots, "_n3s", with_f5=(rel != "labels14"))
        e2 = entries(a.root, rel, rots, "", with_f5=(rel != "labels14"))
        if e3 is None or e2 is None:
            continue
        if rel != "labels14":                                          # N2 worst case over the same attacker set as the arm-B runs (no F4)
            e2 = {k: v for k, v in e2.items() if k != "f4"}
        if pb is None:
            pb = S.PatientBootstrap(np.unique(np.concatenate([next(iter(e3.values()))["cluster"], next(iter(e2.values()))["cluster"]])), B=a.B, seed=0)     # universe = all N2 anchors (N3s drops some)
        def wc(ent):
            names = list(ent); pts = {n: S.auc(ent[n]["y"], ent[n]["score"]) for n in names}
            bs = np.stack([S.auc_boot(ent[n]["y"], ent[n]["score"], pb.codes(ent[n]["cluster"]), pb) for n in names])
            return dict(L=float(max(pts.values())), boot=np.nanmax(bs, axis=0), argmax=max(pts, key=pts.get), points=pts)
        w3 = wc(e3); w2r = wc(sub(e2, ret)); w2 = wc(e2)
        d = S.paired(w2r["boot"], w3["boot"], w2r["L"], w3["L"])
        boots[rel] = w3["boot"]
        res["releases"][rel] = dict(L_N3s=w3["L"], ci=AG.ci(w3["boot"]), ci_v2=AG.ci(w3["boot"], AG.CI_KAPPA), argmax=w3["argmax"], points_N3s={k: float(v) for k, v in w3["points"].items()},
                                    L_N2_retained=w2r["L"], ci_N2r=AG.ci(w2r["boot"]), L_N2_all=w2["L"], drop_N2r_minus_N3s=d["diff"], drop_ci=AG.ci(w2r["boot"] - w3["boot"]), drop_ci_v2=AG.ci(w2r["boot"] - w3["boot"], AG.CI_KAPPA), drop_mde80=d["mde80"])
        v = res["releases"][rel]
        if rel == "raw":
            v["reading"] = ("identity survives pathology matching" if (v["L_N3s"] >= 0.85 and v["drop_N2r_minus_N3s"] <= 0.04) else ("PARTLY" if v["L_N3s"] >= 0.80 else "the headline needs rewording"))
        if rel == "labels14":
            v["reading"] = "pathology persistence only" if v["L_N3s"] <= 0.58 else ("partly" if v["L_N3s"] <= 0.65 else "carries anatomy / habitus beyond the label")
    if "raw" in res["releases"] and "down_f4" in res["releases"]:
        g = res["releases"]["raw"]["L_N3s"] - res["releases"]["down_f4"]["L_N3s"]
        res["floor_gain_raw_minus_down_f4"] = dict(gap=g, ci=AG.ci(boots["raw"] - boots["down_f4"]), holds_ge_0_06=bool(g >= 0.06))
    if "pcxr_mu0.01_s1_C2" in res["releases"] and "down_f4" in res["releases"]:
        res["pcxr_minus_down_f4_N3s"] = dict(gap=res["releases"]["pcxr_mu0.01_s1_C2"]["L_N3s"] - res["releases"]["down_f4"]["L_N3s"], ci=AG.ci(boots["pcxr_mu0.01_s1_C2"] - boots["down_f4"]))
    # control: const at chance on N3s (rotation 0)
    c = AG.load_attackers(a.root, "const", rots[:1], "_n3s")
    if c:
        ctrl = []
        for k, e in AG.suite_entries(c).items():
            b = S.auc_boot(e["y"], e["score"], S.PatientBootstrap(np.unique(e["cluster"]), B=500, seed=0).codes(e["cluster"]), S.PatientBootstrap(np.unique(e["cluster"]), B=500, seed=0))
            n = len(e["y"]) // 2
            ctrl.append(dict(name=k, auc=S.auc(e["y"], e["score"]), tol=S.control_tolerance(float(np.nanstd(b, ddof=1)), n, n)))
        res["const_control"] = [dict(x, ok=bool(abs(x["auc"] - 0.5) <= max(x["tol"], 0.06 if x["name"].startswith("f1") else 0))) for x in ctrl]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "armb.json").write_text(json.dumps(res, indent=1, default=float))
    print({k: res[k] for k in ("n_pos_N2", "n_pos_N3s", "share_dropped")})
    for rel, v in res["releases"].items():
        print(rel, "N3s %.3f [%.3f,%.3f]" % (v["L_N3s"], *v["ci"]), "N2(retained) %.3f" % v["L_N2_retained"], "drop %.3f (MDE80 %.3f)" % (v["drop_N2r_minus_N3s"], v["drop_mde80"]), v.get("reading", ""))
    print(res.get("floor_gain_raw_minus_down_f4"), res.get("pcxr_minus_down_f4_N3s"), res.get("const_control"))


if __name__ == "__main__":
    main()
