"""Aggregation and the machine-checkable verdicts of T.md 3.8 (FOCUS A2).  Reads the JSON outputs of run_suite.py / run_utility.py, never an image.

   python -m t_cxr.aggregate --root DIR --g0 g0.json --rots 0[,1,2] --out DIR --prereg-commit acdeb7a [--B 2000]
Layout under --root:  <release>_rot<r>/{summary.json,preds_*.json}, <release>_rot<r>_shuf/..., utility_<release>_rot<r>.json, labels14_rot<r>/{summary.json,preds_*.json}.
Shared patient resamples: ONE PatientBootstrap over the T-patient universe (hashed ids) is used for every pair-level and image-level statistic, so worst-cases, utilities and
envelope differences are taken inside the same resample.  Rotations are pooled by concatenation (patients are disjoint across rotations).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import ledger, stats as S

V1 = ["f1_mismatch_ens", "f1_match_ens", "f1_anon2_ens", "f2_match_ens"]
F3 = ["f3a_logreg", "f3b_mlp", "f3c_cosine", "f3c_l2", "f3c_canberra"]
F4 = ["f4_match", "f4_anon2"]
F5 = ["f5a", "f5b"]
LEARNED_PREFIX = ("pcx_", "pcxr_")
CI_KAPPA = 1.10          # S evaluator v2 (S.md v1.4): the QUOTED interval = the percentile interval widened by this factor about its midpoint; we print BOTH (v1 = plain percentile, v2 = widened)


def ci(boot, kappa=1.0):
    lo, hi = float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return [mid - kappa * half, mid + kappa * half]


def _load_json(p):
    return json.loads(Path(p).read_text())


def load_attackers(root, release, rots, suffix=""):
    """pooled over rotations: name -> dict(y, cluster, score, val_auc (mean over rotations), y_val, score_val, cluster_val)."""
    out = {}
    for r in rots:
        d = Path(root) / f"{release}_rot{r}{suffix}"
        if not d.exists():
            return None
        for f in d.glob("preds_*.json"):
            j = _load_json(f); nm = f.stem[len("preds_"):]
            e = out.setdefault(nm, dict(y=[], cluster=[], score=[], y_val=[], score_val=[], cluster_val=[]))
            e["y"] += j["y"]; e["cluster"] += j["cluster"]; e["score"] += j["score_test"]
            e["y_val"] += j["y_val"]; e["score_val"] += j["score_val"]; e["cluster_val"] += j["cluster_val"]
    for e in out.values():
        for k in e:
            e[k] = np.asarray(e[k])
        e["val_auc"] = S.auc(e["y_val"], e["score_val"])
    return out


def suite_entries(att, include_v4=False):
    """declared worst-case suite (T.md 3.5): every V1 present, best-on-V of the F3 variants, F5a / F5b, best-on-V of F4; V4 (f6) only when asked."""
    ent = {k: att[k] for k in V1 + F5 if k in att}
    for grp, nm in ((F3, "f3_best"), (F4, "f4_best")):
        have = [k for k in grp if k in att]
        if have:
            ent[nm] = att[max(have, key=lambda k: att[k]["val_auc"])]
    if include_v4 and "f6" in att:
        ent["f6"] = att["f6"]
    return ent


def worst_case(ent, pb):
    pts, boots = {}, {}
    for nm, e in ent.items():
        pts[nm] = S.auc(e["y"], e["score"])
        boots[nm] = S.auc_boot(e["y"], e["score"], pb.codes(e["cluster"]), pb)
    names = list(ent)
    P = np.array([pts[n] for n in names]); Bm = np.stack([boots[n] for n in names])
    k = int(np.argmax(P))
    return dict(L=float(P.max()), L_boot=np.nanmax(Bm, axis=0), argmax=names[k], points=pts, val_L=float(max(ent[n]["val_auc"] for n in names)),
                selected_by_val=max(names, key=lambda n: ent[n]["val_auc"]), selected_by_val_test_auc=pts[max(names, key=lambda n: ent[n]["val_auc"])], boots=boots)


def load_utility(root, release, rots):
    T = dict(patient=[], Y=[], P=[], P_fixed=[]); V = dict(Y=[], P=[])
    for r in rots:
        p = Path(root) / f"utility_{release}_rot{r}.json"
        if not p.exists():
            return None
        j = _load_json(p)
        for k in T:
            T[k] += j["T"][k]
        V["Y"] += j["V"]["Y"]; V["P"] += j["V"]["P"]
    return dict(T={k: np.asarray(v) for k, v in T.items()}, V={k: np.asarray(v) for k, v in V.items()})


def mean_auc_boot(Y, P, codes, pb):
    per = []
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < len(Y):
            per.append(S.auc_boot(Y[:, k], P[:, k], codes, pb))
    return np.nanmean(np.stack(per), axis=0)


def utility_stats(u, pb, key="P"):
    codes = pb.codes(u["T"]["patient"])
    pts = [S.auc(u["T"]["Y"][:, k], u["T"][key][:, k]) for k in range(u["T"]["Y"].shape[1]) if 0 < u["T"]["Y"][:, k].sum() < len(u["T"]["Y"])]
    ub = mean_auc_boot(u["T"]["Y"], u["T"][key], codes, pb)
    uv = np.nanmean([S.auc(u["V"]["Y"][:, k], u["V"]["P"][:, k]) for k in range(u["V"]["Y"].shape[1]) if 0 < u["V"]["Y"][:, k].sum() < len(u["V"]["Y"])])
    return dict(U=float(np.mean(pts)), U_boot=ub, U_val=float(uv))


def build_release(root, name, rots, pb, include_v4=False):
    att = load_attackers(root, name, rots)
    u = load_utility(root, name, rots)
    if att is None or u is None:
        return None
    wc = worst_case(suite_entries(att, include_v4), pb)
    us = utility_stats(u, pb)
    ref = dict(name=name, L=wc["L"], L_boot=wc["L_boot"], L_val=wc["val_L"], argmax=wc["argmax"], val_selected=wc["selected_by_val"], val_selected_test_auc=wc["selected_by_val_test_auc"],
               attacker_points=wc["points"], U=us["U"], U_boot=us["U_boot"], U_val=us["U_val"], U_fixed=utility_stats(u, pb, "P_fixed")["U"],
               f6=(S.auc(att["f6"]["y"], att["f6"]["score"]) if "f6" in att else None), f5=max([wc["points"].get(k, np.nan) for k in F5] or [np.nan]))
    return ref


def labels14_release(root, rots, pb, raw):
    att = load_attackers(root, "labels14", rots)
    if att is None:
        return None
    ent = {k: att[k] for k in F3 if k in att}
    wc = worst_case(ent, pb)
    return dict(name="labels14", L=wc["L"], L_boot=wc["L_boot"], L_val=wc["val_L"], U=raw["U"], U_boot=raw["U_boot"], U_val=raw["U_val"], code_floor=True)


def run(root, g0_path, rots, B=2000, prereg_commit="", out=None, learned=None, t2=False, g2_path=None):
    root = Path(root)
    u_raw = load_utility(root, "raw", rots)
    if u_raw is None:
        raise SystemExit("utility_raw missing: stage 1 must be complete")
    pb = S.PatientBootstrap(np.unique(u_raw["T"]["patient"]), B=B, seed=0)
    releases = sorted({p.name.split("_rot")[0] for p in root.glob("*_rot*") if p.is_dir() and not p.name.endswith("_shuf") and not p.name.startswith("labels14")})
    R = {}
    for nm in releases:
        r = build_release(root, nm, rots, pb)
        if r is not None:
            R[nm] = r
    learned = learned or [n for n in R if n.startswith(LEARNED_PREFIX)]
    if t2 and "const" not in R and "raw" in R:                                   # T.md 7.3: const is an ANALYTIC anchor (L = 0.5, U = 0.5, zero variance)
        R["const"] = dict(name="const", L=0.5, L_boot=np.full(B, 0.5), L_val=0.5, U=0.5, U_boot=np.full(B, 0.5), U_val=0.5, U_fixed=0.5, argmax="analytic", val_selected="analytic",
                          val_selected_test_auc=0.5, attacker_points={}, f6=None, f5=np.nan)
    floors = {n: r for n, r in R.items() if n not in learned}
    res = dict(evaluator_sha=ledger.evaluator_sha(), prereg_commit=prereg_commit, rots=list(rots), B=B, n_patients_universe=int(len(pb.patients)), releases={}, KT={})
    for n, r in R.items():
        res["releases"][n] = dict(L=r["L"], L_ci=ci(r["L_boot"]), L_ci_v2=ci(r["L_boot"], CI_KAPPA), L_mde80_single=float(S.Z_MDE * np.nanstd(r["L_boot"], ddof=1)), U=r["U"], U_ci=ci(r["U_boot"]), U_ci_v2=ci(r["U_boot"], CI_KAPPA), U_mde80_single=float(S.Z_MDE * np.nanstd(r["U_boot"], ddof=1)),
                                  attacker_points={k: float(v) for k, v in r["attacker_points"].items()},
                                  U_fixed=r["U_fixed"], worst_attacker=r["argmax"], val_selected=r["val_selected"], val_selected_test_auc=r["val_selected_test_auc"], f6=r["f6"], learned=n in learned)
    g0 = _load_json(g0_path) if g0_path and Path(g0_path).exists() else None
    # ---- KT0 / KT1
    nT = int(round(len(next(iter(load_attackers(root, "raw", rots).values()))["y"]) / 2)) if "raw" in R else 0
    ctrl = []
    for nm in ("const",):
        a = load_attackers(root, nm, rots)
        if a:
            for k, e in suite_entries(a).items():
                b = S.auc_boot(e["y"], e["score"], pb.codes(e["cluster"]), pb)
                ctrl.append((nm + ":" + k, S.auc(e["y"], e["score"]), float(np.nanstd(b, ddof=1))))
    a = load_attackers(root, "raw", rots, "_shuf")
    if a:
        for k, e in suite_entries(a).items():
            b = S.auc_boot(e["y"], e["score"], pb.codes(e["cluster"]), pb)
            ctrl.append(("raw_shuf:" + k, S.auc(e["y"], e["score"]), float(np.nanstd(b, ddof=1))))
    rim = R.get("rim_w24")
    meta_floor = g0["rot0"]["N2"]["all"]["auc"] if g0 else float("nan")
    if "raw" in R:
        k1 = S.kill_measurability(R["raw"]["L"], meta_floor, [] if t2 else [(a_, s_) for _, a_, s_ in ctrl], nT, nT)
        k1["rim_w24_worst"] = rim["L"] if rim else None
        k1["rim_ok"] = True if t2 else bool(rim is None or rim["L"] <= 0.56)           # T2 (D2): the border-only clause is replaced by G2; rim is descriptive
        k1["fires"] = bool(k1["fires"] or not k1["rim_ok"])
        k1["controls"] = [dict(name=n, auc=a_, se=s_) for n, a_, s_ in ctrl]
        res["KT"]["KT1"] = k1
        f5 = R["raw"]["f5"]
        res["KT"]["KT3"] = S.kill_demographics(f5, R["raw"]["L"]) if np.isfinite(f5) else None
    # ---- paired differences against raw (MDE80 printed next to every difference)
    if "raw" in R:
        res["differences_vs_raw"] = {}
        for n, r in R.items():
            if n == "raw":
                continue
            dl = S.paired(R["raw"]["L_boot"], r["L_boot"], R["raw"]["L"], r["L"]); du = S.paired(R["raw"]["U_boot"], r["U_boot"], R["raw"]["U"], r["U"])
            res["differences_vs_raw"][n] = dict(raw_minus_L=dl, raw_minus_U=du)
    # ---- KT2: envelope per learned defence (FOCUS A2) + sensitivities
    fl = [n for n in floors]
    def env(Dn, floor_names, include_v4=False, extra=None):
        D = R[Dn] if not include_v4 else build_release(root, Dn, rots, pb, True)
        FL = [R[n] if not include_v4 else build_release(root, n, rots, pb, True) for n in floor_names]
        FL = [f for f in FL if f is not None] + ([extra] if extra else [])
        fL = np.stack([f["L_boot"] for f in FL]); fU = np.stack([f["U_boot"] for f in FL])
        bt = S.envelope_boot(fL, fU, D["L_boot"], D["U_boot"])
        mu = max(0.02, S.Z_MDE * float(np.nanstd(bt["dU"], ddof=1)))
        ml = max(0.03, S.Z_MDE * float(np.nanstd(bt["dL"], ddof=1))) if np.isfinite(bt["dL"]).mean() > 0.5 else 0.03
        v = S.envelope_verdict(np.array([f["L"] for f in FL]), np.array([f["U"] for f in FL]), D["L"], D["U"], bt, mu, ml)
        v.update(margin_u=mu, margin_l=ml, n_floors=len(FL))
        return v, D, FL
    def nearest(Dn, floor_names):
        D = R[Dn]; cand = [R[n] for n in floor_names if R[n]["L_val"] <= D["L_val"] + 0.02]
        if not cand:
            return dict(beats=True, note="no floor within +0.02 of D's validation leakage")
        f = max(cand, key=lambda x: x["U_val"]); d = D["U_boot"] - f["U_boot"]
        return dict(floor=f["name"], dU=float(D["U"] - f["U"]), lo=float(np.nanpercentile(d, 2.5)), beats=bool(np.nanpercentile(d, 2.5) >= 0.02))
    verdicts = {}
    for Dn in learned:
        row = {}
        row["primary"], _, _ = env(Dn, fl)
        try:
            row["with_F6"], _, _ = env(Dn, fl, include_v4=True)
        except Exception as e:                                           # f6 not run yet
            row["with_F6"] = dict(error=str(e)[:80])
        l14 = labels14_release(root, rots, pb, R["raw"]) if "raw" in R else None
        if l14:
            row["with_labels14"], _, _ = env(Dn, fl, extra=l14)
        row["nearest_floor_rule"] = nearest(Dn, fl)
        verdicts[Dn] = row
    prim = [v["primary"]["beats_envelope"] for v in verdicts.values()]
    anyfire = lambda key: any(not (v.get(key, {}).get("beats_envelope", v.get(key, {}).get("beats", True))) for v in verdicts.values()) if verdicts else False
    res["KT"]["KT2"] = dict(verdicts=verdicts, fires_primary=bool(verdicts and not any(prim)),
                            fires_any_sensitivity=bool(verdicts and (not any(prim) or all(not v["with_labels14"]["beats_envelope"] for v in verdicts.values() if "with_labels14" in v) and any("with_labels14" in v for v in verdicts.values())
                                                                    or all(not v["nearest_floor_rule"]["beats"] for v in verdicts.values()))))
    res["labels14"] = None
    if "raw" in R:
        l14 = labels14_release(root, rots, pb, R["raw"])
        if l14:
            res["labels14"] = dict(L=l14["L"], L_ci=ci(l14["L_boot"]), L_ci_v2=ci(l14["L_boot"], CI_KAPPA), U=l14["U"], U_ci=ci(l14["U_boot"]), U_ci_v2=ci(l14["U_boot"], CI_KAPPA))
    if not verdicts:
        k1, k3 = res["KT"].get("KT1", {}), res["KT"].get("KT3") or {}
        res["verdict_original"] = "NO-GO (KT1 fires)" if k1.get("fires") else ("NO-GO (KT3 fires)" if k3.get("fires") else "STAGE-0 GATE PASSED (no learned defence evaluated yet)")
        res["verdict_tightened"] = res["verdict_original"]
    else:
      res["verdict_original"] = ("NO-GO (KT1 fires)" if res["KT"].get("KT1", {}).get("fires") else ("NO-GO (KT3 fires)" if (res["KT"].get("KT3") or {}).get("fires") else ("NO-GO (KT2: floors match every learned defence)" if res["KT"]["KT2"]["fires_primary"] else "GO (a learned defence beats the floor envelope)"))) if verdicts or "KT1" in res["KT"] else "incomplete"
      res["verdict_tightened"] = res["verdict_original"] if res["verdict_original"].startswith("NO-GO") else ("NO-GO (KT2 under a sensitivity rule)" if res["KT"]["KT2"]["fires_any_sensitivity"] else res["verdict_original"])
    res["g0"] = dict(passes_all_rotations=g0["passes_all_rotations"]) if g0 else None
    if t2:
        g2 = _load_json(g2_path) if g2_path and Path(g2_path).exists() else None
        res["G2"] = dict(passes=(g2 or {}).get("G2_passes"), checks=(g2 or {}).get("checks"))
        res["verdict_t1"] = "NO-GO (stage 1: KT1 fires, border-only attacker 0.738 > 0.56; deviation D1 recorded)"
        res["verdict_original"] = "NO-GO at G2 (literal T2: control clause 4; D3/A8/A9)" if (g2 and not g2["G2_passes"]) else ("G2 passed" if g2 else "G2 missing")
        res["verdict_tightened"] = res["verdict_original"]
        res["posthoc_sensitivity_note"] = "everything below G2 (KT1', KT2 envelope for pcxr_mu0.01, labels14) is POST-HOC SENSITIVITY after the literal G2 failure; it cannot revive T2"
        res["KT2_posthoc"] = dict(pcxr_fires=res["KT"]["KT2"]["fires_primary"], any_sensitivity_fires=res["KT"]["KT2"]["fires_any_sensitivity"])
    if out:
        o = Path(out); o.mkdir(parents=True, exist_ok=True)
        (o / "result.json").write_text(json.dumps(res, indent=1, default=lambda x: float(x) if np.isscalar(x) else None))
        lines = ["| release | L worst (95% CI) | worst attacker | U (95% CI) | U_fixed | F6 |", "|---|---|---|---|---|---|"]
        for n, r in sorted(res["releases"].items(), key=lambda kv: kv[1]["L"]):
            lines.append(f"| {n}{' (learned)' if r['learned'] else ''} | {r['L']:.3f} [{r['L_ci'][0]:.3f}, {r['L_ci'][1]:.3f}] / v2 [{r['L_ci_v2'][0]:.3f}, {r['L_ci_v2'][1]:.3f}] | {r['worst_attacker']} | {r['U']:.3f} [{r['U_ci'][0]:.3f}, {r['U_ci'][1]:.3f}] | {r['U_fixed']:.3f} | {'' if r['f6'] is None else format(r['f6'], '.3f')} |")
        (o / "floors_table.md").write_text("\n".join(lines) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--g0", default=None); ap.add_argument("--rots", default="0"); ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--prereg-commit", default=""); ap.add_argument("--t2", action="store_true"); ap.add_argument("--g2", default=None); ap.add_argument("--learned", default=None)
    a = ap.parse_args()
    r = run(a.root, a.g0, [int(x) for x in a.rots.split(",")], a.B, a.prereg_commit, a.out, learned=a.learned.split(",") if a.learned else None, t2=a.t2, g2_path=a.g2)
    print(r["verdict_original"], "|", r["verdict_tightened"])


if __name__ == "__main__":
    main()
