"""T7 (T.md section 15): learned image defender against F1+.  Reads saved files only.
   python -m t_cxr.t7_aggregate --root ROOT --out DIR [--gates-only | --controls-only | --sealed]
ROOT = the T6 root (floors: `<release>_rot0_screen_ladderall_r5`, `utility_<release>_rot0.json`, raw / down_f4 anchors) plus the T7 files: `train_def_<c>.json`, `defl_<c>_rot0_screen_ladderall_r5`, `utility_defl_<c>_rot0.json`,
`defl_<c>_rot0_shuf_screen_ladderall_r5` (control), and with --sealed `defl_<c>_rot0_ladderall_r5`.  Rules: section 15 (fixed before any number)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, stats as S
from .t6_aggregate import RELEASES, preds, summ, utility, util_boot

CONFIGS = ["c1", "c2", "c3"]
FLOORS = RELEASES + ["raw", "down_f4"]
HOLD_AUC_MAX, GAP_MAX, CDROP_MAX, NOISE_TOL, FRESH_MAX = 0.65, 0.05, 0.02, 0.05, 0.85
LINK_MAX, DROP_MAX, BEAT, MATCH, CTL_TOL = 0.80, 0.02, 0.05, 0.01, 0.06


def gate_inloop(root, c):
    p = Path(root) / f"train_def_{c}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text()); f = j["final"]
    g = dict(lam=j["lam"], steps=j["steps"], seconds=j["seconds"], heldout_auc=f["heldout_auc"], heldin_auc=f["heldin_auc"], memorisation_gap=f["memorisation_gap"], C_drop=f["C_drop"], noise_null_auc=j["noise_null_auc"],
             history=[dict(t=h["tag"], heldout_auc=h["heldout_auc"], C_drop=h["C_drop"]) for h in j["history"]])
    g["checks"] = dict(heldout_auc=bool(g["heldout_auc"] <= HOLD_AUC_MAX), memorisation=bool(g["memorisation_gap"] <= GAP_MAX), collapse=bool(g["C_drop"] <= CDROP_MAX), noise_null=bool(abs(g["noise_null_auc"] - 0.5) <= NOISE_TOL))
    g["inloop_pass"] = bool(all(g["checks"].values()))
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--gates-only", action="store_true"); ap.add_argument("--controls-only", action="store_true"); ap.add_argument("--sealed", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = dict(rule=dict(hold_auc_max=HOLD_AUC_MAX, gap_max=GAP_MAX, cdrop_max=CDROP_MAX, noise_tol=NOISE_TOL, fresh_max=FRESH_MAX, link_max=LINK_MAX, drop_max=DROP_MAX, beat=BEAT, match=MATCH, ctl_tol=CTL_TOL), configs={})
    for c in CONFIGS:
        g = gate_inloop(a.root, c)
        if g:
            res["configs"][c] = dict(inloop=g)
    res["inloop_candidates"] = [c for c, v in res["configs"].items() if v["inloop"]["inloop_pass"]]
    if a.gates_only:
        (Path(a.out) / "t7_gates.json").write_text(json.dumps(res, indent=1, default=float))
        for c, v in res["configs"].items():
            print(c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v["inloop"].items() if k != "history"})
        print("in-loop candidates:", res["inloop_candidates"]); return
    # ---- shuffled-label control (read before any fresh-F1+ curve)
    res["controls"] = {}
    for c in CONFIGS:
        p = preds(a.root, f"defl_{c}_rot0_shuf_screen_ladderall_r5")
        if p is not None:
            y, s, cl = np.asarray(p["y_val"]), np.asarray(p["score_val"]), np.asarray(p["cluster_val"]); pb0 = S.PatientBootstrap(np.unique(cl), B=a.B, seed=0)
            auc0 = S.auc(y, s); res["controls"][c] = dict(auc=auc0, ci=AG.ci(S.auc_boot(y, s, pb0.codes(cl), pb0)), passes=bool(abs(auc0 - 0.5) <= CTL_TOL), train=summ(a.root, f"defl_{c}_rot0_shuf_screen_ladderall_r5"))
    if a.controls_only:
        (Path(a.out) / "t7_controls.json").write_text(json.dumps(res["controls"], indent=1, default=float)); print(res["controls"]); return
    # ---- linkage on V for the floors and the fresh F1+ on each defender
    link, ref = {}, None
    for rel in FLOORS + [f"defl_{c}" for c in CONFIGS]:
        p = preds(a.root, f"{rel}_rot0_screen_ladderall_r5" if rel in RELEASES or rel.startswith("defl") else f"{rel}_rot0_ladderall_r5")
        if p is None:
            continue
        ref = ref or p
        assert p["y_val"] == ref["y_val"], "validation pairs differ"
        link[rel] = p
    cl = np.asarray(ref["cluster_val"]); pb = S.PatientBootstrap(np.unique(cl), B=a.B, seed=0)
    pts = {r: S.auc(np.asarray(p["y_val"]), np.asarray(p["score_val"])) for r, p in link.items()}
    boots = {r: S.auc_boot(np.asarray(p["y_val"]), np.asarray(p["score_val"]), pb.codes(cl), pb) for r, p in link.items()}
    U = {rel: utility(a.root, rel) for rel in FLOORS + [f"defl_{c}" for c in CONFIGS]}
    ub, up = {}, {}
    if U.get("raw") is not None:
        Uref = U["raw"]; pat = np.asarray(Uref["V"]["patient"]); uniq, inv = np.unique(pat, return_inverse=True); by = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        rng = np.random.default_rng(0); draws = [np.concatenate([by[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(min(a.B, 500))]
        for rel, u in U.items():
            if u is not None:
                ub[rel], up[rel] = util_boot(u, by, draws)
    floors = [r for r in FLOORS if r in pts and r in up]
    for c in CONFIGS:
        rel = f"defl_{c}"
        if c not in res["configs"] or rel not in pts:
            continue
        row = res["configs"][c]; row["link_val"] = pts[rel]; row["link_ci"] = AG.ci(boots[rel]); row["train"] = summ(a.root, f"{rel}_rot0_screen_ladderall_r5")
        row["fresh_confirms"] = bool(pts[rel] <= FRESH_MAX)
        row["engaged"] = bool(row["inloop"]["inloop_pass"] and row["fresh_confirms"])
        if rel in up and "raw" in up:
            dd = ub["raw"] - ub[rel]
            row.update(util_val=up[rel], drop_vs_raw=up["raw"] - up[rel], drop_ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
            cand = [f for f in floors if up[f] >= up[rel] - MATCH]
            if cand:
                rf = min(cand, key=lambda f: pts[f]); dif = boots[rel] - boots[rf]
                row["reference_floor"] = dict(release=rf, link=pts[rf], util=up[rf], diff=pts[rel] - pts[rf], diff_ci=AG.ci(dif))
            ctl = res["controls"].get(c) or (next(iter(res["controls"].values())) if res["controls"] else None)
            row["protective_original"] = bool(row["engaged"] and row["link_val"] <= LINK_MAX and row["drop_vs_raw"] <= DROP_MAX)
            row["protective_tightened"] = bool(row["engaged"] and row["link_ci"][1] <= LINK_MAX and row["drop_ci"][1] <= DROP_MAX and not row["train"]["still_rising"] and ctl is not None and ctl["passes"])
            rfd = row.get("reference_floor")
            row["beats_floor_original"] = bool(row["engaged"] and rfd and rfd["diff"] <= -BEAT and row["drop_vs_raw"] <= 0.05)
            row["beats_floor_tightened"] = bool(row["engaged"] and rfd and rfd["diff_ci"][1] <= -BEAT and not row["train"]["still_rising"] and ctl is not None and ctl["passes"])
    eng = [c for c, v in res["configs"].items() if v.get("engaged") and "drop_vs_raw" in v]
    res["engaged_configs"] = eng
    res["engagement_never_achieved"] = bool(not eng)
    nom = sorted([c for c in eng if res["configs"][c]["drop_vs_raw"] <= DROP_MAX], key=lambda c: res["configs"][c]["link_val"])
    res["nominated_for_sealed"] = nom[:1]
    if a.sealed:
        res["sealed"] = {}
        for c in res["nominated_for_sealed"]:
            q = preds(a.root, f"defl_{c}_rot0_ladderall_r5")
            if q is None:
                continue
            y, s, cl_ = np.asarray(q["y"]), np.asarray(q["score_test"]), np.asarray(q["cluster"]); pbt = S.PatientBootstrap(np.unique(cl_), B=a.B, seed=0); b = S.auc_boot(y, s, pbt.codes(cl_), pbt)
            ut, ur = utility(a.root, f"defl_{c}"), utility(a.root, "raw")
            row = dict(link_test=S.auc(y, s), link_test_ci=AG.ci(b))
            if ut is not None and ur is not None:
                row.update(util_test=ut["U_test"], drop_test=ur["U_test"] - ut["U_test"], protective_confirmed=bool(row["link_test"] <= LINK_MAX and ur["U_test"] - ut["U_test"] <= DROP_MAX))
            res["sealed"][c] = row
    res["floors_reference"] = {f: dict(link=pts[f], util=up[f]) for f in floors}
    (Path(a.out) / "t7_result.json").write_text(json.dumps(res, indent=1, default=float))
    for c, v in res["configs"].items():
        print(c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k in ("link_val", "util_val", "drop_vs_raw", "engaged", "protective_original", "protective_tightened", "beats_floor_original", "beats_floor_tightened", "reference_floor")})
    print("engaged:", eng, "| never achieved:", res["engagement_never_achieved"], "| nominated:", res["nominated_for_sealed"], "| controls:", res["controls"])


if __name__ == "__main__":
    main()
