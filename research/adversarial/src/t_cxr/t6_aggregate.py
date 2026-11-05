"""T6 stage A (T.md section 14): utility-versus-linkage curve of the floors against F1+, measured on the VALIDATION partition (V role, rotation 0; no ledger entry), the cue ladder, the controls,
the nomination for the sealed look, the stage-B gate.  Reads saved predictions / utility files only.
   python -m t_cxr.t6_aggregate --root ROOT --out DIR [--sealed]
ROOT holds: `<release>_rot0_screen_ladderall_r5` (screens), `down_f32_rot0_shuf_screen_ladderall_r5` (shuffled-label control), `const_rot0_screen_ladderall_r5` (const control),
`utility_<release>_rot0.json` (T6 releases and the anchors raw / down_f4 / pcxr_mu0.01_s1_C2 of T3), the anchors' R5 directories (`raw_rot0_ladderall_r5`, `down_f4_rot0_ladderall_r5`, `pcxr_mu0.01_s1_C2_rot0_ladderall_r5`,
`labels14_rot0_ladderall_r5`), and with --sealed the sealed-phase directories `<release>_rot0_ladderall_r5` of the nominated releases.
Rules are those of section 14 (fixed before any number)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, stats as S

RELEASES = ["down_f8", "down_f16", "down_f32", "jpeg_q10", "hp_s2", "hist"]
CUES = ["hp_s2", "hist"]
ANCHORS = [("raw", "raw_rot0_ladderall_r5"), ("down_f4", "down_f4_rot0_ladderall_r5"), ("pcxr_mu0.01_s1_C2", "pcxr_mu0.01_s1_C2_rot0_ladderall_r5")]
LINK_MAX, DROP_MAX, NOMINATE_DROP, GATE_LINK, GATE_DROP, SHUF_TOL, CONST_TOL = 0.80, 0.02, 0.05, 0.85, 0.10, 0.06, 0.01


def preds(root, d, name="f1_match_ens"):
    p = Path(root) / d / f"preds_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def summ(root, d):
    p = Path(root) / d / "summary.json"
    a = json.loads(p.read_text()).get("attackers", {}).get("f1_match_s0", {}) if p.exists() else {}
    curve = a.get("curve") or []
    va = [c[3] for c in curve if len(c) >= 4]
    best = int(np.argmax(va)) if va else None
    return dict(epochs=a.get("epochs"), truncated=a.get("truncated"), best_epoch=best, best_val_auc=a.get("best_val_auc"),
                still_rising=bool(a.get("truncated") and best is not None and best >= len(va) - 3))


def utility(root, rel):
    p = Path(root) / f"utility_{rel}_rot0.json"
    return json.loads(p.read_text()) if p.exists() else None


def macro_auc(Y, P):
    from sklearn.metrics import roc_auc_score
    out = []
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < len(Y):
            out.append(roc_auc_score(Y[:, k], P[:, k]))
    return float(np.mean(out))


def util_boot(U, groups, draws):
    Y, P = np.asarray(U["V"]["Y"], dtype=float), np.asarray(U["V"]["P"], dtype=float)
    return np.array([macro_auc(Y[ix], P[ix]) for ix in draws]), macro_auc(Y, P)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=1000); ap.add_argument("--sealed", action="store_true")
    a = ap.parse_args()
    res = dict(rule=dict(link_max=LINK_MAX, drop_max=DROP_MAX, nominate_drop=NOMINATE_DROP, gate_link=GATE_LINK, gate_drop=GATE_DROP, shuffled_tol=SHUF_TOL, const_tol=CONST_TOL), rows={}, controls={})
    # --- linkage on V (screens + anchors); shared resamples over V patients
    ref = None
    link = {}
    for rel, d in [(r, f"{r}_rot0_screen_ladderall_r5") for r in RELEASES] + ANCHORS:
        p = preds(a.root, d)
        if p is None:
            continue
        ref = ref or p
        assert p["y_val"] == ref["y_val"], "validation pairs differ between releases"
        link[rel] = p
    if not link:
        raise SystemExit("no linkage predictions found")
    cl = np.asarray(ref["cluster_val"]); pb = S.PatientBootstrap(np.unique(cl), B=a.B, seed=0)
    boots, pts = {}, {}
    for rel, p in link.items():
        y, s = np.asarray(p["y_val"]), np.asarray(p["score_val"])
        pts[rel] = S.auc(y, s); boots[rel] = S.auc_boot(y, s, pb.codes(cl), pb)
    # --- utility on V; patient resamples over the V images of the utility file (identical for every release)
    U = {rel: utility(a.root, rel) for rel in RELEASES + ["raw", "down_f4", "pcxr_mu0.01_s1_C2"]}
    Uref = U.get("raw")
    ub, up = {}, {}
    if Uref is not None:
        pat = np.asarray(Uref["V"]["patient"]); uniq, inv = np.unique(pat, return_inverse=True)
        by = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(min(a.B, 500))]
        for rel, u in U.items():
            if u is not None:
                assert list(u["V"]["patient"]) == list(Uref["V"]["patient"]), "utility V sets differ"
                ub[rel], up[rel] = util_boot(u, by, draws)
    for rel in RELEASES + [r for r, _ in ANCHORS]:
        if rel not in pts:
            continue
        row = dict(link_val=pts[rel], link_ci=AG.ci(boots[rel]), link_ci_v2=AG.ci(boots[rel], AG.CI_KAPPA), train=summ(a.root, dict(ANCHORS).get(rel, f"{rel}_rot0_screen_ladderall_r5")))
        if rel in up and "raw" in up:
            dd = ub["raw"] - ub[rel]
            row.update(util_val=up[rel], util_ci=[float(np.percentile(ub[rel], 2.5)), float(np.percentile(ub[rel], 97.5))], drop_vs_raw=up["raw"] - up[rel], drop_ci=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
        res["rows"][rel] = row
    # --- protective rules (original: points; tightened: bounds + converged attacker)
    for rel in RELEASES:
        r = res["rows"].get(rel)
        if not r or "drop_vs_raw" not in r:
            continue
        r["protective_original"] = bool(r["link_val"] <= LINK_MAX and r["drop_vs_raw"] <= DROP_MAX)
        r["protective_tightened"] = bool(r["link_ci"][1] <= LINK_MAX and r["drop_ci"][1] <= DROP_MAX and not r["train"]["still_rising"])
    ok = [rel for rel in RELEASES if "protective_original" in res["rows"].get(rel, {})]
    res["any_protective_original"] = bool(any(res["rows"][r]["protective_original"] for r in ok)); res["any_protective_tightened"] = bool(any(res["rows"][r]["protective_tightened"] for r in ok))
    # --- nomination for the sealed look: <= 3 releases with utility drop <= 0.05, lowest upper link bound first
    cand = sorted([r for r in ok if res["rows"][r]["drop_vs_raw"] <= NOMINATE_DROP], key=lambda r: res["rows"][r]["link_ci"][1])[:3]
    res["nominated_for_sealed"] = cand
    # --- flat curve and gate B
    near = [r for r in ok if res["rows"][r]["drop_vs_raw"] <= DROP_MAX]
    res["flat_at_ceiling"] = bool(near and all(res["rows"][r]["link_val"] >= 0.95 for r in near))
    gate_cues = [c for c in CUES if c in ok and res["rows"][c]["link_val"] >= GATE_LINK and res["rows"][c]["drop_vs_raw"] >= GATE_DROP]
    res["stage_B_gate"] = dict(no_protective_release=not res["any_protective_original"], removable_cue_candidates=gate_cues, fires=bool((not res["any_protective_original"]) and gate_cues),
                               rule="fires iff NO release is protective (original rule) AND a cue-isolating release (hp_s2 / hist) links at >= 0.85 on V while its utility is >= 0.10 below raw")
    # --- controls
    p = preds(a.root, "down_f32_rot0_shuf_screen_ladderall_r5")
    if p is not None:
        y, s = np.asarray(p["y_val"]), np.asarray(p["score_val"]); b = S.auc_boot(y, s, pb.codes(np.asarray(p["cluster_val"])), pb); auc_ = S.auc(y, s)
        res["controls"]["shuffled_label_down_f32"] = dict(auc=auc_, ci=AG.ci(b), passes=bool(abs(auc_ - 0.5) <= SHUF_TOL), train=summ(a.root, "down_f32_rot0_shuf_screen_ladderall_r5"), rule="|AUC - 0.5| <= 0.06 on V (screen: epoch selected on the same pairs, so an upward selection bias is possible and is stated)")
    p = preds(a.root, "const_rot0_screen_ladderall_r5")
    if p is not None:
        auc_ = S.auc(np.asarray(p["y_val"]), np.asarray(p["score_val"]))
        res["controls"]["const"] = dict(auc=auc_, passes=bool(abs(auc_ - 0.5) <= CONST_TOL), rule="|AUC - 0.5| <= 0.01")
    res["controls_pass"] = bool(res["controls"] and all(c["passes"] for c in res["controls"].values()))
    # --- sealed phase (nominated releases only)
    if a.sealed:
        res["sealed"] = {}
        for rel in cand:
            q = preds(a.root, f"{rel}_rot0_ladderall_r5")
            if q is None:
                continue
            y, s, c = np.asarray(q["y"]), np.asarray(q["score_test"]), np.asarray(q["cluster"])
            pbt = S.PatientBootstrap(np.unique(c), B=a.B, seed=0); b = S.auc_boot(y, s, pbt.codes(c), pbt)
            ut, ur = U.get(rel), U.get("raw")
            row = dict(link_test=S.auc(y, s), link_test_ci=AG.ci(b), link_test_ci_v2=AG.ci(b, AG.CI_KAPPA))
            if ut is not None and ur is not None:
                row.update(util_test=ut["U_test"], util_test_raw=ur["U_test"], drop_test=ur["U_test"] - ut["U_test"])
                row["protective_confirmed"] = bool(row["link_test"] <= LINK_MAX and row["drop_test"] <= DROP_MAX)
            res["sealed"][rel] = row
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "t6_curve.json").write_text(json.dumps(res, indent=1, default=float))
    lines = ["| release | F1+ linkage on V [95%] | epochs (truncated; still rising) | utility U_val [95%] | drop vs raw [95%] | protective (original / tightened) |", "|---|---|---|---|---|---|"]
    for rel, r in res["rows"].items():
        t = r["train"]
        lines.append(f"| {rel} | {r['link_val']:.3f} [{r['link_ci'][0]:.3f}, {r['link_ci'][1]:.3f}] | {t['epochs']} ({t['truncated']}; {t['still_rising']}) | " +
                     (f"{r['util_val']:.3f} [{r['util_ci'][0]:.3f}, {r['util_ci'][1]:.3f}] | {r['drop_vs_raw']:+.3f} [{r['drop_ci'][0]:+.3f}, {r['drop_ci'][1]:+.3f}]" if "util_val" in r else "n/a | n/a") +
                     f" | {r.get('protective_original', '-')} / {r.get('protective_tightened', '-')} |")
    (Path(a.out) / "t6_curve.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("any protective (orig / tight):", res["any_protective_original"], res["any_protective_tightened"], "| nominated:", cand, "| flat at ceiling:", res["flat_at_ceiling"], "| gate B:", res["stage_B_gate"], "| controls:", res["controls"])


if __name__ == "__main__":
    main()
