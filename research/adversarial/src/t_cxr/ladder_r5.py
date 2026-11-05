"""T5 (T.md section 12): the closing rung R5 of the attacker-scaling ladder.  Reads saved predictions only.
   python -m t_cxr.ladder_r5 --root ROOT --rots 0,1 --out DIR [--t4 research/adversarial/results/t_cxr/t4/ladder.json]
ROOT holds the T3 R1 dirs (`<release>_rot<r>`), the R4 dirs (`_ladderall`) and the R5 dirs (`_ladderall_r5`).
Worst case at R5:  L5 = max(F1_R4, F1+_R5, F3_R4 best, F4_R1, F5_R1)  inside every bootstrap resample (the SAME resamples as T4a: B = 2,000, seed 0, cluster universe of the raw R1 test pairs when raw is present),
so L5 >= L4 by construction; D5 = L5 - L4 (paired).  labels14: L5 = max(F3_R4 best, f3x_mlp_R5).
Decision rule (fixed in T.md section 12 before any number), for down_f4:
   SIGNED-WITH-CAVEAT iff point(D5) < 0.02 AND upper 95% bound(D5) < 0.04;  PERMANENTLY UNSIGNABLE iff point(D5) >= 0.02;  otherwise UNRESOLVED (treated as unsignable).
Tightened verdict: the same rule on the total rise R1 -> R5.  Ordering test: reversed iff L5(down_f4) - L5(pcxr_mu0.01_s1_C2) >= 0.03 with the lower 99% bound > 0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, stats as S
from .ladder_aggregate import rung_entries, worst

REL = ["down_f4", "pcxr_mu0.01_s1_C2", "labels14"]
SUF4, SUF5 = "_ladderall", "_ladderall_r5"
POINT_MAX, UPPER_MAX = 0.02, 0.04


def rule(point, upper95):
    if point >= POINT_MAX:
        return "PERMANENTLY UNSIGNABLE"
    if upper95 < UPPER_MAX:
        return "SIGNED-WITH-CAVEAT"
    return "UNRESOLVED (treated as unsignable)"


def r5_entry(root, rel, rots):
    att = AG.load_attackers(root, rel, rots, SUF5)
    if att is None:
        return None
    if rel == "labels14":
        return att.get("f3x_mlp")
    return att.get("f1_match_ens") or att.get("f1_match_s0")


def train_info(root, rel, rots):
    out = {}
    for r in rots:
        p = Path(root) / f"{rel}_rot{r}{SUF5}" / "summary.json"
        if p.exists():
            at = json.loads(p.read_text()).get("attackers", {}).get("f1_match_s0", {})
            out[f"rot{r}"] = {k: at.get(k) for k in ("epochs", "truncated", "best_val_auc", "val_auc")}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rots", default="0,1"); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--t4", default=None, help="T4a ladder.json: the L4 / delta_R4 reproduced here must match it (same resamples)")
    a = ap.parse_args()
    rots = [int(x) for x in a.rots.split(",")]
    res = dict(rots=rots, B=a.B, rule=dict(point_max=POINT_MAX, upper95_max=UPPER_MAX), releases={})
    t4 = json.loads(Path(a.t4).read_text()) if a.t4 else None

    # bootstrap universe: the raw R1 test pairs when present (T4a built its resamples from them), else the first release's
    raw1 = AG.load_attackers(a.root, "raw", rots, "")
    ref = next(iter(raw1.values())) if raw1 else None
    pb, boots, ent_by = None, {}, {}
    for rel in REL:
        a1 = AG.load_attackers(a.root, rel, rots, "")
        if a1 is None:
            continue
        fixed = {}
        if rel != "labels14":
            f4 = [k for k in AG.F4 if k in a1]
            if f4:
                fixed["f4"] = a1[max(f4, key=lambda k: a1[k]["val_auc"])]
            for k in AG.F5:
                if k in a1:
                    fixed[k] = a1[k]
        e1, e4 = rung_entries(a.root, rel, rots, "", fixed), rung_entries(a.root, rel, rots, SUF4, fixed)
        e5x = r5_entry(a.root, rel, rots)
        if e1 is None or e4 is None or e5x is None:
            res["releases"][rel] = dict(missing=True)
            continue
        e5 = dict(e4); e5["f3x" if rel == "labels14" else "f1p"] = e5x
        if pb is None:
            pb = S.PatientBootstrap(np.unique((ref if ref is not None else next(iter(e1.values())))["cluster"]), B=a.B, seed=0)
        y0 = next(iter(e1.values()))["y"]
        for e in list(e1.values()) + list(e4.values()) + [e5x]:
            assert (e["y"] == y0).all(), "test pairs differ between rungs / attackers"
        w1, w4, w5 = worst(e1, pb), worst(e4, pb), worst(e5, pb)
        boots[rel] = dict(R1=w1["boot"], R4=w4["boot"], R5=w5["boot"]); ent_by[rel] = dict(L1=w1["L"], L4=w4["L"], L5=w5["L"])
        row = dict(L1=w1["L"], L4=w4["L"], L5=w5["L"], argmax_R5=w5["argmax"], points_R5={k: float(v) for k, v in w5["points"].items()},
                   ci_R5=AG.ci(w5["boot"]), ci_R5_v2=AG.ci(w5["boot"], AG.CI_KAPPA), train=train_info(a.root, rel, rots))
        if t4 and rel in t4["releases"]:
            row["t4_reproduced"] = dict(L4_t4=t4["releases"][rel]["R4"]["L"], L4_here=w4["L"], match=bool(abs(t4["releases"][rel]["R4"]["L"] - w4["L"]) < 1e-9),
                                        dR4_t4=t4["releases"][rel]["delta_R4"]["diff"], dR4_here=w4["L"] - w1["L"])
        for nm, (hi_, lo_, bh, bl) in dict(D5=("R5", "R4", w5, w4), total_R1_R5=("R5", "R1", w5, w1), D4=("R4", "R1", w4, w1)).items():
            bb = bh["boot"] - bl["boot"]; p = S.paired(bh["boot"], bl["boot"], bh["L"], bl["L"])
            row[nm] = dict(diff=p["diff"], ci=AG.ci(bb), ci_v2=AG.ci(bb, AG.CI_KAPPA), mde80=p["mde80"], upper95=float(AG.ci(bb)[1]), upper95_v2=float(AG.ci(bb, AG.CI_KAPPA)[1]))
        # the F1+ attacker itself against the R4 F1 / best-F3 (point values, descriptive)
        if rel != "labels14":
            row["f1p_minus_f1_R4_point"] = float(S.auc(e5["f1p"]["y"], e5["f1p"]["score"]) - S.auc(e4["f1"]["y"], e4["f1"]["score"]))
        else:
            row["f3x_minus_f3_R4_point"] = float(S.auc(e5["f3x"]["y"], e5["f3x"]["score"]) - S.auc(e4["f3"]["y"], e4["f3"]["score"]))
        res["releases"][rel] = row

    d = res["releases"].get("down_f4", {})
    if "D5" in d:
        res["leg2_R5"] = dict(
            verdict_original=rule(d["D5"]["diff"], d["D5"]["upper95"]),
            verdict_tightened=rule(d["total_R1_R5"]["diff"], d["total_R1_R5"]["upper95"]),
            verdict_original_v2=rule(d["D5"]["diff"], d["D5"]["upper95_v2"]),
            D5=d["D5"], total_R1_R5=d["total_R1_R5"],
            literal_T4a_clause=dict(rise_R1_R4=d["D4"]["diff"], status="FAILED" if d["D4"]["diff"] >= 0.04 else "met"),
            reading=("every chest X-ray linkage number is quoted as a LOWER BOUND, permanently" if d["D5"]["diff"] >= POINT_MAX else
                     ("the curve has flattened enough that more F1-family capacity is unlikely to move the level by 0.02 or more; this does not repeal the T4a clause" if d["D5"]["upper95"] < UPPER_MAX else "unresolved: treated as unsignable")))
    if "down_f4" in boots and "pcxr_mu0.01_s1_C2" in boots:
        for k in ("R4", "R5"):
            bd, bp = boots["down_f4"][k], boots["pcxr_mu0.01_s1_C2"][k]
            gap = ent_by["down_f4"]["L" + k[1]] - ent_by["pcxr_mu0.01_s1_C2"]["L" + k[1]]
            lo99 = float(np.nanpercentile(bd - bp, 0.5))
            res.setdefault("ordering", {})[k] = dict(L_down_f4_minus_L_defender=gap, lower99=lo99, upper99=float(np.nanpercentile(bd - bp, 99.5)), defender_beats_floor=bool(gap >= 0.03 and lo99 > 0),
                                                     floor_ahead_by=float(-gap), paired_mde80=float(S.paired(bd, bp, ent_by["down_f4"]["L" + k[1]], ent_by["pcxr_mu0.01_s1_C2"]["L" + k[1]])["mde80"]))
        o = res["ordering"]["R5"]
        res["ordering_reading"] = ("REVERSED: the defender leaks less than down_f4 by >= 0.03 (lower 99% bound > 0) at R5" if o["defender_beats_floor"] else
                                   "the T3 result 'no defender beats the floors' SURVIVES at R5: the floor is %s the defender by %.3f" % ("ahead of" if o["floor_ahead_by"] > 0 else "behind", abs(o["floor_ahead_by"])))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ladder_r5.json").write_text(json.dumps(res, indent=1, default=float))
    for rel, v in res["releases"].items():
        if "L5" in v:
            print(rel, "L1 %.3f L4 %.3f L5 %.3f" % (v["L1"], v["L4"], v["L5"]), "D5 %.3f [%.3f,%.3f] mde80 %.3f" % (v["D5"]["diff"], *v["D5"]["ci"], v["D5"]["mde80"]), "argmax", v["argmax_R5"], v["train"])
    print("leg2_R5:", {k: res.get("leg2_R5", {}).get(k) for k in ("verdict_original", "verdict_tightened", "reading")})
    print("ordering:", res.get("ordering"), res.get("ordering_reading"))


if __name__ == "__main__":
    main()
