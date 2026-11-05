"""T4a (T.md section 10): attacker-scaling ladder on the T3 partition.  Reads saved predictions only.
   python -m t_cxr.ladder_aggregate --root ROOT --rots 0,1 --out DIR
ROOT holds the T3 R1 directories (`<release>_rot<r>`, `labels14_rot<r>`) and the ladder directories (`<release>_rot<r>_ladder2500`, `_ladderall`, `labels14_rot<r>_ladder...`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, stats as S

RUNGS = [("R1", ""), ("R2", "_ladder2500"), ("R4", "_ladderall")]
IMG = ["raw", "down_f4", "pcxr_mu0.01_s1_C2"]


def rung_entries(root, release, rots, suffix, fixed):
    """F1 and best-F3 of this rung plus the R1 values of F4 / F5 (`fixed`)."""
    att = AG.load_attackers(root, release, rots, suffix)
    if att is None:
        return None
    ent = {}
    if release != "labels14" and "f1_match_ens" in att:
        ent["f1"] = att["f1_match_ens"]
    f3 = [k for k in AG.F3 if k in att]
    ent["f3"] = att[max(f3, key=lambda k: att[k]["val_auc"])]
    ent.update(fixed)
    return ent


def worst(ent, pb):
    names = list(ent)
    pts = {n: S.auc(ent[n]["y"], ent[n]["score"]) for n in names}
    boots = {n: S.auc_boot(ent[n]["y"], ent[n]["score"], pb.codes(ent[n]["cluster"]), pb) for n in names}
    P = np.array([pts[n] for n in names]); B = np.stack([boots[n] for n in names])
    return dict(L=float(P.max()), boot=np.nanmax(B, axis=0), argmax=names[int(np.argmax(P))], points=pts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rots", default="0,1"); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    rots = [int(x) for x in a.rots.split(",")]
    res = dict(rots=rots, B=a.B, releases={})
    pb = None
    boots = {}
    for rel in IMG + ["labels14"]:
        # R1 reference (the T3 attackers) -> fixed F4 / F5 entries
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
        rows = {}
        for rname, sfx in RUNGS:
            ent = rung_entries(a.root, rel, rots, sfx, fixed)
            if ent is None:
                continue
            if pb is None:
                pb = S.PatientBootstrap(np.unique(next(iter(ent.values()))["cluster"]), B=a.B, seed=0)
            y0 = next(iter(ent.values()))["y"]
            for e in ent.values():
                assert (e["y"] == y0).all(), "test pairs differ between rungs / attackers"
            w = worst(ent, pb)
            rows[rname] = w; boots[(rel, rname)] = w["boot"]
        res["releases"][rel] = {k: dict(L=v["L"], ci=AG.ci(v["boot"]), ci_v2=AG.ci(v["boot"], AG.CI_KAPPA), argmax=v["argmax"], points={a_: float(b_) for a_, b_ in v["points"].items()}) for k, v in rows.items()}
        out_rel = res["releases"][rel]
        for rname in ("R2", "R4"):
            if "R1" in rows and rname in rows:
                d = S.paired(rows[rname]["boot"], rows["R1"]["boot"], rows[rname]["L"], rows["R1"]["L"])
                ci = AG.ci(rows[rname]["boot"] - rows["R1"]["boot"]); ci2 = AG.ci(rows[rname]["boot"] - rows["R1"]["boot"], AG.CI_KAPPA)
                out_rel[f"delta_{rname}"] = dict(diff=d["diff"], ci=ci, ci_v2=ci2, mde80=d["mde80"], lo=d["lo"], hi=d["hi"])
        if "delta_R4" in out_rel:
            dl = out_rel["delta_R4"]
            out_rel["reading"] = ("levels are LOWER BOUNDS" if (dl["diff"] >= 0.03 and dl["ci"][0] > 0.01) else ("CONVERGED" if dl["ci"][1] < 0.02 else "PARTLY CONVERGED / unresolved"))
        if "R2" in rows and "R4" in rows and "R1" in rows:
            out_rel["increments"] = dict(R1_to_R2=rows["R2"]["L"] - rows["R1"]["L"], R2_to_R4=rows["R4"]["L"] - rows["R2"]["L"], note="training-set sizes grow about x2 (R1->R2) and x2.6 (R2->R4) in triples")
    # Leg-2 clause for the floors, ordering test at R4
    leg = {}
    for rel in ("raw", "down_f4"):
        dr = res["releases"].get(rel, {}).get("delta_R4")
        if dr:
            leg[rel] = dict(point=dr["diff"], upper95=dr["ci"][1], upper95_v2=dr["ci_v2"][1])
    if leg:
        res["leg2_clause"] = dict(per_floor=leg, holds=bool(all(v["upper95"] < 0.04 for v in leg.values())), fails=bool(any(v["point"] >= 0.04 for v in leg.values())), rule="holds iff the upper 95% bound of DL(R4) < 0.04 for raw AND down_f4; fails iff either point estimate >= 0.04")
    for rname in ("R1", "R2", "R4"):
        if ("down_f4", rname) in boots and ("pcxr_mu0.01_s1_C2", rname) in boots:
            bd, bp = boots[("down_f4", rname)], boots[("pcxr_mu0.01_s1_C2", rname)]
            gap = res["releases"]["down_f4"][rname]["L"] - res["releases"]["pcxr_mu0.01_s1_C2"][rname]["L"]      # > 0: the FLOOR leaks MORE than the defender
            lo99 = float(np.nanpercentile(bd - bp, 0.5))
            res.setdefault("ordering", {})[rname] = dict(L_down_f4_minus_L_defender=gap, lower99=lo99, upper99=float(np.nanpercentile(bd - bp, 99.5)),
                                                         defender_beats_floor=bool(gap >= 0.03 and lo99 > 0), floor_ahead_by=float(-gap))
    if "R4" in res.get("ordering", {}):
        o = res["ordering"]["R4"]
        res["ordering_reading"] = ("REVERSED: the defender leaks less than down_f4 by >= 0.03 (lower 99% bound > 0) at R4" if o["defender_beats_floor"]
                                   else "the T3 result 'no defender beats the floors' SURVIVES: at R4 the floor is %s the defender by %.3f" % ("ahead of" if o["floor_ahead_by"] > 0 else "behind", abs(o["floor_ahead_by"])))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ladder.json").write_text(json.dumps(res, indent=1, default=float))
    for rel, v in res["releases"].items():
        print(rel, {k: round(x["L"], 3) for k, x in v.items() if isinstance(x, dict) and "L" in x}, {k: (round(x["diff"], 3), round(x["mde80"], 3)) for k, x in v.items() if k.startswith("delta")}, v.get("reading"))
    print("leg2:", res.get("leg2_clause", {}).get("holds"), res.get("leg2_clause", {}).get("per_floor"))
    print("ordering:", res.get("ordering"))


if __name__ == "__main__":
    main()
