"""POST-HOC descriptive add-on to H3-001 staged (F.md section 23.2): the V-selected adapter's dU_H against the full floor envelope at EVERY budget, trained or not,
per fold and protocol (P0 stored AP, Pabc aliased), to show the shape of the budget curve when the memorisation-calibrated trained check removes cells.
  python h3001_untrained_desc.py --runs blk=dir,... --scratch S --out out.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

import aggregate as AG
import aggregate_f2 as F2
import aggregate_f4 as F4
import aggregate_h3001 as H

AG.K, AG.CHANCE = 10, 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args()
    runs = dict(x.split("=") for x in a.runs.split(","))
    res = {}
    for proto in ("P0", "Pabc"):
        for nm, dp in runs.items():
            d = dp if proto == "P0" else H.alias(dp, a.salt, Path(a.scratch) / f"alias_{nm}")
            sp = AG.Split(d, a.salt, a.B, 900, True, "ldaT2")
            P = {k: v for k, v in np.load(Path(d) / f"preds_{a.salt}.npz").items()}
            meta = json.loads((Path(d) / f"meta_{a.salt}.json").read_text())
            regime = meta["regime"]
            floors = [x for x in sp.names(AG.FAM_IND) if sp.has_att(x) and sp.U("V", x) is not None]
            matched = [x for x in sp.arms if sp.fam[x] == "matched" and sp.has_att(x)]
            env = floors + matched
            adapters = [x for x in sp.arms if sp.fam[x] == "adapter"]
            row = {}
            for R, _ in H.F4.BUD:
                arms_R = [x for x in adapters if regime[x] == R]
                if not arms_R:
                    continue
                tf = {x: F2.compare(sp, P, x, env) for x in arms_R}
                af = max(arms_R, key=lambda x: tf[x]["dU_V"])
                t = tf[af]
                row[R] = dict(arm=af, F_star=t.get("F_star"), U_H=t["U_H"], L_H=t["L_H"], dU_H=t.get("dU_H"), dU_H_ci=t.get("dU_H_ci"), win=t["win"],
                              U_V=t["U_V"], U_V_perm=sp.U("V", f"perm_{R}_d64"))
            best_floor = max(floors, key=lambda x: sp.U("V", x))
            row["_best_floor_by_V"] = dict(arm=best_floor, U_H=sp.U("H", best_floor), L_H=sp.L("H", best_floor))
            row["_raw"] = dict(U_H=sp.U("H", "raw"), L_H=sp.L("H", "raw"))
            res[f"{proto}|{nm}"] = row
            print(proto, nm, {R: (round(v["dU_H"], 4) if v.get("dU_H") is not None else None, v["win"]) for R, v in row.items() if not R.startswith("_")}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
