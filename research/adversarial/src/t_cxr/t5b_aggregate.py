"""T5b (T.md section 13): the shuffled-label F1+ control and raw at R5, rotation 0.  Reads saved predictions only.
   python -m t_cxr.t5b_aggregate --root ROOT --out DIR
ROOT holds `down_f4_rot0_shuf_ladderall_r5` (control), `raw_rot0_ladderall_r5`, `down_f4_rot0_ladderall_r5`, `pcxr_mu0.01_s1_C2_rot0_ladderall_r5` and, for context, `raw_rot0`, `raw_rot0_ladderall`, `down_f4_rot0_ladderall`.
Rule (fixed in section 13 before any number): CONTROL PASSES iff |AUC - 0.5| <= 0.06 (D4) for the shuffled-label F1+ on the unshuffled rotation-0 test pairs (point estimate); otherwise every R5 number is INVALID and withdrawn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import aggregate as AG, stats as S

TOL = 0.06


def get(root, rel, suffix, name="f1_match_ens"):
    att = AG.load_attackers(root, rel, [0], suffix)
    if att is None or name not in att:
        return None
    return att[name]


def summ(root, rel, suffix):
    p = Path(root) / f"{rel}_rot0{suffix}" / "summary.json"
    a = json.loads(p.read_text()).get("attackers", {}).get("f1_match_s0", {}) if p.exists() else {}
    return {k: a.get(k) for k in ("val_auc", "epochs", "truncated", "best_val_auc")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    ent = dict(control=get(a.root, "down_f4", "_shuf_ladderall_r5"), raw5=get(a.root, "raw", "_ladderall_r5"), down5=get(a.root, "down_f4", "_ladderall_r5"), pcxr5=get(a.root, "pcxr_mu0.01_s1_C2", "_ladderall_r5"),
               raw4=get(a.root, "raw", "_ladderall"), down4=get(a.root, "down_f4", "_ladderall"))
    ref = next(e for e in ent.values() if e is not None)
    pb = S.PatientBootstrap(np.unique(ref["cluster"]), B=a.B, seed=0)
    pts, boots = {}, {}
    for k, e in ent.items():
        if e is None:
            continue
        assert (e["y"] == ref["y"]).all(), "test pairs differ between runs"
        pts[k] = S.auc(e["y"], e["score"]); boots[k] = S.auc_boot(e["y"], e["score"], pb.codes(e["cluster"]), pb)
    res = dict(rot=0, B=a.B, tolerance=TOL, auc={k: dict(point=pts[k], ci=AG.ci(boots[k]), ci_v2=AG.ci(boots[k], AG.CI_KAPPA)) for k in pts})
    if "control" in pts:
        c = pts["control"]
        res["control"] = dict(auc=c, ci=AG.ci(boots["control"]), offset=c - 0.5, passes=bool(abs(c - 0.5) <= TOL), rule="passes iff |AUC - 0.5| <= 0.06 (D4), point estimate, rotation 0",
                              train=summ(a.root, "down_f4", "_shuf_ladderall_r5"), reference_unshuffled_F1plus=pts.get("down5"),
                              verdict=("CONTROL PASSES: the R5 numbers stand (shuffled-label F1+ at chance within D4)" if abs(c - 0.5) <= TOL else "CONTROL FAILS: every R5 number is INVALID and withdrawn"))
    for a_, b_, nm in (("raw5", "down5", "raw_minus_down_f4"), ("raw5", "pcxr5", "raw_minus_pcxr_C2"), ("down5", "pcxr5", "down_f4_minus_pcxr_C2")):
        if a_ in pts and b_ in pts:
            bb = boots[a_] - boots[b_]
            res.setdefault("rot0_differences_F1plus", {})[nm] = dict(diff=pts[a_] - pts[b_], ci=AG.ci(bb), ci_v2=AG.ci(bb, AG.CI_KAPPA), lower_gt_0=bool(AG.ci(bb)[0] > 0))
    if "raw5" in pts:
        res["raw_R5"] = dict(train=summ(a.root, "raw", "_ladderall_r5"), rise_vs_R4_F1=(pts["raw5"] - pts["raw4"]) if "raw4" in pts else None)
        d = res.get("rot0_differences_F1plus", {}).get("raw_minus_down_f4")
        if d:
            res["raw_reading"] = ("down_f4 is measurably BELOW raw at F1+ capacity (lower 95% bound of raw - down_f4 > 0)" if d["lower_gt_0"] else "raw and down_f4 are NOT distinguishable at F1+ capacity on rotation 0 (lower 95% bound of raw - down_f4 <= 0)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "t5b.json").write_text(json.dumps(res, indent=1, default=float))
    print({k: round(v["point"], 4) for k, v in res["auc"].items()})
    print(res.get("control"), res.get("raw_reading"), res.get("rot0_differences_F1plus"))


if __name__ == "__main__":
    main()
