"""T2 nomination (T.md 7.3): from VALIDATION-only screens choose which floors go to the sealed evaluation.  No sealed material is read.
For each defender D: per floor family, the floor with the largest L_val not above L_val(D) and the one with the smallest L_val not below it; keep the 2 families whose bracketing
floors are closest in |L_val - L_val(D)|, at most 4 floors per D.  L_val = max over the screen attackers of the validation AUC.
   python -m t_cxr.nominate --root DIR --defenders pcxr_mu0.01,pcxr_mu0.005 --out nominate.json"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FAMILY = [("warp", r"^(rw[123]_|warp0)"), ("blur", r"^blur_"), ("down", r"^down_"), ("noise", r"^noise_"), ("quant", r"^quantd?_"), ("jpeg", r"^jpeg_"), ("pca", r"^pca_"), ("dppix", r"^dppix_")]
SCREEN_ATT = ("f3a_logreg", "f3b_mlp", "f3c_cosine", "f3c_l2", "f3c_canberra")


def family(name):
    for f, rx in FAMILY:
        if re.match(rx, name):
            return f
    return None


def l_val(root, release):
    p = Path(root) / f"{release}_rot0_screen" / "summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())["attackers"]
    return max(j[k]["val_auc"] for k in SCREEN_ATT if k in j)


def nominate(root, defenders, k_fam=2):
    rel = sorted({p.name[: -len("_rot0_screen")] for p in Path(root).glob("*_rot0_screen")})
    L = {r: l_val(root, r) for r in rel}
    out = dict(L_val={r: v for r, v in L.items() if v is not None}, nominated={}, union=[])
    for D in defenders:
        if L.get(D) is None:
            out["nominated"][D] = None
            continue
        ld = L[D]; cand = []
        fams = {}
        for r, v in L.items():
            f = family(r)
            if f is None or v is None or r in defenders:
                continue
            fams.setdefault(f, []).append((r, v))
        for f, items in fams.items():
            below = [(r, v) for r, v in items if v <= ld]; above = [(r, v) for r, v in items if v >= ld]
            b = max(below, key=lambda x: x[1]) if below else None
            a_ = min(above, key=lambda x: x[1]) if above else None
            dist = min([abs(x[1] - ld) for x in (b, a_) if x])
            cand.append((dist, f, [x[0] for x in (b, a_) if x]))
        cand.sort()
        chosen = [r for _, _, rs in cand[:k_fam] for r in rs][:4]
        out["nominated"][D] = dict(L_val_D=ld, floors=chosen, families=[f for _, f, _ in cand[:k_fam]])
        out["union"] += chosen
    out["union"] = sorted(set(out["union"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--defenders", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = nominate(a.root, a.defenders.split(","))
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r["nominated"], indent=1), "\nunion:", r["union"])
