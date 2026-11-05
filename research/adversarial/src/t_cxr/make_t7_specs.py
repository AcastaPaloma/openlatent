"""T7 job specs (T.md section 15).  Names t_k_*, priority 5, hf_allow json/log/md, threads <= 8, T3 salt.  Defender weights stay on the VM ($HOME/work/t_data/pcxr/generator_defl_<tag>.pth), never uploaded.
   python -m t_cxr.make_t7_specs --kind train|eval|control|sealed --out DIR --configs c1,c2,c3
train  : the three defender configurations (lambda 1 / 3 / 10), 3,000 s each
eval   : per engaged candidate: the fresh F1+ screen on the VALIDATION partition (no ledger entry, weights kept locally) and the utility fit
control: shuffled-label F1+ on the released images of one configuration
sealed : scoring of the saved F1+ of the nominated configuration on the sealed rotation-0 T pairs (one ledger entry)"""
from __future__ import annotations

import argparse
from pathlib import Path

from .make_t4_specs import COMMON, DATA, BANK, GEN, SALT, EX, PY, mk
from .make_t6_specs import CK, BUDGET, RS, UTIL

LAM = {"c1": 1.0, "c2": 3.0, "c3": 10.0}
TRAIN_BUDGET = 3000
TRAIN = f"{PY} -m t_cxr.train_def --budget-s {TRAIN_BUDGET} --eval-every 600 {DATA} {BANK} --exclude {EX} --wdir $HOME/work/t_data/pcxr --out $OUT"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["train", "eval", "control", "sealed"]); ap.add_argument("--out", required=True); ap.add_argument("--configs", default="c1,c2,c3")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in [x for x in a.configs.split(",") if x]:
        if a.kind == "train":
            mk(a.out, f"t_k_def_{c}", f"{TRAIN} --tag {c} --lam {LAM[c]:g}")
        elif a.kind == "eval":
            mk(a.out, f"t_k_scr_{c}", f"{RS} --release defl_{c} --rot 0 --screen --ladder-s -1 --r5 --r5-budget {BUDGET} --attackers f1 --conds match --seeds 0 --save-model {CK}/defl_{c}_rot0.pt {COMMON}")
            mk(a.out, f"t_k_util_{c}", f"{UTIL} --release defl_{c}")
        elif a.kind == "control":
            mk(a.out, f"t_k_ctl_{c}", f"{RS} --release defl_{c} --rot 0 --screen --shuffle-labels --ladder-s -1 --r5 --r5-budget {BUDGET} --attackers f1 --conds match --seeds 0 {COMMON}")
        else:
            mk(a.out, f"t_k_seal_{c}", f"{RS} --release defl_{c} --rot 0 --ladder-s -1 --r5 --r5-budget {BUDGET} --load-model {CK}/defl_{c}_rot0.pt --attackers f1 --conds match --seeds 0 {COMMON}")


if __name__ == "__main__":
    main()
