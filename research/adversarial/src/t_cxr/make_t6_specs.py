"""T6 job specs (T.md section 14).  Names t_v_*, priority 5, hf_allow json/log/md, threads <= 8, T3 salt.
   python -m t_cxr.make_t6_specs --kind screens|sealed --out DIR [--releases a,b]
screens: six F1+ screens on the VALIDATION partition (no ledger entry; weights kept LOCALLY under $HOME/work/t_work/t6_ckpt, never uploaded), the shuffled-label control on down_f32, the const control, six utility jobs.
sealed:  scoring of the saved F1+ model of each NOMINATED release on the sealed rotation-0 T pairs (one ledger entry each; no training)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .make_t4_specs import COMMON, DATA, BANK, GEN, SALT, EX, PY, mk

REL = ["down_f8", "down_f16", "down_f32", "jpeg_q10", "hp_s2", "hist"]
CK = "$HOME/work/t_work/t6_ckpt"
BUDGET = 1200
RS = f"{PY} -m t_cxr.run_suite"
UTIL = f"{PY} -m t_cxr.run_utility_t2 --rots 0 --clf-init $HOME/work/t_data/pcxr/classifier.pth {DATA} {BANK} {SALT} {GEN} --exclude {EX} --out $OUT"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["screens", "sealed"]); ap.add_argument("--out", required=True); ap.add_argument("--releases", default="")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.kind == "screens":
        for rel in REL:
            mk(a.out, f"t_v_scr_{rel}", f"{RS} --release {rel} --rot 0 --screen --ladder-s -1 --r5 --r5-budget {BUDGET} --attackers f1 --conds match --seeds 0 --save-model {CK}/{rel}_rot0.pt {COMMON}")
        mk(a.out, "t_v_ctl_down_f32", f"{RS} --release down_f32 --rot 0 --screen --shuffle-labels --ladder-s -1 --r5 --r5-budget {BUDGET} --attackers f1 --conds match --seeds 0 {COMMON}")
        mk(a.out, "t_v_const", f"{RS} --release const --rot 0 --screen --ladder-s -1 --r5 --r5-budget 300 --max-epochs 2 --attackers f1 --conds match --seeds 0 {COMMON}")
        for rel in REL:
            mk(a.out, f"t_v_util_{rel}", f"{UTIL} --release {rel}")
    else:
        for rel in [r for r in a.releases.split(",") if r]:
            mk(a.out, f"t_v_seal_{rel}", f"{RS} --release {rel} --rot 0 --ladder-s -1 --r5 --r5-budget {BUDGET} --load-model {CK}/{rel}_rot0.pt --attackers f1 --conds match --seeds 0 {COMMON}")


if __name__ == "__main__":
    main()
