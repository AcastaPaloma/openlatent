"""T5 job specs (T.md section 12).  Names t_w_*, priority 5, hf_allow json/log/md, threads <= 8, T3 salt, R4 training pairs (A + all S), F1+ attacker (--r5).
One job per (release, rotation) so the cut order of the pre-registration (pcxr rotation 1 first, then labels14) can be applied by simply not running a job.
   python -m t_cxr.make_t5_specs --out DIR"""
from __future__ import annotations

import argparse
from pathlib import Path

from .make_t4_specs import COMMON, PY, UT, mk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--kind", default="main", choices=["main", "control", "seed2"])
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.kind == "control":                  # T.md section 13 (order after T5): shuffled-label F1+ control on down_f4 rot0 and raw at R5 on rot0 (same recipe and budget)
        mk(a.out, "t_w_ctl_down_r0", f"{PY} -m t_cxr.run_suite --release down_f4 --rot 0 --ladder-s -1 --r5 --shuffle-labels --attackers f1 --conds match --seeds 0 {COMMON}")
        mk(a.out, "t_w_f1p_raw_r0", f"{PY} -m t_cxr.run_suite --release raw --rot 0 --ladder-s -1 --r5 --attackers f1 --conds match --seeds 0 {COMMON}")
        return
    if a.kind == "seed2":                    # only if section 13 rule (3) is met
        mk(a.out, "t_w_f1p_down_r0_seed1", f"{PY} -m t_cxr.run_suite --release down_f4 --rot 0 --ladder-s -1 --r5 --attackers f1 --conds match --seeds 1 {COMMON}")
        return
    for short, rel in (("down", "down_f4"), ("pcxr", "pcxr_mu0.01_s1_C2")):
        for r in (0, 1):
            mk(a.out, f"t_w_f1p_{short}_r{r}", f"{PY} -m t_cxr.run_suite --release {rel} --rot {r} --ladder-s -1 --r5 --attackers f1 --conds match --seeds 0 {COMMON}")
    mk(a.out, "t_w_lab14_r5", f"{UT} --ladder-s -1 --r5")


if __name__ == "__main__":
    main()
