"""T4 job specs (T.md sections 10 and 11).  Names t_z_*, priority 5, hf_allow json/log/md, threads <= 8, T3 salt.
   python -m t_cxr.make_t4_specs --kind ladder|armb --out DIR"""
from __future__ import annotations

import argparse
from pathlib import Path

EXPORT = "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=research/adversarial/src;"
PY = "$HOME/venvs/t/bin/python"
EX = "research/adversarial/results/t_cxr/exclude_dups.json"
SALT = "--salt-file research/adversarial/sealed/t3_cxr_salt.txt"
DATA = "--data $HOME/work/t_data/cxr_meta --pri $HOME/work/t_data/cxr_meta"
BANK = "--bank $HOME/work/t_data/cxr/cxr256.npy"
GEN = "--gen-dir $HOME/work/t_data/pricheXy_weights --pcxr-dir $HOME/work/t_data/pcxr"
COMMON = f"{DATA} {BANK} {SALT} {GEN} --work $HOME/work/t_work --exclude {EX} --out $OUT"
UT = f"{PY} -m t_cxr.run_utility_t2 --rots 0,1 --clf-init $HOME/work/t_data/pcxr/classifier.pth {DATA} {BANK} {SALT} {GEN} --exclude {EX} --out $OUT --release raw --labels14 --skip-utility-json"
IMG = [("raw", "raw"), ("down_f4", "down_f4"), ("pcxr", "pcxr_mu0.01_s1_C2")]


def mk(out, name, cmd, prio=5):
    (Path(out) / f"{name}.yaml").write_text(f'name: {name}\npriority: {prio}\ncmd: "{EXPORT} {cmd}"\nhf_path: runs/{name}\nhf_allow: ["*.json","*.log","*.md"]\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["ladder", "armb"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.kind == "ladder":
        for tag, n in (("2500", 2500), ("all", -1)):
            for short, rel in IMG:
                chain = [f"{PY} -m t_cxr.run_suite --release {rel} --rot {r} --ladder-s {n} --attackers f1,f3 --conds match --seeds 0 {COMMON}" for r in (0, 1)]
                mk(a.out, f"t_z_lad_{short}_{tag}", " && ".join(chain))
            mk(a.out, f"t_z_lad_labels14_{tag}", f"{UT} --ladder-s {n}")
    else:
        for short, rel in IMG:
            chain = [f"{PY} -m t_cxr.run_suite --release {rel} --rot {r} --design N3s --attackers f1,f3,f5 --conds match --seeds 0 {COMMON}" for r in (0, 1)]
            mk(a.out, f"t_z_n3_{short}", " && ".join(chain))
        mk(a.out, "t_z_n3_const", f"{PY} -m t_cxr.run_suite --release const --rot 0 --design N3s --attackers f1,f3,f5 --conds match --seeds 0 {COMMON}")
        mk(a.out, "t_z_n3_labels14", f"{UT} --design N3s")


if __name__ == "__main__":
    main()
