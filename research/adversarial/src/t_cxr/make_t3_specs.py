"""T3 job specs (T.md section 8).  Names t_y_*, priority 5, hf_allow json/log/md, threads <= 8, new salt file.
   python -m t_cxr.make_t3_specs --kind adv|screens --out DIR
   python -m t_cxr.make_t3_specs --kind sealed --nominate nominate_t3.json --out DIR      (after the nomination table is committed)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPORT = "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=research/adversarial/src;"
PY = "$HOME/venvs/t/bin/python"
EX = "research/adversarial/results/t_cxr/exclude_dups.json"
SALT = "--salt-file research/adversarial/sealed/t3_cxr_salt.txt"
DATA = "--data $HOME/work/t_data/cxr_meta --pri $HOME/work/t_data/cxr_meta"
BANK = "--bank $HOME/work/t_data/cxr/cxr256.npy"
GEN = "--gen-dir $HOME/work/t_data/pricheXy_weights --pcxr-dir $HOME/work/t_data/pcxr"
FS = "$HOME/work/t_data/pcxr/flowstats.pt"
COMMON = f"{DATA} {BANK} {SALT} {GEN} --work $HOME/work/t_work --exclude {EX} --out $OUT"
DEFENDERS = ["pcxr_mu0.01", "pcxr_mu0.01_s1_C1", "pcxr_mu0.01_s1_C2", "pcxr_mu0.005_s0_C1", "pcxr_mu0.005_s0_C2"]
GROUPS = {"a": ["raw", "warp0", "rw1_mu0.005", "rw1_mu0.01", "rw1_mu0.02", "rw2_mu0.005", "rw2_mu0.01", "rw2_mu0.02", "rw2_mu0.04"],
          "b": ["blur_s0.5", "blur_s1", "blur_s2", "blur_s4", "down_f2", "down_f4", "down_f8"],
          "c": ["noise_s0.02", "noise_s0.05", "noise_s0.1", "noise_s0.2", "quant_l32", "quant_l8", "quant_l4"],
          "d": ["quantd_l8", "quantd_l4", "jpeg_q30", "jpeg_q10", "jpeg_q3", "pca_k8", "pca_k32", "pca_k128", "dppix_b8"]}


def mk(out, name, cmd, prio=5):
    (Path(out) / f"{name}.yaml").write_text(f'name: {name}\npriority: {prio}\ncmd: "{EXPORT} {cmd}"\nhf_path: runs/{name}\nhf_allow: ["*.json","*.log","*.md"]\n')


def screen_cmd(r, extra=""):
    return f"{PY} -m t_cxr.run_suite --release {r} --rot 0 --screen --attackers f3 --conds match {extra} {COMMON}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["adv", "screens", "dscreens", "sealed"]); ap.add_argument("--nominate", default=None); ap.add_argument("--out", required=True); ap.add_argument("--defenders", default=",".join(DEFENDERS)); ap.add_argument("--g2lite", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.kind == "adv":
        for mu, seed, tag in (("0.01", 1, "s1"), ("0.005", 0, "s0")):
            mk(a.out, f"t_y_adv_mu{mu}_{tag}", f"{PY} -m t_cxr.train_pcx --stage adv --mu {mu} --seed {seed} --tag {tag} {DATA} {BANK} --exclude {EX} --wdir $HOME/work/t_data/pcxr --out $OUT --max-hours 1.8")
    elif a.kind == "screens":
        for g, names in GROUPS.items():
            mk(a.out, f"t_y_screen_{g}", "for r in " + " ".join(names) + f"; do {screen_cmd('$r')}; done")
        mk(a.out, "t_y_screen_e", " && ".join([screen_cmd("pcxr_mu0.01"), screen_cmd("rw3_mu0.005", f"--flow-stats {FS}"), screen_cmd("rw3_mu0.01", f"--flow-stats {FS}")]))
    elif a.kind == "dscreens":
        for r in DEFENDERS[1:]:
            mk(a.out, f"t_y_dscreen_{r.removeprefix('pcxr_')}", screen_cmd(r))                   # one job per new checkpoint, spread over the workers
    else:
        nom = json.loads(Path(a.nominate).read_text())
        floors = nom["union"]; D = a.defenders.split(",")
        for r in ["raw"] + D + floors:
            fs = f"--flow-stats {FS}" if r.startswith("rw3_") else ""
            attackers = "f1,f3,f4,f5,f6" if (r in D or r == "raw") else "f1,f3,f4,f5"
            chain = [f"{PY} -m t_cxr.run_suite --release {r} --rot {rot} --attackers {attackers} --conds match --seeds 0 {fs} {COMMON}" for rot in (0, 1)]
            mk(a.out, f"t_y_ev_{r}", " && ".join(chain))
        UT = f"{PY} -m t_cxr.run_utility_t2 --rots 0,1 --clf-init $HOME/work/t_data/pcxr/classifier.pth {DATA} {BANK} {SALT} {GEN} --exclude {EX} --out $OUT"
        mk(a.out, "t_y_util_raw", f"{UT} --release raw --labels14")
        for r in D + floors:
            fs = f"--flow-stats {FS}" if r.startswith("rw3_") else ""
            mk(a.out, f"t_y_util_{r}", f"{UT} --release {r} {fs}")
        # validity extras on the new partition: frame-zeroed (rotations 0, 1), const (rotation 0 only in the lite plan), shuffled-label controls (train labels; both label sets only if not lite)
        g2 = [f"{PY} -m t_cxr.run_suite --release core_w24 --rot {rot} --attackers f1,f3,f4 --conds match --seeds 0 {COMMON}" for rot in (0, 1)]
        g2 += [f"{PY} -m t_cxr.run_suite --release const --rot {rot} --attackers f1,f3,f5 --conds match --seeds 0 {COMMON}" for rot in ((0,) if a.g2lite else (0, 1))]
        g2 += [f"{PY} -m t_cxr.run_suite --release raw --rot {rot} --shuffle-labels {sv} --attackers f1,f3,f5 --conds match --seeds 0 {COMMON}" for rot in (0, 1) for sv in (("",) if a.g2lite else ("", "--shuffle-val"))]
        mk(a.out, "t_y_g2", " && ".join(g2))


if __name__ == "__main__":
    main()
