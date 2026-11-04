"""Generate the T2 job specs that depend on earlier results (written into --out, validated by the caller with yaml.safe_load and `sh -n`).
   python -m t_cxr.make_t2_specs --kind post_adv --out DIR
   python -m t_cxr.make_t2_specs --kind sealed --nominate nominate.json --out DIR
post_adv: flowstats of the retrained / released generators, then the validation-only screens of pcxr_* and of the rw3 floors (flow statistics from pcxr).
sealed:   run_suite on rotations 0,1,2 for each pcxr release and each nominated floor (raw rotations come from stage 1 and the G2 jobs), and the T2 utility jobs (raw with labels14)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPORT = "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=research/adversarial/src;"
PY = "$HOME/venvs/t/bin/python"
EX = "research/adversarial/results/t_cxr/exclude_dups.json"
DATA = "--data $HOME/work/t_data/cxr_meta --pri $HOME/work/t_data/cxr_meta"
BANK = "--bank $HOME/work/t_data/cxr/cxr256.npy"
SALT = "--salt-file research/adversarial/sealed/t_cxr_salt.txt"
GEN = "--gen-dir $HOME/work/t_data/pricheXy_weights --pcxr-dir $HOME/work/t_data/pcxr"
FS = "$HOME/work/t_data/pcxr/flowstats.pt"
COMMON = f"{DATA} {BANK} {SALT} {GEN} --work $HOME/work/t_work --exclude {EX} --out $OUT"


def mk(out, name, prio, cmd):
    (Path(out) / f"{name}.yaml").write_text(f'name: {name}\npriority: {prio}\ncmd: "{EXPORT} {cmd}"\nhf_path: runs/{name}\nhf_allow: ["*.json","*.log","*.md"]\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["post_adv", "sealed"]); ap.add_argument("--nominate", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--defenders", default="pcxr_mu0.01,pcxr_mu0.005")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    D = a.defenders.split(",")
    if a.kind == "post_adv":
        gens = ",".join(D + ["pcx_mu0.01"])
        chain = [f"{PY} -m t_cxr.flowstats --generators {gens} {DATA} {BANK} {SALT} --exclude {EX} {GEN} --out $OUT --wout {FS}"]
        for r in D:
            chain.append(f"{PY} -m t_cxr.run_suite --release {r} --rot 0 --screen --attackers f3 --conds match {COMMON}")
        for r in ("rw3_mu0.005", "rw3_mu0.01"):
            chain.append(f"{PY} -m t_cxr.run_suite --release {r} --rot 0 --screen --attackers f3 --conds match --flow-stats {FS} {COMMON}")
        mk(a.out, "t_x_post_adv", 5, " && ".join(chain))
        return
    nom = json.loads(Path(a.nominate).read_text())
    floors = nom["union"]
    for r in D + floors:
        fs = f"--flow-stats {FS}" if r.startswith("rw3_") else ""
        extra = "f1,f2,f3,f4,f5,f6" if r in D else "f1,f3,f4,f5"
        chain = [f"{PY} -m t_cxr.run_suite --release {r} --rot {rot} --attackers {extra} --conds match --seeds 0 {fs} {COMMON}" for rot in (0, 1, 2)]
        mk(a.out, f"t_x_ev_{r}", 5, " && ".join(chain))
    UT = f"{PY} -m t_cxr.run_utility_t2 --rots 0,1,2 --clf-init $HOME/work/t_data/pcxr/classifier.pth {DATA} {BANK} {SALT} {GEN} --exclude {EX} --out $OUT"
    mk(a.out, "t_x_util_raw", 5, f"{UT} --release raw --labels14")
    for r in D + floors:
        fs = f"--flow-stats {FS}" if r.startswith("rw3_") else ""
        mk(a.out, f"t_x_util_{r}", 5, f"{UT} --release {r} {fs}")


if __name__ == "__main__":
    main()
