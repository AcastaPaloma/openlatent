"""Write Focus D3 job specs (PII-Masking-300K).  EVERY spec sets hf_allow [json, log, md] (WHITELIST condition c): per-document arrays (units.npz,
records.jsonl) and all caches stay on the VM scratch disk.  Priority 5, <= 8 threads.

  python make_jobs_pii.py embed|s1|s2|eng|adv <args>
"""
import json
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "jobs" / "specs"
ENV = "USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
PY = "$HOME/venvs/d/bin/python research/fields/src/d_embedpriv"
GUARD = "n=0; while [ ! -f $HOME/work/cache/d3_pii/gtr_base/done.txt ] && [ $n -lt 240 ]; do sleep 15; n=$((n+1)); done; "


def write(name, cmd, prio=5):
    (SPEC / f"{name}.yaml").write_text(f"name: {name}\npriority: {prio}\ncmd: {json.dumps(cmd)}\nhf_path: runs/{name}\nhf_allow: [\"*.json\", \"*.log\", \"*.md\"]\n")
    return name


def run(name, grid, seeds):
    return write(name, f"{GUARD}{ENV} {PY}/pii_run.py --name {name} --grid {grid} --seeds {seeds} --out $OUT")


if __name__ == "__main__":
    mode = sys.argv[1]
    out = []
    if mode == "embed":
        out.append(write("d_pii_embed", f"{ENV} {PY}/pii_embed.py"))
    elif mode == "s1":  # stage 1: controls + reference attackers + oracle + record-split demonstration, 3 seeds
        for s in (0, 1, 2):
            out.append(run(f"d_pii_s1_s{s}", "p_ctrl", s))
    elif mode == "s2":  # stage 2: floors and controls, 3 seeds, two jobs per seed
        for s in (0, 1, 2):
            out.append(run(f"d_pii_fa_s{s}", "p_fa", s))
            out.append(run(f"d_pii_fb_s{s}", "p_fb", s))
    elif mode == "eng":  # stage 3a: engagement check (seed 0)
        out.append(run("d_pii_eng_s0", "p_eng", 0))
    elif mode == "adv":  # stage 3b: only if engaged; args: lambda list
        for lam in sys.argv[2].split(","):
            for s in (0, 1):
                out.append(run(f"d_pii_adv{lam.replace('.', 'p')}_s{s}", f"p_adv_{lam}", s))
    print("\n".join(out))
