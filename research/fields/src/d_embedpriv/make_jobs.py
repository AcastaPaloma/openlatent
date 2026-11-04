"""Write Focus D job specs (research/fields/jobs/specs/d_*.yaml).  The cmd string is JSON-encoded (a valid YAML double-quoted scalar) so quoting
cannot break the YAML; always validate with queue_job.sh before queueing.

  python make_jobs.py headroom        # controls + oracle + floors, seeds 0,1,2 (no learned privacy arm)
  python make_jobs.py learned 0,1,2   # bott_mlp control + adv grid (only after the margin is frozen in D.md)
"""
import json
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "jobs" / "specs"
PY = "USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false $HOME/venvs/d/bin/python research/fields/src/d_embedpriv/run_kill.py"


def write(name, grids, seeds, prio, embedder="gtr_base"):
    # one job runs several named grids back to back via --specs assembled by grids.py (grid names joined with '+')
    guard = "n=0; while [ ! -f $HOME/work/cache/d_embedpriv/%s/meta.json ] && [ $n -lt 120 ]; do sleep 15; n=$((n+1)); done; " % embedder
    cmd = guard + f"{PY} --name {name} --embedder {embedder} --grid {'+'.join(grids)} --seeds {seeds} --out $OUT"
    (SPEC / f"{name}.yaml").write_text(f"name: {name}\npriority: {prio}\ncmd: {json.dumps(cmd)}\nhf_path: runs/{name}\n")
    return name


if __name__ == "__main__":
    mode = sys.argv[1]
    out = []
    if mode == "headroom":
        for s in (0, 1, 2):
            out.append(write(f"d_hr1_gtr_s{s}", ["ctrl", "oracle", "floors_b", "suplin"], s, 11 + 2 * s))
            out.append(write(f"d_hr2_gtr_s{s}", ["floors_a"], s, 12 + 2 * s))
    elif mode == "learned":
        seeds = sys.argv[2] if len(sys.argv) > 2 else "0,1,2"
        for s in seeds.split(","):
            s = int(s)
            out.append(write(f"d_bott_gtr_s{s}", ["bott"], s, 20 + s))
            for lam in (0.1, 0.3, 1, 3):
                out.append(write(f"d_adv_l{lam:g}_gtr_s{s}", [f"adv_l{lam:g}"], s, 23 + s))
    print("\n".join(out))
