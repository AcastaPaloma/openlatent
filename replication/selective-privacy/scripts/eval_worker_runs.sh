#!/usr/bin/env bash
# On the VM that holds the derived maps: evaluate models trained by other workers.
#
# Every 10 minutes, pull runs/ from Drive (other workers' checkpoints) and run the queue
# in eval mode on the given lists. Models whose training has finished (DONE.json) get
# their original, default and targeted evals; the loop ends once every listed model is
# trained and evaluated.
#
# Usage: PRIVACY="..." ACTION="..." eval_worker_runs.sh <upstream_dir> <out_dir>
set -uo pipefail

UP="${1:?usage: eval_worker_runs.sh <upstream_dir> <out_dir>}"
OUT="${2:?usage: eval_worker_runs.sh <upstream_dir> <out_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PRIVACY ACTION

while true; do
  bash "$HERE/sync_drive.sh" pull-runs
  MODE=eval bash "$HERE/run_kth_all_models.sh" "$UP" "$OUT"
  [ $? -eq 3 ] || break
  sleep 600
done
echo "worker runs evaluated: $(date -u +%H:%M:%S)"
