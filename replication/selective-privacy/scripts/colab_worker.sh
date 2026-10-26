#!/usr/bin/env bash
# Turn a fresh Colab VM into an extra training worker for the KTH model queue.
#
# Workers only train: training needs the KTH frames (from the raw archives on Drive) but
# not the derived AFD and similarity maps, which are slow to move through Drive. Each
# worker's checkpoints reach Drive through the push loop, and the first VM, which holds
# the derived maps, evaluates them (eval_worker_runs.sh).
# Workers must be given disjoint model lists: they share one Drive folder, and each
# trains only its own models.
#
# Prerequisites: A100 runtime, Drive mounted at /content/drive, this repo cloned at
# /content/openlatent (branch replicate-selective-privacy).
#
# Usage: PRIVACY="..." ACTION="..." bash colab_worker.sh <worker_name>
#   e.g. PRIVACY="" ACTION="r2plus1d_r50:8:8 mvit_base_16x4:8:8" bash colab_worker.sh b
set -euo pipefail

NAME="${1:?usage: colab_worker.sh <worker_name>}"
: "${PRIVACY?set PRIVACY (may be empty)}" "${ACTION?set ACTION (may be empty)}"
WORK=/content/work
UP="$WORK/SelectivePrivacyPreservation"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$WORK/logs"

SKIP_DERIVED=1 bash "$HERE/colab_bootstrap.sh"

if ! pgrep -f "[w]hile true; do bash .*sync_drive.sh" >/dev/null; then
  setsid nohup bash -c "while true; do bash '$HERE/sync_drive.sh' push; sleep 300; done" \
    >> "$LOGS/sync_$NAME.log" 2>&1 < /dev/null &
fi

export MODE=train PRIVACY ACTION PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup bash "$HERE/run_kth_all_models.sh" "$UP" "$WORK/runs" \
  >> "$LOGS/all_models_$NAME.log" 2>&1 < /dev/null &

sleep 2
echo "worker $NAME started: PRIVACY=[$PRIVACY] ACTION=[$ACTION]; log $LOGS/all_models_$NAME.log"
