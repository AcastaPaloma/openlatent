#!/usr/bin/env bash
# Start the KTH pipeline's background jobs, detached from the notebook kernel.
#
#   - Drive push loop every 5 min (checkpoints, results, logs)
#   - derived inputs (AFD + similarity maps), mirrored to Drive one clip at a time
#   - evaluation chain (original + obfuscated, for each trained model)
#
# Each job runs in its own session with no inherited pipes, so interrupting or
# losing a notebook cell cannot kill it. Jobs already running are not started twice.
#
# Usage: bash colab_launch.sh
set -euo pipefail

WORK=/content/work
UP="$WORK/SelectivePrivacyPreservation"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$WORK/logs"
export DERIVED_MIRROR=/content/drive/MyDrive/openlatent-replication/derived_clips
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS" "$DERIVED_MIRROR"

running() { pgrep -f "$1" >/dev/null; }

if ! running "[w]hile true; do bash .*sync_drive.sh"; then
  setsid nohup bash -c "while true; do bash '$HERE/sync_drive.sh' push; sleep 300; done" \
    >> "$LOGS/sync.log" 2>&1 < /dev/null &
fi

if ! running "[p]repare_derived_kth.sh"; then
  (cd "$UP" && setsid nohup bash "$HERE/prepare_derived_kth.sh" "$UP" \
    >> "$LOGS/derived_kth_test.log" 2>&1 < /dev/null &)
fi

if ! running "[r]un_kth_eval.sh"; then
  setsid nohup bash "$HERE/run_kth_eval.sh" "$UP" "$WORK/runs" "$LOGS/derived_kth_test.log" \
    >> "$LOGS/eval_kth.log" 2>&1 < /dev/null &
fi

sleep 2
pgrep -fa "[s]ync_drive|[p]repare_derived|[r]un_kth_eval|[d]erive_kth" | cut -c1-120
