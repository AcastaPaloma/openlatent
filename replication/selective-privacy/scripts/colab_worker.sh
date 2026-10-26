#!/usr/bin/env bash
# Turn a fresh Colab VM into an extra worker for the KTH model queue.
#
# Restores state and derived inputs from Drive (colab_bootstrap.sh; the AFD and similarity
# maps come from bundles/, written by `sync_drive.sh bundle` on the first VM), starts the
# Drive push loop, then runs run_kth_all_models.sh on this worker's model lists, detached.
# Workers must be given disjoint model lists: they share one Drive folder, and each
# trains and evaluates only its own models.
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
REMOTE=/content/drive/MyDrive/openlatent-replication

for d in kth_afd kth_torso_sim kth_arm_sim kth_leg_sim kth_hair_sim kth_hairT_sim; do
  test -f "$REMOTE/bundles/$d.tar" || { echo "missing bundle $d.tar on Drive" >&2; exit 1; }
done

bash "$HERE/colab_bootstrap.sh"

if ! pgrep -f "[w]hile true; do bash .*sync_drive.sh" >/dev/null; then
  setsid nohup bash -c "while true; do bash '$HERE/sync_drive.sh' push; sleep 300; done" \
    >> "$LOGS/sync_$NAME.log" 2>&1 < /dev/null &
fi

export PRIVACY ACTION PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup bash "$HERE/run_kth_all_models.sh" "$UP" "$WORK/runs" \
  >> "$LOGS/all_models_$NAME.log" 2>&1 < /dev/null &

sleep 2
echo "worker $NAME started: PRIVACY=[$PRIVACY] ACTION=[$ACTION]; log $LOGS/all_models_$NAME.log"
