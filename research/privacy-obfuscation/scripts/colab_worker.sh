#!/usr/bin/env bash
# Turn a Colab A100 runtime into a queue worker. Needs Drive mounted at /content/drive and this
# repo cloned at ~/openlatent (branch privacy-research).
#
# Jobs come from Drive (MyDrive/ppv/queue_<name>/pending), written by the controller; results
# are mirrored to MyDrive/ppv/runs/<run>/ (and uploaded to HF when an HF token is configured).
# Idempotent: rerun after a disconnect; the worker requeues jobs that were running.
#
# Usage: bash colab_worker.sh <name> [slots]
set -euo pipefail

NAME="${1:?usage: colab_worker.sh <name> [slots]}"
SLOTS="${2:-2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVE=/content/drive/MyDrive/ppv
test -d /content/drive/MyDrive || { echo "mount Drive first" >&2; exit 1; }
mkdir -p "$DRIVE/queue_$NAME" "$DRIVE/runs" ~/work/logs

bash "$HERE/setup_vm.sh"
bash "$HERE/download_data.sh" kth
cd "$HERE/.."
[ -f ~/work/data/kth/meta.csv ] || python3 -c "from ppv import kth; import os; kth.prepare(os.path.expanduser('~/work/data/raw/kth'), os.path.expanduser('~/work/data/kth'))"

if ! pgrep -f "^python3 .*worker.py" >/dev/null; then
  PPV_QUEUE="$DRIVE/queue_$NAME" PPV_MIRROR="$DRIVE/runs" setsid nohup python3 "$HERE/worker.py" --slots "$SLOTS" \
    >> ~/work/logs/worker.log 2>&1 < /dev/null &
fi
sleep 2
echo "colab worker $NAME up: queue $DRIVE/queue_$NAME, results $DRIVE/runs"
