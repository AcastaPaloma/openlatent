#!/usr/bin/env bash
# Rebuild a fresh Colab VM from GitHub + Google Drive, then verify the KTH data.
# Idempotent: every step skips work that is already done. Run after any disconnect.
#
# Prerequisites: GPU runtime, Drive mounted at /content/drive, and this repo cloned at
# /content/openlatent (branch replicate-selective-privacy).
#
# SKIP_DERIVED=1 skips restoring the derived AFD and similarity maps (training-only workers).
#
# Usage: bash colab_bootstrap.sh
set -euo pipefail

WORK=/content/work
UP="$WORK/SelectivePrivacyPreservation"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE=/content/drive/MyDrive/openlatent-replication

test -d "$REMOTE" || { echo "Drive folder $REMOTE missing; mount Drive first" >&2; exit 1; }
mkdir -p "$WORK/logs"

echo "== upstream at 7e8b98d =="
if [ ! -d "$UP/.git" ]; then
  git clone -q https://github.com/f-ilic/SelectivePrivacyPreservation "$UP"
fi
git -C "$UP" checkout -q 7e8b98d

echo "== python deps =="
pip install -q --no-deps pytorchvideo==0.1.3
pip install -q av fvcore iopath parameterized tbparse termplotlib h5py seaborn

echo "== restore state from Drive =="
bash "$HERE/sync_drive.sh" pull

echo "== KTH frames (archives come from Drive, so no download) =="
JOBS=12 bash "$HERE/prepare_kth.sh" "$UP" "$WORK/kth_raw"
cd "$UP"
python "$HERE/fix_kth_metadata.py"
python "$HERE/verify_kth.py"

echo "bootstrap done"
