#!/usr/bin/env bash
# Mirror replication state between the VM's local disk and Google Drive.
#
#   push : local -> Drive. Run checkpoints, DONE/provenance/eval JSON, logs, raw KTH
#          archives, and the expensive derived inputs (AFD, flow, similarity maps).
#   pull : Drive -> local. Use after a VM restart, before resuming the jobs.
#
# Local disk stays the working copy, so training and I/O stay fast. Drive is the
# durable copy. Frames under data/kth are not synced: they are re-extracted from the
# archives in a few minutes. rsync never deletes, so a partial sync cannot remove
# finished work on either side.
#
# Usage: sync_drive.sh push|pull
set -euo pipefail

MODE="${1:?usage: sync_drive.sh push|pull}"
LOCAL="${LOCAL:-/content/work}"
UP="$LOCAL/SelectivePrivacyPreservation"
REMOTE="${REMOTE:-/content/drive/MyDrive/openlatent-replication}"
DERIVED="kth_afd kth_flow kth_torso_sim kth_arm_sim kth_leg_sim kth_hair_sim"

test -d "$REMOTE" || { echo "Drive not mounted at $REMOTE" >&2; exit 1; }

# sync SRC DST: copy a directory's contents, creating DST. Missing SRC is skipped.
sync_dir() {
  local src="$1" dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    rsync -a --partial "$src/" "$dst/"
  fi
}

push() {
  sync_dir "$LOCAL/runs" "$REMOTE/runs"
  sync_dir "$LOCAL/logs" "$REMOTE/logs"
  mkdir -p "$REMOTE/raw"
  shopt -s nullglob
  for z in "$LOCAL"/kth_raw/*.zip; do
    rsync -a --partial "$z" "$REMOTE/raw/"
  done
  shopt -u nullglob
  for d in $DERIVED; do
    sync_dir "$UP/data/$d" "$REMOTE/derived/$d"
  done
  if [ -f "$UP/data/splits/kth/metadata.csv" ]; then
    mkdir -p "$REMOTE/splits/kth"
    rsync -a "$UP/data/splits/kth/metadata.csv" "$REMOTE/splits/kth/"
  fi
}

pull() {
  sync_dir "$REMOTE/runs" "$LOCAL/runs"
  sync_dir "$REMOTE/logs" "$LOCAL/logs"
  mkdir -p "$LOCAL/kth_raw"
  shopt -s nullglob
  for z in "$REMOTE"/raw/*.zip; do
    rsync -a --partial "$z" "$LOCAL/kth_raw/"
  done
  shopt -u nullglob
  for d in $DERIVED; do
    sync_dir "$REMOTE/derived/$d" "$UP/data/$d"
  done
  if [ -f "$REMOTE/splits/kth/metadata.csv" ]; then
    mkdir -p "$UP/data/splits/kth"
    rsync -a "$REMOTE/splits/kth/metadata.csv" "$UP/data/splits/kth/"
  fi
}

case "$MODE" in
  push) push ;;
  pull) pull ;;
  *) echo "usage: sync_drive.sh push|pull" >&2; exit 2 ;;
esac

echo "sync $MODE done: $(date -u +%H:%M:%S)"
