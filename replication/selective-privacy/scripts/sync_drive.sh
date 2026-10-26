#!/usr/bin/env bash
# Mirror replication state between the VM's local disk and Google Drive.
#
#   push    : local -> Drive. Run checkpoints, DONE/provenance/eval JSON, logs, raw KTH
#             archives. Cheap, so it can run on a timer.
#   derived : local -> Drive for the derived inputs (AFD, flow, similarity maps). Run
#             once when a stage completes; it is slow on tens of thousands of files.
#   bundle  : local -> Drive, one tar per derived directory (bundles/<dir>.tar). A few
#             large files copy to Drive in minutes; tens of thousands of PNGs take hours.
#   pull    : Drive -> local. Restores everything after a VM restart, before resuming.
#             Derived directories come from bundles when present; SKIP_DERIVED=1 skips
#             them (training-only workers).
#   pull-runs : Drive -> local for runs/ only (picks up other workers' checkpoints).
#
# Local disk stays the working copy, so training and I/O stay fast. Drive is the
# durable copy. Frames under data/kth are not synced: they are re-extracted from the
# archives in a few minutes. rsync never deletes, so a partial sync cannot remove
# finished work on either side.
#
# Several VMs can share one Drive folder (see colab_worker.sh): each trains different
# models, and --update keeps a VM from overwriting a newer file on the other side with
# the older copy it pulled at startup.
#
# Usage: sync_drive.sh push|derived|bundle|pull
set -euo pipefail

MODE="${1:?usage: sync_drive.sh push|derived|bundle|pull}"
LOCAL="${LOCAL:-/content/work}"
UP="$LOCAL/SelectivePrivacyPreservation"
REMOTE="${REMOTE:-/content/drive/MyDrive/openlatent-replication}"
DERIVED="kth_afd kth_flow kth_torso_sim kth_arm_sim kth_leg_sim kth_hair_sim"
TARGETED="kth_torsoT_sim kth_armT_sim kth_legT_sim kth_hairT_sim"

test -d "$REMOTE" || { echo "Drive not mounted at $REMOTE" >&2; exit 1; }

# sync SRC DST: copy a directory's contents, creating DST. Missing SRC is skipped.
sync_dir() {
  local src="$1" dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    rsync -a --partial --update "$src/" "$dst/"
  fi
}

# push: small, frequently changing state. Safe to run every few minutes. Derived inputs
# are excluded: they hold tens of thousands of files, and rescanning them on Drive on
# every pass made the push take 20+ minutes. Sync them with `derived` once per stage.
push() {
  sync_dir "$LOCAL/runs" "$REMOTE/runs"
  sync_dir "$LOCAL/logs" "$REMOTE/logs"
  mkdir -p "$REMOTE/raw"
  shopt -s nullglob
  for z in "$LOCAL"/kth_raw/*.zip; do
    rsync -a --partial "$z" "$REMOTE/raw/"
  done
  shopt -u nullglob
  if [ -f "$UP/data/splits/kth/metadata.csv" ]; then
    mkdir -p "$REMOTE/splits/kth"
    rsync -a "$UP/data/splits/kth/metadata.csv" "$REMOTE/splits/kth/"
  fi
}

# derived: the expensive inputs. Run once a stage is complete, not on a timer.
derived() {
  for d in $DERIVED; do
    sync_dir "$UP/data/$d" "$REMOTE/derived/$d"
  done
}

# bundle: tar each complete derived directory to Drive, written under a temporary name
# and renamed, so a bundle on Drive is never partial.
bundle() {
  mkdir -p "$REMOTE/bundles"
  for d in $DERIVED $TARGETED; do
    if [ -d "$UP/data/$d" ]; then
      tar -C "$UP/data" -cf "$REMOTE/bundles/.$d.tar.tmp" "$d"
      mv "$REMOTE/bundles/.$d.tar.tmp" "$REMOTE/bundles/$d.tar"
      echo "  bundled $d: $(du -h "$REMOTE/bundles/$d.tar" | cut -f1)"
    fi
  done
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
  if [ -f "$REMOTE/splits/kth/metadata.csv" ]; then
    mkdir -p "$UP/data/splits/kth"
    rsync -a "$REMOTE/splits/kth/metadata.csv" "$UP/data/splits/kth/"
  fi
  # training-only workers do not need the derived inputs; skip the slow restore
  [ "${SKIP_DERIVED:-0}" = 1 ] && return 0
  mkdir -p "$UP/data"
  for d in $DERIVED $TARGETED; do
    if [ -f "$REMOTE/bundles/$d.tar" ]; then
      if [ ! -d "$UP/data/$d" ]; then
        # extract beside the target and rename, so an interrupted pull leaves no partial dir
        rm -rf "$UP/data/.extract_$d" && mkdir -p "$UP/data/.extract_$d"
        tar -C "$UP/data/.extract_$d" -xf "$REMOTE/bundles/$d.tar"
        mv "$UP/data/.extract_$d/$d" "$UP/data/$d" && rmdir "$UP/data/.extract_$d"
      fi
    else
      sync_dir "$REMOTE/derived/$d" "$UP/data/$d"
    fi
  done
}

case "$MODE" in
  push) push ;;
  pull) pull ;;
  pull-runs) sync_dir "$REMOTE/runs" "$LOCAL/runs" ;;
  derived) derived ;;
  bundle) bundle ;;
  *) echo "usage: sync_drive.sh push|derived|bundle|pull|pull-runs" >&2; exit 2 ;;
esac

echo "sync $MODE done: $(date -u +%H:%M:%S)"
