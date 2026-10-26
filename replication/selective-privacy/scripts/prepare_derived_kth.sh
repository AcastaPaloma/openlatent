#!/usr/bin/env bash
# Build the KTH inputs the selective "Ours" evaluation needs, for the test split only:
#   1. motion-consistent noise (AFD) and optical flow: data/kth_afd, data/kth_flow
#      (derive_kth_afd.py: upstream AFD math with batched RAFT)
#   2. per-template DINO similarity maps: data/kth_<t>_sim for t in torso, arm, leg, hair
#      (derive_kth_sims.py: upstream process_video on the test clips)
#
# Obfuscation is applied only at test time, so train-split clips need neither.
# Both drivers skip finished videos, so rerunning after an interruption resumes.
#
# Usage: prepare_derived_kth.sh <upstream_dir>
set -euo pipefail

UP="${1:?usage: prepare_derived_kth.sh <upstream_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATES="${TEMPLATES:-torso arm leg hair}"
AFD_BATCH="${AFD_BATCH:-8}"

test -d "$UP/data/kth" || { echo "missing $UP/data/kth; extract frames first" >&2; exit 1; }

cd "$UP"

echo "== AFD + optical flow, test split, RAFT batch $AFD_BATCH =="
python "$HERE/derive_kth_afd.py" --batch "$AFD_BATCH"

echo "== similarity maps, test split, templates: $TEMPLATES (one process per template) =="
LOG_DIR="${LOG_DIR:-$UP/derived_logs}"
mkdir -p "$LOG_DIR"
pids=""
for t in $TEMPLATES; do
  python "$HERE/derive_kth_sims.py" --templates "$t" > "$LOG_DIR/sims_$t.log" 2>&1 &
  pids="$pids $!"
done
failed=0
for pid in $pids; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "similarity map generation failed; see $LOG_DIR/sims_*.log" >&2
  exit 1
fi

echo "KTH derived inputs (test split) done:"
for d in kth_afd kth_flow $(for t in $TEMPLATES; do echo kth_${t}_sim; done); do
  if [ -d "$UP/data/$d" ]; then
    echo "  $d: $(find "$UP/data/$d" -type f | wc -l) files"
  fi
done
