#!/usr/bin/env bash
# Build the KTH inputs the selective "Ours" evaluation needs, for the test split only:
#   1. motion-consistent noise (AFD) and optical flow: data/kth_afd, data/kth_flow
#      (derive_kth_afd.py: upstream AFD math with batched RAFT)
#   2. DINO similarity maps for the four KTH templates: data/kth_<t>_sim, t in torso, arm,
#      leg, hair (derive_kth_sims.py: one DINO pass per frame shared by all templates)
#
# Obfuscation is applied only at test time, so train-split clips need neither.
# Both drivers skip finished work, so rerunning after an interruption resumes. The
# similarity driver fails unless every test clip has one map per frame for every template.
#
# Usage: prepare_derived_kth.sh <upstream_dir>
set -euo pipefail

UP="${1:?usage: prepare_derived_kth.sh <upstream_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AFD_BATCH="${AFD_BATCH:-8}"
SIM_BATCH="${SIM_BATCH:-8}"

test -d "$UP/data/kth" || { echo "missing $UP/data/kth; extract frames first" >&2; exit 1; }

cd "$UP"

echo "== AFD + optical flow, test split, RAFT batch $AFD_BATCH =="
python "$HERE/derive_kth_afd.py" --batch "$AFD_BATCH"

echo "== similarity maps, test split, all templates, DINO batch $SIM_BATCH =="
python "$HERE/derive_kth_sims.py" --batch "$SIM_BATCH"

echo "KTH derived inputs (test split) done:"
for d in kth_afd kth_flow kth_torso_sim kth_arm_sim kth_leg_sim kth_hair_sim; do
  if [ -d "$UP/data/$d" ]; then
    echo "  $d: $(find "$UP/data/$d" -type f | wc -l) files"
  fi
done
