#!/usr/bin/env bash
# Build the KTH inputs that the selective "Ours" path needs, on top of extracted frames:
#   1. motion-consistent noise (AFD) and optical flow: data/kth_afd, data/kth_flow
#   2. per-template similarity maps for the KTH obfuscation set: data/kth_<t>_sim
#      for t in torso, arm, leg, hair (the paper's KTH/SBU template set)
#
# Both upstream tools skip work that is already done (generate_kth.py compares file
# counts per class; similarity_from_template_for_dataset.py checks each video folder),
# so rerunning after an interruption resumes instead of starting over.
#
# Usage: prepare_derived_kth.sh <upstream_dir>
set -euo pipefail

UP="${1:?usage: prepare_derived_kth.sh <upstream_dir>}"
TEMPLATES="${TEMPLATES:-torso arm leg hair}"

test -d "$UP/data/kth" || { echo "missing $UP/data/kth; extract frames first" >&2; exit 1; }

echo "== AFD + optical flow for KTH =="
( cd "$UP/afd" && python generate_kth.py --src ../data/kth )

echo "== similarity maps for templates: $TEMPLATES =="
cd "$UP"
for t in $TEMPLATES; do
  test -f "output/descriptors/$t.pt" || { echo "missing descriptor output/descriptors/$t.pt" >&2; exit 1; }
  echo "-- $t"
  python matcher/similarity_from_template_for_dataset.py \
    -descriptorpath "output/descriptors/$t.pt" -dataset kth
done

echo "KTH derived inputs done:"
for d in kth_afd kth_flow kth_torso_sim kth_arm_sim kth_leg_sim kth_hair_sim; do
  if [ -d "$UP/data/$d" ]; then
    echo "  $d: $(find "$UP/data/$d" -type f | wc -l) files"
  fi
done
