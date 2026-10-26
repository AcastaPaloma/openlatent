#!/usr/bin/env bash
# Evaluate the KTH models on original and obfuscated test frames, as their inputs finish.
#
# Original-frame evaluations run as soon as each model's training run writes DONE.json.
# Obfuscated evaluations wait for prepare_derived_kth.sh to report that the test-split
# AFD and similarity maps are complete. Each eval writes its own JSON, so the script
# can be rerun safely.
#
# Usage: run_kth_eval.sh <upstream_dir> <out_dir> <derived_log>
set -euo pipefail

UP="${1:?usage: run_kth_eval.sh <upstream_dir> <out_dir> <derived_log>}"
OUT="${2:?usage: run_kth_eval.sh <upstream_dir> <out_dir> <derived_log>}"
DERIVED_LOG="${3:?usage: run_kth_eval.sh <upstream_dir> <out_dir> <derived_log>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATES="${TEMPLATES:-torso arm leg hair}"

ACT_ARCH="x3d_s"
ACT_NAME="x3d_s__13x6____pretrained__True____train_backbone__True"
PRIV_ARCH="resnet18"
PRIV_NAME="resnet18____pretrained__True____train_backbone__True"

wait_for() {
  until [ -f "$1" ]; do sleep 60; done
}

cd "$UP"

wait_for "$OUT/action/kth/$ACT_NAME/DONE.json"
python "$HERE/eval_model.py" --architecture "$ACT_ARCH" --datasetname kth -pretrained -train_backbone \
  --batch_size 32 --out "$OUT"

wait_for "$OUT/privacy/kth/$PRIV_NAME/DONE.json"
python "$HERE/eval_model.py" --architecture "$PRIV_ARCH" --datasetname kth -pretrained -train_backbone -privacy \
  --batch_size 128 --out "$OUT"

until grep -q "derived inputs (test split) done" "$DERIVED_LOG" 2>/dev/null; do sleep 60; done

python "$HERE/eval_model.py" --architecture "$ACT_ARCH" --datasetname kth -pretrained -train_backbone \
  --batch_size 32 --out "$OUT" -selectively_mask --obfuscate $TEMPLATES

python "$HERE/eval_model.py" --architecture "$PRIV_ARCH" --datasetname kth -pretrained -train_backbone -privacy \
  --batch_size 128 --out "$OUT" -selectively_mask --obfuscate $TEMPLATES

echo "KTH evaluations done"
