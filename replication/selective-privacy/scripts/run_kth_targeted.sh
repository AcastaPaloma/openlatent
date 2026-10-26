#!/usr/bin/env bash
# Ablation: upstream's use_targeted=True matcher setting on the KTH test split.
#
# The default-setting run gave obfuscation weights of at most 0.42, so privacy barely
# dropped. With use_targeted=True each template is replaced by its best-matching token in
# the clip's first frame, which can only raise similarities. This writes kth_<t>T_sim
# (mirrored per clip to Drive), then evaluates both trained models with
# --obfuscate torsoT armT legT hairT. Finished evaluations are skipped on rerun.
#
# Usage: run_kth_targeted.sh <upstream_dir> <out_dir>
set -euo pipefail

UP="${1:?usage: run_kth_targeted.sh <upstream_dir> <out_dir>}"
OUT="${2:?usage: run_kth_targeted.sh <upstream_dir> <out_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TEMPLATES="torsoT armT legT hairT"
TAG="obf_armT_hairT_legT_torsoT"
ACT_NAME="x3d_s__13x6____pretrained__True____train_backbone__True"
PRIV_NAME="resnet18____pretrained__True____train_backbone__True"

cd "$UP"
python "$HERE/derive_kth_sims.py" --targeted --batch "${SIM_BATCH:-8}"

if [ ! -f "$OUT/eval/action/kth/$ACT_NAME/$TAG.json" ]; then
  python "$HERE/eval_model.py" --architecture x3d_s --datasetname kth -pretrained -train_backbone \
    --batch_size 32 --out "$OUT" -selectively_mask --obfuscate $TEMPLATES
fi
if [ ! -f "$OUT/eval/privacy/kth/$PRIV_NAME/$TAG.json" ]; then
  python "$HERE/eval_model.py" --architecture resnet18 --datasetname kth -pretrained -train_backbone -privacy \
    --batch_size 128 --out "$OUT" -selectively_mask --obfuscate $TEMPLATES
fi
echo "KTH targeted ablation done"
