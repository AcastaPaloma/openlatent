#!/usr/bin/env bash
# Train and evaluate the paper's model sets on KTH, one model at a time.
#
# Privacy: resnet18, resnet50, resnet101, vit, vit_b_32 (upstream privacy_models).
# Action:  every upstream action model except e2s_x3d_{s,m,l}, which need optical flow
#          for the training split and upstream's separate E2S trainer (not built here).
#
# Privacy models run first: they decide the privacy verdict and are much cheaper.
# For each model: resumable training (train_model.py), then evaluation on original
# frames, default-matcher obfuscation and targeted-matcher obfuscation. A model whose
# training or eval fails is logged and skipped; the queue continues. Finished runs
# (DONE.json) and finished evals (their JSON) are skipped, so the queue can be relaunched.
#
# Batch sizes keep the effective batch at upstream's 64 (micro-batch x accumulation),
# sized so each model fits on an A100-40GB.
#
# To split the queue across VMs (or across lanes on one GPU), set PRIVACY and ACTION to
# disjoint subsets; an empty string runs none of that kind. See colab_worker.sh.
#
# Usage: [PRIVACY="..."] [ACTION="..."] run_kth_all_models.sh <upstream_dir> <out_dir>
set -uo pipefail

UP="${1:?usage: run_kth_all_models.sh <upstream_dir> <out_dir>}"
OUT="${2:?usage: run_kth_all_models.sh <upstream_dir> <out_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_OBF="torso arm leg hair"
TARGETED_OBF="torsoT armT legT hairT"

# name:micro_batch:accumulation
PRIVACY="${PRIVACY-resnet18:128:1 resnet50:128:1 resnet101:128:1 vit:64:2 vit_b_32:128:1}"
ACTION="${ACTION-x3d_s:32:2 x3d_m:16:4 x3d_l:8:8 slow_r50:16:4 i3d_r50:16:4 c2d_r50:16:4 \
r2plus1d_r50:8:8 slowfast_r50:8:8 csn_r101:8:8 mvit_base_16x4:8:8}"

cd "$UP"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

targeted_ready() {
  # every targeted template has one map per frame for every test clip
  # derive_kth_sims.py --targeted logs plain template names ("[hair] verified")
  grep -q "\[hair\] verified" "$UP/../logs/targeted_kth.log" 2>/dev/null
}

run_model() {
  local kind="$1" arch="$2" bs="$3" acc="$4"
  local flags=(--architecture "$arch" --datasetname kth -pretrained -train_backbone --out "$OUT")
  local epochs=80
  if [ "$kind" = privacy ]; then
    flags+=(-privacy)
    epochs=500
  fi

  log "== $kind $arch: train (batch $bs x $acc, $epochs epochs)"
  if ! python "$HERE/train_model.py" "${flags[@]}" --num_epochs "$epochs" \
      --batch_size "$bs" --accumulate_grad_batches "$acc"; then
    log "!! $kind $arch: training failed; skipping its evals"
    return 0
  fi

  local name
  name=$(ls "$OUT/$kind/kth" | grep -E "^${arch}(__|____)" | head -1)
  local evaldir="$OUT/eval/$kind/kth/$name"

  for cond in original default targeted; do
    local json extra=()
    case "$cond" in
      original) json="$evaldir/original.json" ;;
      default)  json="$evaldir/obf_arm_hair_leg_torso.json"; extra=(-selectively_mask --obfuscate $DEFAULT_OBF) ;;
      targeted) json="$evaldir/obf_armT_hairT_legT_torsoT.json"; extra=(-selectively_mask --obfuscate $TARGETED_OBF)
                until targeted_ready; do sleep 60; done ;;
    esac
    if [ -f "$json" ]; then
      log "   $cond: exists"
      continue
    fi
    if ! python "$HERE/eval_model.py" "${flags[@]}" --batch_size "$bs" "${extra[@]}"; then
      log "!! $kind $arch: $cond eval failed"
    fi
  done
}

for spec in $PRIVACY; do
  IFS=: read -r arch bs acc <<< "$spec"
  run_model privacy "$arch" "$bs" "$acc"
done
for spec in $ACTION; do
  IFS=: read -r arch bs acc <<< "$spec"
  run_model action "$arch" "$bs" "$acc"
done
log "all KTH models done"
