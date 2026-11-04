#!/bin/bash
# usage: make_jobs.sh <tag> <encoder-short> <enc-full> <priority> [extra run_sweep args]   (writes specs to research/fields/jobs/specs)
tag=$1; short=$2; enc=$3; pr=$4; shift 4
for ds in camelyon tcga tolkach_esca; do for s in 0 1 2; do
  n=a_${tag}_${short}_${ds}_s${s}
  cat > research/fields/jobs/specs/$n.yaml <<EOT
name: $n
priority: $pr
cmd: USE_TF=0 \$HOME/venvs/a/bin/python research/fields/src/a_pathorob/run_sweep.py --encoder $enc --dataset $ds --seed $s --grid all $@
hf_path: runs/$n
EOT
done; done
