#!/bin/bash
# Final aggregation of SC-0 for one dataset on the VM: pre-registered grid (floors s0-2 + T_scvi arm) and the post-hoc dense-frontier sensitivity.
#   bash run_final.sh immune|pancreas|hlca [B] [Bcluster]
ds=$1; B=${2:-1000}; BC=${3:-0}
R=$HOME/work/runs; O=$HOME/work/e_agg; mkdir -p $O
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY=$HOME/venvs/e/bin/python; SRC=$HOME/openlatent/research/fields/src/e_singlecell
RUNS="$R/e_floors_${ds}_s0 $R/e_floors_${ds}_s1 $R/e_floors_${ds}_s2 $R/e_tscvi_${ds}/s0 $R/e_tscvi_${ds}/s1 $R/e_tscvi_${ds}/s2"
nice -n 10 $PY $SRC/aggregate.py --dataset $ds --runs $RUNS --B $B --cluster $BC --json $O/${ds}_final.json --md $O/${ds}_final.md > $O/${ds}_final.log 2>&1
# sensitivity: same plus the dense floor frontier (seed 0 only): changes only the inductive candidate pool of K3
nice -n 10 $PY $SRC/aggregate.py --dataset $ds --runs $RUNS $R/e_dense_${ds}_s0 --B 200 --json $O/${ds}_dense.json --md $O/${ds}_dense.md > $O/${ds}_dense.log 2>&1
echo done $ds
