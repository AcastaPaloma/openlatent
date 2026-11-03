#!/bin/bash
# runone.sh GPU ARM SEED [STEPS] [SIGMA]  -- dev-half run on one GPU
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12; CUDA_VISIBLE_DEVICES=$1 python3 train_arms.py $2 $3 ${4:-1500} $5 > logs/arm_$2_s$3.log 2>&1
