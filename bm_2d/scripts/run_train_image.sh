#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  "linear_conditional"
  "linear_ot_conditional"
  "linear_isotropic_conditional"
  "linear_brownian_conditional"
  "stochastic_interpolant_conditional"
  "linear_anisotropic_conditional"
  "linear_multibridge_conditional"
)

DATASET="cifar10"
LOG_DIR="logs/image_targets"
mkdir -p "${LOG_DIR}"

for target in "${TARGETS[@]}"; do
  echo "=================================================="
  echo "Training image BM with target_type=${target}"
  echo "Dataset=${DATASET}"
  echo "=================================================="

  python scripts/train_bm_on_image.py \
    --dataset "${DATASET}" \
    --target-type "${target}" \
    --batch-size 128 \
    --n-epochs 10 \
    --learning-rate 1e-3 \
    --beta-value 1e-2 \
    --sigma-floor 5e-2 \
    --t-eps 1e-2 \
    --lambda-d 1.0 \
    --lambda-forward-align 0.0 \
    --posterior-temperature 1.0 \
    --posterior-topk 0 \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/${DATASET}_${target}.log"
done