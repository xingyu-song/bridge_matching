#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Path setup
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# Python entry
# ============================================================
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="scripts/train_bm_2d.py"

# ============================================================
# Common experiment config
# ============================================================
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

LEARNING_RATE="${LEARNING_RATE:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ITERATIONS="${ITERATIONS:-20000}"
LOG_EVERY="${LOG_EVERY:-2000}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"

BETA_VALUE="${BETA_VALUE:-0.01}"
SIGMA_FLOOR="${SIGMA_FLOOR:-0.05}"
T_EPS="${T_EPS:-1e-2}"
LAMBDA_D="${LAMBDA_D:-1.0}"

POSTERIOR_TEMPERATURE="${POSTERIOR_TEMPERATURE:-1.0}"
POSTERIOR_TOPK="${POSTERIOR_TOPK:-0}"
SEED="${SEED:-42}"

# ============================================================
# Target type options
# Uncomment / edit as needed
# ============================================================
TARGET_TYPES=(
  "diffusion_cfm_decomp"
  # "diffusion_conditional"
  # "diffusion_marginal"
  # "linear_tube_conditional"
  # "linear_tube_symmetric_conditional"
  # "linear_kde_marginal"
  # "linear_conditional"
  # "linear_ot_conditional"
  # "linear_isotropic_conditional"
  # "linear_brownian_conditional"
  # "stochastic_interpolant_conditional"
  # "linear_anisotropic_conditional"
  # "linear_multibridge_conditional"
)

# ============================================================
# Source dataset options
# Keep one or many
# ============================================================
SOURCE_DATASETS=(
  "gaussian"
  # "moons"
  # "mixture"
  # "siggraph"
  # "checkerboard"
  # "invertocat"
)

# ============================================================
# Target dataset options
# Keep one or many
# ============================================================
TARGET_DATASETS=(
  "moons"
  "mixture"
  "siggraph"
  "checkerboard"
  "invertocat"
  "gaussian"
)

# ============================================================
# Logging
# ============================================================
SAVE_LOGS="${SAVE_LOGS:-1}"
LOG_ROOT="${LOG_ROOT:-logs/train_bm_2d}"
mkdir -p "${LOG_ROOT}"

# ============================================================
# Run mode
# 0 = run everything
# 1 = skip source == target
# ============================================================
SKIP_SAME_DATASET="${SKIP_SAME_DATASET:-0}"

# ============================================================
# Dry run
# 1 = print commands only
# 0 = actually run
# ============================================================
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Main loop
# ============================================================
for target_type in "${TARGET_TYPES[@]}"; do
  for source_dataset in "${SOURCE_DATASETS[@]}"; do
    for target_dataset in "${TARGET_DATASETS[@]}"; do

      if [[ "${SKIP_SAME_DATASET}" == "1" && "${source_dataset}" == "${target_dataset}" ]]; then
        echo "Skipping ${source_dataset} -> ${target_dataset} (same dataset)"
        continue
      fi

      cmd=(
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
        --source-dataset "${source_dataset}"
        --target-dataset "${target_dataset}"
        --output-dir "${OUTPUT_DIR}"
        --learning-rate "${LEARNING_RATE}"
        --batch-size "${BATCH_SIZE}"
        --iterations "${ITERATIONS}"
        --log-every "${LOG_EVERY}"
        --hidden-dim "${HIDDEN_DIM}"
        --beta-value "${BETA_VALUE}"
        --sigma-floor "${SIGMA_FLOOR}"
        --t-eps "${T_EPS}"
        --lambda-d "${LAMBDA_D}"
        --target-type "${target_type}"
        --posterior-temperature "${POSTERIOR_TEMPERATURE}"
        --posterior-topk "${POSTERIOR_TOPK}"
        --seed "${SEED}"
      )

      echo "=================================================="
      echo "Running:"
      echo "  target_type    = ${target_type}"
      echo "  source_dataset = ${source_dataset}"
      echo "  target_dataset = ${target_dataset}"
      echo "=================================================="

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%q ' "${cmd[@]}"
        echo
        continue
      fi

      if [[ "${SAVE_LOGS}" == "1" ]]; then
        log_file="${LOG_ROOT}/${target_type}__${source_dataset}2${target_dataset}.log"
        "${cmd[@]}" 2>&1 | tee "${log_file}"
      else
        "${cmd[@]}"
      fi

    done
  done
done