#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_DIR="outputs/bm/cifar10"

RUN_DIRS=(
  "diffusion_cfm_decomp/dcd_20260419123216"
)

LAMBDA_PAIRS=(
  # "1 1"
  # "1 0"
  "1 0.1"
  "1 0.2"
  "1 0.3"
  "1 0.4"
  "1 0.5"
  "1 0.6"
  "1 0.7"
  "1 0.8"
  "1 0.9"
  "0 1"
)

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-80}"
NUM_OUTPUT_STEPS="${NUM_OUTPUT_STEPS:-101}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-10}"
FPS="${FPS:-20}"
METRICS_BATCH_SIZE="${METRICS_BATCH_SIZE:-64}"
NUM_METRIC_SAMPLES="${NUM_METRIC_SAMPLES:-1000}"
METRICS_SUBSET_SIZE="${METRICS_SUBSET_SIZE:-1000}"
METRICS_NUM_SPLITS="${METRICS_NUM_SPLITS:-5}"
METHOD="${METHOD:-midpoint}"
SEED="${SEED:-42}"

DO_SAMPLE="${DO_SAMPLE:-1}"
DO_BACKWARD="${DO_BACKWARD:-0}"
DO_GRID="${DO_GRID:-0}"
DO_TRAJECTORY="${DO_TRAJECTORY:-0}"
DO_METRICS="${DO_METRICS:-1}"

for run_rel in "${RUN_DIRS[@]}"; do
  ckpt_path="${BASE_DIR}/${run_rel}/ckpt.pth"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt_path}"
    continue
  fi

  method_name="$(dirname "${run_rel}")"
  run_name="$(basename "${run_rel}")"

  for pair in "${LAMBDA_PAIRS[@]}"; do
    read -r lambda_u lambda_d <<< "${pair}"

    echo "=================================================="
    echo "Evaluating: ${run_rel}"
    echo "lambda_u=${lambda_u}, lambda_d=${lambda_d}"
    echo "=================================================="

    python scripts/eval_bm_on_image.py \
      --ckpt-path "${ckpt_path}" \
      --lambda_eval_u "${lambda_u}" \
      --lambda_eval_d "${lambda_d}" \
      $( [[ "${DO_SAMPLE}" == "1" ]] && echo --do-sample ) \
      $( [[ "${DO_BACKWARD}" == "1" ]] && echo --do-backward ) \
      $( [[ "${DO_GRID}" == "1" ]] && echo --do-grid ) \
      $( [[ "${DO_TRAJECTORY}" == "1" ]] && echo --do-trajectory ) \
      $( [[ "${DO_METRICS}" == "1" ]] && echo --do-metrics ) \
      --batch-size "${BATCH_SIZE}" \
      --num-inference-steps "${NUM_INFERENCE_STEPS}" \
      --num-output-steps "${NUM_OUTPUT_STEPS}" \
      --samples-per-class "${SAMPLES_PER_CLASS}" \
      --fps "${FPS}" \
      --metrics-batch-size "${METRICS_BATCH_SIZE}" \
      --num-metric-samples "${NUM_METRIC_SAMPLES}" \
      --metrics-subset-size "${METRICS_SUBSET_SIZE}" \
      --metrics-num-splits "${METRICS_NUM_SPLITS}" \
      --method "${METHOD}" \
      --seed "${SEED}" \
      # 2>&1 | tee "${log_file}"
  done
done