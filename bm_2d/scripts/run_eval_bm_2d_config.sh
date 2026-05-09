#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Path setup
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="scripts/eval_bm_2d.py"

# ============================================================
# Base directory (auto search root)
# ============================================================
BASE_DIR="${BASE_DIR:-outputs/bm}"

# ============================================================
# Dataset filters (lists; comment out entries you don't want)
# ============================================================
SOURCE_DATASETS=(
  "gaussian"
  # "moons"
  # "mixture"
  # "siggraph"
  # "checkerboard"
  # "invertocat"
)

TARGET_DATASETS=(
  "moons"
  "mixture"
  "siggraph"
  "checkerboard"
  "invertocat"
  # "gaussian"
)

# specific run names (e.g. lc_2026xxxx). Leave empty to disable manual filtering.
RUN_NAMES=(
  # "lc_2026xxxx"
  # "dcd_2026xxxx"
)

# automatic run-name discovery by prefix inside each <source>2<target> folder.
# example: "sic" will match sic_2026xxxx runs.
RUN_NAME_PREFIXES=(
  "dcd"
  # "sic"
)

# 1 = auto-discover run names using RUN_NAME_PREFIXES
# 0 = use RUN_NAMES only
AUTO_DISCOVER_RUN_NAMES="${AUTO_DISCOVER_RUN_NAMES:-0}"

# Skip source==target pairs (0/1)
SKIP_SAME_DATASET="${SKIP_SAME_DATASET:-0}"

# ============================================================
# Lambda settings (multi-run)
# ============================================================
LAMBDA_PAIRS=(
  "1 1"
  "1 0"
  "0 1"
  # "0.5 1"
)

# ============================================================
# Evaluation flags
# ============================================================
DO_SAMPLING_PLOT="${DO_SAMPLING_PLOT:-1}"
DO_GIF="${DO_GIF:-0}"
DO_LIKELIHOOD="${DO_LIKELIHOOD:-0}"
DO_BACKWARD="${DO_BACKWARD:-1}"

# ============================================================
# Sampling parameters
# ============================================================
SAMPLING_NUM_SAMPLES="${SAMPLING_NUM_SAMPLES:-1000000}"
SAMPLING_STEP_SIZE="${SAMPLING_STEP_SIZE:-0.05}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"

GIF_SAMPLE_STEPS="${GIF_SAMPLE_STEPS:-101}"
GIF_GRID_SIZE="${GIF_GRID_SIZE:-15}"
GIF_NUM_SAMPLES="${GIF_NUM_SAMPLES:-500000}"
GIF_INTERVAL="${GIF_INTERVAL:-50}"

# ============================================================
# Logging
# ============================================================
SAVE_LOGS="${SAVE_LOGS:-1}"
LOG_ROOT="${LOG_ROOT:-logs/eval_bm_2d}"
mkdir -p "${LOG_ROOT}"

# ============================================================
# Dry run (debug)
# ============================================================
DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# Build optional flags
# ============================================================
FLAGS=()

[[ "${DO_SAMPLING_PLOT}" == "1" ]] && FLAGS+=(--do-sampling-plot)
[[ "${DO_GIF}" == "1" ]] && FLAGS+=(--do-gif)
[[ "${DO_LIKELIHOOD}" == "1" ]] && FLAGS+=(--do-likelihood)
[[ "${DO_BACKWARD}" == "1" ]] && FLAGS+=(--do-backward)

collect_run_names_for_pair() {
  local pair_dir="$1"
  local discovered=()

  if [[ "${AUTO_DISCOVER_RUN_NAMES}" == "1" ]]; then
    if [[ -d "${pair_dir}" ]]; then
      for prefix in "${RUN_NAME_PREFIXES[@]}"; do
        [[ -n "${prefix}" ]] || continue
        # Current layout: outputs/bm/<source>2<target>/<run_name>/ckpt.pth
        while IFS= read -r run_dir; do
          discovered+=("$(basename "${run_dir}")")
        done < <(find "${pair_dir}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" | sort)
      done
    fi
  fi

  if [[ ${#discovered[@]} -gt 0 ]]; then
    printf '%s\n' "${discovered[@]}" | awk '!seen[$0]++'
    return
  fi

  if [[ ${#RUN_NAMES[@]} -gt 0 ]]; then
    printf '%s\n' "${RUN_NAMES[@]}"
    return
  fi

  if [[ -d "${pair_dir}" ]]; then
    while IFS= read -r run_dir; do
      basename "${run_dir}"
    done < <(find "${pair_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
    return
  fi

  return 0
}

# ============================================================
# Main loop (datasets x lambda sweep)
# ============================================================
for source_dataset in "${SOURCE_DATASETS[@]}"; do
  for target_dataset in "${TARGET_DATASETS[@]}"; do

    if [[ "${SKIP_SAME_DATASET}" == "1" && "${source_dataset}" == "${target_dataset}" ]]; then
      echo "Skipping ${source_dataset} -> ${target_dataset} (same dataset)"
      continue
    fi

    pair_dir="${BASE_DIR}/${source_dataset}2${target_dataset}"
    mapfile -t pair_run_names < <(collect_run_names_for_pair "${pair_dir}")

    if [[ ${#pair_run_names[@]} -eq 0 ]]; then
      echo "Skipping ${source_dataset} -> ${target_dataset}: no matching run folders found under ${pair_dir}"
      continue
    fi

    for run_name in "${pair_run_names[@]}"; do
      for pair in "${LAMBDA_PAIRS[@]}"; do
        read -r lambda_u lambda_d <<< "${pair}"

        echo "=================================================="
        echo "Eval: ${source_dataset} -> ${target_dataset} | run=${run_name:-ALL} | (u,d)=(${lambda_u},${lambda_d})"
        echo "=================================================="

        ckpt_path="${pair_dir}/${run_name}/ckpt.pth"

        if [[ ! -f "${ckpt_path}" ]]; then
          echo "Skipping missing ckpt: ${ckpt_path}"
          continue
        fi

        cmd=(
          "${PYTHON_BIN}" "${EVAL_SCRIPT}"
          --ckpt-path "${ckpt_path}"
          --lambda-eval-u "${lambda_u}"
          --lambda-eval-d "${lambda_d}"

          --sampling-num-samples "${SAMPLING_NUM_SAMPLES}"
          --sampling-step-size "${SAMPLING_STEP_SIZE}"
          --sampling-steps "${SAMPLING_STEPS}"

          --gif-sample-steps "${GIF_SAMPLE_STEPS}"
          --gif-grid-size "${GIF_GRID_SIZE}"
          --gif-num-samples "${GIF_NUM_SAMPLES}"
          --gif-interval "${GIF_INTERVAL}"
        )

        # Boolean flags
        cmd+=("${FLAGS[@]}")

        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '%q ' "${cmd[@]}"; echo
          continue
        fi

        if [[ "${SAVE_LOGS}" == "1" ]]; then
          run_tag="${run_name:-all}"
          log_file="${LOG_ROOT}/${source_dataset}2${target_dataset}_${run_tag}_u${lambda_u}_d${lambda_d}.log"
          "${cmd[@]}" 2>&1 | tee "${log_file}"
        else
          "${cmd[@]}"
        fi

      done
    done
  done
done