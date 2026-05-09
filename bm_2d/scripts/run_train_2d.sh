#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Extract all registered target names from targets.py
mapfile -t TARGET_TYPES < <(
  python - <<'PY'
import re
from pathlib import Path
text = Path("bridge_matching/targets.py").read_text()
for name in re.findall(r'@register_target\("([^"]+)"\)', text):
    print(name)
PY
)

# You asked: gaussian -> moons/checkerboard/mixture
DATASETS=("moons" "checkerboard" "mixture")

run_idx=0
total=$(( ${#TARGET_TYPES[@]} * ${#DATASETS[@]} ))
echo "Total runs: $total"

for target_type in "${TARGET_TYPES[@]}"; do
  for target_dataset in "${DATASETS[@]}"; do
    run_idx=$((run_idx + 1))
    echo "[$run_idx/$total] target_type=${target_type}, gaussian->${target_dataset}"
    python scripts/train_bm_2d.py \
      --source-dataset gaussian \
      --target-dataset "${target_dataset}" \
      --target-type "${target_type}"
  done
done