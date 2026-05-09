# Bridge Matching (2D + Image Experiments)

PyTorch implementation of bridge-matching style training and evaluation for:
- **2D synthetic distributions** (e.g., gaussian, moons, checkerboard)
- **image datasets** (e.g., CIFAR-10, MNIST, FashionMNIST, CelebA)

The codebase trains two vector fields (`u` and `d`) and evaluates forward/backward transport behavior, visualization artifacts, and sample quality metrics.

## Repository Layout

- `bridge_matching/`: core library code
  - `datasets/`: toy 2D samplers and image dataset loaders
  - `models/`: UNet and model utilities
  - `targets.py`: registered bridge-matching target definitions
  - `solver.py`: ODE solver wrappers
  - `metrics.py`: FID/KID/Inception Score helpers
  - `visualization.py`: plotting and GIF generation
- `scripts/`: training/evaluation entrypoints
  - `train_bm_2d.py`, `eval_bm_2d.py`
  - `train_bm_image.py`, `eval_bm_image.py`
  - shell helpers for running sweeps

## Requirements

This project uses Python with PyTorch and torchvision.

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install tyro jaxtyping tqdm matplotlib scipy flow-matching pytest
```

Notes:
- `scipy` is required for the OT-based target (`bm_linear_marginal_ot`).
- image metrics download Inception weights from torchvision on first use.
- datasets are downloaded under `data/` automatically by torchvision loaders.

## Quick Start

Run commands from the repository root.

### 1) Train on 2D toy data

```bash
python scripts/train_bm_2d.py \
  --source-dataset gaussian \
  --target-dataset checkerboard \
  --target-type bm_diffusion_conditional
```

Outputs are saved under:

`outputs/bm/<source>2<target>/<target_short>_<timestamp>/`

Typical artifacts:
- `ckpt.pth`
- `train.log`
- `losses.png`

### 2) Evaluate a 2D checkpoint

```bash
python scripts/eval_bm_2d.py \
  --ckpt-path outputs/bm/gaussian2checkerboard/<run_name>/ckpt.pth \
  --do-field-decomposition-plot \
  --do-metrics \
  --lambda-eval-u 1.0 \
  --lambda-eval-d 1.0
```

Evaluation artifacts are saved under:

`outputs/bm/<source>2<target>/<run_name>/eval/`

### 3) Train on image data

```bash
python scripts/train_bm_image.py \
  --dataset cifar10 \
  --target-type cfm_linear \
  --n-epochs 10 \
  --batch-size 128
```

Outputs are saved under:

`outputs/bm/<dataset>/<target_type>/<target_short>_<timestamp>/`

### 4) Evaluate an image checkpoint

```bash
python scripts/eval_bm_image.py \
  --ckpt-path outputs/bm/cifar10/<target_type>/<run_name>/ckpt.pth \
  --dataset cifar10 \
  --do-metrics \
  --lambda-eval-u 1.0 \
  --lambda-eval-d 1.0
```

## Target Types

Target classes are registered in `bridge_matching/targets.py`.

To inspect available target names quickly:

```bash
python - <<'PY'
import re
from pathlib import Path
text = Path("bridge_matching/targets.py").read_text()
names = re.findall(r'@register_target\("([^"]+)"\)', text)
print("\n".join(names))
PY
```

## Testing

Run dataset tests:

```bash
pytest scripts/test_datasets.py -q
```

## Reproducibility

- Use `--seed` on train/eval scripts.
- Training scripts write config and logs per run.
- Checkpoint files include model states and saved script arguments.

## Common Tips

- If running on CPU, reduce `--batch-size` and sample counts.
- For image metrics, start with small `--num-metric-samples` for quick checks.
- For 2D evaluation sweeps, use `scripts/run_eval_bm_2d_config.sh` as a template.

