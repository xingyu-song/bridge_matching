# Bridge Matching: 2D and Image Experiments

This repository combines Bridge Matching (BM) experiments across:
- **2D synthetic distributions** (e.g., gaussian, moons, checkerboard)
- **Image datasets** (e.g., CIFAR-10, MNIST/FashionMNIST, ImageNet-style data)

The codebase focuses on training and evaluating transport vector fields (`u` and `d`), and provides scripts for visualization, checkpointing, and sample-quality metrics.

## Codebase Origin and Attribution

- **2D pipeline** is based on [keishihara/flow-matching](https://github.com/keishihara/flow-matching).
- **Image pipeline** is based on [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).

This repo extends/adapts those foundations for BM-style objectives and evaluation workflows.

## Repository Layout

- `bm_2d/`: 2D BM training/evaluation code and scripts
- `bm_image/`: image BM training/evaluation code and examples

Typical components include:
- dataset loaders/samplers
- BM target definitions
- model and solver utilities
- metrics (FID/KID/IS) and visualization helpers

## Environment Setup

Use Python + PyTorch. A minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
```

Then install each subproject's dependencies as needed:

```bash
# 2D side (example)
pip install tyro jaxtyping tqdm matplotlib scipy flow-matching pytest

# Image side
cd bm_image/examples/image
pip install -r requirements.txt
```

## Quick Start: 2D

From `bm_2d/` (or run paths from repo root), train and evaluate BM on toy distributions.

### Train

```bash
python scripts/train_bm_2d.py \
  --source-dataset gaussian \
  --target-dataset checkerboard \
  --target-type bm_diffusion_conditional
```

### Evaluate

```bash
python scripts/eval_bm_2d.py \
  --ckpt-path outputs/bm/gaussian2checkerboard/<run_name>/ckpt.pth \
  --do-field-decomposition-plot \
  --do-metrics \
  --lambda-eval-u 1.0 \
  --lambda-eval-d 1.0
```

Outputs are typically written under:
- `outputs/bm/<source>2<target>/<target_short>_<timestamp>/` (train)
- `outputs/bm/<source>2<target>/<run_name>/eval/` (eval)

## Quick Start: Image

Image experiments live under `bm_image/examples/image`.

### 1) Prepare data (example: blurred ImageNet)

Follow the dataset preparation flow from the image README, including:
- downloading blurred ImageNet
- optional resizing/downsampling for target resolution
- setting `IMAGENET_DIR` and resolution-specific paths

### 2) Train BM

```bash
python train_bm.py \
  --dataset=cifar10 \
  --data_path=./data/image_generation \
  --target_type=cbm_diffusion \
  --output_dir=./output_cbm_diffusion_cifar10 \
  --batch_size=64 \
  --epochs=3000 \
  --eval_frequency=100 \
  --compute_fid
```

### 3) Evaluate only (from checkpoint)

```bash
python train_bm.py \
  --dataset=cifar10 \
  --data_path=./data/image_generation \
  --target_type=cbm_diffusion \
  --output_dir=./output_cbm_diffusion_cifar10 \
  --resume=./output_cbm_diffusion_cifar10/checkpoint-899.pth \
  --eval_only \
  --compute_fid
```

Common output artifacts:
- `args.json`
- `log.txt` / eval logs
- `checkpoint-<epoch>.pth` and `checkpoint.pth`
- `snapshots/`

## BM Target Types

Common target names used across scripts:
- `cbm_diffusion`
- `bm_diffusion_conditional`
- `bm_linear_marginal`
- `bm_linear_marginal_ot`
- `cfm_linear`
- `cfm_diffusion`

Exact registration and behavior can differ by subproject implementation.

## Reproducibility Tips

- Set `--seed` for train/eval commands.
- Start with smaller batch sizes/sample counts for quick validation.
- For metrics (especially image), run low sample counts first, then scale up.

## Acknowledgements

- **2D experiments** are based on [keishihara/flow-matching](https://github.com/keishihara/flow-matching).
- **Image experiments** are based on [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).

Additional references and components:
- [Guided Diffusion](https://github.com/openai/guided-diffusion/)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

## References

- Lipman et al., *Flow Matching for Generative Modeling* (2023)
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (2023)
