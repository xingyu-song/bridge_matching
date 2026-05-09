# Image example

## Training instructions

1. Download and unpack blurred ImageNet from the [official website](https://image-net.org/download.php).

```
export IMAGENET_DIR=~/flow_matching/examples/image/data/
export IMAGENET_RES=64
tar -xf ~/Downloads/train_blurred.tar.gz -C $IMAGENET_DIR
```

2. Downsample Imagenet to the desired resolution.

```
cd ~/
git clone git@github.com:PatrykChrabaszcz/Imagenet32_Scripts.git
python Imagenet32_Scripts/image_resizer_imagent.py -i ${IMAGENET_DIR}train_blurred -o ${IMAGENET_DIR}train_blurred_$IMAGENET_RES -s $IMAGENET_RES -a box  -r -j 10 
```

3. Set up the virtual environment. First, set up the virtual environment by following the steps in the repository's `README.md`. Then,

```
conda activate flow_matching

cd examples/image
pip install -r requirements.txt
```

4. [Optional] Test-run training locally. A test run executes one step of training followed by one step of evaluation.

```
python train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --test_run
```

5. Launch training on a SLURM cluster

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ 
```

6. Evaluate the model using the `--eval_only` flag. The evaluation script will generate snapshots under the `/snapshots` folder. Specify the `--compute_fid` flag to also compute the FID with respect to the training set. Make sure to specify your most recent checkpoint to resume from. The results are printed to `log.txt`.

```
python submitit_train.py --data_path=${IMAGENET_DIR}train_blurred_$IMAGENET_RES/box/ --resume=./output_dir/checkpoint-899.pth --compute_fid --eval_only
```

## Bridge Matching (BM) with `train_bm.py`

Use this section for BM training/evaluation (instead of `train.py` / `submitit_train.py`).

### 1) BM environment setup

From this folder:

```
cd examples/image
conda activate flow_matching
pip install -r requirements.txt
```

`train_bm.py` needs access to the BM repo (`bridge_matching/targets.py`):

- Either place BM at the default location expected by the script (`<workspace>/BM`)
- Or pass `--bm_repo_path=/absolute/path/to/BM`

### 2) Recommended output naming

Use output folders like:

- `output_<target_type>_<dataset_name>`
- Examples:
  - `output_cbm_diffusion_cifar10`
  - `output_cbm_diffusion_imagenette32`
  - `output_cfm_diffusion_imagenette64`

During runs, each output folder stores:

- `args.json` (full run config)
- `log.txt` (json-line metrics)
- `checkpoint-<epoch>.pth` and `checkpoint.pth`
- `snapshots/` (generated eval samples)

### 3) Train commands (different datasets)

#### CIFAR-10 (downloaded automatically)

```
python train_bm.py \
  --dataset=cifar10 \
  --data_path=./data/image_generation \
  --target_type=cbm_diffusion \
  --output_dir=./output_cbm_diffusion_cifar10 \
  --batch_size=64 \
  --epochs=3000 \
  --eval_frequency=100 \
  --compute_fid \
  --use_ema \
  --cfg_scale=0.0 \
  --ode_method=heun2 \
  --ode_options='{"nfe": 50}' \
  --skewed_timesteps \
  --edm_schedule
```

#### ImageNet32-style batches (`train_data_batch_1..10`)

```
python train_bm.py \
  --dataset=imagenet32 \
  --data_path=${IMAGENET_DIR}train_blurred_32/box/ \
  --target_type=cbm_diffusion \
  --output_dir=./output_cbm_diffusion_imagenette32 \
  --batch_size=32 \
  --epochs=900 \
  --eval_frequency=100 \
  --decay_lr \
  --compute_fid \
  --ode_method=dopri5 \
  --ode_options='{"atol": 1e-5, "rtol": 1e-5}'
```

#### ImageNet64-style folder (`ImageFolder` layout)

```
python train_bm.py \
  --dataset=imagenet \
  --data_path=${IMAGENET_DIR}train_blurred_64/box/ \
  --target_type=cfm_diffusion \
  --output_dir=./output_cfm_diffusion_imagenette64 \
  --batch_size=32 \
  --epochs=900 \
  --eval_frequency=100 \
  --decay_lr \
  --compute_fid \
  --ode_method=dopri5 \
  --ode_options='{"atol": 1e-5, "rtol": 1e-5}'
```

### 4) Evaluation-only command

Resume from a checkpoint and evaluate into the same `output_*` folder:

```
python train_bm.py \
  --dataset=cifar10 \
  --data_path=./data/image_generation \
  --target_type=cbm_diffusion \
  --output_dir=./output_cbm_diffusion_cifar10 \
  --resume=./output_cbm_diffusion_cifar10/checkpoint-899.pth \
  --eval_only \
  --compute_fid
```

### 5) Useful BM arguments

- `--target_type`: one of `cbm_diffusion`, `bm_diffusion_conditional`, `bm_linear_marginal`, `bm_linear_marginal_ot`, `cfm_linear`, `cfm_diffusion`
- `--save_frequency`: checkpoint frequency (`-1` follows `eval_frequency`, `0` disables periodic saves)
- `--lambda_d`, `--lambda_forward_align`: BM objective weights
- `--posterior_temperature`, `--posterior_topk`: posterior controls for BM targets
- `--sigma_floor`, `--beta_value`, `--t_eps`: BM diffusion/bridge stability settings


## Results

BM results below are taken from the `eval_log.txt` files in each `output_*` folder.

| Output folder | Dataset | Target | Evaluated checkpoint | Best eval setting (`lambda_eval_u`, `lambda_eval_d`) | FID (lower better) | KID mean | Inception Score |
|---|---|---|---|---|---:|---:|---:|
| `output_cbm_diffusion_cifar10` | CIFAR-10 | `cbm_diffusion` | `checkpoint-299` | `(1.0, 0.75)` | **5.338** | 0.002527 | 8.889 |
| `output_cfm_diffusion_cifar10` | CIFAR-10 | `cfm_diffusion` | `checkpoint-299` | `(1.0, 0.0)` | **5.205** | 0.002808 | 8.795 |
| `output_cbm_diffusion_imagenette32` | Imagenette32 | `cbm_diffusion` | `checkpoint-999` | `(1.0, 0.75)` | **12.752** | 0.002426 | 10.780 |
| `output_cfm_diffusion_imagenette32` | Imagenette32 | `cfm_diffusion` | `checkpoint-999` | `(1.0, 1.0)` | **13.105** | 0.002648 | 10.321 |
| `output_cbm_diffusion_imagenette64` | Imagenette64 | `cbm_diffusion` | `checkpoint-499` | `(1.0, 1.0)` | **13.299** | 0.002261 | 11.836 |
| `output_cfm_diffusion_imagenette64` | Imagenette64 | `cfm_diffusion` | `checkpoint-499` | `(1.0, 1.0)` | **13.387** | 0.002198 | 11.798 |

Notes:
- Some folders include multi-`lambda_eval_d` sweeps; table reports the best-FID setting from each folder log.
- Metrics are rounded to 3 decimals for readability.



## Acknowledgements

This example partially use code from:
- [Guided diffusion](https://github.com/openai/guided-diffusion/)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)

## License

The majority of the code in this example is licensed under CC-BY-NC, however portions of the project are available under separate license terms: 
- The UNet model is under MIT license.
- The distributed computing and the grad scaler code is under MIT license.

## Citations

Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." Advances in neural information processing systems 35 (2022): 26565-26577.

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.
