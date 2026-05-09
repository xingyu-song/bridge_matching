# `train_bm.py` Optional Arguments

This script uses:
- Base Flow Matching args from `train_arg_parser.py`
- BM-specific args from `train_bm_arg_parser.py`

## Quick Notes

- `--discrete_flow_matching` exists in the parser but is **not supported** by `train_bm.py` (the script raises an error if enabled).
- `--target_type` controls whether both branches (`u`, `d`) are trained. `cfm_*` keeps `d` fixed at zero; `cbm_diffusion` trains both branches on the same VP diffusion path as `cfm_diffusion`.

## Core Training

- `--batch_size` (int, default: `32`)
- `--epochs` (int, default: `921`)
- `--accum_iter` (int, default: `1`)
- `--lr` (float, default: `0.0001`)
- `--optimizer_betas` (float list, default: `[0.9, 0.95]`)
- `--decay_lr` (flag, default: `False`)
- `--use_ema` (flag, default: `False`)

## Dataset / Dataloader

- `--dataset` (str, default: first key in `MODEL_CONFIGS`, choices from `MODEL_CONFIGS.keys()`)
- `--data_path` (str, default: `./data/image_generation`)
- `--num_workers` (int, default: `10`)
- `--pin_mem` (flag, default: `True`)
- `--no_pin_mem` (flag, sets `pin_mem=False`)

## Output / Resume / Eval

- `--output_dir` (str, default: `./output_dir`)
- `--resume` (str, default: `""`)
- `--start_epoch` (int, default: `0`)
- `--eval_only` (flag, default: `False`)
- `--eval_frequency` (int, default: `50`)
- `--save_frequency` (int, default: `-1`; `-1` follows `eval_frequency`, `0` disables periodic checkpointing)
- `--compute_fid` (flag, default: `False`)
- `--save_fid_samples` (flag, default: `False`)
- `--fid_samples` (int, default: `50000`)
- `--test_run` (flag, default: `False`)

## ODE / Sampling Controls

- `--ode_method` (str, default: `midpoint`, choices: `SOLVERS.keys() + ["edm_heun"]`)
- `--ode_options` (JSON, default: `{"step_size": 0.01}`)
- `--sampling_dtype` (str, default: `float32`, choices: `float32|float64`)
- `--cfg_scale` (float, default: `0.2`)
- `--sym` (float, default: `0.0`)
- `--temp` (float, default: `1.0`)
- `--sym_func` (flag, default: `False`)
- `--skewed_timesteps` (flag, default: `False`)
- `--edm_schedule` (flag, default: `False`)
- `--class_drop_prob` (float, default: `0.2`)

## Device / Repro / Distributed

- `--device` (str, default: `cuda`)
- `--seed` (int, default: `0`)
- `--world_size` (int, default: `1`)
- `--local_rank` (int, default: `-1`)
- `--dist_on_itp` (flag, default: `False`)
- `--dist_url` (str, default: `env://`)

## Discrete FM (Parser Includes, BM Disallows)

- `--discrete_flow_matching` (flag, default: `False`) **unsupported in BM script**
- `--discrete_fm_steps` (int, default: `1024`)

## Bridge Matching (BM) Specific

- `--target_type` (str, default: `cbm_diffusion`, choices:
  `cbm_diffusion`, `bm_diffusion_conditional`, `bm_linear_marginal`, `bm_linear_marginal_ot`, `cfm_linear`, `cfm_diffusion`)
- `--beta_value` (float, default: `1e-2`)
- `--sigma_floor` (float, default: `5e-2`)
- `--t_eps` (float, default: `1e-4`)
- `--save_frequency` (int, default: `-1`; `-1` follows `eval_frequency`, `0` disables periodic checkpointing)
- `--lambda_d` (float, default: `1.0`)
- `--lambda_forward_align` (float, default: `0.0`)
- `--posterior_temperature` (float, default: `1.0`)
- `--posterior_topk` (int, default: `0`; `0` disables top-k subsetting)
- `--cfm_beta_min` (float, default: `0.1`)
- `--cfm_beta_max` (float, default: `20.0`)
- `--bm_repo_path` (str, default: `""`; if empty, script falls back to `<repo>/BM`)

