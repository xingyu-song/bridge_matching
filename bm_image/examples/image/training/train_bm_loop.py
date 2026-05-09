"""Bridge Matching one-epoch training loop.

Trains two networks (`flow_u`, `flow_d`) by regressing the BM targets
`(u*, d*)` produced by `bridge_matching.targets.make_bridge_matching_targets`.
Designed to mirror `training.train_loop.train_one_epoch` so that the rest
of the pipeline (NativeScaler, distributed mode, EMA wrapper) remains
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def train_one_epoch_bm(
    flow_u: torch.nn.Module,
    flow_d: torch.nn.Module,
    bm_targets,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    flow_u.train(True)
    train_d_branch = args.target_type != "cfm_linear"
    if train_d_branch:
        flow_d.train(True)
    else:
        flow_d.train(False)

    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    accum_iter = args.accum_iter
    valid_steps = 0
    skipped_steps = 0
    oom_steps = 0

    trainable_params = list(flow_u.parameters())
    if train_d_branch:
        trainable_params += list(flow_d.parameters())

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        try:
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Match FM convention: scale to [-1, 1]
            samples = samples * 2.0 - 1.0
            x_0 = torch.randn_like(samples)

            if torch.rand(1) < args.class_drop_prob:
                conditioning = {}
            else:
                conditioning = {"label": labels}

            # Time sampling with BM-style endpoint clipping
            t = args.t_eps + (1.0 - 2.0 * args.t_eps) * torch.rand(
                samples.shape[0], device=device, dtype=samples.dtype
            )
            t_bm = t.view(-1, 1)

            # BM target construction can become numerically unstable in fp16/bf16.
            # Keep it in fp32, then let autocast handle model/loss compute.
            with torch.cuda.amp.autocast(enabled=False):
                x_t, u_star, d_star = bm_targets.compute(
                    x_0=x_0.float(),
                    x_1=samples.float(),
                    t=t_bm.float(),
                )

            if not (
                torch.isfinite(x_t).all()
                and torch.isfinite(u_star).all()
                and torch.isfinite(d_star).all()
            ):
                logger.warning(
                    "Skipping non-finite BM targets at epoch %d step %d",
                    epoch,
                    data_iter_step,
                )
                skipped_steps += 1
                optimizer.zero_grad()
                continue

            x_t = x_t.to(device=device, dtype=samples.dtype)
            u_star = u_star.to(device=device, dtype=samples.dtype)
            d_star = d_star.to(device=device, dtype=samples.dtype)

            with torch.cuda.amp.autocast():

                u_pred = flow_u(x_t, t, extra=conditioning)
                if train_d_branch:
                    d_pred = flow_d(x_t, t, extra=conditioning)
                    v_pred = u_pred + d_pred
                    v_star = u_star + d_star

                    loss_u = torch.pow(u_pred - u_star, 2).mean()
                    loss_d = torch.pow(d_pred - d_star, 2).mean()
                    loss_v = torch.pow(v_pred - v_star, 2).mean()
                    loss = (
                        loss_u
                        + args.lambda_d * loss_d
                        + args.lambda_forward_align * loss_v
                    )
                else:
                    loss_u = torch.pow(u_pred - u_star, 2).mean()
                    loss_d = torch.zeros((), device=device)
                    loss_v = torch.zeros((), device=device)
                    loss = loss_u
        except RuntimeError as err:
            if "out of memory" not in str(err).lower():
                raise
            oom_steps += 1
            skipped_steps += 1
            logger.warning(
                "CUDA OOM at epoch %d step %d; skipping batch and clearing cache.",
                epoch,
                data_iter_step,
            )
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.warning(
                "Skipping non-finite loss at epoch %d step %d "
                "(loss_u=%s, loss_d=%s, loss_v=%s)",
                epoch,
                data_iter_step,
                float(loss_u.detach().cpu()) if torch.isfinite(loss_u) else "nan/inf",
                float(loss_d.detach().cpu()) if torch.isfinite(loss_d) else "nan/inf",
                float(loss_v.detach().cpu()) if torch.isfinite(loss_v) else "nan/inf",
            )
            skipped_steps += 1
            optimizer.zero_grad()
            continue

        batch_loss.update(loss)
        epoch_loss.update(loss)
        valid_steps += 1

        loss = loss / accum_iter

        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=trainable_params,
            update_grad=apply_update,
        )

        if apply_update:
            _maybe_update_ema(flow_u)
            if train_d_branch:
                _maybe_update_ema(flow_d)

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss={batch_loss.compute():.4f} "
                f"loss_u={loss_u.item():.4f} loss_d={loss_d.item():.4f} "
                f"loss_v={loss_v.item():.4f} lr={lr:.2e}"
            )

    if valid_steps == 0:
        raise ValueError(
            "All training steps produced non-finite values in this epoch. "
            "Try safer settings, e.g. larger --sigma_floor (0.01~0.05), "
            "larger --t_eps (1e-4~1e-3), or smaller --lr."
        )

    if skipped_steps > 0:
        logger.warning(
            "Epoch %d skipped %d/%d non-finite steps",
            epoch,
            skipped_steps,
            valid_steps + skipped_steps,
        )
    if oom_steps > 0:
        logger.warning("Epoch %d encountered %d CUDA OOM step(s)", epoch, oom_steps)

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}


def _maybe_update_ema(model: torch.nn.Module) -> None:
    if isinstance(model, EMA):
        model.update_ema()
    elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
        model.module.update_ema()
