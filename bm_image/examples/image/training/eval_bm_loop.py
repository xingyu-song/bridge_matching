"""Bridge Matching FID evaluation loop.

Combines the trained `flow_u` and `flow_d` into a single velocity field
`v = lambda_eval_u * u + lambda_eval_d * d`, then samples with
`flow_matching.solver.ODESolver` (same library API the FM trainer uses).
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Iterable

import torch
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import save_image
from training.edm_time_discretization import get_time_discretization

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledBridgeModel(ModelWrapper):
    """Wraps (flow_u, flow_d) into a single velocity field with CFG support."""

    def __init__(
        self,
        flow_u: torch.nn.Module,
        flow_d: torch.nn.Module,
        lambda_eval_u: float = 1.0,
        lambda_eval_d: float = 1.0,
        use_d: bool = True,
    ):
        super().__init__(flow_u)
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.lambda_eval_u = lambda_eval_u
        self.lambda_eval_d = lambda_eval_d
        self.use_d = use_d
        self.nfe_counter = 0

    def _call_flow(self, flow, x, t, extra):
        with torch.cuda.amp.autocast(), torch.no_grad():
            return flow(x, t, extra=extra)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            cond_u = self._call_flow(self.flow_u, x, t, {"label": label})
            uncond_u = self._call_flow(self.flow_u, x, t, {})
            u = (1.0 + cfg_scale) * cond_u - cfg_scale * uncond_u

            if self.use_d:
                cond_d = self._call_flow(self.flow_d, x, t, {"label": label})
                uncond_d = self._call_flow(self.flow_d, x, t, {})
                d = (1.0 + cfg_scale) * cond_d - cfg_scale * uncond_d
            else:
                d = torch.zeros_like(u)
        else:
            u = self._call_flow(self.flow_u, x, t, {"label": label})
            if self.use_d:
                d = self._call_flow(self.flow_d, x, t, {"label": label})
            else:
                d = torch.zeros_like(u)

        self.nfe_counter += 1
        return (self.lambda_eval_u * u + self.lambda_eval_d * d).to(torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_bm_model(
    flow_u: torch.nn.Module,
    flow_d: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: argparse.Namespace,
):
    gc.collect()
    use_d = not args.target_type.startswith("cfm_")
    model = CFGScaledBridgeModel(
        flow_u=flow_u,
        flow_d=flow_d,
        lambda_eval_u=args.lambda_eval_u,
        lambda_eval_d=args.lambda_eval_d,
        use_d=use_d,
    )
    model.train(False)

    solver = ODESolver(velocity_model=model)
    ode_opts = args.ode_options

    compute_fid = getattr(args, "compute_fid", False)
    compute_kid = getattr(args, "compute_kid", False)
    compute_inception_score = getattr(args, "compute_inception_score", False)
    compute_any_metric = compute_fid or compute_kid or compute_inception_score

    fid_metric = None
    kid_metric = None
    inception_score_metric = None
    if compute_fid:
        fid_metric = FrechetInceptionDistance(normalize=True).to(
            device=device, non_blocking=True
        )
    if compute_kid:
        kid_metric = KernelInceptionDistance(
            subsets=args.inception_splits,
            subset_size=min(1000, fid_samples),
            normalize=True,
        ).to(device=device, non_blocking=True)
    if compute_inception_score:
        inception_score_metric = InceptionScore(
            splits=args.inception_splits,
            normalize=True,
        ).to(device=device, non_blocking=True)

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if fid_metric is not None:
            fid_metric.update(samples, real=True)
        if kid_metric is not None:
            kid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            model.reset_nfe_counter()
            x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

            if args.edm_schedule:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)

            synthetic = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                return_intermediates=False,
                atol=ode_opts.get("atol", 1e-5),
                rtol=ode_opts.get("rtol", 1e-5),
                step_size=ode_opts.get("step_size", None),
                label=labels,
                cfg_scale=args.cfg_scale,
            )

            synthetic = torch.clamp(synthetic * 0.5 + 0.5, min=0.0, max=1.0)
            synthetic = torch.floor(synthetic * 255).to(torch.float32) / 255.0

            logger.info(
                f"{samples.shape[0]} samples generated in {model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic.shape[0] > fid_samples:
                synthetic = synthetic[: fid_samples - num_synthetic]
            if fid_metric is not None:
                fid_metric.update(synthetic, real=False)
            if kid_metric is not None:
                kid_metric.update(synthetic, real=False)
            if inception_score_metric is not None:
                inception_score_metric.update(synthetic)
            num_synthetic += synthetic.shape[0]

            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True

        if not compute_any_metric:
            return {}

        if fid_metric is not None and data_iter_step % PRINT_FREQUENCY == 0:
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Eval [{data_iter_step}/{len(data_loader)}] "
                f"samples [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        if args.test_run:
            break

    stats = {
        "lambda_eval_u": float(args.lambda_eval_u),
        "lambda_eval_d": float(args.lambda_eval_d),
    }
    if fid_metric is not None:
        stats["fid"] = float(fid_metric.compute().detach().cpu())
    if kid_metric is not None:
        kid_mean, kid_std = kid_metric.compute()
        stats["kid_mean"] = float(kid_mean.detach().cpu())
        stats["kid_std"] = float(kid_std.detach().cpu())
    if inception_score_metric is not None:
        is_mean, is_std = inception_score_metric.compute()
        stats["inception_score_mean"] = float(is_mean.detach().cpu())
        stats["inception_score_std"] = float(is_std.detach().cpu())
    return stats
