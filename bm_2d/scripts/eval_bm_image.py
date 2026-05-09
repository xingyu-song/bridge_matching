
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import tyro
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from bridge_matching.datasets.image_datasets import get_image_dataset, get_test_transform
from bridge_matching.metrics import (
    build_inception_models,
    collect_real_images,
    compute_fid,
    compute_inception_score,
    compute_kid,
    extract_inception_features_and_probs_batched,
    generate_fake_images,
)
from bridge_matching.models import UNetModel
from bridge_matching.solver import ModelWrapper, ODESolver
from bridge_matching.utils import set_seed


@dataclass
class EvalArguments:
    # I/O
    ckpt_path: Path = Path("outputs/bm/cifar10/bm_diffusion_conditional/bdc_20260427213113/ckpt.pth")  # checkpoint (.pth) containing flow_u/flow_d and training args
    output_dir: Path | None = None  # eval artifact directory; None => <ckpt_parent>/eval
    dataset: str = "cifar10"  # dataset name

    # Bridge mixing weights
    lambda_eval_u: float = 1.0  # scale for u-flow during bridge evaluation
    lambda_eval_d: float = 1.0  # scale for d-flow during bridge evaluation

    # What to run / save
    do_sample: bool = False  # save final generated sample image (forward pass)
    do_backward: bool = False  # run backward bridge from forward final samples
    do_grid: bool = False  # save side-by-side grid snapshots
    do_trajectory: bool = False  # save GIF trajectory across time
    do_metrics: bool = True  # run quantitative metrics (FID/KID/Inception Score)

    # Visualization batch / trajectory controls
    batch_size: int = 128  # batch size for initial forward visualization pass
    num_inference_steps: int = 80  # fixed integration steps in [0,1] for fixed-step solvers
    num_output_steps: int = 101  # output time points in [0,1] for saved visualization/GIF frames
    samples_per_class: int = 10  # generated examples per class for visualization
    fps: int = 20  # frame rate for trajectory GIF files

    # Metrics controls
    metrics_batch_size: int = 256  # chunk size for sample generation and Inception extraction
    num_metric_samples: int = 100  # total fake images generated for metrics
    metrics_subset_size: int | None = None  # cap real images for metrics; None => full test set
    metrics_num_splits: int = 10  # splits/subsets for IS and KID averaging

    # ODE solver controls
    method: str = "midpoint"  # ODE solver method (e.g., dopri5/euler/midpoint/rk4/heun3)
    ode_atol: float = 1e-5  # absolute tolerance for adaptive ODE solvers
    ode_rtol: float = 1e-5  # relative tolerance for adaptive ODE solvers
    seed: int = 42  # random seed for reproducible evaluation

    # UNet architecture controls.
    num_channels: int = 128  # UNet base channels
    num_res_blocks: int = 2  # residual blocks per level
    dropout: float = 0.0  # UNet dropout
    channel_mult: str = "1,2,2,2"  # channel multipliers as comma-separated ints
    attention_resolutions: str = "16"  # attention resolutions string like "16,8"
    class_cond: bool = False  # class conditioning


# Compatibility alias for checkpoints that saved the training ScriptArguments dataclass.
ScriptArguments = EvalArguments


class ForwardBridgeWrapper(torch.nn.Module):
    def __init__(self, flow_u: torch.nn.Module, flow_d: torch.nn.Module, lambda_eval_d: float = 1.0, lambda_eval_u: float = 1.0):
        super().__init__()
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.lambda_eval_d = lambda_eval_d
        self.lambda_eval_u = lambda_eval_u

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.lambda_eval_u * self.flow_u(t=t, x=x, y=y) + self.lambda_eval_d * self.flow_d(t=t, x=x, y=y)


class BackwardBridgeWrapper(torch.nn.Module):
    def __init__(self, flow_u: torch.nn.Module, flow_d: torch.nn.Module, lambda_eval_d: float = 1.0, lambda_eval_u: float = 1.0):
        super().__init__()
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.lambda_eval_d = lambda_eval_d
        self.lambda_eval_u = lambda_eval_u

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        t_rev = 1.0 - t
        return -(
            self.lambda_eval_u * self.flow_u(t=t_rev, x=x, y=y)
            - self.lambda_eval_d * self.flow_d(t=t_rev, x=x, y=y)
        )



def _save_trajectory_visuals(
    sol: torch.Tensor,
    t_eval: torch.Tensor,
    num_classes: int,
    eval_dir: Path,
    prefix: str,
    fps: int,
    do_samples: bool = True,
    do_grid: bool = True,
    do_trajectory: bool = True,
) -> None:
    if do_samples:
        final_samples = sol[-1]
        save_image(final_samples, eval_dir / f"{prefix}_final_samples.png", nrow=num_classes, normalize=True)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        final_grid = make_grid(final_samples, nrow=num_classes, normalize=True)
        ax[0].imshow(final_grid.permute(1, 2, 0))
        ax[0].set_title("Final samples (t = 1.0)", fontsize=16)
        ax[0].axis("off")

        def update(frame: int) -> None:
            grid = make_grid(sol[frame], nrow=num_classes, normalize=True)
            ax[1].clear()
            ax[1].imshow(grid.permute(1, 2, 0))
            ax[1].set_title(f"t = {t_eval[frame].item():.2f}", fontsize=16)
            ax[1].axis("off")
        update(0)

    if do_grid:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.1)
        fig.savefig(eval_dir / f"{prefix}_grid.png", bbox_inches="tight")

    if do_trajectory:
        ani = animation.FuncAnimation(fig, update, frames=sol.shape[0])
        ani.save(eval_dir / f"{prefix}_trajectory.gif", writer="pillow", fps=fps)
    
    if do_samples or do_grid or do_trajectory:
        plt.close(fig)


def _run_quantitative_metrics(
    dataset,
    solver: ODESolver,
    input_shape: torch.Size,
    num_classes: int,
    class_cond: bool,
    args: EvalArguments,
    device: torch.device,
    eval_dir: Path,
) -> None:
    print("Running quantitative image-generation metrics...")
    print(f"num_metric_samples = {args.num_metric_samples}")
    print(f"metrics_subset_size = {args.metrics_subset_size}")
    print(f"metrics_batch_size = {args.metrics_batch_size}")
    print(f"num_inference_steps = {args.num_inference_steps}")
    print(f"method = {args.method}")
    print(f"ode_atol = {args.ode_atol}")
    print(f"ode_rtol = {args.ode_rtol}")
    real_images = collect_real_images(dataset, args.metrics_subset_size, batch_size=args.metrics_batch_size)
    fake_images = generate_fake_images(
        solver=solver,
        input_shape=input_shape,
        num_classes=num_classes,
        num_samples=args.num_metric_samples,
        batch_size=args.metrics_batch_size,
        num_inference_steps=args.num_inference_steps,
        method=args.method,
        class_cond=class_cond,
        atol=args.ode_atol,
        rtol=args.ode_rtol,
        device=device,
    )

    inception_feature_model, inception_logit_model = build_inception_models(device)
    real_features, _ = extract_inception_features_and_probs_batched(
        feature_model=inception_feature_model,
        logit_model=inception_logit_model,
        images=real_images,
        device=device,
        batch_size=args.metrics_batch_size,
    )
    fake_features, fake_probs = extract_inception_features_and_probs_batched(
        feature_model=inception_feature_model,
        logit_model=inception_logit_model,
        images=fake_images,
        device=device,
        batch_size=args.metrics_batch_size,
    )

    fid = compute_fid(real_features, fake_features)
    kid = compute_kid(
        real_features,
        fake_features,
        num_subsets=args.metrics_num_splits,
        subset_size=min(100, real_features.shape[0], fake_features.shape[0]),
    )
    inception_score = compute_inception_score(fake_probs, num_splits=args.metrics_num_splits)

    metrics_text = (
        f"FID: {fid:.6f}\n"
        f"KID: {kid:.6f}\n"
        f"Inception Score: {inception_score:.6f}\n"
        f"num_metric_samples: {args.num_metric_samples}\n"
        f"metrics_subset_size: {args.metrics_subset_size}\n"
        f"method: {args.method}\n"
        f"ode_atol: {args.ode_atol}\n"
        f"ode_rtol: {args.ode_rtol}\n"
    )
    print(metrics_text)
    (eval_dir / f"metrics_{args.seed}_{args.method}_({args.lambda_eval_u},{args.lambda_eval_d}).txt").write_text(metrics_text)


def main(args: EvalArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    add_safe_globals([ScriptArguments])
    # ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False) # load on CPU with weights_only to avoid issues with missing classes/objects in the checkpoint
    saved_args = ckpt.get("args", None)

    if saved_args is None:
        raise ValueError(f"Checkpoint {args.ckpt_path} does not contain saved training args.")

    dataset_name = args.dataset
    class_cond = args.class_cond

    output_dir = args.output_dir if args.output_dir is not None else (args.ckpt_path.parent / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_image_dataset(
        dataset_name,
        train=False,
        transform=get_test_transform(),
    )
    input_shape = dataset[0][0].size()
    num_classes = len(dataset.classes)

    flow_u = UNetModel(
        input_shape,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_classes=num_classes,
        class_cond=class_cond,
        dropout=args.dropout,
        channel_mult=tuple(int(x.strip()) for x in args.channel_mult.split(",") if x.strip()),
        attention_resolutions=args.attention_resolutions,
    ).to(device)
    flow_d = UNetModel(
        input_shape,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_classes=num_classes,
        class_cond=class_cond,
        dropout=args.dropout,
        channel_mult=tuple(int(x.strip()) for x in args.channel_mult.split(",") if x.strip()),
        attention_resolutions=args.attention_resolutions,
    ).to(device)

    flow_u.load_state_dict(ckpt["flow_u"])
    flow_d.load_state_dict(ckpt["flow_d"])
    flow_u.eval()
    flow_d.eval()

    with torch.no_grad():
        x_1, y = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)))
        x_1, y = x_1.to(device), y.to(device)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.size(0), device=device, dtype=x_1.dtype)
        cond_y = y if class_cond else None

        u = flow_u(t=t, x=x_1, y=cond_y)
        d = flow_d(t=t, x=x_1, y=cond_y)

        u_norm = u.flatten(1).norm(dim=1).mean().item()
        d_norm = d.flatten(1).norm(dim=1).mean().item()
        ratio = d_norm / (u_norm + 1e-8)

        print(f"mean ||u|| = {u_norm}")
        print(f"mean ||d|| = {d_norm}")
        print(f"ratio ||d|| / ||u|| = {ratio}")

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset: {dataset_name}")
    print(f"num_channels: {args.num_channels}")
    print(f"num_res_blocks: {args.num_res_blocks}")
    print(f"dropout: {args.dropout}")
    print(f"channel_mult: {args.channel_mult}")
    print(f"attention_resolutions: {args.attention_resolutions}")
    print(f"class_cond: {class_cond}")
    print(f"lambda_eval_u: {args.lambda_eval_u}")
    print(f"lambda_eval_d: {args.lambda_eval_d}")
    print(f"Output dir: {output_dir}")


    t_eval = torch.linspace(0, 1, args.num_output_steps, device=device)
    class_list = torch.arange(num_classes, device=device).repeat(args.samples_per_class) if class_cond else None
    num_viz_samples = (
        class_list.size(0)
        if class_list is not None
        else max(1, args.samples_per_class * num_classes)
    )
    x_init = torch.randn((num_viz_samples, *input_shape), dtype=torch.float32, device=device)
    fixed_step_methods = {"euler", "midpoint", "rk4", "heun2", "heun3", "explicit_adams", "implicit_adams"}
    step_size = (1.0 / args.num_inference_steps) if args.method in fixed_step_methods else None

    forward_model = ModelWrapper(
        ForwardBridgeWrapper(
            flow_u=flow_u,
            flow_d=flow_d,
            lambda_eval_d=args.lambda_eval_d,
            lambda_eval_u=args.lambda_eval_u,
        )
    )
    solver = ODESolver(forward_model)
    sol = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=args.method,
        atol=args.ode_atol,
        rtol=args.ode_rtol,
        time_grid=t_eval,
        return_intermediates=True,
        y=class_list,
    ).detach().cpu()
    _save_trajectory_visuals(
        sol=sol,
        t_eval=t_eval.detach().cpu(),
        num_classes=num_classes,
        eval_dir=output_dir,
        prefix=f"forward_({args.lambda_eval_u},{args.lambda_eval_d})",
        fps=args.fps,
        do_samples=args.do_sample,
        do_grid=args.do_grid,
        do_trajectory=args.do_trajectory,
    )

    if args.do_metrics:
        # Release temporary tensors from visualization path before running metrics.
        del x_init, sol, t_eval, class_list
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _run_quantitative_metrics(
            dataset=dataset,
            solver=solver,
            input_shape=input_shape,
            num_classes=num_classes,
            class_cond=class_cond,
            args=args,
            device=device,
            eval_dir=output_dir,
        )

    if args.do_backward:
        backward_model = ModelWrapper(
            BackwardBridgeWrapper(
                flow_u=flow_u,
                flow_d=flow_d,
                lambda_eval_d=args.lambda_eval_d,
                lambda_eval_u=args.lambda_eval_u,
            )
        )
        backward_solver = ODESolver(backward_model)
        backward_sol = backward_solver.sample(
            x_init=sol[-1].to(device),
            step_size=step_size,
            method=args.method,
            atol=args.ode_atol,
            rtol=args.ode_rtol,
            time_grid=t_eval,
            return_intermediates=True,
            y=class_list,
        ).detach().cpu()
        _save_trajectory_visuals(
            sol=backward_sol,
            t_eval=t_eval.detach().cpu(),
            num_classes=num_classes,
            eval_dir=output_dir,
            prefix=f"backward_({args.lambda_eval_u},{args.lambda_eval_d})",
            fps=args.fps,
            do_grid=args.do_grid,
            do_trajectory=args.do_trajectory,
        )


if __name__ == "__main__":
    main(tyro.cli(EvalArguments))
