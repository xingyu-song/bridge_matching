from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import tyro
from torch.serialization import add_safe_globals
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module

from bridge_matching import visualization
from bridge_matching.datasets import TOY_DATASETS, SyntheticDataset, ToyDatasetName
from bridge_matching.solver import ODESolver, TimeBroadcastWrapper
from bridge_matching.utils import set_seed

# Magnitude heatmaps in plot_u_d_vector_fields_over_time (dark yellow → dark red).
_FIELD_DECOMP_MAG_CMAP = LinearSegmentedColormap.from_list(
    "field_dark_yellow_red",
    [
        "#eaf3c6",
        "#f0f1bf",
        "#f7efb8",
        "#ffeead",
        "#f8d3a7",
        "#f2b8a1",
        "#eb9f9a",
        "#e78993",
    ],
    N=256,
)


@dataclass
class EvalArguments:
    ckpt_path: Path = Path("outputs/2d/bm/gaussian2checkerboard/bdc_20260428172130/ckpt.pth")
    output_dir: Path | None = None

    # dataset: ToyDatasetName = "checkerboard" # "moons", "mixture", "siggraph", "checkerboard", "invertocat"
    hidden_dim: int = 512
    ####### change here #######
    lambda_eval_u: float = 1.0 # change here ******
    lambda_eval_d: float = 1.0 # change here ******
    lambda_eval_d_sweep: str = ""  # e.g. "0.0,0.5,1.0,1.5"
    seed: int = 42
    deterministic_metrics: bool = True
    ############################

    do_backward: bool = False
    do_sampling_plot: bool = False
    do_gif: bool = False
    do_likelihood: bool = False
    do_field_decomposition_plot: bool = True
    do_metrics: bool = False

    field_grid_size: int = 45
    field_time_steps: int = 5
    field_num_context_samples: int = 20_000
    field_quiver_scale: float = 1.0
    field_quiver_width: float = 0.0025
    field_arrow_length: float = 0.12
    field_magnitude_eps: float = 1e-8
    field_arrow_stride: int = 2
    field_heatmap_levels: int = 120
    field_use_streamplot: bool = True
    field_stream_density: float = 1.6
    field_stream_linewidth: float = 1.2
    # Scale learned u and d before plotting (sum row shows s*u + s*d = s*(u+d)).
    field_visual_scale: float = 0.5

    sampling_num_samples: int = 1_000_000
    sampling_step_size: float = 0.01
    sampling_steps: int = 50
    sampling_method: str = "midpoint"
    sampling_atol: float = 1e-5
    sampling_rtol: float = 1e-5 
    metrics_num_samples: int = 10_000

    gif_sample_steps: int = 101
    gif_grid_size: int = 15
    gif_num_samples: int = 500_000
    gif_interval: int = 50



# Compatibility alias for checkpoints that saved training args as
# __main__.ScriptArguments. This keeps loading working without maintaining
# a second nearly identical dataclass in this file.
ScriptArguments = EvalArguments


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, dim),
        )

    def forward(
        self,
        x_t: Float[Tensor, "batch dim"],
        t: Float[Tensor, "batch time_dim"],
    ) -> Float[Tensor, "batch dim"]:
        h = torch.cat([x_t, t], dim=1)
        return self.layers(h)


class ForwardBridgeWrapper(torch.nn.Module):
    def __init__(self, flow_u: Module, flow_d: Module, lambda_eval_d: float = 1.0, lambda_eval_u: float = 1.0):
        super().__init__()
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.lambda_eval_d = lambda_eval_d
        self.lambda_eval_u = lambda_eval_u

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.lambda_eval_u * self.flow_u(x_t=x_t, t=t) + self.lambda_eval_d * self.flow_d(x_t=x_t, t=t)
    

class BackwardBridgeWrapper(torch.nn.Module):
    def __init__(self, flow_u: Module, flow_d: Module, lambda_eval_d: float = 1.0, lambda_eval_u: float = 1.0):
        super().__init__()
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.lambda_eval_d = lambda_eval_d
        self.lambda_eval_u = lambda_eval_u

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_rev = 1.0 - t
        return -(
            self.lambda_eval_u * self.flow_u(x_t=x_t, t=t_rev)
            - self.lambda_eval_d * self.flow_d(x_t=x_t, t=t_rev)
        )


def _matrix_sqrt_psd(mat: Tensor, eps: float = 1e-12) -> Tensor:
    mat = 0.5 * (mat + mat.T)
    if not torch.isfinite(mat).all():
        return torch.full_like(mat, torch.nan)

    jitter = 0.0
    eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
    for _ in range(6):
        try:
            evals, evecs = torch.linalg.eigh(mat + jitter * eye)
            evals = torch.clamp(evals, min=0.0)
            sqrt_evals = torch.sqrt(evals)
            return (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T
        except RuntimeError:
            jitter = eps if jitter == 0.0 else jitter * 10.0

    return torch.full_like(mat, torch.nan)


def _trace_sqrt_cov_product(cov_r: Tensor, cov_f: Tensor) -> Tensor:
    cov_r = cov_r.to(torch.float64)
    cov_f = cov_f.to(torch.float64)
    cov_r_sqrt = _matrix_sqrt_psd(cov_r)
    cov_prod = cov_r_sqrt @ cov_f @ cov_r_sqrt
    cov_prod_sqrt = _matrix_sqrt_psd(cov_prod)
    return torch.trace(cov_prod_sqrt)


def _compute_mean_and_cov(features: Tensor) -> tuple[Tensor, Tensor]:
    mu = features.mean(dim=0)
    centered = features - mu
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    return mu, cov


def _compute_fid(real_features: Tensor, fake_features: Tensor) -> float:
    mu_r, cov_r = _compute_mean_and_cov(real_features)
    mu_f, cov_f = _compute_mean_and_cov(fake_features)
    mean_diff = ((mu_r.to(torch.float64) - mu_f.to(torch.float64)) ** 2).sum()
    fid = mean_diff + torch.trace(cov_r.to(torch.float64) + cov_f.to(torch.float64)) - 2.0 * _trace_sqrt_cov_product(cov_r, cov_f)
    fid = torch.clamp(fid, min=0.0)
    return float(fid.item())


def _compute_rbf_mmd(real_features: Tensor, fake_features: Tensor) -> float:
    pairwise_rf = torch.cdist(real_features, fake_features, p=2) ** 2
    pairwise_rr = torch.cdist(real_features, real_features, p=2) ** 2
    pairwise_ff = torch.cdist(fake_features, fake_features, p=2) ** 2

    # Median heuristic for RBF bandwidth on cross distances.
    sigma2 = torch.median(pairwise_rf).clamp_min(1e-6)

    k_rf = torch.exp(-pairwise_rf / (2.0 * sigma2))
    k_rr = torch.exp(-pairwise_rr / (2.0 * sigma2))
    k_ff = torch.exp(-pairwise_ff / (2.0 * sigma2))

    n = real_features.shape[0]
    m = fake_features.shape[0]
    sum_rr = (k_rr.sum() - torch.diagonal(k_rr).sum()) / max(n * (n - 1), 1)
    sum_ff = (k_ff.sum() - torch.diagonal(k_ff).sum()) / max(m * (m - 1), 1)
    mmd2 = sum_rr + sum_ff - 2.0 * k_rf.mean()
    return float(mmd2.item())


def _sample_final_from_flow(
    flow: Module,
    source_dataset: SyntheticDataset,
    num_samples: int,
    step_size: float | None,
    method: str,
    atol: float,
    rtol: float,
    device: torch.device,
    x_init: Tensor | None = None,
) -> Tensor:
    solver = ODESolver(flow)
    if x_init is None:
        x_init = source_dataset.sample(num_samples).to(device)
    else:
        x_init = x_init.to(device)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    sol = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        atol=atol,
        rtol=rtol,
        time_grid=time_grid,
        return_intermediates=False,
    )
    return sol.detach()


def _run_2d_metrics(
    wrapped_model: Module,
    source_dataset: SyntheticDataset,
    target_dataset: SyntheticDataset,
    args: EvalArguments,
    device: torch.device,
    output_dir: Path,
    direction_name: str,
    lambda_eval_u: float,
    lambda_eval_d: float,
    cached_source_samples: Tensor | None = None,
    cached_real_samples: Tensor | None = None,
) -> None:
    fake_samples = _sample_final_from_flow(
        flow=wrapped_model,
        source_dataset=source_dataset,
        num_samples=args.metrics_num_samples,
        step_size=args.sampling_step_size,
        method=args.sampling_method,
        atol=args.sampling_atol,
        rtol=args.sampling_rtol,
        device=device,
        x_init=cached_source_samples,
    )
    if cached_real_samples is None:
        real_samples = target_dataset.sample(args.metrics_num_samples).to(device)
    else:
        real_samples = cached_real_samples.to(device)

    fid = _compute_fid(real_samples, fake_samples)
    mmd_rbf = _compute_rbf_mmd(real_samples, fake_samples)

    metrics_text = (
        f"Direction: {direction_name}\n"
        f"FID_2D: {fid:.6f}\n"
        f"MMD_RBF_2D: {mmd_rbf:.6f}\n"
        f"metrics_num_samples: {args.metrics_num_samples}\n"
        f"sampling_method: {args.sampling_method}\n"
        f"sampling_step_size: {args.sampling_step_size}\n"
        f"sampling_atol: {args.sampling_atol}\n"
        f"sampling_rtol: {args.sampling_rtol}\n"
        f"lambda_eval_u: {lambda_eval_u}\n"
        f"lambda_eval_d: {lambda_eval_d}\n"
        f"seed: {args.seed}\n"
        f"deterministic_metrics: {args.deterministic_metrics}\n"
    )
    print(metrics_text)
    metrics_path = output_dir / (
        f"metrics_{direction_name}_{args.sampling_method}_"
        f"({lambda_eval_u},{lambda_eval_d}).txt"
    )
    metrics_path.write_text(metrics_text)
    print(f"Saved metrics to {metrics_path}")


# === Helper for field decomposition plot ===
def plot_u_d_vector_fields_over_time(
    flow_u: Module,
    flow_d: Module,
    source_dataset: 'SyntheticDataset',
    target_dataset: 'SyntheticDataset',
    output_dir: Path,
    filename: str,
    device: torch.device,
    grid_size: int = 45,
    time_steps: int = 5,
    num_context_samples: int = 20_000,
    quiver_scale: float = 1.0,
    quiver_width: float = 0.0025,
    arrow_length: float = 0.12,
    magnitude_eps: float = 1e-8,
    arrow_stride: int = 2,
    heatmap_levels: int = 120,
    use_streamplot: bool = True,
    stream_density: float = 1.6,
    stream_linewidth: float = 1.2,
    field_visual_scale: float = 0.5,
) -> None:
    """Visualize learned u, d, and s*u + s*d (s = field_visual_scale) on the same 2D grid.

    The plot range is estimated from source and target samples so the field plot
    covers the region where trajectories usually live. At each time step, scaled
    fields s*u and s*d and their sum are visualized in separate panels: heatmap
    intensity shows field magnitude, while streamlines or sparse equal-length
    arrows show field direction. Axis tick labels are hidden for a cleaner figure.
    """
    if source_dataset.dim != 2 or target_dataset.dim != 2:
        raise ValueError("plot_u_d_vector_fields_over_time only supports 2D datasets.")

    output_dir.mkdir(parents=True, exist_ok=True)

    flow_u.eval()
    flow_d.eval()

    with torch.no_grad():
        source_samples = source_dataset.sample(num_context_samples).detach()
        target_samples = target_dataset.sample(num_context_samples).detach()
        context_samples = torch.cat([source_samples, target_samples], dim=0)

        xy_min = context_samples.min(dim=0).values
        xy_max = context_samples.max(dim=0).values
        xy_center = 0.5 * (xy_min + xy_max)
        xy_half_width = 0.5 * (xy_max - xy_min)
        xy_half_width = torch.clamp(xy_half_width * 1.15, min=1.0)
        xy_min = xy_center - xy_half_width
        xy_max = xy_center + xy_half_width

        xs = torch.linspace(xy_min[0].item(), xy_max[0].item(), grid_size, device=device)
        ys = torch.linspace(xy_min[1].item(), xy_max[1].item(), grid_size, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        if time_steps <= 1:
            ts = torch.tensor([0.5], device=device)
        else:
            ts = torch.linspace(0.0, 1.0, time_steps, device=device)

        # (evolution of samples removed)

        ncols = time_steps
        nrows = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 10.5), squeeze=False)

        xx_np = xx.detach().cpu().numpy()
        yy_np = yy.detach().cpu().numpy()
        stride = max(1, arrow_stride)
        xx_sparse_np = xx_np[::stride, ::stride]
        yy_sparse_np = yy_np[::stride, ::stride]

        s_tex = f"{float(field_visual_scale):g}"

        def style_field_axis(ax: plt.Axes) -> None:
            """No tick marks or numeric labels; grid still uses major tick locations."""
            ax.tick_params(
                axis="both",
                which="both",
                length=0,
                labelbottom=False,
                labelleft=False,
            )

        for idx, t_value in enumerate(ts):
            t_grid = torch.full((grid.shape[0], 1), float(t_value.item()), device=device)
            u = field_visual_scale * flow_u(x_t=grid, t=t_grid).reshape(
                grid_size, grid_size, 2
            )
            d = field_visual_scale * flow_d(x_t=grid, t=t_grid).reshape(
                grid_size, grid_size, 2
            )
            ud = u + d

            u_mag = torch.linalg.norm(u, dim=-1)
            d_mag = torch.linalg.norm(d, dim=-1)
            ud_mag = torch.linalg.norm(ud, dim=-1)

            u_dir = u / torch.clamp(u_mag.unsqueeze(-1), min=magnitude_eps)
            d_dir = d / torch.clamp(d_mag.unsqueeze(-1), min=magnitude_eps)
            ud_dir = ud / torch.clamp(ud_mag.unsqueeze(-1), min=magnitude_eps)

            u_plot = (arrow_length * u_dir).detach().cpu().numpy()
            d_plot = (arrow_length * d_dir).detach().cpu().numpy()
            ud_plot = (arrow_length * ud_dir).detach().cpu().numpy()
            u_mag_np = u_mag.detach().cpu().numpy()
            d_mag_np = d_mag.detach().cpu().numpy()
            ud_mag_np = ud_mag.detach().cpu().numpy()

            u_ax = axes[0][idx]
            d_ax = axes[1][idx]
            ud_ax = axes[2][idx]

            u_heat = u_ax.contourf(
                xx_np, yy_np, u_mag_np, levels=heatmap_levels, cmap=_FIELD_DECOMP_MAG_CMAP
            )
            if use_streamplot:
                u_ax.streamplot(
                    xx_np,
                    yy_np,
                    u_dir.detach().cpu().numpy()[:, :, 0],
                    u_dir.detach().cpu().numpy()[:, :, 1],
                    color="black",
                    density=stream_density,
                    linewidth=stream_linewidth,
                    arrowsize=1.0,
                )
            else:
                u_ax.quiver(
                    xx_sparse_np,
                    yy_sparse_np,
                    u_plot[::stride, ::stride, 0],
                    u_plot[::stride, ::stride, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    color="black",
                    alpha=0.85,
                    width=quiver_width,
                    headwidth=4.0,
                    headlength=5.0,
                    headaxislength=4.0,
                )
            u_ax.set_title(rf"${s_tex}\,u$ field, t = {float(t_value.item()):.2f}")
            u_ax.set_xlim(xy_min[0].item(), xy_max[0].item())
            u_ax.set_ylim(xy_min[1].item(), xy_max[1].item())
            u_ax.set_aspect("equal", adjustable="box")
            u_ax.grid(alpha=0.15)
            style_field_axis(u_ax)
            if idx == time_steps - 1:
                cbar_u = fig.colorbar(u_heat, ax=u_ax, fraction=0.046, pad=0.04)
                cbar_u.set_label(rf"$\|{s_tex}\,u\|$", fontsize=8)

            d_heat = d_ax.contourf(
                xx_np, yy_np, d_mag_np, levels=heatmap_levels, cmap=_FIELD_DECOMP_MAG_CMAP
            )
            if use_streamplot:
                d_ax.streamplot(
                    xx_np,
                    yy_np,
                    d_dir.detach().cpu().numpy()[:, :, 0],
                    d_dir.detach().cpu().numpy()[:, :, 1],
                    color="black",
                    density=stream_density,
                    linewidth=stream_linewidth,
                    arrowsize=1.0,
                )
            else:
                d_ax.quiver(
                    xx_sparse_np,
                    yy_sparse_np,
                    d_plot[::stride, ::stride, 0],
                    d_plot[::stride, ::stride, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    color="black",
                    alpha=0.85,
                    width=quiver_width,
                    headwidth=4.0,
                    headlength=5.0,
                    headaxislength=4.0,
                )
            d_ax.set_title(rf"${s_tex}\,d$ field, t = {float(t_value.item()):.2f}")
            d_ax.set_xlim(xy_min[0].item(), xy_max[0].item())
            d_ax.set_ylim(xy_min[1].item(), xy_max[1].item())
            d_ax.set_aspect("equal", adjustable="box")
            d_ax.grid(alpha=0.15)
            style_field_axis(d_ax)
            if idx == time_steps - 1:
                cbar_d = fig.colorbar(d_heat, ax=d_ax, fraction=0.046, pad=0.04)
                cbar_d.set_label(rf"$\|{s_tex}\,d\|$", fontsize=8)

            ud_heat = ud_ax.contourf(
                xx_np, yy_np, ud_mag_np, levels=heatmap_levels, cmap=_FIELD_DECOMP_MAG_CMAP
            )
            if use_streamplot:
                ud_ax.streamplot(
                    xx_np,
                    yy_np,
                    ud_dir.detach().cpu().numpy()[:, :, 0],
                    ud_dir.detach().cpu().numpy()[:, :, 1],
                    color="black",
                    density=stream_density,
                    linewidth=stream_linewidth,
                    arrowsize=1.0,
                )
            else:
                ud_ax.quiver(
                    xx_sparse_np,
                    yy_sparse_np,
                    ud_plot[::stride, ::stride, 0],
                    ud_plot[::stride, ::stride, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=quiver_scale,
                    color="black",
                    alpha=0.85,
                    width=quiver_width,
                    headwidth=4.0,
                    headlength=5.0,
                    headaxislength=4.0,
                )
            ud_ax.set_title(
                rf"${s_tex}\,u + {s_tex}\,d$ field, t = {float(t_value.item()):.2f}"
            )
            ud_ax.set_xlim(xy_min[0].item(), xy_max[0].item())
            ud_ax.set_ylim(xy_min[1].item(), xy_max[1].item())
            ud_ax.set_aspect("equal", adjustable="box")
            ud_ax.grid(alpha=0.15)
            style_field_axis(ud_ax)
            if idx == time_steps - 1:
                cbar_ud = fig.colorbar(ud_heat, ax=ud_ax, fraction=0.046, pad=0.04)
                cbar_ud.set_label(rf"$\|{s_tex}\,u + {s_tex}\,d\|$", fontsize=8)

        direction_style = "streamlines" if use_streamplot else "direction arrows"
        fig.suptitle(
            f"Magnitude heatmap and {direction_style} for "
            f"{s_tex}·u, {s_tex}·d, and {s_tex}·u + {s_tex}·d",
            y=1.02,
        )
        fig.tight_layout(pad=0.15, w_pad=0.001, h_pad=0.12)
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main(args: EvalArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # These local training checkpoints include the pickled ScriptArguments object.
    # PyTorch 2.6 defaults to weights_only=True, which rejects that metadata.
    add_safe_globals([ScriptArguments])
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    saved_args = ckpt.get("args", None)
    folder_name = args.ckpt_path.parent.parent.name  # "[source]2[target]"
    source_dataset_name, target_dataset_name = folder_name.split("2")
    hidden_dim = saved_args.hidden_dim if saved_args is not None and hasattr(saved_args, "hidden_dim") else args.hidden_dim

    source_dataset: SyntheticDataset = TOY_DATASETS[source_dataset_name](device=device)
    target_dataset: SyntheticDataset = TOY_DATASETS[target_dataset_name](device=device)

    dim = source_dataset.dim
    flow_u = Mlp(dim=dim, time_dim=1, h=hidden_dim).to(device)
    flow_d = Mlp(dim=dim, time_dim=1, h=hidden_dim).to(device)

    flow_u.load_state_dict(ckpt["flow_u"])
    flow_d.load_state_dict(ckpt["flow_d"])

    flow_u.eval()
    flow_d.eval()

# ===== DEBUG CHECK HERE =====
    with torch.no_grad():
        x = source_dataset.sample(2048)  # shape (2048, 2)
        t = torch.rand(2048, 1, device=device)

        u = flow_u(x_t=x, t=t)
        d = flow_d(x_t=x, t=t)

        print("mean ||u|| =", u.norm(dim=1).mean().item())
        print("mean ||d|| =", d.norm(dim=1).mean().item())
        print("ratio ||d|| / ||u|| =", (d.norm(dim=1).mean() / u.norm(dim=1).mean()).item())
# ============================

    # Always save under the checkpoint folder: .../<run>/eval
    output_dir = args.ckpt_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Dataset: {source_dataset_name} -> {target_dataset_name}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"lambda_eval_u: {args.lambda_eval_u}")
    print(f"lambda_eval_d: {args.lambda_eval_d}")
    print(f"lambda_eval_d_sweep: {args.lambda_eval_d_sweep if args.lambda_eval_d_sweep else '(disabled)'}")
    print(f"seed: {args.seed}")
    print(f"deterministic_metrics: {args.deterministic_metrics}")
    print(f"Output dir: {output_dir}")
    print(f"do_field_decomposition_plot: {args.do_field_decomposition_plot}")
    print(f"field_grid_size: {args.field_grid_size}")
    print(f"field_time_steps: {args.field_time_steps}")
    print(f"field_visual_scale: {args.field_visual_scale}")
    print(f"do_metrics: {args.do_metrics}")
    print(f"metrics_num_samples: {args.metrics_num_samples}")
    print(f"sampling_method: {args.sampling_method}")

    if args.do_field_decomposition_plot:
        plot_u_d_vector_fields_over_time(
            flow_u=flow_u,
            flow_d=flow_d,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            output_dir=output_dir,
            filename=f"u_d_vector_fields_{source_dataset_name}_to_{target_dataset_name}.png",
            device=device,
            grid_size=args.field_grid_size,
            time_steps=args.field_time_steps,
            num_context_samples=args.field_num_context_samples,
            quiver_scale=args.field_quiver_scale,
            quiver_width=args.field_quiver_width,
            arrow_length=args.field_arrow_length,
            magnitude_eps=args.field_magnitude_eps,
            arrow_stride=args.field_arrow_stride,
            heatmap_levels=args.field_heatmap_levels,
            use_streamplot=args.field_use_streamplot,
            stream_density=args.field_stream_density,
            stream_linewidth=args.field_stream_linewidth,
            field_visual_scale=args.field_visual_scale,
        )

# ===== FORWARD =====
    lambda_eval_d_values = [args.lambda_eval_d]
    if args.lambda_eval_d_sweep.strip():
        lambda_eval_d_values = [
            float(x.strip()) for x in args.lambda_eval_d_sweep.split(",") if x.strip()
        ]

    cached_source_samples_forward = None
    cached_target_samples_forward = None
    if args.deterministic_metrics and args.do_metrics:
        cached_source_samples_forward = source_dataset.sample(args.metrics_num_samples).to(device)
        cached_target_samples_forward = target_dataset.sample(args.metrics_num_samples).to(device)

    wrapped_model = TimeBroadcastWrapper(
        ForwardBridgeWrapper(
            flow_u=flow_u,
            flow_d=flow_d,
            lambda_eval_d=lambda_eval_d_values[0],
            lambda_eval_u=args.lambda_eval_u,
        )
    )

    if args.do_metrics:
        for lambda_eval_d in lambda_eval_d_values:
            wrapped_model = TimeBroadcastWrapper(
                ForwardBridgeWrapper(
                    flow_u=flow_u,
                    flow_d=flow_d,
                    lambda_eval_d=lambda_eval_d,
                    lambda_eval_u=args.lambda_eval_u,
                )
            )
            _run_2d_metrics(
                wrapped_model=wrapped_model,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                args=args,
                device=device,
                output_dir=output_dir,
                direction_name=f"{source_dataset_name}_to_{target_dataset_name}",
                lambda_eval_u=args.lambda_eval_u,
                lambda_eval_d=lambda_eval_d,
                cached_source_samples=cached_source_samples_forward,
                cached_real_samples=cached_target_samples_forward,
            )

    if args.do_sampling_plot:
        visualization.plot_ode_sampling_evolution(
            flow=wrapped_model,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            num_samples=args.sampling_num_samples,
            step_size=args.sampling_step_size,
            sample_steps=args.sampling_steps,
            output_dir=output_dir,
            filename=f"ode_sampling_evolution_{source_dataset_name}_to_{target_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).png",
        )

    if args.do_gif:
        visualization.save_vector_field_and_samples_as_gif(
            flow=wrapped_model,
            target_dataset=target_dataset,
            source_dataset=source_dataset,
            sample_steps=args.gif_sample_steps,
            grid_size=args.gif_grid_size,
            num_samples=args.gif_num_samples,
            interval=args.gif_interval,
            output_dir=output_dir,
            filename=f"vector_field_and_samples_{source_dataset_name}_to_{target_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).gif",
        )

    if args.do_likelihood:
        visualization.plot_likelihood(
            flow=wrapped_model,
            target_dataset=target_dataset,
            source_dataset=source_dataset,
            output_dir=output_dir,
            filename=f"likelihood_{source_dataset_name}_to_{target_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).png",
        )

# ===== BACKWARD =====
    if args.do_backward:
        cached_source_samples_backward = None
        cached_target_samples_backward = None
        if args.deterministic_metrics and args.do_metrics:
            cached_source_samples_backward = target_dataset.sample(args.metrics_num_samples).to(device)
            cached_target_samples_backward = source_dataset.sample(args.metrics_num_samples).to(device)

        wrapped_model = TimeBroadcastWrapper(
            BackwardBridgeWrapper(
                flow_u=flow_u,
                flow_d=flow_d,
                lambda_eval_d=lambda_eval_d_values[0],
                lambda_eval_u=args.lambda_eval_u,
            )
        )
        if args.do_metrics:
            for lambda_eval_d in lambda_eval_d_values:
                wrapped_model = TimeBroadcastWrapper(
                    BackwardBridgeWrapper(
                        flow_u=flow_u,
                        flow_d=flow_d,
                        lambda_eval_d=lambda_eval_d,
                        lambda_eval_u=args.lambda_eval_u,
                    )
                )
                _run_2d_metrics(
                    wrapped_model=wrapped_model,
                    source_dataset=target_dataset,
                    target_dataset=source_dataset,
                    args=args,
                    device=device,
                    output_dir=output_dir,
                    direction_name=f"{target_dataset_name}_to_{source_dataset_name}",
                    lambda_eval_u=args.lambda_eval_u,
                    lambda_eval_d=lambda_eval_d,
                    cached_source_samples=cached_source_samples_backward,
                    cached_real_samples=cached_target_samples_backward,
                )
        if args.do_sampling_plot:
            visualization.plot_ode_sampling_evolution(
                flow=wrapped_model,
                source_dataset=target_dataset,
                target_dataset=source_dataset,
                num_samples=args.sampling_num_samples,
                step_size=args.sampling_step_size,
                sample_steps=args.sampling_steps,
                output_dir=output_dir,
                filename=f"ode_sampling_evolution_{target_dataset_name}_to_{source_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).png",
            )

        if args.do_gif:
            visualization.save_vector_field_and_samples_as_gif(
                flow=wrapped_model,
                target_dataset=source_dataset,
                source_dataset=target_dataset,
                sample_steps=args.gif_sample_steps,
                grid_size=args.gif_grid_size,
                num_samples=args.gif_num_samples,
                interval=args.gif_interval,
                output_dir=output_dir,
                filename=f"vector_field_and_samples_{target_dataset_name}_to_{source_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).gif",
            )

        if args.do_likelihood:
            visualization.plot_likelihood(
                flow=wrapped_model,
                target_dataset=source_dataset,
                source_dataset=target_dataset,
                output_dir=output_dir,
                filename=f"likelihood_{target_dataset_name}_to_{source_dataset_name}_({args.lambda_eval_u},{args.lambda_eval_d}).png",
            )

if __name__ == "__main__":
    main(tyro.cli(EvalArguments))