from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module
from tqdm.auto import tqdm

from bridge_matching import visualization
from bridge_matching.datasets import TOY_DATASETS, SyntheticDataset, ToyDatasetName
from bridge_matching.solver import TimeBroadcastWrapper
from bridge_matching.utils import set_seed
from bridge_matching.targets import make_bridge_matching_targets


@dataclass
class ScriptArguments:
    
    source_dataset: ToyDatasetName | None = "gaussian" # "moons", "mixture", "siggraph", "checkerboard", "invertocat", "gaussian"
    target_dataset: ToyDatasetName = "checkerboard" # "moons", "mixture", "siggraph", "checkerboard", "invertocat", "gaussian"
    
    output_dir: Path = Path("outputs")
    learning_rate: float = 1e-3
    batch_size: int = 4096  
    iterations: int = 100000
    log_every: int = 2000
    hidden_dim: int = 512
    beta_value: float = 0.01   # 1e-2 for diffusion targets, 0.1 for linear targets
    sigma_floor: float = 0.05 # 5e-2 for diffusion targets, 0.1 for linear targets
    t_eps: float = 1e-2
    lambda_d: float = 1 # Core
    
    target_type: str = "bm_diffusion_conditional" 
    # ------------------------------------------------------------
    # Available Target Types
    # ------------------------------------------------------------
    # Diffusion-based targets:
    #   - diffusion_conditional
    #   - diffusion_marginal
    #   - diffusion_cfm_decomp **
    #
    # Linear-based targets:
    #   - linear_tube_conditional
    #   - linear_tube_symmetric_conditional
    #   - linear_kde_marginal **
    #   - linear_conditional
    #   - linear_ot_conditional
    #   - linear_isotropic_conditional
    #   - linear_brownian_conditional
    #   - linear_anisotropic_conditional
    #   - linear_multibridge_conditional
    #
    # Stochastic-interpolant-based targets:
    #   - stochastic_interpolant_conditional

    posterior_temperature: float = 1.0
    posterior_topk: int = 0  # 0 means disabled
    seed: int = 42


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


def main(args: ScriptArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    source_dataset_name = args.source_dataset if args.source_dataset is not None else "gaussian"
    target_dataset_name = args.target_dataset

    # Short target name (e.g., diffusion_cfm_decomp -> dcd)
    target_short = ''.join([word[0] for word in args.target_type.split('_')])

    output_dir = args.output_dir / "bm" / f"{source_dataset_name}2{target_dataset_name}" / f"{target_short}_{run_time}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    log_f = open(log_file, "w")

    # Log full configuration
    log_f.write("=== Configuration ===\n")
    for key, value in vars(args).items():
        log_f.write(f"{key}: {value}\n")
    log_f.write("=====================\n\n")
    log_f.flush()


    print(f"Using device: {device}")
    print(f"Dataset: {args.source_dataset} -> {args.target_dataset}")
    print(f"Posterior temperature: {args.posterior_temperature}")
    print(f"Posterior top-k: {args.posterior_topk if args.posterior_topk > 0 else None}")
    log_f.write(f"Using device: {device}\n")
    log_f.write(f"Dataset: {args.source_dataset} -> {args.target_dataset}\n")
    log_f.write(f"Posterior temperature: {args.posterior_temperature}\n")
    log_f.write(f"Posterior top-k: {args.posterior_topk if args.posterior_topk > 0 else None}\n")

    source_dataset: SyntheticDataset = TOY_DATASETS[source_dataset_name](device=device) 
    dataset: SyntheticDataset = TOY_DATASETS[target_dataset_name](device=device)
    

    flow_u = Mlp(dim=dataset.dim, time_dim=1, h=args.hidden_dim).to(device)
    flow_d = Mlp(dim=dataset.dim, time_dim=1, h=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(list(flow_u.parameters()) + list(flow_d.parameters()), args.learning_rate)

    posterior_topk = args.posterior_topk if args.posterior_topk > 0 else None

    bm_targets = make_bridge_matching_targets(
        dataset=dataset,
        beta_value=args.beta_value,
        sigma_floor=args.sigma_floor,
        t_eps=args.t_eps,
        lambda_d=args.lambda_d,
        target_type=args.target_type,
        posterior_temperature=args.posterior_temperature,
        posterior_topk=posterior_topk,
    )

    losses = []
    for global_step in tqdm(range(args.iterations), desc="Training", dynamic_ncols=True):
        x_1 = dataset.sample(args.batch_size)
        x_0 = source_dataset.sample(args.batch_size)
        t = torch.rand(x_1.size(0), 1, device=device)

        optimizer.zero_grad()

        x_t, u_t, d_t = bm_targets.compute(x_0=x_0, x_1=x_1, t=t)

        u_pred = flow_u(x_t=x_t, t=t)
        d_pred = flow_d(x_t=x_t, t=t)

        loss_u = F.mse_loss(u_pred, u_t)
        loss_d = F.mse_loss(d_pred, d_t)

        loss = loss_u + args.lambda_d * loss_d

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (global_step + 1) % args.log_every == 0:
            tqdm.write(
                f"| step: {global_step + 1:6d} | loss: {loss.item():8.4f} | "
                f"loss_u: {loss_u.item():8.4f} | loss_d: {loss_d.item():8.4f} |"
            )
            log_f.write(
                f"| step: {global_step + 1:6d} | loss: {loss.item():8.4f} | "
                f"loss_u: {loss_u.item():8.4f} | loss_d: {loss_d.item():8.4f} |\n"
            )
            log_f.flush()

    flow_u.eval()
    flow_d.eval()
    # ===== Compute mean norms of u and d =====
    with torch.no_grad():
        x_1 = dataset.sample(args.batch_size)
        x_0 = source_dataset.sample(args.batch_size)
        t = torch.rand(x_1.size(0), 1, device=device)

        x_t, u_t, d_t = bm_targets.compute(x_0=x_0, x_1=x_1, t=t)
        u_pred = flow_u(x_t=x_t, t=t)
        d_pred = flow_d(x_t=x_t, t=t)

        u_norm = torch.norm(u_pred, dim=1).mean().item()
        d_norm = torch.norm(d_pred, dim=1).mean().item()
        ratio = d_norm / (u_norm + 1e-8)

        print(f"mean ||u|| = {u_norm}")
        print(f"mean ||d|| = {d_norm}")
        print(f"ratio ||d|| / ||u|| = {ratio}")

        log_f.write(f"mean ||u|| = {u_norm}\n")
        log_f.write(f"mean ||d|| = {d_norm}\n")
        log_f.write(f"ratio ||d|| / ||u|| = {ratio}\n")
        log_f.flush()
    torch.save(
        {
            "flow_u": flow_u.state_dict(),
            "flow_d": flow_d.state_dict(),
            "args": args,
        },
        output_dir / "ckpt.pth",
    )
    visualization.plot_loss_curve(losses=losses, output_path=output_dir / "losses.png")

    # Sampling with ODE solver and visualization
    # Keep the existing visualization pipeline by wrapping the forward field u + d.

    # class ForwardBridgeWrapper(torch.nn.Module):
    #     def __init__(self, flow_u, flow_d):
    #         super().__init__()
    #         self.flow_u = flow_u
    #         self.flow_d = flow_d

    #     def forward(self, x_t, t):
    #         return self.flow_u(x_t=x_t, t=t) + self.flow_d(x_t=x_t, t=t)

    # wrapped_model = TimeBroadcastWrapper(ForwardBridgeWrapper(flow_u, flow_d))

    # visualization.plot_ode_sampling_evolution(
    #     flow=wrapped_model,
    #     dataset=dataset,
    #     output_dir=output_dir,
    #     filename=f"ode_sampling_evolution_{args.dataset}.png",
    # )

    # visualization.save_vector_field_and_samples_as_gif(
    #     flow=wrapped_model,
    #     dataset=dataset,
    #     output_dir=output_dir,
    #     filename=f"vector_field_and_samples_{args.dataset}.gif",
    # )

    # visualization.plot_likelihood(
    #     flow=wrapped_model,
    #     dataset=dataset,
    #     output_dir=output_dir,
    #     filename=f"likelihood_{args.dataset}.png",
    # )

    log_f.close()


if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))
