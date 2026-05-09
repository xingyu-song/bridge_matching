from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from contextlib import nullcontext
from typing import TextIO

import torch
import torch.nn.functional as F
import tyro
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bridge_matching import visualization
from bridge_matching.datasets.image_datasets import (
    get_image_dataset,
    get_test_transform,
    get_train_transform,
)
from bridge_matching.metrics import (
    build_inception_feature_model,
    collect_real_images,
    compute_fid,
    extract_inception_features,
    generate_fake_images,
)
from bridge_matching.models import UNetModel
from bridge_matching.solver import ModelWrapper, ODESolver
from bridge_matching.targets import make_bridge_matching_targets
from bridge_matching.utils import model_size_summary, set_seed
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class ScriptArguments:
    # Data / run setup
    dataset: str = "cifar10"
    output_dir: Path = Path("outputs")
    seed: int = 42
    horizontal_flip: bool = True

    # Optimization
    batch_size: int = 128
    n_epochs: int = 10
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.9999
    mixed_precision: bool = False
    use_lr_scheduler: bool = False
    warmup_ratio: float = 0.2  # used if warmup_steps <= 0
    warmup_steps: int = 45000
    lr_decay: str = "polynomial"  # options: "linear", "cosine", "polynomial", "none"
    min_learning_rate: float = 1e-8

    # Bridge-matching target / path config
    beta_value: float = 1e-2
    sigma_floor: float = 0.05
    sigma_min: float = 0.0
    t_eps: float = 0.01
    lambda_d: float = 1.0
    lambda_forward_align: float = 0.0
    target_type: str = "cfm_linear"  # e.g. "diffusion_cfm_decomp", "cfm_linear", or "cfm_diffusion"
    cfm_beta_min: float = 0.1  # song.md VP diffusion config
    cfm_beta_max: float = 20.0  # song.md VP diffusion config
    posterior_temperature: float = 1.0
    posterior_topk: int = 0  # 0 means disabled
    class_cond: bool = False

    # UNet architecture
    num_channels: int = 128
    num_res_blocks: int = 2
    dropout: float = 0.0
    channel_mult: str = "1,2,2,2"
    attention_resolutions: str = "16"
    # Checkpoint + on-the-fly FID eval
    ckpt_every_epochs: int = 20
    eval_fid_on_ckpt: bool = True
    fid_num_samples: int = 500
    fid_metrics_subset_size: int = 500
    fid_batch_size: int = 256
    fid_num_inference_steps: int = 30
    fid_method: str = "midpoint"
    fid_atol: float = 1e-5
    fid_rtol: float = 1e-5
    resume_from_ckpt: Path | None = None


class ForwardBridgeWrapper(torch.nn.Module):
    def __init__(self, flow_u: torch.nn.Module, flow_d: torch.nn.Module, use_d: bool = True):
        super().__init__()
        self.flow_u = flow_u
        self.flow_d = flow_d
        self.use_d = use_d

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        u = self.flow_u(t=t, x=x, y=y)
        if not self.use_d:
            return u
        return u + self.flow_d(t=t, x=x, y=y)


def _save_checkpoint(
    path: Path,
    flow_u: torch.nn.Module,
    flow_d: torch.nn.Module,
    ema_flow_u: torch.nn.Module | None,
    ema_flow_d: torch.nn.Module | None,
    args: ScriptArguments,
    epoch: int,
) -> None:
    torch.save(
        {
            "flow_u": flow_u.state_dict(),
            "flow_d": flow_d.state_dict(),
            "ema_flow_u": (ema_flow_u.state_dict() if ema_flow_u is not None else None),
            "ema_flow_d": (ema_flow_d.state_dict() if ema_flow_d is not None else None),
            "args": args,
            "epoch": epoch,
        },
        path,
    )


def _jsonable_args(args: ScriptArguments) -> dict:
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if not k.startswith("_")
    }


def _init_logging(output_dir: Path, args: ScriptArguments) -> TextIO:
    with open(output_dir / "config.json", "w") as f:
        json.dump(_jsonable_args(args), f, indent=4)

    log_f = open(output_dir / "train.log", "w")
    log_f.write("=== Configuration ===\n")
    for key, value in vars(args).items():
        if not key.startswith("_"):
            log_f.write(f"{key}: {value}\n")
    log_f.write("=====================\n\n")
    log_f.flush()
    return log_f


def _log_line(log_f: TextIO, msg: str) -> None:
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()


def _parse_channel_mult(channel_mult: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in channel_mult.split(",") if x.strip())


def _build_flow_model(
    input_shape: torch.Size,
    num_classes: int,
    args: ScriptArguments,
    channel_mult: tuple[int, ...],
    device: torch.device,
) -> torch.nn.Module:
    return UNetModel(
        input_shape,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        num_classes=num_classes,
        class_cond=args.class_cond,
        dropout=args.dropout,
        channel_mult=channel_mult,
        attention_resolutions=args.attention_resolutions,
    ).to(device)


def _build_scheduler(args: ScriptArguments, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    if not args.use_lr_scheduler:
        return None

    total_steps = args.n_epochs * steps_per_epoch
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if args.lr_decay == "linear":
            return max(0.0, 1.0 - progress)
        if args.lr_decay == "cosine":
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if args.lr_decay == "polynomial":
            lr = args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * max(0.0, 1.0 - progress)
            return lr / args.learning_rate
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)


def _run_periodic_fid_eval(
    args: ScriptArguments,
    eval_flow_u: torch.nn.Module,
    eval_flow_d: torch.nn.Module,
    input_shape: torch.Size,
    num_classes: int,
    device: torch.device,
    inception_feature_model: torch.nn.Module,
    real_features_for_fid: torch.Tensor,
) -> float:
    use_d = args.target_type != "cfm_linear"
    solver = ODESolver(ModelWrapper(ForwardBridgeWrapper(eval_flow_u, eval_flow_d, use_d=use_d)))
    fake_images = generate_fake_images(
        solver=solver,
        input_shape=input_shape,
        num_classes=num_classes,
        num_samples=args.fid_num_samples,
        batch_size=args.fid_batch_size,
        num_inference_steps=args.fid_num_inference_steps,
        method=args.fid_method,
        class_cond=args.class_cond,
        atol=args.fid_atol,
        rtol=args.fid_rtol,
        device=device,
    )
    fake_features = extract_inception_features(inception_feature_model, fake_images.to(device)).detach().cpu()
    return compute_fid(real_features_for_fid, fake_features)


def _run_final_norm_eval(
    dataset,
    batch_size: int,
    device: torch.device,
    bm_targets,
    flow_u: torch.nn.Module,
    flow_d: torch.nn.Module,
    class_cond: bool,
) -> tuple[float, float, float]:
    with torch.no_grad():
        x_1, y = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)))
        x_1, y = x_1.to(device), y.to(device)
        cond_y = y if class_cond else None
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.size(0), 1, device=device, dtype=x_1.dtype)
        x_t, _, _ = bm_targets.compute(x_0=x_0, x_1=x_1, t=t)
        u_pred = flow_u(t=t.view(-1), x=x_t, y=cond_y)
        d_pred = flow_d(t=t.view(-1), x=x_t, y=cond_y)
        u_norm = u_pred.flatten(1).norm(dim=1).mean().item()
        d_norm = d_pred.flatten(1).norm(dim=1).mean().item()
        ratio = d_norm / (u_norm + 1e-8)
    return u_norm, d_norm, ratio


def get_run_dir(args: ScriptArguments) -> Path:
    """Return the per-run training output directory."""
    if hasattr(args, "_run_dir") and args._run_dir is not None:
        return args._run_dir

    base_dir = args.output_dir / "bm" / args.dataset / args.target_type
    base_dir.mkdir(parents=True, exist_ok=True)

    target_short = ''.join([word[0] for word in args.target_type.split('_')])
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{target_short}_{run_time}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args._run_dir = run_dir
    return run_dir


def train(args: ScriptArguments):
    """Train the flow matching model on the given dataset."""
    if args.resume_from_ckpt is not None:
        args.resume_from_ckpt = Path(args.resume_from_ckpt)
        args._run_dir = args.resume_from_ckpt.parent

    output_dir = get_run_dir(args)
    log_f = _init_logging(output_dir, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    _log_line(log_f, f"Using device: {device}")
    _log_line(log_f, f"Dataset: {args.dataset}")
    _log_line(log_f, f"Target type: {args.target_type}")
    _log_line(log_f, f"Posterior temperature: {args.posterior_temperature}")
    _log_line(log_f, f"Posterior top-k: {args.posterior_topk if args.posterior_topk > 0 else None}")

    # Load the dataset
    dataset = get_image_dataset(
        args.dataset,
        train=True,
        transform=get_train_transform(horizontal_flip=args.horizontal_flip),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    _log_line(log_f, f"Loaded {args.dataset} dataset with {len(dataset):,} samples")

    num_classes = len(dataset.classes)
    input_shape = dataset[0][0].size()
    _log_line(log_f, f"{input_shape=}, {num_classes=}")
    eval_dataset = None
    real_features_for_fid = None
    inception_feature_model = None
    if args.eval_fid_on_ckpt:
        eval_dataset = get_image_dataset(
            args.dataset,
            train=False,
            transform=get_test_transform(),
        )
        real_images_for_fid = collect_real_images(eval_dataset, args.fid_metrics_subset_size)
        inception_feature_model = build_inception_feature_model(device)
        real_features_for_fid = extract_inception_features(
            inception_feature_model,
            real_images_for_fid.to(device),
        ).detach().cpu()
        del real_images_for_fid

    channel_mult = _parse_channel_mult(args.channel_mult)

    # Build trainable flows.
    flow_u = _build_flow_model(input_shape, num_classes, args, channel_mult, device)
    flow_d = _build_flow_model(input_shape, num_classes, args, channel_mult, device)
    train_d_branch = args.target_type != "cfm_linear"
    if not train_d_branch:
        # Paper-style CFM linear trains only one vector field (u / v).
        for p in flow_d.parameters():
            p.requires_grad_(False)
    ema_flow_u = None
    ema_flow_d = None
    if args.use_ema:
        ema_flow_u = _build_flow_model(input_shape, num_classes, args, channel_mult, device)
        ema_flow_d = _build_flow_model(input_shape, num_classes, args, channel_mult, device)
        ema_flow_u.load_state_dict(flow_u.state_dict())
        ema_flow_d.load_state_dict(flow_d.state_dict())
        ema_flow_u.eval()
        ema_flow_d.eval()
        for p in ema_flow_u.parameters():
            p.requires_grad_(False)
        for p in ema_flow_d.parameters():
            p.requires_grad_(False)

    posterior_topk = args.posterior_topk if args.posterior_topk > 0 else None

    bm_targets = make_bridge_matching_targets(
        dataset=None,
        beta_value=args.beta_value,
        sigma_floor=args.sigma_floor,
        t_eps=args.t_eps,
        lambda_d=args.lambda_d,
        target_type=args.target_type,
        cfm_beta_min=args.cfm_beta_min,
        cfm_beta_max=args.cfm_beta_max,
        posterior_temperature=args.posterior_temperature,
        posterior_topk=posterior_topk,
    )

    trainable_params = list(flow_u.parameters())
    if train_d_branch:
        trainable_params += list(flow_d.parameters())

    # Load the optimizer
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and args.mixed_precision))
    _log_line(log_f, f"GradScaler enabled: {scaler._enabled}")
    _log_line(
        log_f,
        "Optimizer config: "
        + str(
            {
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "grad_clip_norm": args.grad_clip_norm,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
            }
        ),
    )
    model_size_summary(flow_u)
    if train_d_branch:
        model_size_summary(flow_d)

    scheduler = _build_scheduler(args, optimizer, len(dataloader))

    start_epoch = 0
    if args.resume_from_ckpt is not None:
        ckpt = torch.load(args.resume_from_ckpt, map_location=device, weights_only=False)
        flow_u.load_state_dict(ckpt["flow_u"])
        flow_d.load_state_dict(ckpt["flow_d"])
        if args.use_ema and ema_flow_u is not None and ema_flow_d is not None:
            if ckpt.get("ema_flow_u") is not None:
                ema_flow_u.load_state_dict(ckpt["ema_flow_u"])
            if ckpt.get("ema_flow_d") is not None:
                ema_flow_d.load_state_dict(ckpt["ema_flow_d"])
        start_epoch = int(ckpt.get("epoch", 0))
        _log_line(
            log_f,
            f"[resume] loaded checkpoint: {args.resume_from_ckpt} (epoch={start_epoch})",
        )

    losses: list[float] = []
    for epoch in range(start_epoch, args.n_epochs):
        flow_u.train()
        flow_d.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1:2d}/{args.n_epochs}", dynamic_ncols=True)

        for x_1, y in pbar:

            # print("x_1 shape:", x_1.shape, "dtype:", x_1.dtype, "y shape:", y.shape)
            # print("x_1 min/max:", x_1.min().item(), x_1.max().item())
            # break

            x_1, y = x_1.to(device), y.to(device)
            cond_y = y if args.class_cond else None

            x_0 = torch.randn_like(x_1)
            t = args.t_eps + (1.0 - 2.0 * args.t_eps) * torch.rand(
                x_1.size(0), device=device, dtype=x_1.dtype
            )
            t_bm = t.view(-1, 1)
            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if (device.type == "cuda" and args.mixed_precision)
                else nullcontext()
            )
            with autocast_ctx:
                x_t, u_t, d_t = bm_targets.compute(x_0=x_0, x_1=x_1, t=t_bm)

                u_pred = flow_u(t=t, x=x_t, y=cond_y)
                if args.target_type == "cfm_linear":
                    # Paper-style CFM linear: train only one vector field target.
                    d_pred = torch.zeros_like(u_pred)
                    v_pred = u_pred
                    v_t = u_t
                    loss_u = F.mse_loss(u_pred, u_t)
                    loss_d = torch.zeros_like(loss_u)
                    loss_v = torch.zeros_like(loss_u)
                    loss = loss_u
                else:
                    d_pred = flow_d(t=t, x=x_t, y=cond_y)
                    v_pred = u_pred + d_pred
                    v_t = u_t + d_t

                    loss_u = F.mse_loss(u_pred, u_t)
                    loss_d = F.mse_loss(d_pred, d_t)
                    loss_v = F.mse_loss(v_pred, v_t)
                    loss = loss_u + args.lambda_d * loss_d + args.lambda_forward_align * loss_v

            # Gradient scaling and backprop
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.use_ema:
                _update_ema(ema_flow_u, flow_u, args.ema_decay)
                _update_ema(ema_flow_d, flow_d, args.ema_decay)

            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            losses.append(loss_item)
            pbar.set_postfix(
                {
                    "loss": loss_item,
                    "loss_u": loss_u.item(),
                    "loss_d": loss_d.item(),
                    "loss_v": loss_v.item(),
                }
            )
            if (len(losses) % 200) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"| step: {len(losses):6d} | loss: {loss.item():8.4f} | "
                    f"loss_u: {loss_u.item():8.4f} | loss_d: {loss_d.item():8.4f} | loss_v: {loss_v.item():8.4f} | "
                    f"lr: {current_lr:.6f} |"
                )
                tqdm.write(msg)
                log_f.write(msg + "\n")
                log_f.flush()

        epoch_idx = epoch + 1
        if args.ckpt_every_epochs > 0 and (epoch_idx % args.ckpt_every_epochs) == 0:
            eval_flow_u = ema_flow_u if args.use_ema else flow_u
            eval_flow_d = ema_flow_d if args.use_ema else flow_d
            ckpt_path = output_dir / f"ckpt_epoch_{epoch_idx:04d}.pth"
            _save_checkpoint(
                path=ckpt_path,
                flow_u=flow_u,
                flow_d=flow_d,
                ema_flow_u=ema_flow_u,
                ema_flow_d=ema_flow_d,
                args=args,
                epoch=epoch_idx,
            )
            _log_line(log_f, f"[checkpoint] saved: {ckpt_path}")
            if args.eval_fid_on_ckpt:
                eval_flow_u.eval()
                eval_flow_d.eval()
                fid = _run_periodic_fid_eval(
                    args=args,
                    eval_flow_u=eval_flow_u,
                    eval_flow_d=eval_flow_d,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    device=device,
                    inception_feature_model=inception_feature_model,
                    real_features_for_fid=real_features_for_fid,
                )
                fid_msg = (
                    f"[eval] epoch={epoch_idx} | FID={fid:.6f} | "
                    f"num_samples={args.fid_num_samples} | steps={args.fid_num_inference_steps} | method={args.fid_method}"
                )
                _log_line(log_f, fid_msg)

    eval_flow_u = ema_flow_u if args.use_ema else flow_u
    eval_flow_d = ema_flow_d if args.use_ema else flow_d
    eval_flow_u.eval()
    eval_flow_d.eval()
    u_norm, d_norm, ratio = _run_final_norm_eval(
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        bm_targets=bm_targets,
        flow_u=eval_flow_u,
        flow_d=eval_flow_d,
        class_cond=args.class_cond,
    )
    _log_line(log_f, f"mean ||u|| = {u_norm}")
    _log_line(log_f, f"mean ||d|| = {d_norm}")
    _log_line(log_f, f"ratio ||d|| / ||u|| = {ratio}")

    _save_checkpoint(
        path=output_dir / "ckpt.pth",
        flow_u=flow_u,
        flow_d=flow_d,
        ema_flow_u=ema_flow_u,
        ema_flow_d=ema_flow_d,
        args=args,
        epoch=args.n_epochs,
    )
    print(f"Final checkpoint saved to {output_dir / 'ckpt.pth'}")
    visualization.plot_loss_curve(losses=losses, output_path=output_dir / "losses.png")
    log_f.close()




if __name__ == "__main__":
    train(tyro.cli(ScriptArguments))
