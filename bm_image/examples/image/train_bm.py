"""Bridge Matching training entry point.

Mirrors `train.py` but trains two networks (`flow_u`, `flow_d`) using the
BM targets from the BM repo (`BM/bridge_matching/targets.py`). Reuses
flow_matching utilities (UNetModel, EMA, NativeScaler, distributed mode,
ODESolver, ModelWrapper) without modifying any existing flow_matching code.
"""

import datetime
import bisect
import importlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from PIL import Image

from models.model_configs import instantiate_model
from train_bm_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform
from training.eval_bm_loop import eval_bm_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.train_bm_loop import train_one_epoch_bm

logger = logging.getLogger(__name__)


class ImageNet32BatchDataset(torch.utils.data.Dataset):
    """Read public ImageNet32 batch files without expanding them into images."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.data_batches = []
        self.label_batches = []
        self.cumulative_sizes = []
        total = 0

        for batch_idx in range(1, 11):
            batch_path = self.root / f"train_data_batch_{batch_idx}"
            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            data = batch["data"]
            labels = np.asarray(batch["labels"], dtype=np.int64) - 1
            if data.shape[1] != 3 * 32 * 32:
                raise ValueError(f"Unexpected ImageNet32 shape in {batch_path}: {data.shape}")

            self.data_batches.append(data)
            self.label_batches.append(labels)
            total += data.shape[0]
            self.cumulative_sizes.append(total)

    @classmethod
    def can_read(cls, root):
        root = Path(root)
        return all((root / f"train_data_batch_{idx}").exists() for idx in range(1, 11))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        batch_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_total = 0 if batch_idx == 0 else self.cumulative_sizes[batch_idx - 1]
        local_idx = index - prev_total

        image = self.data_batches[batch_idx][local_idx].reshape(3, 32, 32)
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode="RGB")
        label = int(self.label_batches[batch_idx][local_idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _import_bm_targets(args):
    """Locate BM repo and import `make_bridge_matching_targets`."""
    if args.bm_repo_path:
        bm_root = Path(args.bm_repo_path).resolve()
    else:
        # default: <gen_model>/BM, two levels above flow_matching repo root.
        bm_root = Path(__file__).resolve().parents[3] / "BM"

    if not (bm_root / "bridge_matching" / "targets.py").exists():
        raise FileNotFoundError(
            f"Could not find BM targets at {bm_root}/bridge_matching/targets.py. "
            "Pass --bm_repo_path to the BM repo root."
        )
    if str(bm_root) not in sys.path:
        sys.path.insert(0, str(bm_root))

    targets_module = importlib.import_module("bridge_matching.targets")
    return targets_module.make_bridge_matching_targets


def _save_bm_checkpoint(
    args, epoch, flow_u, flow_d, optimizer, lr_schedule, loss_scaler
):
    if not distributed_mode.is_main_process() or not args.output_dir:
        return
    output_dir = Path(args.output_dir)
    payload = {
        "flow_u": _unwrap_ddp(flow_u).state_dict(),
        "flow_d": _unwrap_ddp(flow_d).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }
    torch.save(payload, output_dir / f"checkpoint-{epoch}.pth")
    torch.save(payload, output_dir / "checkpoint.pth")


def _load_bm_checkpoint(
    args, flow_u, flow_d, optimizer, lr_schedule, loss_scaler
):
    if not args.resume:
        return
    # Checkpoints saved by this script include argparse.Namespace in `args`.
    # PyTorch>=2.6 defaults to weights_only=True, which rejects that payload.
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    _unwrap_ddp(flow_u).load_state_dict(ckpt["flow_u"])
    _unwrap_ddp(flow_d).load_state_dict(ckpt["flow_d"])
    if (
        "optimizer" in ckpt
        and "epoch" in ckpt
        and not getattr(args, "eval_only", False)
        and not getattr(args, "resume_model_only", False)
    ):
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_schedule.load_state_dict(ckpt["lr_schedule"])
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        args.start_epoch = ckpt["epoch"] + 1
    logger.info(f"Resumed BM checkpoint: {args.resume}")


def _unwrap_ddp(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process() and args.output_dir:
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f)

    if args.discrete_flow_matching:
        raise ValueError("BM training does not support --discrete_flow_matching.")

    make_bridge_matching_targets = _import_bm_targets(args)

    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset in ("imagenet", "imagenet32"):
        if ImageNet32BatchDataset.can_read(args.data_path):
            dataset_train = ImageNet32BatchDataset(args.data_path, transform=transform_train)
        else:
            dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform_train
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")
    logger.info(dataset_train)

    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    logger.info("Initializing BM models")
    flow_u = instantiate_model(
        architechture=args.dataset, is_discrete=False, use_ema=args.use_ema
    ).to(device)
    flow_d = instantiate_model(
        architechture=args.dataset, is_discrete=False, use_ema=args.use_ema
    ).to(device)

    train_d_branch = not args.target_type.startswith("cfm_")
    if not train_d_branch:
        for p in flow_d.parameters():
            p.requires_grad_(False)
        flow_d.eval()

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        flow_u = torch.nn.parallel.DistributedDataParallel(
            flow_u, device_ids=[args.gpu], find_unused_parameters=True
        )
        if train_d_branch:
            flow_d = torch.nn.parallel.DistributedDataParallel(
                flow_d, device_ids=[args.gpu], find_unused_parameters=True
            )

    posterior_topk = args.posterior_topk if args.posterior_topk > 0 else None
    bm_targets = make_bridge_matching_targets(
        target_type=args.target_type,
        beta_value=args.beta_value,
        sigma_floor=args.sigma_floor,
        posterior_temperature=args.posterior_temperature,
        posterior_topk=posterior_topk,
        cfm_beta_min=args.cfm_beta_min,
        cfm_beta_max=args.cfm_beta_max,
    )

    trainable_params = list(_unwrap_ddp(flow_u).parameters())
    if train_d_branch:
        trainable_params += list(_unwrap_ddp(flow_d).parameters())

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=tuple(args.optimizer_betas)
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=args.epochs, start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    loss_scaler = NativeScaler()
    _load_bm_checkpoint(args, flow_u, flow_d, optimizer, lr_schedule, loss_scaler)

    logger.info(f"Start from epoch {args.start_epoch} to {args.epochs}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if not args.eval_only:
            train_stats = train_one_epoch_bm(
                flow_u=flow_u,
                flow_d=flow_d,
                bm_targets=bm_targets,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        else:
            log_stats = {"epoch": epoch}

        save_frequency = (
            args.eval_frequency if args.save_frequency < 0 else args.save_frequency
        )
        should_save = (
            args.output_dir
            and not args.eval_only
            and (
                (save_frequency > 0 and (epoch + 1) % save_frequency == 0)
                or args.test_run
            )
        )
        should_eval = args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        )

        if should_save:
            _save_bm_checkpoint(
                args, epoch, flow_u, flow_d, optimizer, lr_schedule, loss_scaler
            )

        if should_eval:
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)

            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks

            eval_stats = eval_bm_model(
                flow_u=_unwrap_ddp(flow_u),
                flow_d=_unwrap_ddp(flow_d),
                data_loader=data_loader_train,
                device=device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total = time.time() - start_time
    logger.info(f"Training time {datetime.timedelta(seconds=int(total))}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
