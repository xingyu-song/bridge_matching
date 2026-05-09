from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.models import Inception_V3_Weights, inception_v3


def matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=eps)
    return (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.T


def compute_mean_and_cov(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = features.mean(dim=0)
    centered = features - mu
    cov = centered.T @ centered / max(features.shape[0] - 1, 1)
    return mu, cov


def compute_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    mu_r, cov_r = compute_mean_and_cov(real_features)
    mu_f, cov_f = compute_mean_and_cov(fake_features)
    mean_diff = ((mu_r - mu_f) ** 2).sum()
    cov_prod_sqrt = matrix_sqrt_psd(cov_r @ cov_f)
    fid = mean_diff + torch.trace(cov_r + cov_f - 2.0 * cov_prod_sqrt)
    return float(fid.item())


def polynomial_mmd(features_x: torch.Tensor, features_y: torch.Tensor) -> torch.Tensor:
    dim = features_x.shape[1]
    k_xx = ((features_x @ features_x.T) / dim + 1.0) ** 3
    k_yy = ((features_y @ features_y.T) / dim + 1.0) ** 3
    k_xy = ((features_x @ features_y.T) / dim + 1.0) ** 3

    n = features_x.shape[0]
    m = features_y.shape[0]
    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / max(n * (n - 1), 1)
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / max(m * (m - 1), 1)
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def compute_kid(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    num_subsets: int = 10,
    subset_size: int = 100,
) -> float:
    max_subset = min(subset_size, real_features.shape[0], fake_features.shape[0])
    if max_subset < 2:
        return float("nan")

    print(
        f"[metrics] KID: computing with {num_subsets} subsets "
        f"(subset_size={max_subset})"
    )
    vals = []
    for _ in tqdm(range(num_subsets), desc="KID subsets", unit="subset"):
        idx_r = torch.randperm(real_features.shape[0], device=real_features.device)[:max_subset]
        idx_f = torch.randperm(fake_features.shape[0], device=fake_features.device)[:max_subset]
        vals.append(polynomial_mmd(real_features[idx_r], fake_features[idx_f]))
    print("[metrics] KID: done")
    return float(torch.stack(vals).mean().item())


def compute_inception_score(probs: torch.Tensor, num_splits: int = 5) -> float:
    num_samples = probs.shape[0]
    if num_samples < num_splits:
        num_splits = max(1, num_samples)

    split_scores = []
    for split in torch.chunk(probs, num_splits, dim=0):
        if split.numel() == 0:
            continue
        p_y = split.mean(dim=0, keepdim=True)
        kl = split * (torch.log(split.clamp_min(1e-8)) - torch.log(p_y.clamp_min(1e-8)))
        split_scores.append(torch.exp(kl.sum(dim=1).mean()))

    if not split_scores:
        return float("nan")
    return float(torch.stack(split_scores).mean().item())


def build_inception_feature_model(device: torch.device) -> torch.nn.Module:
    weights = Inception_V3_Weights.DEFAULT
    feature_model = inception_v3(weights=weights, transform_input=False).to(device)
    feature_model.fc = torch.nn.Identity()
    feature_model.eval()
    return feature_model


def build_inception_models(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    weights = Inception_V3_Weights.DEFAULT

    feature_model = inception_v3(weights=weights, transform_input=False).to(device)
    feature_model.fc = torch.nn.Identity()
    feature_model.eval()

    logit_model = inception_v3(weights=weights, transform_input=False).to(device)
    logit_model.eval()
    return feature_model, logit_model


def _preprocess_images_for_inception(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_inception_features(feature_model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    images = _preprocess_images_for_inception(images)
    with torch.no_grad():
        features = feature_model(images)
    if features.ndim > 2:
        features = torch.flatten(features, 1)
    return features


def extract_inception_features_and_probs(
    feature_model: torch.nn.Module,
    logit_model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = _preprocess_images_for_inception(images)
    with torch.no_grad():
        features = feature_model(images)
        logits = logit_model(images)
        probs = torch.softmax(logits, dim=1)

    if features.ndim > 2:
        features = torch.flatten(features, 1)
    return features, probs


def extract_inception_features_and_probs_batched(
    feature_model: torch.nn.Module,
    logit_model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    print("[metrics] Inception: extracting features/probs (batched)")
    features_batches = []
    probs_batches = []
    n = images.shape[0]
    if n == 0:
        raise ValueError("Expected at least one image for metrics computation.")

    for start in tqdm(range(0, n, batch_size), desc="Inception batches", unit="batch"):
        end = min(start + batch_size, n)
        batch = images[start:end].to(device, non_blocking=True)
        batch_features, batch_probs = extract_inception_features_and_probs(
            feature_model=feature_model,
            logit_model=logit_model,
            images=batch,
        )
        features_batches.append(batch_features.detach().cpu())
        probs_batches.append(batch_probs.detach().cpu())

        del batch, batch_features, batch_probs

    print("[metrics] Inception: extraction done")
    return torch.cat(features_batches, dim=0), torch.cat(probs_batches, dim=0)


def collect_real_images(dataset, limit: int | None, batch_size: int = 256) -> torch.Tensor:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    if n <= 0:
        raise ValueError("Expected a positive number of real images.")
    print(f"[metrics] Collecting real images: n={n}, batch_size={batch_size}")
    subset = dataset if n == len(dataset) else torch.utils.data.Subset(dataset, range(n))
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    xs = []
    for batch_x, _ in tqdm(loader, desc="Real image batches", unit="batch"):
        xs.append(batch_x)
    print("[metrics] Real image collection done")
    return torch.cat(xs, dim=0)


def generate_fake_images(
    solver,
    input_shape: torch.Size,
    num_classes: int,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    method: str,
    class_cond: bool,
    atol: float,
    rtol: float,
    device: torch.device,
) -> torch.Tensor:
    print(
        f"[metrics] Generating fake images: num_samples={num_samples}, "
        f"batch_size={batch_size}, method={method}, steps={num_inference_steps}"
    )
    out = []
    remaining = num_samples
    adaptive_methods = {"dopri5", "dopri8", "bosh3", "adaptive_heun"}
    step_size = None if method in adaptive_methods else (1.0 / num_inference_steps)
    t_eval = torch.linspace(0, 1, 2, device=device)
    pbar = tqdm(total=num_samples, desc="Fake image generation", unit="img")
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        labels = (torch.arange(cur_bs, device=device) % num_classes) if class_cond else None
        x_init = torch.randn((cur_bs, *input_shape), dtype=torch.float32, device=device)
        sol = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=t_eval,
            return_intermediates=True,
            atol=atol,
            rtol=rtol,
            y=labels,
        )
        out.append(sol[-1].detach().cpu())
        remaining -= cur_bs
        pbar.update(cur_bs)
    pbar.close()
    print("[metrics] Fake image generation done")
    return torch.cat(out, dim=0)
