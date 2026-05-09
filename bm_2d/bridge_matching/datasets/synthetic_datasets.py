from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn import datasets


class SyntheticDataset:
    """Base class for synthetic datasets"""

    def __init__(self, dim: int = 2, device: torch.device = "cpu"):
        self.dim = dim
        self.device = device

    def sample(self, n: int, **kwargs) -> torch.Tensor:
        """
        Generate n samples from the dataset.

        Args:
            n (int): Number of samples to generate.
            **kwargs: Additional parameters for sampling (e.g., noise level).

        Returns:
            torch.Tensor: A tensor of shape (n, dim) representing the samples.
        """
        raise NotImplementedError

    def get_square_range(self, samples: torch.Tensor | None = None) -> list[list[float]]:
        """Compute the range of the samples for plotting"""

        assert self.dim == 2, "Only 2D datasets are supported for now"

        if samples is None:
            samples = self.sample(10000)

        if samples.numel() == 0:
            raise ValueError("No samples provided to compute the range.")

        x_min, x_max = samples[:, 0].min().item(), samples[:, 0].max().item()
        y_min, y_max = samples[:, 1].min().item(), samples[:, 1].max().item()
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        range_max = max(x_max - x_min, y_max - y_min) / 2
        offset = range_max * 0.05
        square_range = [
            [x_center - range_max - offset, x_center + range_max + offset],
            [y_center - range_max - offset, y_center + range_max + offset],
        ]
        return square_range


class DatasetMoons(SyntheticDataset):
    """Two half-moons"""

    def sample(self, n: int, noise: float = 0.05) -> torch.Tensor:
        moons = datasets.make_moons(n_samples=n, noise=noise)[0].astype(np.float32)
        return torch.from_numpy(moons).to(self.device)


class DatasetMixture(SyntheticDataset):
    """4 mixture of gaussians"""

    def sample(self, n: int) -> torch.Tensor:
        assert n % 4 == 0
        r = np.r_[
            np.random.randn(n // 4, 2) * 0.5 + np.array([0, -2]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([0, 0]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([2, 2]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([-2, 2]),
        ]
        return torch.from_numpy(r.astype(np.float32)).to(self.device)


class DatasetSiggraph(SyntheticDataset):
    """
    Created by Eric from https://blog.evjang.com/2018/01/nf2.html
    Source: https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(Path(__file__).parents[2] / "data" / "siggraph.pkl", "rb") as f:
            XY = np.array(pickle.load(f), dtype=np.float32)
            XY -= np.mean(XY, axis=0)  # center
        self.XY = torch.from_numpy(XY).to(self.device)

    def sample(self, n: int) -> torch.Tensor:
        X = self.XY[np.random.randint(self.XY.shape[0], size=n)]
        return X


class DatasetCheckerboard(SyntheticDataset):
    """Checkerboard"""

    def sample(self, n: int) -> torch.Tensor:
        x1 = torch.rand(n) * 4 - 2
        x2 = torch.rand(n) - torch.randint(high=2, size=(n,)) * 2
        x2 = x2 + (torch.floor(x1) % 2)
        data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
        return data.float().to(self.device)


class DatasetGaussian(SyntheticDataset):
    """Isotropic or diagonal Gaussian distribution"""

    def __init__(
        self,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if mean is None:
            mean = [0.0, 0.0]
        if std is None:
            std = [1.0, 1.0]

        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device)

        assert self.mean.shape[0] == self.dim, "Mean dimension mismatch"
        assert self.std.shape[0] == self.dim, "Std dimension mismatch"

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn((n, self.dim), device=self.device) * self.std + self.mean


class DatasetInvertocat(SyntheticDataset):
    """Github Logo, The Invertocat
    Source: https://github.com/rtqichen/ffjord/blob/master/imgs/github.png
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        image = Image.open(Path(__file__).parents[2] / "data" / "invertocat.png").convert("L")
        image = np.array(image)
        h, w = image.shape
        x = np.linspace(-4, 4, w)
        y = np.linspace(4, -4, h)
        xs, ys = np.meshgrid(x, y)
        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)
        self.means = np.concatenate([xs, ys], axis=1)
        image = image.max() - image
        self.probs = image.reshape(-1) / image.sum()
        self.std = np.array([8 / w / 2, 8 / h / 2])

    def sample(self, n: int) -> torch.Tensor:
        indices = np.random.choice(len(self.probs), size=n, replace=True, p=self.probs)
        means = self.means[indices]
        return torch.from_numpy(means + np.random.randn(n, 2) * self.std).float().to(self.device)


if __name__ == "__main__":
    # Plot the above datasets

    import argparse
    from pathlib import Path

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    from bridge_matching.datasets import TOY_DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="checkerboard")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ds = TOY_DATASETS[args.dataset]()
    assert ds.dim == 2, "Only 2D datasets are supported"

    print(f"Dataset: {args.dataset}")
    print(f"Sample size: {args.sample_size}")
    print(f"Dataset dim: {ds.dim}")

    samples = ds.sample(args.sample_size)
    square_range = ds.get_square_range(samples)

    fig, ax = plt.subplots(figsize=(4, 4))
    H = ax.hist2d(samples[:, 0], samples[:, 1], bins=300, range=square_range)
    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    norm = cm.colors.Normalize(vmin=cmin, vmax=cmax)
    ax.hist2d(samples[:, 0], samples[:, 1], bins=300, norm=norm, range=square_range)
    ax.set_aspect("equal")
    ax.axis("off")

    path = Path(args.output_dir) / f"{args.dataset}.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved to {path}")
