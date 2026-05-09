from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, FashionMNIST
from torchvision.transforms.v2 import Compose, Normalize, RandomHorizontalFlip, ToDtype, ToImage


def get_image_dataset(
    dataset_name: str,
    root: str = Path(__file__).parents[2] / "data",
    train: bool = True,
    transform: Callable | None = None,
) -> Dataset:
    if dataset_name == "mnist":
        return MNIST(root, train, transform, download=True)
    elif dataset_name == "fashion_mnist":
        return FashionMNIST(root, train, transform, download=True)
    elif dataset_name == "cifar10":
        return CIFAR10(root, train, transform, download=True)
    elif dataset_name == "celeba":
        return CelebA(root, train, transform, download=True)  # gdown is required to download
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_transform(horizontal_flip: bool = False, normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if horizontal_flip:
        transform_list.append(RandomHorizontalFlip())
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)


def get_test_transform(normalize: bool = True) -> Callable:
    transform_list = [
        ToImage(),  # convert to torchvision.tv_tensors.Image
        ToDtype(torch.float32, scale=True),  # scale to [0, 1]
    ]
    if normalize:
        transform_list.append(Normalize((0.5,), (0.5,)))  # normalize to [-1, 1]
    return Compose(transform_list)
