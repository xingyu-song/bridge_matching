import pytest
import torch

from bridge_matching.datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_size():
    return 1000


@pytest.mark.parametrize(
    "dataset_class",
    [
        (DatasetMoons),
        (DatasetMixture),
        (DatasetCheckerboard),
        (DatasetSiggraph),
        (DatasetInvertocat),
    ],
    ids=["moons", "mixture", "checkerboard", "siggraph", "invertocat"],
)
def test_dataset(dataset_class, device, sample_size):
    dataset = dataset_class(device=device)
    samples = dataset.sample(sample_size)

    assert samples.shape == (sample_size, dataset.dim)
    assert isinstance(samples, torch.Tensor)
    assert samples.device == device

    square_range = dataset.get_square_range(samples)

    assert len(square_range) == 2
    assert len(square_range[0]) == 2
    assert len(square_range[1]) == 2
    assert square_range[0][0] < square_range[0][1]
    assert square_range[1][0] < square_range[1][1]


def test_toy_datasets_registry(device):
    from bridge_matching.datasets import TOY_DATASETS

    for name, dataset_cls in TOY_DATASETS.items():
        dataset = dataset_cls(device=device)
        samples = dataset.sample(40)

        assert samples.shape == (40, 2)
        assert isinstance(samples, torch.Tensor)
