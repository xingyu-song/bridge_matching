from typing import Literal

from bridge_matching.datasets.synthetic_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    DatasetGaussian,
    SyntheticDataset,
)

ToyDatasetName = Literal["moons", "mixture", "siggraph", "checkerboard", "invertocat", "gaussian"]

TOY_DATASETS: dict[str, type[SyntheticDataset]] = {
    "gaussian": DatasetGaussian,
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
}
