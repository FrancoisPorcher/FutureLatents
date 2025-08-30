"""Dataset factory utilities."""

from __future__ import annotations

from typing import Any, Dict
from omegaconf import DictConfig

from .kinetics_400 import Kinetics400
from .kinetics_400_cached import Kinetics400Cached

_DATASETS = {
    "kinetics_400": Kinetics400,
    "kinetics_400_cached": Kinetics400Cached,
}


def build_dataset(config: DictConfig):
    """Build a dataset from a configuration dictionary.

    The configuration must contain a single entry under the ``datasets`` key
    specifying which dataset to construct. The entry name is used to select the
    appropriate dataset class.
    """

    datasets_cfg = config.DATASETS
    if len(datasets_cfg) != 1:
        raise ValueError("Config must contain exactly one dataset specification")

    name = next(iter(datasets_cfg)).lower()
    if name not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")

    dataset_cls = _DATASETS[name]
    return dataset_cls(config)
