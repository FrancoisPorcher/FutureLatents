"""Dataset factory utilities."""

from __future__ import annotations

from omegaconf import DictConfig

from .kinetics_400 import Kinetics400
from .kinetics_400_cached import Kinetics400Cached
from .SyntheticBouncingShapes import SyntheticBouncingShapes

_DATASETS = {
    "kinetics_400": Kinetics400,
    "kinetics_400_cached": Kinetics400Cached,
    "synthetic_bouncing_shapes": SyntheticBouncingShapes,
}


def build_dataset(config: DictConfig, split: str = "train"):
    """Build a dataset according to the requested split.

    Only handles the ``kinetics_400`` dataset and swaps the CSV path based on
    the requested ``split``.
    """

    dataset_cfg = config.DATASETS.KINETICS_400
    if str(split).lower() == "train":
        dataset_cfg.PATHS.CSV = dataset_cfg.PATHS.TRAIN_CSV
    else:
        dataset_cfg.PATHS.CSV = dataset_cfg.PATHS.VAL_CSV
    return _DATASETS["kinetics_400"](config)
