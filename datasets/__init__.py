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
    """
    s = str(split).lower()
    # Minimal: only handle Kinetics-400 and select CSV by split
    if s == "train":
        config.DATASETS.KINETICS_400.PATHS.CSV = (
            config.DATASETS.KINETICS_400.PATHS.TRAIN_CSV
        )
        return Kinetics400(config)
    elif s in ("val", "visualisation"):
        config.DATASETS.KINETICS_400.PATHS.CSV = (
            config.DATASETS.KINETICS_400.PATHS.VAL_CSV
        )
        return Kinetics400(config)
    else:
        # Minimal behaviour: default to validation split
        config.DATASETS.KINETICS_400.PATHS.CSV = (
            config.DATASETS.KINETICS_400.PATHS.VAL_CSV
        )
        return Kinetics400(config)
