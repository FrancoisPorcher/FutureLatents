"""Dataset factory utilities."""

from __future__ import annotations

from omegaconf import DictConfig

from .kinetics_400 import Kinetics400
from .kinetics_400_cached import Kinetics400Cached

_DATASETS = {
    "kinetics_400": Kinetics400,
    "kinetics_400_cached": Kinetics400Cached,
}


def build_dataset(config: DictConfig, split: str = "train"):
    """Build a dataset from a configuration dictionary.

    Parameters
    ----------
    config:
        Configuration object that contains a single dataset entry under the
        ``datasets`` key.
    split:
        Which split of the dataset to build (``"train"`` or ``"val"``).
    """
    # Check if we're in single video overfitting mode
    if config.TRAINER.TRAINING.get("overfit_single_video", False):
        from .kinetics_400 import SingleVideoDataset

        video_path = config.TRAINER.TRAINING.overfit_video_path
        if not video_path:
            raise ValueError(
                "overfit_video_path must be specified when overfit_single_video is True"
            )

        datasets_cfg = config.DATASETS
        if len(datasets_cfg) != 1:
            raise ValueError("Config must contain exactly one dataset specification")
        dataset_cfg = next(iter(datasets_cfg.values()))

        return SingleVideoDataset(
            video_path=video_path,
            n_frame=dataset_cfg.N_FRAME,
            stride=dataset_cfg.STRIDE,
        )

    # Regular dataset building
    datasets_cfg = config.DATASETS
    if len(datasets_cfg) != 1:
        raise ValueError("Config must contain exactly one dataset specification")

    name = next(iter(datasets_cfg))
    dataset_cfg = datasets_cfg[name]
    name_lower = name.lower()

    # Select the appropriate CSV path based on the split
    if split == "train":
        dataset_cfg.PATHS.CSV = dataset_cfg.PATHS.TRAIN_CSV
    elif split == "val":
        dataset_cfg.PATHS.CSV = dataset_cfg.PATHS.VAL_CSV
    else:
        raise ValueError("split must be 'train' or 'val'")

    if name_lower not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {name_lower}")

    dataset_cls = _DATASETS[name_lower]
    return dataset_cls(config)
