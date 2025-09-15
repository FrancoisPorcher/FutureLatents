"""Dataset factory utilities."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from .kinetics_400 import Kinetics400
from .kinetics_400_cached import Kinetics400Cached
from .syntheticbouncingshapes import (
    SyntheticBouncingShapesVideo,
    SyntheticBouncingShapesImage,
)

_DATASETS = {
    "kinetics_400": Kinetics400,
    "kinetics_400_cached": Kinetics400Cached,
    # Backward-compatible name for the video variant
    "synthetic_bouncing_shapes": SyntheticBouncingShapesVideo,
    # Explicit image variant (single-frame clips)
    "synthetic_bouncing_shapes_images": SyntheticBouncingShapesImage,
}


def build_dataset(config: DictConfig, split: str = "train"):
    if split=='train':
        config_datasets_train = config.DATASETS.TRAIN
        dataset_name = config_datasets_train.NAME
        if dataset_name == 'kinetics_400':
            config_kinetics_400 = config_datasets_train.KINETICS_400
            dataset = Kinetics400(config_kinetics_400)
        elif dataset_name == 'kinetics_400_cached':
            config_kinetics_400_cached = config_datasets_train.KINETICS_400_CACHED
            dataset = Kinetics400Cached(config_kinetics_400_cached)
        elif dataset_name == 'synthetic_bouncing_shapes':
            config_sbs = config_datasets_train.SYNTHETIC_BOUNCING_SHAPES
            dataset = SyntheticBouncingShapesVideo(config_sbs)
        elif dataset_name == 'synthetic_bouncing_shapes_images':
            config_sbs_img = config_datasets_train.SYNTHETIC_BOUNCING_SHAPES_IMAGES
            dataset = SyntheticBouncingShapesImage(config_sbs_img)
        return dataset
    elif split=='val':
        config_datasets_val = config.DATASETS.VAL
        dataset_name = config_datasets_val.NAME
        if dataset_name == 'kinetics_400':
            config_kinetics_400 = config_datasets_val.KINETICS_400
            dataset = Kinetics400(config_kinetics_400)
        elif dataset_name == 'kinetics_400_cached':
            config_kinetics_400_cached = config_datasets_val.KINETICS_400_CACHED
            dataset = Kinetics400Cached(config_kinetics_400_cached)
        elif dataset_name == 'synthetic_bouncing_shapes':
            config_sbs = config_datasets_val.SYNTHETIC_BOUNCING_SHAPES
            dataset = SyntheticBouncingShapesVideo(config_sbs)
        elif dataset_name == 'synthetic_bouncing_shapes_images':
            config_sbs_img = config_datasets_val.SYNTHETIC_BOUNCING_SHAPES_IMAGES
            dataset = SyntheticBouncingShapesImage(config_sbs_img)
        return dataset
    elif split=='visualisation':
        config_datasets_visualisation = config.DATASETS.VISUALISATION
        dataset_name = config_datasets_visualisation.NAME
        if dataset_name == 'synthetic_bouncing_shapes':
            config_synthetic_bouncing_shapes = config_datasets_visualisation.SYNTHETIC_BOUNCING_SHAPES
            dataset = SyntheticBouncingShapesVideo(config_synthetic_bouncing_shapes)
        elif dataset_name == 'synthetic_bouncing_shapes_images':
            config_synthetic_bouncing_shapes_images = config_datasets_visualisation.SYNTHETIC_BOUNCING_SHAPES_IMAGES
            dataset = SyntheticBouncingShapesImage(config_synthetic_bouncing_shapes_images)
        return dataset
    
