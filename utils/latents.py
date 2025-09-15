from __future__ import annotations

from omegaconf import DictConfig


def infer_latent_dimensions(cfg: DictConfig) -> tuple[int, int, int]:
    """Infer temporal and spatial latent dimensions from configuration.

    Returns a tuple ``(t, h, w)`` where ``t`` is the number of temporal latents
    and ``h``/``w`` are the number of spatial tokens along each dimension.
    """
    train_cfg = cfg["DATASETS"]["TRAIN"]
    dataset_cfg = train_cfg[train_cfg["NAME"].upper()]
    n_frames = dataset_cfg["N_FRAME"]
    t_stride = dataset_cfg["STRIDE"]
    temporal = n_frames // t_stride

    backbone_cfg = cfg["BACKBONE"]
    image_size = backbone_cfg["IMAGE_SIZE"]
    patch_size = backbone_cfg["PATCH_SIZE"]
    spatial = image_size // patch_size

    return temporal, spatial, spatial
