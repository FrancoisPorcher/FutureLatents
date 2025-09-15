from __future__ import annotations

from omegaconf import DictConfig


def infer_latent_dimensions(cfg: DictConfig) -> tuple[int, int, int]:
    """Return context/target temporal latents and spatial tokens from config.

    The configuration already specifies ``NUM_CONTEXT_LATENTS`` and
    ``NUM_TARGET_LATENTS`` under ``MODEL``. This helper simply reads those
    values and computes the spatial token count per dimension using the
    backbone's image and patch sizes.

    Returns a tuple ``(n_context, n_target, spatial)`` where ``spatial`` is the
    number of spatial tokens along a single dimension (height/width).
    """
    model_cfg = cfg["MODEL"]
    n_context = int(model_cfg["NUM_CONTEXT_LATENTS"])
    n_target = int(model_cfg["NUM_TARGET_LATENTS"])

    backbone_cfg = cfg["BACKBONE"]
    image_size = int(backbone_cfg["IMAGE_SIZE"])
    patch_size = int(backbone_cfg["PATCH_SIZE"])
    spatial = image_size // patch_size

    return n_context, n_target, spatial
