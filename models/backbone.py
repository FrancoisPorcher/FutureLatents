"""Utilities for constructing optional backbone encoders."""

from typing import Any, Tuple

import logging
from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor

logger = logging.getLogger(__name__)


def build_backbone(backbone_cfg: Any) -> Tuple[Any, Any]:
    """Construct the optional backbone encoder and preprocessor.

    Returns a tuple ``(encoder, preprocessor)``. When no backbone is
    configured (e.g., missing or empty configuration), both are ``None``.
    """
    if backbone_cfg is None:
        logger.info("No backbone configuration provided; using cached latents only")
        return None, None

    backbone_type = backbone_cfg.BACKBONE_TYPE.lower()
    hf_repo = backbone_cfg.HF_REPO
    image_size = backbone_cfg.IMAGE_SIZE if "IMAGE_SIZE" in backbone_cfg else None

    if backbone_type == "vjepa2":
        logger.info("Loading '%s' backbone from %s", backbone_type, hf_repo)
        encoder = AutoModel.from_pretrained(hf_repo)
        preprocessor = AutoVideoProcessor.from_pretrained(
            hf_repo,
            size={"height": image_size, "width": image_size} if image_size else None,
        )
        return encoder, preprocessor
    if backbone_type == "dinov3":
        logger.info("Loading '%s' backbone from %s", backbone_type, hf_repo)
        encoder = AutoModel.from_pretrained(hf_repo)
        preprocessor = AutoImageProcessor.from_pretrained(
            hf_repo,
            size={"height": image_size, "width": image_size} if image_size else None,
        )
        return encoder, preprocessor

    logger.warning("Unknown backbone type '%s'; no encoder will be loaded", backbone_type)
    return None, None
