"""Utilities for constructing optional backbone encoders."""

from typing import Any, Tuple

import logging
from transformers import AutoModel, AutoVideoProcessor

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

    if backbone_type == "vjepa2":
        hf_repo = backbone_cfg.get("HF_REPO")
        logger.info("Loading '%s' backbone from %s", backbone_type, hf_repo)
        encoder = AutoModel.from_pretrained(hf_repo)
        # AutoVideoProcessor also covers many video/image processors on HF
        preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
        return encoder, preprocessor

    logger.warning("Unknown backbone type '%s'; no encoder will be loaded", backbone_type)
    return None, None
