"""Latent video generative model."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from transformers import AutoModel


class LatentVideoModel(nn.Module):
    """Video generative model that operates in latent space.

    Parameters
    ----------
    config: dict
        Configuration dictionary that must include ``"backbone"`` with a
        ``"hf_repo"`` entry specifying the pretrained model to use as
        encoder.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # Backbone encoder producing latent representations
        self.encoder = AutoModel.from_pretrained(config["backbone"]["hf_repo"])
        # Placeholder for future diffusion transformer component
        self.diffusion_transformer = None
