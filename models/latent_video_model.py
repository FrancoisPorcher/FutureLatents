"""Latent video generative model."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import logging

import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor

from models.DiT import DiT

logger = logging.getLogger(__name__)


class LatentVideoModel(nn.Module):
    """Video generative model that operates in latent space.

    Parameters
    ----------
    config: dict
        Configuration dictionary. If ``"backbone"`` with an ``"hf_repo"``
        entry is provided, the corresponding pretrained model is loaded as
        encoder. Otherwise, no encoder is initialised and the model operates
        directly on latent tokens.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        # Backbone encoder producing latent representations (optional)
        backbone_cfg = config.BACKBONE
        hf_repo = getattr(backbone_cfg, "HF_REPO", None)
        if hf_repo:
            self.encoder = AutoModel.from_pretrained(hf_repo)
            # Use the original Hugging Face video preprocessor tied to the encoder
            self.preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
            logger.info(f"Loaded backbone encoder from {hf_repo}")
        else:
            self.encoder = None
            self.preprocessor = None
            logger.info("No backbone encoder, operating directly on latents")
        # Flow matching transformer component
        fm_cfg = config.FLOW_MATCHING
        dit_cfg = getattr(fm_cfg, "DIT", None) or {}
        self.flow_transformer = DiT(**dit_cfg) if dit_cfg else None
        # Configure whether the encoder should be trainable
        trainable = config.ENCODER_TRAINABLE
        self.set_encoder_trainable(trainable)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def set_encoder_trainable(self, trainable: bool) -> None:
        """Enable or disable training for the encoder parameters."""
        if self.encoder is None:
            logger.info("No encoder to (un)freeze")
            return
        for param in self.encoder.parameters():
            param.requires_grad = trainable
        if trainable:
            logger.info("Encoder is trainable")
        else:
            logger.info("Encoder is frozen")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple generator
        """Yield parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    # Backwards compatibility with the example API
    def trainable_modules(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple wrapper
        return self.trainable_parameters()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, latents=None, timesteps=None, video=None):
        """Encode videos or run the flow transformer on latent tokens.

        When ``video`` is provided and a backbone encoder is available, the
        video is first preprocessed and encoded into latent features. In all
        other cases the inputs are assumed to already be latent tokens and are
        passed directly to the flow transformer.
        """
        if video is not None and self.encoder is not None:
            inputs = self.preprocessor(video, return_tensors="pt")[
                "pixel_values_videos"
            ]
            latents = self.encoder.get_vision_features(inputs)
        elif video is not None:
            latents = video

        if self.flow_transformer is None:
            raise RuntimeError("Flow transformer is not initialised")
        return self.flow_transformer(latents, timesteps)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:  # pragma: no cover - simple reporting
        """Return parameter counts for sub-modules and the total.

        Each module's parameter count is printed as ``"<name>: <count>"`` and a
        dictionary with the counts is returned. Counts are formatted in groups of
        three digits separated by spaces for readability.
        """
        counts: Dict[str, int] = {}
        if self.encoder is not None:
            counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        if self.flow_transformer is not None:
            counts["flow_transformer"] = sum(
                p.numel() for p in self.flow_transformer.parameters()
            )
        total = sum(counts.values()) if counts else 0
        counts["total"] = total
        for name, num in counts.items():
            formatted = f"{num:,}".replace(",", " ")
            print(f"{name}: {formatted} parameters")
        return counts
