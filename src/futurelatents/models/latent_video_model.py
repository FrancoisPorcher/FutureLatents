"""Latent video generative model."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor


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
        hf_repo = config["backbone"]["hf_repo"]
        self.encoder = AutoModel.from_pretrained(hf_repo)
        # Use the original Hugging Face video preprocessor tied to the encoder
        self.preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
        # Placeholder for future diffusion transformer component
        self.diffusion_transformer = None
        # Configure whether the encoder should be trainable
        trainable = config.get("encoder_trainable", False)
        breakpoint()
        self.set_encoder_trainable(trainable)
        breakpoint()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def set_encoder_trainable(self, trainable: bool) -> None:
        """Enable or disable training for the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def trainable_parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple generator
        """Yield parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    # Backwards compatibility with the example API
    def trainable_modules(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple wrapper
        return self.trainable_parameters()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def encode_video(self, video):  # pragma: no cover - thin wrapper
        """Preprocess and encode a batch of video frames."""
        inputs = self.preprocessor(video, return_tensors="pt")
        return self.encoder(inputs["pixel_values"])
