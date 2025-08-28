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
        self.set_encoder_trainable(trainable)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def set_encoder_trainable(self, trainable: bool) -> None:
        """Enable or disable training for the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = trainable
        print("Froze encoder")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple generator
        """Yield parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    # Backwards compatibility with the example API
    def trainable_modules(self) -> Iterable[nn.Parameter]:  # pragma: no cover - simple wrapper
        return self.trainable_parameters()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def encode_video_with_backbone(self, video):  # pragma: no cover - thin wrapper
        """Preprocess and encode a batch of video frames."""
        inputs = self.preprocessor(video, return_tensors="pt")["pixel_values_videos"]
        bacbone_video_features = self.encoder.get_vision_features(inputs)  # [B, N_tokens, embed_dim] = [B, 8192, 1024]
        return bacbone_video_features

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:  # pragma: no cover - simple reporting
        """Return parameter counts for sub-modules and the total.

        Each module's parameter count is printed as ``"<name>: <count>"`` and a
        dictionary with the counts is returned.
        """
        counts: Dict[str, int] = {}
        counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        if self.diffusion_transformer is not None:
            counts["diffusion_transformer"] = sum(
                p.numel() for p in self.diffusion_transformer.parameters()
            )
        total = sum(counts.values())
        counts["total"] = total
        for name, num in counts.items():
            print(f"{name}: {num}")
        return counts
