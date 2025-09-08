"""Latent video generative model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor

from models.DiT import DiT
from einops import rearrange
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
        fm_cfg = config.MODEL.FLOW_MATCHING
        dit_cfg = getattr(fm_cfg, "DIT", None) or {}
        # Config files may specify DIT parameters using upper-case keys.
        # Normalize keys to match the DiT constructor signature.
        dit_cfg = {k.lower(): v for k, v in dit_cfg.items()}
        gc = config.TRAINER.TRAINING.GRADIENT_CHECKPOINTING
        dit_cfg["gradient_checkpointing"] = gc
        self.flow_transformer = DiT(**dit_cfg) if dit_cfg else None
        self.gradient_checkpointing = gc

        # Number of context latents used during training
        self.num_context_latents = config.MODEL.NUM_CONTEXT_LATENTS

        # Optionally normalise embeddings after extraction
        self.normalize_embeddings = config.TRAINER.TRAINING.NORMALIZE_EMBEDDINGS

        # Store maximum timestep embedding range for continuous flow matching
        self.num_train_timesteps = int(fm_cfg.NUM_TRAIN_TIMESTEPS)

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
    def encode_inputs(self, inputs: Dict[str, Any]):
        """Return latent representations extracted from ``inputs``.

        ``inputs`` must contain either ``"video"`` frames or an ``"embedding``
        tensor.  If raw video is supplied an encoder must be available and the
        frames are preprocessed before feature extraction.  When ``embedding``
        is provided, no encoder may be initialised since the features are
        assumed to be pre-computed.
        """
        if "video" in inputs:
            if self.encoder is None:
                raise ValueError(
                    "`video` provided in inputs but no encoder is initialised"
                )
            video = self.preprocessor(inputs["video"], return_tensors="pt")[
                "pixel_values_videos"
            ]
            return self.encoder.get_vision_features(video, keep_spatio_temporal_dimension=True)
        if "embedding" in inputs:
            if self.encoder is not None:
                raise ValueError(
                    "`embedding` provided in inputs but encoder is initialised; remove the encoder to use cached embeddings"
                )
            return inputs["embedding"]
        raise ValueError("`inputs` must contain either 'video' or 'embedding'")

    def split_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``latents`` into context and target parts."""

        n = int(self.num_context_latents)
        if n < 0 or n > latents.shape[2]:
            raise ValueError("`num_context_latents` is out of bounds")
        context = latents[:, :, :n, :, :]
        target = latents[:, :, n:, :, :]
        return context, target

    def forward(self, batch: Dict[str, Any]):
        """Encode ``batch`` and predict the velocity field.

        This implements the flow matching objective where the model learns the
        constant velocity transporting samples from a base distribution ``x0``
        to data samples ``x1``. A random time ``t`` is drawn uniformly from
        ``[0, 1]`` and the network receives the interpolated state ``xt`` along
        with ``t``. The target is the velocity field ``x1 - x0``.
        """

        latents = self.encode_inputs(batch)  # [B, D, T, H, W]
        if self.normalize_embeddings:
            latents = F.normalize(latents, dim=1)
        context_latents, target_latents = self.split_latents(latents)  # [B, D, n, H, W], [B, D, T-n, H, W]
        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents = rearrange(target_latents, "b d t h w -> b (t h w) d")

        # Sample initial noise x0 and blend with data x1 using a random timestep
        x0 = torch.randn_like(target_latents)
        t = torch.rand(target_latents.shape[0], device=target_latents.device)
        xt = x0 + (t[:, None, None]) * (target_latents - x0)
        velocity = target_latents - x0
        timesteps = t * self.num_train_timesteps

        if self.flow_transformer is None:
            raise RuntimeError("Flow transformer is not initialised")
        prediction = self.flow_transformer(context_latents, xt, timesteps)
        return prediction, velocity

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
