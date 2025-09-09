"""Wrapper models for latent video prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
from einops import rearrange

from .DiT import DiT, PredictorTransformer

logger = logging.getLogger(__name__)


class LatentVideoModel(nn.Module):
    """Video generative model that operates in latent space."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        backbone_cfg = config.BACKBONE
        hf_repo = backbone_cfg.HF_REPO
        if hf_repo:
            self.encoder = AutoModel.from_pretrained(hf_repo)
            self.preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
            logger.info(f"Loaded backbone encoder from {hf_repo}")
        else:
            self.encoder = None
            self.preprocessor = None
            logger.info("No backbone encoder, operating directly on latents")

        dit_cfg = {k.lower(): v for k, v in config.MODEL.DIT.items()}
        gc = config.TRAINER.TRAINING.GRADIENT_CHECKPOINTING
        dit_cfg["gradient_checkpointing"] = gc
        self.flow_transformer = DiT(**dit_cfg)
        self.gradient_checkpointing = gc

        self.num_context_latents = config.MODEL.NUM_CONTEXT_LATENTS
        self.normalize_embeddings = config.TRAINER.TRAINING.NORMALIZE_EMBEDDINGS

        trainable = config.ENCODER_TRAINABLE
        self.set_encoder_trainable(trainable)

        self.num_train_timesteps = int(config.MODEL.NUM_TRAIN_TIMESTEPS)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def set_encoder_trainable(self, trainable: bool) -> None:
        if self.encoder is None:
            logger.info("No encoder to (un)freeze")
            return
        for param in self.encoder.parameters():
            param.requires_grad = trainable
        logger.info("Encoder is %s", "trainable" if trainable else "frozen")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_modules(self) -> Iterable[nn.Parameter]:  # pragma: no cover
        return self.trainable_parameters()

    def count_parameters(self) -> int:
        """Return the number of trainable parameters and log it."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %.2fM", n_params / 1e6)
        return n_params

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def encode_inputs(self, inputs: Dict[str, Any]):
        if "video" in inputs:
            if self.encoder is None:
                raise ValueError("`video` provided in inputs but no encoder is initialised")
            video = self.preprocessor(inputs["video"], return_tensors="pt")[
                "pixel_values_videos"
            ]
            return self.encoder.get_vision_features(
                video, keep_spatio_temporal_dimension=True
            )
        if "embedding" in inputs:
            if self.encoder is not None:
                raise ValueError(
                    "`embedding` provided in inputs but encoder is initialised; remove the encoder to use cached embeddings"
                )
            return inputs["embedding"]
        raise ValueError("`inputs` must contain either 'video' or 'embedding'")

    def split_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(self.num_context_latents)
        if n < 0 or n > latents.shape[2]:
            raise ValueError("`num_context_latents` is out of bounds")
        context = latents[:, :, :n, :, :]
        target = latents[:, :, n:, :, :]
        return context, target

    def forward(self, batch: Dict[str, Any]):
        latents = self.encode_inputs(batch)  # [B, D, T, H, W]
        if self.normalize_embeddings:
            latents = F.normalize(latents, dim=1)
        context_latents, target_latents = self.split_latents(latents)
        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents = rearrange(target_latents, "b d t h w -> b (t h w) d")

        x0 = torch.randn_like(target_latents)
        t = torch.rand(target_latents.shape[0], device=target_latents.device)
        xt = x0 + (t[:, None, None]) * (target_latents - x0)
        velocity = target_latents - x0
        timesteps = t * self.num_train_timesteps

        if self.flow_transformer is None:
            raise RuntimeError("Flow transformer is not initialised")
        prediction = self.flow_transformer(context_latents, xt, timesteps)
        return prediction, velocity


class DeterministicLatentVideoModel(LatentVideoModel):
    """Predict future latent tokens directly with an L1 objective."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        pred_cfg = config.MODEL.PREDICTOR
        dit_cfg = {k.lower(): v for k, v in pred_cfg.DIT.items()}
        gc = config.TRAINER.TRAINING.GRADIENT_CHECKPOINTING
        dit_cfg["gradient_checkpointing"] = gc
        self.predictor = PredictorTransformer(**dit_cfg) if dit_cfg else None

    def forward(self, batch: Dict[str, Any]):
        latents = self.encode_inputs(batch)
        if self.normalize_embeddings:
            latents = F.normalize(latents, dim=1)
        context_latents, target_latents = self.split_latents(latents)
        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents = rearrange(target_latents, "b d t h w -> b (t h w) d")
        if self.predictor is None:
            raise RuntimeError("Predictor transformer is not initialised")
        prediction = self.predictor(context_latents, target_latents)
        return prediction, target_latents


__all__ = ["LatentVideoModel", "DeterministicLatentVideoModel"]
