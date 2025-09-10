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

# to compute norms, temp
from torch import linalg as LA


class LatentVideoBase(nn.Module):
    """Lightweight base: only shared utilities + (optional) backbone. No algorithm-specific parts."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        # --- Optional backbone ---
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

        # --- Shared knobs ---
        self.num_context_latents = config.MODEL.NUM_CONTEXT_LATENTS
        self.normalize_embeddings = config.TRAINER.TRAINING.NORMALIZE_EMBEDDINGS

        # --- Encoder freeze policy ---
        self.set_encoder_trainable(config.ENCODER_TRAINABLE)

    # ------------------------------
    # Training / utility helpers
    # ------------------------------
    def set_encoder_trainable(self, trainable: bool) -> None:
        if self.encoder is None:
            logger.info("No encoder to (un)freeze")
            return
        for p in self.encoder.parameters():
            p.requires_grad = trainable
        logger.info("Encoder is %s", "trainable" if trainable else "frozen")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:  # pragma: no cover
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_modules(self) -> Iterable[nn.Module]:  # pragma: no cover
        # By default, everything with trainable params.
        # Subclasses can override to expose a curated list.
        yield from {m for m in self.modules() if any(p.requires_grad for p in m.parameters(recurse=False))}

    def count_parameters(self) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %.2fM", n / 1e6)
        return n

    # ------------------------------
    # Forward helpers
    # ------------------------------
    def encode_inputs(self, inputs: Dict[str, Any]):
        if "video" in inputs:
            if self.encoder is None:
                raise ValueError("`video` provided but no encoder is initialised")
            video = self.preprocessor(inputs["video"], return_tensors="pt")["pixel_values_videos"]
            return self.encoder.get_vision_features(video, keep_spatio_temporal_dimension=True)

        if "embedding" in inputs:
            if self.encoder is not None:
                raise ValueError("`embedding` provided but encoder is initialised; remove encoder to use cached embeddings")
            return inputs["embedding"]

        raise ValueError("`inputs` must contain either 'video' or 'embedding'")

    def split_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(self.num_context_latents)
        if n < 0 or n > latents.shape[2]:
            raise ValueError("`num_context_latents` is out of bounds")
        context = latents[:, :, :n, :, :]
        target  = latents[:, :, n:, :, :]
        return context, target


class FlowMatchingLatentVideoModel(LatentVideoBase):
    """Video generative model with flow matching in latent space."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        dit_cfg = {k.lower(): v for k, v in config.MODEL.DIT.items()}
        self.flow_transformer = DiT(**dit_cfg)

        # Flow-specific knobs
        self.num_train_timesteps = int(config.MODEL.NUM_TRAIN_TIMESTEPS)

    def forward(self, batch: Dict[str, Any], return_norms: bool = False):
        latents = self.encode_inputs(batch)  # [B, D, T, H, W]
        norms = None
        if self.normalize_embeddings:
            norm_per_token_l1 = LA.vector_norm(latents, ord=1, dim=1)
            norm_per_token_l2 = LA.vector_norm(latents, ord=2, dim=1)
            norms = {"l1": norm_per_token_l1, "l2": norm_per_token_l2}
            latents = F.normalize(latents, dim=1)
        context_latents, target_latents = self.split_latents(latents)

        # Flatten tokens: [B, (T*H*W), D]
        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents  = rearrange(target_latents,  "b d t h w -> b (t h w) d")

        # Flow matching noise / schedule
        x0 = torch.randn_like(target_latents)                # initial noise
        t  = torch.rand(target_latents.shape[0], device=target_latents.device)  # [B]
        xt = x0 + (t[:, None, None]) * (target_latents - x0)  # linear interp
        velocity = target_latents - x0
        timesteps = t * self.num_train_timesteps

        prediction = self.flow_transformer(context_latents, xt, timesteps)  # predict velocity
        if return_norms and norms is not None:
            return prediction, velocity, norms
        return prediction, velocity

    def trainable_modules(self) -> Iterable[nn.Module]:  # optional: curated list
        yield from [m for m in [self.encoder, self.flow_transformer] if m is not None]


class DeterministicLatentVideoModel(LatentVideoBase):
    """Predict future latent tokens directly (no time steps, no flow)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        dit_cfg = {k.lower(): v for k, v in config.MODEL.DIT.items()}
        self.predictor = PredictorTransformer(**dit_cfg)

    def forward(self, batch: Dict[str, Any], return_norms: bool = False):
        latents = self.encode_inputs(batch)  # [B, D, T, H, W]

        norms = None
        if self.normalize_embeddings:
            norm_per_token_l1 = LA.vector_norm(latents, ord=1, dim=1)
            norm_per_token_l2 = LA.vector_norm(latents, ord=2, dim=1)
            norms = {"l1": norm_per_token_l1, "l2": norm_per_token_l2}
            latents = F.normalize(latents, dim=1)

        context_latents, target_latents = self.split_latents(latents)

        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents  = rearrange(target_latents,  "b d t h w -> b (t h w) d")

        prediction = self.predictor(context_latents)  # direct next-token prediction
        if return_norms and norms is not None:
            return prediction, target_latents, norms
        return prediction, target_latents

    def trainable_modules(self) -> Iterable[nn.Module]:  # optional: curated list
        yield from [m for m in [self.encoder, self.predictor] if m is not None]


__all__ = ["LatentVideoBase", "FlowMatchingLatentVideoModel", "DeterministicLatentVideoModel"]
