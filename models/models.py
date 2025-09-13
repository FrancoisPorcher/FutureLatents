"""Wrapper models for latent video prediction."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .backbone import build_backbone

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
        # Read family/type directly from config (already defined there)
        self.backbone_type = (
            backbone_cfg.BACKBONE_TYPE.lower() if backbone_cfg else None
        )
        # Assemble encoder and preprocessor via helper
        self.encoder, self.preprocessor = build_backbone(backbone_cfg)

        # --- Shared knobs ---
        self.num_context_latents = config.MODEL.NUM_CONTEXT_LATENTS
        self.embedding_norm = config.TRAINER.TRAINING.EMBEDDING_NORM.lower()

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

    def _forward_video_vjepa2(self, inputs: Dict[str, Any]) -> torch.Tensor:
        if "video" not in inputs:
            raise ValueError("Missing 'video' in inputs for VJEPA2 forward")
        if self.encoder is None or self.preprocessor is None:
            raise ValueError("`video` provided but no encoder is initialised")
        processed = self.preprocessor(inputs["video"], return_tensors="pt")
        video = processed["pixel_values_videos"]  # type: ignore[index]
        # Keep (T,H,W) spatial-temporal structure in the output if supported
        return self.encoder.get_vision_features(video)

    def _forward_video_dinov3(self, inputs: Dict[str, Any]) -> torch.Tensor:  # pragma: no cover
        """Encode a batch of videos using a DINOv3 image backbone.

        DINOv3 operates on individual images, so we flatten the temporal
        dimension and treat all frames across the batch as a single batch of
        images.  The resulting features are reshaped back to the
        ``[B, D, T, H, W]`` layout expected by the rest of the model.
        """

        if "video" not in inputs:
            raise ValueError("Missing 'video' in inputs for DINOv3 forward")
        if self.encoder is None or self.preprocessor is None:
            raise ValueError("`video` provided but no encoder is initialised")

        video = inputs["video"]
        if video.dim() == 4:  # allow [T, C, H, W] without batch dim
            video = video.unsqueeze(0)

        b, t, _, _, _ = video.shape

        # Collapse batch and time to process all frames in one pass
        frames = rearrange(video, "b t c h w -> (b t) c h w")
        processed = self.preprocessor(images=frames, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(video.device)

        with torch.inference_mode():
            outputs = self.encoder(pixel_values)
            hs = outputs.last_hidden_state[:, 5:, :]  # discard global tokens

        spatial_tokens = hs.shape[1]
        side = int(spatial_tokens ** 0.5)
        if side * side != spatial_tokens:  # pragma: no cover - sanity check
            raise ValueError("DINOv3 returned non-square token grid")

        feats = rearrange(
            hs, "(b t) (h w) d -> b d t h w", b=b, t=t, h=side, w=side
        )
        return feats

    def encode_video_with_backbone(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Public wrapper used by models to obtain [B, D, T, H, W] latents.

        Encode raw video or return cached embeddings based on inputs.

        - When ``inputs`` contains ``"video"``, route to the appropriate
          backbone forward according to ``self.backbone_type``.
        - When ``inputs`` contains ``"embedding"``, return it directly (and
          require that no encoder is initialised).
        """
        # Cached embeddings path
        if "embedding" in inputs:
            if self.encoder is not None:
                raise ValueError(
                    "`embedding` provided but encoder is initialised; remove encoder to use cached embeddings"
                )
            return inputs["embedding"]

        # Raw video path
        if "video" in inputs:
            if self.encoder is None:
                raise ValueError("`video` provided but no encoder is initialised")
            if self.backbone_type == "vjepa2":
                return self._forward_video_vjepa2(inputs)
            if self.backbone_type == "dinov3":
                return self._forward_video_dinov3(inputs)
            raise ValueError(
                f"Unknown or unsupported backbone '{self.backbone_type}'. Expected 'vjepa2' or 'dinov3'."
            )

        raise ValueError("`inputs` must contain either 'video' or 'embedding'")

    def split_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(self.num_context_latents)
        if n < 0 or n > latents.shape[2]:
            raise ValueError("`num_context_latents` is out of bounds")
        context = latents[:, :, :n, :, :]
        target  = latents[:, :, n:, :, :]
        return context, target

    # ------------------------------
    # Embedding helpers
    # ------------------------------
    def norm_embeddings(
        self, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply optional normalization to encoder embeddings.

        Returns the normalized latents and a dict with L1/L2 norms of the
        original latents for monitoring purposes.
        """

        norms = {
            "l1": LA.vector_norm(latents, ord=1, dim=1),
            "l2": LA.vector_norm(latents, ord=2, dim=1),
        }

        if self.embedding_norm == "none":
            return latents, norms
        if self.embedding_norm == "l1":
            latents = F.normalize(latents, p=1, dim=1)
            return latents, norms
        if self.embedding_norm == "l2":
            latents = F.normalize(latents, p=2, dim=1)
            return latents, norms
        if self.embedding_norm == "layer":
            mean = latents.mean(dim=1, keepdim=True)
            std = latents.std(dim=1, keepdim=True, unbiased=False)
            latents = (latents - mean) / (std + 1e-5)
            return latents, norms
        raise ValueError(
            f"Unknown embedding norm '{self.embedding_norm}'. Expected one of ['none','l1','l2','layer']"
        )


class FlowMatchingLatentVideoModel(LatentVideoBase):
    """Video generative model with flow matching in latent space."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        dit_cfg = {k.lower(): v for k, v in config.MODEL.DIT.items()}
        self.flow_transformer = DiT(**dit_cfg)

        # Flow-specific knobs
        self.num_train_timesteps = int(config.MODEL.NUM_TRAIN_TIMESTEPS)

    def forward(self, batch: Dict[str, Any], return_norms: bool = False):
        latents = self.encode_video_with_backbone(batch)  # [B, D, T, H, W]
        latents, norms = self.norm_embeddings(latents)
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
        if return_norms:
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
        latents = self.encode_video_with_backbone(batch)  # [B, D, T, H, W]

        latents, norms = self.norm_embeddings(latents)

        context_latents, target_latents = self.split_latents(latents)

        context_latents = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_latents  = rearrange(target_latents,  "b d t h w -> b (t h w) d")

        prediction = self.predictor(context_latents)  # direct next-token prediction
        if return_norms:
            return prediction, target_latents, norms
        return prediction, target_latents

    def trainable_modules(self) -> Iterable[nn.Module]:  # optional: curated list
        yield from [m for m in [self.encoder, self.predictor] if m is not None]


__all__ = ["LatentVideoBase", "FlowMatchingLatentVideoModel", "DeterministicLatentVideoModel"]
