"""Transformer backbone for flow models."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Iterable, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoVideoProcessor
from einops import rearrange

logger = logging.getLogger(__name__)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) helper."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.sin()[None, None, :, :], emb.cos()[None, None, :, :]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLP(nn.Module):
    """Simple feed-forward network used inside transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """Self-attention using ``scaled_dot_product_attention``.

    The attention backend is selected automatically based on available
    kernels. Flash attention is preferred, followed by the XFormers
    memory-efficient kernel and finally the math implementation as a
    fallback.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        sin, cos = self.rope(N, x.device)
        q, k = self.rope.apply_rotary(q, k, sin, cos)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(x)


class MultiheadCrossAttention(nn.Module):
    """Cross-attention between ``x`` and ``context`` tokens."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            x = scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """Transformer block consisting of self-attention and an MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadSelfAttention(dim, num_heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        residual = x
        x = self.norm1(x)
        x = self.modulate(x, shift_msa, scale_msa)
        x = self.attn(x)
        x = residual + gate_msa[:, None, :] * x
        residual = x
        x = self.norm_cross(x)
        x = self.modulate(x, shift_ca, scale_ca)
        x = self.cross_attn(x, context)
        x = residual + gate_ca[:, None, :] * x
        residual = x
        x = self.norm2(x)
        x = self.modulate(x, shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = residual + gate_mlp[:, None, :] * x
        return x


class FinalLayer(nn.Module):
    """Final DiT layer with adaptive layer norm."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = DiTBlock.modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Minimal DiT backbone operating on latent tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        max_time_embeddings: int = 1000,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_time_embeddings = max_time_embeddings
        self.gradient_checkpointing = gradient_checkpointing

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(input_dim, hidden_dim),
        )
        self.in_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, input_dim)

    def forward(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the transformer to ``target_latents`` conditioned on ``timesteps``.

        The ``context_latents`` are attended to via cross attention while the
        adaptive layer normalisation (AdaLN) is conditioned only on the
        ``timesteps``.
        
        Args:
            context_latents: Latents used for conditioning of shape (B, N_context, D)
            target_latents: Latents to be transformed of shape (B, N_target, D)
            timesteps: Timesteps for each element in the batch of shape (B,)
        """
        context_tokens = self.context_mlp(context_latents)
        t_emb = timestep_embedding(timesteps, self.hidden_dim)
        c = self.time_mlp(t_emb)

        x = self.in_proj(target_latents)
        x = self.in_norm(x)

        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, c, context_tokens, use_reentrant=False)
            else:
                x = block(x, c, context_tokens)
        x = self.final_layer(x, c)
        return x


# -----------------------------------------------------------------------------
# Deterministic predictor components
# -----------------------------------------------------------------------------


class PredictorBlock(nn.Module):
    """Transformer block with self- and cross-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadSelfAttention(dim, num_heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadCrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + self.mlp(self.norm2(x))
        return x


class PredictorFinalLayer(nn.Module):
    """Final projection layer for the predictor transformer."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear(x)


class PredictorTransformer(nn.Module):
    """Minimal transformer operating on latent tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(input_dim, hidden_dim),
        )
        self.in_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [PredictorBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = PredictorFinalLayer(hidden_dim, input_dim)

    def forward(
        self, context_latents: torch.Tensor, target_latents: torch.Tensor
    ) -> torch.Tensor:
        context_tokens = self.context_mlp(context_latents)
        x = self.in_proj(target_latents)
        x = self.in_norm(x)
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x, context_tokens, use_reentrant=False)
            else:
                x = block(x, context_tokens)
        x = self.final_layer(x)
        return x


# -----------------------------------------------------------------------------
# Latent video models
# -----------------------------------------------------------------------------


class LatentVideoModel(nn.Module):
    """Video generative model that operates in latent space."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        backbone_cfg = config.BACKBONE
        hf_repo = getattr(backbone_cfg, "HF_REPO", None)
        if hf_repo:
            self.encoder = AutoModel.from_pretrained(hf_repo)
            self.preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
            logger.info(f"Loaded backbone encoder from {hf_repo}")
        else:
            self.encoder = None
            self.preprocessor = None
            logger.info("No backbone encoder, operating directly on latents")

        fm_cfg = config.MODEL.FLOW_MATCHING
        dit_cfg = getattr(fm_cfg, "DIT", None) or {}
        dit_cfg = {k.lower(): v for k, v in dit_cfg.items()}
        gc = config.TRAINER.TRAINING.GRADIENT_CHECKPOINTING
        dit_cfg["gradient_checkpointing"] = gc
        self.flow_transformer = DiT(**dit_cfg) if dit_cfg else None
        self.gradient_checkpointing = gc

        self.num_context_latents = config.MODEL.NUM_CONTEXT_LATENTS
        self.normalize_embeddings = config.TRAINER.TRAINING.NORMALIZE_EMBEDDINGS

        trainable = config.ENCODER_TRAINABLE
        self.set_encoder_trainable(trainable)

        self.num_train_timesteps = int(fm_cfg.NUM_TRAIN_TIMESTEPS)

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
        # ``LatentVideoModel`` expects a flow matching configuration.  When
        # working with deterministic prediction we provide a minimal stub so we
        # can reuse the parent initialisation logic.
        fm_cfg = getattr(config.MODEL, "FLOW_MATCHING", None)
        if fm_cfg is None:
            config.MODEL.FLOW_MATCHING = SimpleNamespace(
                NUM_TRAIN_TIMESTEPS=1, DIT=None
            )
        elif getattr(fm_cfg, "NUM_TRAIN_TIMESTEPS", None) is None:
            fm_cfg.NUM_TRAIN_TIMESTEPS = 1

        super().__init__(config)

        pred_cfg = config.MODEL.PREDICTOR
        dit_cfg = getattr(pred_cfg, "DIT", None) or {}
        dit_cfg = {k.lower(): v for k, v in dit_cfg.items()}
        gc = config.TRAINER.TRAINING.GRADIENT_CHECKPOINTING
        dit_cfg["gradient_checkpointing"] = gc
        self.predictor = PredictorTransformer(**dit_cfg) if dit_cfg else None

        # The flow transformer and training timesteps are not used for
        # deterministic prediction.
        self.flow_transformer = None
        self.num_train_timesteps = None

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


__all__ = [
    "DiT",
    "PredictorTransformer",
    "LatentVideoModel",
    "DeterministicLatentVideoModel",
]
