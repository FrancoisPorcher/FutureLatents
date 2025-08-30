"""Transformer backbone for flow models."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """Transformer block consisting of self-attention and an MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


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

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Apply the transformer to ``x`` conditioned on ``timesteps``."""

        t_emb = timestep_embedding(timesteps, self.hidden_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.in_proj(x)
        x = x + t_emb[:, None, :]
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.norm(x)
        x = self.out_proj(x)
        return x


__all__ = ["DiT"]

