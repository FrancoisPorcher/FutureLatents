"""Loss utilities for training.

This module provides helpers to select and configure loss functions
used by trainers.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn.functional as F


def get_criterion(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a loss function for ``name``.

    Supported values: ``"mse"``, ``"l1"``, ``"bce_logits"``, ``"cross_entropy"``.
    """

    loss_map = {
        "mse": F.mse_loss,
        "l1": F.l1_loss,
        "bce_logits": F.binary_cross_entropy_with_logits,
        "cross_entropy": F.cross_entropy,
    }
    criterion = loss_map.get(name.lower())
    if criterion is None:  # pragma: no cover - config error
        raise ValueError("LOSS must be 'mse', 'l1', 'bce_logits', or 'cross_entropy'")
    return criterion


def compute_locator_step_losses(
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    coord_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    heatmap_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reshape_heatmap_target: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prepare_heatmap_cross_entropy_inputs: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, coordinate, and heatmap losses for locator batches."""

    coord_losses: list[torch.Tensor] = []
    heatmap_losses: list[torch.Tensor] = []

    for prefix in ("square", "circle"):
        coord_pred = outputs[f"{prefix}_pred"]
        coord_target = batch[f"target_{prefix}_position_normalized"].to(coord_pred)
        coord_losses.append(coord_loss_fn(coord_pred, coord_target))

        logits = outputs[f"{prefix}_heatmap_logits"]
        heatmap_target = reshape_heatmap_target(
            batch[f"target_{prefix}_heatmap_patch"].to(logits),
            logits,
        )
        logits_flat, target_idx = prepare_heatmap_cross_entropy_inputs(logits, heatmap_target)
        heatmap_losses.append(heatmap_loss_fn(logits_flat, target_idx))

    coord_loss = torch.stack(coord_losses).sum()
    heatmap_loss = torch.stack(heatmap_losses).sum()
    total_loss = coord_loss + heatmap_loss

    return total_loss, coord_loss, heatmap_loss


__all__ = ["get_criterion", "compute_locator_step_losses"]
