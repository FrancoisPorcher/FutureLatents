"""Loss utilities for training.

This module provides helpers to select and configure loss functions
used by trainers.
"""

from __future__ import annotations

from typing import Callable

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


__all__ = ["get_criterion"]
