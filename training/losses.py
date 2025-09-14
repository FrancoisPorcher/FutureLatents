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

    Supported values: ``"mse"``, ``"l1"``.
    """

    loss_map = {"mse": F.mse_loss, "l1": F.l1_loss}
    criterion = loss_map.get(name.lower())
    if criterion is None:  # pragma: no cover - config error
        raise ValueError("LOSS must be 'mse' or 'l1'")
    return criterion


__all__ = ["get_criterion"]

