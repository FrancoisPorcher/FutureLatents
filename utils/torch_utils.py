"""Utility helpers for working with PyTorch tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import torch

def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Mapping):
        return {key: _move_to_device(value, device) for key, value in batch.items()}

    if isinstance(batch, tuple):
        return tuple(_move_to_device(value, device) for value in batch)

    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]

    if isinstance(batch, set):
        return {_move_to_device(value, device) for value in batch}

    return _move_to_device(batch, device)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=True)
    return obj


__all__ = ["_move_batch_to_device"]
