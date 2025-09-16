"""Utility helpers for working with PyTorch tensors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from accelerate.state import AcceleratorState
import torch


def _move_batch_to_device(batch: Any) -> Any:
    """Recursively move tensors in ``batch`` to the current accelerator device."""

    device = AcceleratorState().device
    return _move_batch_with_device(batch, device)


def _move_batch_with_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, Mapping):
        return {key: _move_batch_with_device(value, device) for key, value in obj.items()}

    if isinstance(obj, tuple):
        return tuple(_move_batch_with_device(value, device) for value in obj)

    if isinstance(obj, list):
        return [_move_batch_with_device(value, device) for value in obj]

    if isinstance(obj, set):
        return {_move_batch_with_device(value, device) for value in obj}

    return _move_to_device(obj, device)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=True)
    return obj


__all__ = ["_move_batch_to_device"]
