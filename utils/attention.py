from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _available_backends() -> list[SDPBackend]:
    """Return attention backends in preference order.

    The function inspects which scaled dot-product attention (SDPA) kernels are
    *enabled* in the current PyTorch environment and returns them ordered by
    preference: Flash Attention > Memory Efficient Attention > math fallback.

    Returning a list rather than a single backend allows PyTorch to gracefully
    fall back to a supported kernel when higher-priority implementations are not
    available at run time (e.g. when the GPU architecture does not support Flash
    Attention).  This avoids ``RuntimeError: No available kernel`` failures on
    older GPUs.
    """

    backends: list[SDPBackend] = []
    if torch.backends.cuda.flash_sdp_enabled():
        backends.append(SDPBackend.FLASH_ATTENTION)
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    backends.append(SDPBackend.MATH)
    return backends


@contextmanager
def sdpa_auto_backend() -> Iterable[None]:
    """Context manager selecting the best attention backend automatically."""
    with sdpa_kernel(_available_backends()):
        yield
