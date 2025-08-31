from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _available_backends() -> SDPBackend | list[SDPBackend]:
    """Return the optimal attention backend(s) for the current environment.

    Preference order is Flash Attention > Efficient Attention > Math fallback.
    The function inspects the PyTorch CUDA SDP utilities to determine which
    kernels are compiled and enabled.  When neither specialised kernel is
    available the math implementation is used as a safe default.
    """

    # Prefer flash attention when compiled and enabled
    if torch.backends.cuda.flash_sdp_enabled():
        return SDPBackend.FLASH_ATTENTION

    backends: list[SDPBackend] = []
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    backends.append(SDPBackend.MATH)
    return backends


@contextmanager
def sdpa_auto_backend() -> Iterable[None]:
    """Context manager selecting the best attention backend automatically."""
    with sdpa_kernel(_available_backends()):
        yield
