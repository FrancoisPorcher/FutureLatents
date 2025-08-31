from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _select_backend() -> Optional[SDPBackend]:
    """Select an attention backend in order of preference.

    The function checks which scaled dot-product attention (SDPA) kernels are
    available in the current environment and returns the most suitable backend
    following the priority: Flash Attention > XFormers > math.  If none of these
    are available ``None`` is returned and the regular PyTorch implementation is
    used.  The chosen backend is printed for transparency.
    """

    if torch.backends.cuda.flash_sdp_enabled():
        print("Using Flash Attention backend")
        return SDPBackend.FLASH_ATTENTION

    try:  # check for xformers
        import xformers.ops  # noqa: F401
    except Exception:
        pass
    else:
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            print("Using xformers backend")
            return SDPBackend.EFFICIENT_ATTENTION

    if torch.backends.cuda.math_sdp_enabled():
        print("Using math attention backend")
        return SDPBackend.MATH

    print("No specialized attention backend available; using regular PyTorch")
    return None


@contextmanager
def sdpa_auto_backend() -> Iterable[None]:
    """Context manager selecting and reporting the best attention backend."""
    backend = _select_backend()
    if backend is not None:
        with sdpa_kernel(backend):
            yield
    else:
        yield
