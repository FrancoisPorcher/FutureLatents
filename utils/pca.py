from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type {type(x)}; expected np.ndarray or torch.Tensor")


def _reshape_to_video_tensor(arr_2d: np.ndarray, T: int, H: int, W: int) -> torch.Tensor:
    """(N,3) -> torch.Tensor[T,3,H,W] using row-major order.

    Assumes N == T*H*W.
    """
    N, D = arr_2d.shape
    if D != 3:
        raise ValueError(f"Expected 3 components, got {D}")
    if N != T * H * W:
        raise ValueError(f"N={N} does not match T*H*W={T*H*W}")
    arr = arr_2d.reshape(T, H, W, 3).transpose(0, 3, 1, 2)  # (T,3,H,W)
    return torch.from_numpy(arr).float()


def pca_latents_to_video_tensors(
    context_latents: np.ndarray | torch.Tensor,
    target_latents: np.ndarray | torch.Tensor,
    prediction_latents: np.ndarray | torch.Tensor,
    *,
    T: int,
    H: int,
    W: int,
    n_components: int = 3,
    fit_on: Literal["context", "target", "prediction"] = "context",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project latents to 3D with PCA and reshape to video tensors.

    Returns three tensors shaped (T, 3, H, W) corresponding to context, target
    and prediction latents after PCA projection (fitted on ``fit_on``).
    Inputs can be numpy arrays or torch tensors shaped (N, D).
    """
    c = _to_numpy(context_latents)
    t = _to_numpy(target_latents)
    p = _to_numpy(prediction_latents)

    if c.ndim != 2 or t.ndim != 2 or p.ndim != 2:
        raise ValueError("All inputs must be 2D arrays shaped (N, D)")

    if n_components != 3:
        raise ValueError("n_components must be 3 for video RGB mapping")

    ref = {"context": c, "target": t, "prediction": p}[fit_on]
    pca = PCA(n_components=n_components)
    pca.fit(ref)

    c_p = pca.transform(c)
    t_p = pca.transform(t)
    p_p = pca.transform(p)

    c_vid = _reshape_to_video_tensor(c_p, T, H, W)
    t_vid = _reshape_to_video_tensor(t_p, T, H, W)
    p_vid = _reshape_to_video_tensor(p_p, T, H, W)
    return c_vid, t_vid, p_vid

