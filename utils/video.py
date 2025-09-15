from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import matplotlib.animation as animation


def save_tensor(
    tensor: torch.Tensor,
    out_path: Path | str,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save a tensor to ``.pt`` and log the event.

    - Accepts tensors shaped ``[T,C,H,W]`` or ``[B,T,C,H,W]`` (or any tensor).
    - Detaches and moves to CPU before saving for portability.
    """
    path = Path(out_path)
    t = tensor.detach().cpu()
    torch.save(t, path)
    if logger is not None:
        logger.info("Saved tensor -> %s [shape=%s]", str(path), str(tuple(t.shape)))
    return path


def save_visualisation_tensors(
    video: torch.Tensor,
    context_latents: torch.Tensor,
    target_latents: torch.Tensor,
    prediction_latents: torch.Tensor,
    out_dir: Path | str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save visualisation tensors into ``out_dir``.

    This convenience wrapper stores the input ``video`` and the three latent
    tensors used during visualisation (``context_latents``, ``target_latents``
    and ``prediction_latents``) using :func:`save_tensor`.
    """
    directory = Path(out_dir)
    save_tensor(video, directory / "video.pt", logger=logger)
    save_tensor(context_latents, directory / "context_latents.pt", logger=logger)
    save_tensor(target_latents, directory / "target_latents.pt", logger=logger)
    save_tensor(prediction_latents, directory / "prediction_latents.pt", logger=logger)


def convert_video_tensor_to_mp4(
    video_tensor: torch.Tensor,
) -> Tuple[List[np.ndarray], int]:
    """Convert a video tensor to a list of frame arrays suitable for MP4 writing.

    Returns (frames_per_sample, fps). ``frames_per_sample`` is a list where each
    item is a numpy array shaped ``[T, H, W, C]``. A default preview fps of 10
    is returned, matching prior behavior.
    """
    vt = video_tensor
    if vt.dim() == 4:
        vt = vt.unsqueeze(0)  # [1,T,C,H,W]
    if vt.dim() != 5:
        raise ValueError(f"Unexpected video dims: {vt.dim()} (expected 4 or 5)")

    bsz = vt.shape[0]
    frames_per_sample: List[np.ndarray] = []
    for b in range(bsz):
        single = vt[b]  # [T,C,H,W]
        frames = single.permute(0, 2, 3, 1).contiguous().numpy()  # [T,H,W,C]
        frames_per_sample.append(frames)
    fps = 10
    return frames_per_sample, fps


def save_mp4_video(
    frames_per_sample: Sequence[np.ndarray],
    out_base_path: Path | str,
    fps: int,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Write MP4 file(s) next to ``out_base_path`` using Matplotlib/ffmpeg.

    - If ``len(frames_per_sample) == 1``, writes ``<base>.mp4``.
    - Otherwise writes ``<base>_sampleXX.mp4`` per sample.
    - Logs each save if a logger is provided.
    """
    out_base = Path(out_base_path)

    # Optional: gracefully skip if ffmpeg is unavailable
    if shutil.which("ffmpeg") is None:
        if logger is not None:
            logger.warning(
                "ffmpeg not found; skipping MP4 export for base %s", str(out_base)
            )
        return []

    # Import locally to avoid global backend effects
    import matplotlib  # type: ignore
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import animation  # type: ignore

    written: List[Path] = []

    for idx, frames in enumerate(frames_per_sample):
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0])
        ax.axis("off")

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"],
        )

        mp4_path = (
            out_base.with_suffix(".mp4")
            if len(frames_per_sample) == 1
            else out_base.parent / f"{out_base.stem}_sample{idx:02d}.mp4"
        )
        with writer.saving(fig, str(mp4_path), dpi=100):
            for f in frames:
                im.set_data(f)
                writer.grab_frame()
        plt.close(fig)

        if logger is not None:
            logger.info("Saved MP4 -> %s (fps=%d)", str(mp4_path), fps)
        written.append(mp4_path)

    return written


def visualise_video_pt(video_pt, fps: int = 24, loop: bool = True, figsize=(6, 6)):
    """
    Visualise a video tensor inline in a notebook.

    Parameters
    ----------
    video_pt : torch.Tensor
        Video tensor of shape (T, C, H, W) or (1, T, C, H, W). C should be 1 or >=3.
        Values can be in [0,1], [0,255], or arbitrary float range (auto-normalised).
    fps : int
        Frames per second for playback.
    loop : bool
        Whether the animation should loop.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    IPython.display.HTML
        HTML object that renders the animation inline.
    """
    if not isinstance(video_pt, torch.Tensor):
        raise TypeError("video_pt must be a torch.Tensor")

    x = video_pt.detach().cpu()

    # Accept (1, T, C, H, W) by squeezing batch dim if present
    if x.ndim == 5 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 4:
        raise ValueError(f"Expected tensor with 4 dims (T, C, H, W), got shape {tuple(x.shape)}")

    T, C, H, W = x.shape
    if C == 1:
        pass  # grayscale supported
    elif C >= 3:
        x = x[:, :3, ...]  # take first 3 channels if more are present
    else:
        raise ValueError(f"Unsupported channel count C={C}. Expected 1 (grayscale) or >=3 (RGB).")

    # (T, C, H, W) -> (T, H, W, C)
    x = x.permute(0, 2, 3, 1).contiguous().float().numpy()

    # Robust normalisation to uint8
    x_min, x_max = x.min(), x.max()
    if x_min >= 0 and x_max <= 1.5:
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
    elif x_min >= 0 and x_max <= 255.0:
        x = x.clip(0, 255).astype(np.uint8)
    else:
        # Auto min-max scale across the whole clip
        x = ((x - x_min) / (x_max - x_min + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)

    is_gray = (x.shape[-1] == 1)

    # Build animation
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if is_gray:
        im = ax.imshow(x[0, ..., 0], cmap="gray", vmin=0, vmax=255, animated=True)
    else:
        im = ax.imshow(x[0], animated=True)

    def _update(i):
        if is_gray:
            im.set_data(x[i, ..., 0])
        else:
            im.set_data(x[i])
        return (im,)

    interval_ms = int(1000 / max(1, fps))
    ani = animation.FuncAnimation(
        fig, _update, frames=T, interval=interval_ms, blit=True, repeat=loop
    )

    # Render as JS/HTML (no external encoders needed)
    html = ani.to_jshtml(fps=fps, default_mode="loop" if loop else "once")
    plt.close(fig)  # prevent duplicate static figure display
    return HTML(html)
