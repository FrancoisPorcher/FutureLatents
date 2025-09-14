from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging
import shutil

import numpy as np
import torch


def save_video_tensor(
    video_tensor: torch.Tensor,
    out_path: Path | str,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save a video tensor to ``.pt`` and log the event.

    - Accepts tensors shaped ``[T,C,H,W]`` or ``[B,T,C,H,W]`` (or any tensor).
    - Detaches and moves to CPU before saving for portability.
    """
    path = Path(out_path)
    vt = video_tensor.detach().cpu()
    torch.save(vt, path)
    if logger is not None:
        logger.info("Saved tensor -> %s [shape=%s]", str(path), str(tuple(vt.shape)))
    return path


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

