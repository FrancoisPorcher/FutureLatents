from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import logging
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
from PIL import Image, ImageDraw


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


def save_position_targets_and_predictions(
    batch: Dict[str, Any],
    outputs: Dict[str, Any],
    out_directory: Path,
    batch_idx: int,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    images = batch["image"].detach().cpu()
    square_targets = batch["target_square_position_pixel"].detach().cpu().numpy()
    circle_targets = batch["target_circle_position_pixel"].detach().cpu().numpy()
    square_predictions = outputs["denormalized_square_pred"].detach().cpu().numpy()
    circle_predictions = outputs["denormalized_circle_pred"].detach().cpu().numpy()
    def _heatmap_softmax(logits: torch.Tensor) -> torch.Tensor:
        flat = logits.flatten(start_dim=1)
        probs = torch.softmax(flat, dim=1)
        return probs.view_as(logits)

    square_heatmap_probs = _heatmap_softmax(outputs["square_heatmap_logits"]).detach().cpu()
    circle_heatmap_probs = _heatmap_softmax(outputs["circle_heatmap_logits"]).detach().cpu()

    indices = batch.get("index")
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if indices is not None:
        index_array = np.asarray(indices).reshape(-1)
    else:
        index_array = None

    def _extract_coords(arr: np.ndarray, sample_idx: int) -> tuple[float, float]:
        coords = arr[sample_idx]
        if coords.ndim > 1:
            coords = coords[0]
        return float(coords[0]), float(coords[1])

    def _to_pixel_coordinates(coords: tuple[float, float], width: int, height: int) -> tuple[int, int]:
        x, y = coords
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return x, y

    def _draw_cross(
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
        size: int = 6,
        thickness: int = 3,
    ) -> None:
        x_min = max(0, x - size)
        x_max = min(width - 1, x + size)
        y_min = max(0, y - size)
        y_max = min(height - 1, y + size)
        draw.line((x_min, y, x_max, y), fill=color, width=thickness)
        draw.line((x, y_min, x, y_max), fill=color, width=thickness)
        draw.ellipse(
            (x - thickness // 2, y - thickness // 2, x + thickness // 2, y + thickness // 2),
            fill=color,
        )

    sample_stems: List[str] = []

    def _upsample_heatmap(prob_map: np.ndarray, height: int, width: int) -> np.ndarray:
        h_in, w_in = prob_map.shape
        if h_in == 0 or w_in == 0:
            return np.zeros((height, width), dtype=np.float32)
        scale_h = max(height // h_in, 1)
        scale_w = max(width // w_in, 1)
        upsampled = np.repeat(np.repeat(prob_map, scale_h, axis=0), scale_w, axis=1)
        return upsampled[:height, :width]

    for sample_idx in range(images.shape[0]):
        img = images[sample_idx].clamp(0, 255).round().to(torch.uint8)
        arr = img.permute(1, 2, 0).contiguous()
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        pil_image = Image.fromarray(arr.numpy())
        pil_image = pil_image.convert("RGB")

        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        overlays = (
            (_to_pixel_coordinates(_extract_coords(square_targets, sample_idx), width, height), (170, 85, 255)),
            (_to_pixel_coordinates(_extract_coords(square_predictions, sample_idx), width, height), (215, 160, 255)),
            (_to_pixel_coordinates(_extract_coords(circle_targets, sample_idx), width, height), (255, 140, 0)),
            (_to_pixel_coordinates(_extract_coords(circle_predictions, sample_idx), width, height), (255, 205, 120)),
        )
        for coord, color in overlays:
            _draw_cross(draw, coord[0], coord[1], width, height, color)

        if index_array is not None:
            sample_index = int(index_array[sample_idx])
            sample_name = f"sample_{sample_index:06d}.png"
        else:
            sample_name = f"batch{batch_idx:04d}_sample{sample_idx:02d}.png"

        out_path = out_directory / sample_name
        sq_prob = square_heatmap_probs[sample_idx, 0].numpy()
        circ_prob = circle_heatmap_probs[sample_idx, 0].numpy()
        heatmap_red = _upsample_heatmap(sq_prob, height, width)
        heatmap_blue = _upsample_heatmap(circ_prob, height, width)
        heatmap_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        heatmap_rgb[..., 0] = np.clip(heatmap_red * 255.0, 0, 255).astype(np.uint8)
        heatmap_rgb[..., 2] = np.clip(heatmap_blue * 255.0, 0, 255).astype(np.uint8)
        heatmap_image = Image.fromarray(heatmap_rgb)
        combined = Image.new("RGB", (width + heatmap_image.width, height))
        combined.paste(pil_image, (0, 0))
        combined.paste(heatmap_image, (width, 0))

        combined.save(out_path)
        if logger is not None:
            logger.info("Saved image -> %s", str(out_path))

        sample_stems.append(Path(sample_name).stem)

    return sample_stems


def save_batch(
    batch: Dict[str, Any],
    outputs: Dict[str, Any],
    out_dir: Path | str,
    batch_idx: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    out_directory = Path(out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)
    save_position_targets_and_predictions(
        batch,
        outputs,
        out_directory,
        batch_idx,
        logger,
    )
    return None


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
