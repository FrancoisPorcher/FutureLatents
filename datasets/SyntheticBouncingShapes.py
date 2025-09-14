"""Synthetic video dataset: bouncing square + circular disk.

Generates a single video on-the-fly with two moving shapes:
- a red square bouncing elastically on the image borders
- a blue disk moving along a circular trajectory

Frames are returned as a uint8 tensor shaped [T, C, H, W].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


@dataclass
class SyntheticConfig:
    n_frame: int = 64
    size: int = 256
    square_size: int = 64
    square_init: tuple[float, float] = (16.0, 32.0)
    square_vel: tuple[float, float] = (3.0, 2.0)
    disk_radius: int = 24


class SyntheticBouncingShapes(Dataset):
    """Dataset producing a single synthetic video.

    Parameters
    ----------
    n_frame: int
        Number of frames to generate.
    size: int
        Height and width of each frame in pixels.
    square_size: int
        Edge length of the bouncing square.
    square_init: tuple[float, float]
        Initial (x, y) position of the square's top-left corner.
    square_vel: tuple[float, float]
        Velocity (vx, vy) of the square in pixels per frame.
    disk_radius: int
        Radius of the disk following a circular path.
    """

    def __init__(
        self,
        n_frame: int = 64,
        size: int = 256,
        square_size: int = 64,
        square_init: tuple[float, float] = (16.0, 32.0),
        square_vel: tuple[float, float] = (3.0, 2.0),
        disk_radius: int = 24,
    ) -> None:
        super().__init__()
        self.cfg = SyntheticConfig(
            n_frame=n_frame,
            size=size,
            square_size=square_size,
            square_init=square_init,
            square_vel=square_vel,
            disk_radius=disk_radius,
        )

        # Pre-generate once to keep __getitem__ lightweight
        self._video = self._generate_video()

    def __len__(self) -> int:
        # Single custom video for now
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return a clone to avoid accidental in-place edits by callers
        video = self._video.clone()
        t, c, h, w = video.shape
        return {
            "video": video,               # [T, C, H, W] uint8
            "index": 0,
            "label": "synthetic_bounce_circle",
            "video_path": "synthetic://bouncing_square_circle",
            "n_frames_original_video": t,
            "n_frames": t,
            "C_original": c,
            "H_original": h,
            "W_original": w,
            "padded": False,
        }

    # ---------------------------
    # Generation helpers
    # ---------------------------
    def _generate_video(self) -> torch.Tensor:
        T = int(self.cfg.n_frame)
        size = int(self.cfg.size)
        sq = int(self.cfg.square_size)
        r = int(self.cfg.disk_radius)

        x, y = float(self.cfg.square_init[0]), float(self.cfg.square_init[1])
        vx, vy = float(self.cfg.square_vel[0]), float(self.cfg.square_vel[1])
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(size - sq), float(size - sq)

        cx0, cy0 = size / 2.0, size / 2.0
        R = (size / 2.0) - r - 8.0  # keep disk within borders

        frames: list[Image.Image] = []
        for t in range(T):
            # Update square position with elastic wall bounces
            x += vx
            y += vy
            if x < xmin:
                x = xmin
                vx = -vx
            elif x > xmax:
                x = xmax
                vx = -vx
            if y < ymin:
                y = ymin
                vy = -vy
            elif y > ymax:
                y = ymax
                vy = -vy

            # Render frame
            img = Image.new("RGB", (size, size), (20, 20, 20))
            draw = ImageDraw.Draw(img)

            # Bouncing square
            xi, yi = int(round(x)), int(round(y))
            draw.rectangle([xi, yi, xi + sq, yi + sq], fill=(200, 50, 50))

            # Disk along circular trajectory
            theta = 2 * np.pi * (t / T)
            cx = int(round(cx0 + R * np.cos(theta)))
            cy = int(round(cy0 + R * np.sin(theta)))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 180, 250))

            frames.append(img)

        # Convert list of PIL images -> [T, H, W, C] uint8 -> [T, C, H, W]
        arr = np.stack([np.array(f, dtype=np.uint8) for f in frames], axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return tensor


__all__ = ["SyntheticBouncingShapes"]

