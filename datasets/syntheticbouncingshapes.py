"""Synthetic video dataset: bouncing square + circular disk.

Generates videos on-the-fly with two moving shapes:
- a red square bouncing elastically on the image borders
- a blue disk moving along a circular trajectory

Frames are returned as a uint8 tensor shaped [T, C, H, W].
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class SyntheticBouncingShapes(Dataset):
    """Dataset producing synthetic videos from config.

    Mirrors the config-driven pattern used in ``Kinetics400``: pass a
    dataset-specific config object and read parameters from it.
    Expected config keys (after uppercasing by the loader):
    - ``N_FRAME``: number of frames
    - ``LENGTH``: number of samples to expose via ``__len__``
    - ``IMAGE_SIZE``: height/width in pixels
    - ``SQUARE_SIZE``: edge length of the square
    - ``SQUARE_INIT``: [x0, y0] top-left initial position (floats)
    - ``SQUARE_VEL``: [vx, vy] velocity in px/frame (floats)
    - ``DISK_RADIUS``: radius of the disk
    """

    def __init__(self, config) -> None:
        self.config = config

        # Read parameters from config (following kinetics_400 style)
        self.n_frame = int(config.N_FRAME)
        # Dataset length is configurable; default to 4 if unspecified
        self.length = int(getattr(config, "LENGTH", 4))
        self.image_size = int(config.IMAGE_SIZE)
        self.square_size = int(config.SQUARE_SIZE)
        # Lists from YAML become ListConfig -> cast to floats
        self.square_init = (
            float(config.SQUARE_INIT[0]),
            float(config.SQUARE_INIT[1]),
        )
        self.square_vel = (
            float(config.SQUARE_VEL[0]),
            float(config.SQUARE_VEL[1]),
        )
        self.disk_radius = int(config.DISK_RADIUS)

        # Randomness source
        #
        # We intentionally rely on NumPy's *global* RNG rather than keeping a
        # ``RandomState`` instance on the dataset.  PyTorch's ``DataLoader``
        # automatically seeds the global NumPy generator for each worker.  A
        # per-instance ``RandomState`` would be cloned with identical state in
        # every worker process, producing the exact same "random" video on
        # different workers.  Using the global RNG avoids this behaviour while
        # still allowing users to control determinism via ``numpy.random.seed``.

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate a fresh random sample each time
        video = self._generate_video_random()
        t, c, h, w = video.shape
        return {
            "video": video,               # [T, C, H, W] uint8
            "index": idx,
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
    def _generate_video_random(self) -> torch.Tensor:
        T = self.n_frame
        size = self.image_size
        sq = self.square_size
        r = self.disk_radius

        # Sample random initial square position uniformly in the valid range
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(size - sq), float(size - sq)
        # Sample random initial square position uniformly in the valid range.
        #
        # ``numpy.random`` is seeded automatically for each ``DataLoader``
        # worker, ensuring different workers generate independent sequences.
        x = float(np.random.uniform(xmin, xmax))
        y = float(np.random.uniform(ymin, ymax))
        vx, vy = self.square_vel

        cx0, cy0 = size / 2.0, size / 2.0
        R = (size / 2.0) - r - 8.0  # keep disk within borders
        # Random initial phase for the circular trajectory
        theta0 = float(np.random.uniform(0.0, 2 * np.pi))

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
            theta = theta0 + 2 * np.pi * (t / T)
            cx = int(round(cx0 + R * np.cos(theta)))
            cy = int(round(cy0 + R * np.sin(theta)))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 180, 250))

            frames.append(img)

        # Convert list of PIL images -> [T, H, W, C] uint8 -> [T, C, H, W]
        arr = np.stack([np.array(f, dtype=np.uint8) for f in frames], axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return tensor


__all__ = ["SyntheticBouncingShapes"]
