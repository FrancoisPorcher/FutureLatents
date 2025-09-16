"""Synthetic shapes datasets.

- ``SyntheticBouncingShapesVideo``: generates short synthetic videos with two
  moving shapes (a red square bouncing on borders, and a blue disk following a
  circular trajectory). Frames are returned as uint8 ``[T, C, H, W]``.
- ``SyntheticBouncingShapesImage``: generates single images where the square
  and disk are placed at random valid positions. Returns a single-frame video
  tensor ``[1, C, H, W]`` for compatibility.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class SyntheticBouncingShapesVideo(Dataset):
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
        sample = self._generate_video_random()
        video = sample["video"]
        centers_px = sample["centers_px"]
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
            # Supervision targets: absolute pixel centers per frame
            # Shape: dict of tensors with shape [T, 2] storing (x, y) in pixels
            "centers_px": centers_px,
        }

    # ---------------------------
    # Generation helpers
    # ---------------------------
    def _generate_video_random(self) -> Dict[str, Any]:
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
        # Accumulate absolute pixel centers for each object at every frame.
        # We use integer pixel indices in (x, y) order, within [0, size-1].
        square_centers = np.zeros((T, 2), dtype=np.int64)
        disk_centers = np.zeros((T, 2), dtype=np.int64)
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
            # Square center (absolute px). If sq is odd, rounding picks nearest px.
            square_cx = int(round(xi + sq / 2.0))
            square_cy = int(round(yi + sq / 2.0))
            # Clamp to valid image bounds just in case of edge rounding
            square_cx = int(np.clip(square_cx, 0, size - 1))
            square_cy = int(np.clip(square_cy, 0, size - 1))
            square_centers[t] = (square_cx, square_cy)

            # Disk along circular trajectory
            theta = theta0 + 2 * np.pi * (t / T)
            cx = int(round(cx0 + R * np.cos(theta)))
            cy = int(round(cy0 + R * np.sin(theta)))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 180, 250))
            # Disk center is directly (cx, cy) in absolute pixels
            cx = int(np.clip(cx, 0, size - 1))
            cy = int(np.clip(cy, 0, size - 1))
            disk_centers[t] = (cx, cy)

            frames.append(img)

        # Convert list of PIL images -> [T, H, W, C] uint8 -> [T, C, H, W]
        arr = np.stack([np.array(f, dtype=np.uint8) for f in frames], axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

        centers_px: Dict[str, torch.Tensor] = {
            # [T, 2] with (x, y) absolute pixel indices
            "square": torch.from_numpy(square_centers),
            "disk": torch.from_numpy(disk_centers),
        }

        # Return a compact, well-structured dictionary
        return {
            "video": tensor,           # [T, C, H, W] uint8
            "centers_px": centers_px, # dict[name]->[T,2] (x,y) in pixels
        }


class SyntheticBouncingShapesImage(Dataset):
    """Dataset producing single-frame images with randomly placed shapes.

    The square and disk are positioned uniformly at random within valid
    borders so that the full shapes fit inside the image. Returned sample
    mirrors the video dataset structure for compatibility, using ``T=1``.

    Expected config keys (after uppercasing by the loader):
    - ``LENGTH``: number of samples to expose via ``__len__``
    - ``IMAGE_SIZE``: height/width in pixels
    - ``SQUARE_SIZE``: edge length of the square
    - ``DISK_RADIUS``: radius of the disk
    """

    def __init__(self, config) -> None:
        self.config = config
        # Dataset length is configurable; default to 4 if unspecified
        self.length = int(getattr(config, "LENGTH", 4))
        self.image_size = int(config.IMAGE_SIZE)
        self.square_size = int(config.SQUARE_SIZE)
        self.disk_radius = int(config.DISK_RADIUS)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._generate_image_random()
        video = sample["video"]  # [1, C, H, W]
        centers_px = sample["centers_px"]  # dict[name]->[1,2]
        t, c, h, w = video.shape
        return {
            "video": video,               # [1, C, H, W] uint8 for compatibility
            "index": idx,
            "label": "synthetic_random_shapes_image",
            "video_path": "synthetic://random_square_circle_image",
            "n_frames_original_video": t,
            "n_frames": t,
            "C_original": c,
            "H_original": h,
            "W_original": w,
            "padded": False,
            # Supervision targets: absolute pixel centers per frame [T,2]
            "centers_px": centers_px,
        }

    def _generate_image_random(self) -> Dict[str, Any]:
        size = self.image_size
        sq = self.square_size
        r = self.disk_radius

        # Sample square top-left uniformly where it fully fits
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(size - sq), float(size - sq)
        x = float(np.random.uniform(xmin, xmax))
        y = float(np.random.uniform(ymin, ymax))

        # Sample disk center uniformly where it fully fits
        cx = float(np.random.uniform(r, size - r))
        cy = float(np.random.uniform(r, size - r))

        # Render image
        img = Image.new("RGB", (size, size), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        # Square
        xi, yi = int(round(x)), int(round(y))
        draw.rectangle([xi, yi, xi + sq, yi + sq], fill=(200, 50, 50))
        square_cx = int(round(xi + sq / 2.0))
        square_cy = int(round(yi + sq / 2.0))
        square_cx = int(np.clip(square_cx, 0, size - 1))
        square_cy = int(np.clip(square_cy, 0, size - 1))

        # Disk
        cxi = int(round(cx))
        cyi = int(round(cy))
        draw.ellipse([cxi - r, cyi - r, cxi + r, cyi + r], fill=(50, 180, 250))
        cxi = int(np.clip(cxi, 0, size - 1))
        cyi = int(np.clip(cyi, 0, size - 1))

        # Convert to tensor with T=1
        arr = np.array(img, dtype=np.uint8)  # [H, W, C]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,C,H,W]

        centers_px: Dict[str, torch.Tensor] = {
            "square": torch.tensor([[square_cx, square_cy]], dtype=torch.int64),
            "disk": torch.tensor([[cxi, cyi]], dtype=torch.int64),
        }

        return {
            "image": tensor,           # [1, C, H, W] uint8
            "centers_px": centers_px, # dict[name]->[1,2]
        }


__all__ = [
    "SyntheticBouncingShapesVideo",
    "SyntheticBouncingShapesImage",
]
