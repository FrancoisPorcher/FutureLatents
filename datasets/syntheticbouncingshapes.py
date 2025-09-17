"""Synthetic shapes datasets.

- ``SyntheticBouncingShapesVideo``: generates short synthetic videos with two
  moving shapes (a red square bouncing on borders, and a blue circle following
  a circular trajectory). Frames are returned as uint8 ``[T, C, H, W]``.
- ``SyntheticBouncingShapesImage``: generates single RGB images where the
  square and circle are placed at random valid positions. Returned samples
  expose an ``"image"`` tensor with shape ``[C, H, W]`` alongside supervision
  metadata.
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
    - ``DISK_RADIUS``: radius of the circle (kept for backward compatibility)
    """

    def __init__(self, config) -> None:
        self.config = config

        # Read parameters from config (following kinetics_400 style)
        self.n_frame = config.N_FRAME
        # Dataset length is configurable; default to 4 if unspecified
        self.length = config.LENGTH
        self.image_size = config.IMAGE_SIZE
        self.square_size = config.SQUARE_SIZE
        # Lists from YAML become ListConfig -> cast to floats
        self.square_init = (
            float(config.SQUARE_INIT[0]),
            float(config.SQUARE_INIT[1]),
        )
        self.square_vel = (
            float(config.SQUARE_VEL[0]),
            float(config.SQUARE_VEL[1]),
        )
        self.circle_radius = int(config.DISK_RADIUS)

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
            # Shape: tensors with shape [T, 2] storing (x, y) in pixels
            "target_square_position_pixel": sample["target_square_position_pixel"],
            "target_circle_position_pixel": sample["target_circle_position_pixel"],
            "target_square_position_normalized": sample["target_square_position_normalized"],
            "target_circle_position_normalized": sample["target_circle_position_normalized"],
            "target_square_heatmap": sample["target_square_heatmap"],
            "target_circle_heatmap": sample["target_circle_heatmap"],
            "target_square_heatmap_patch": sample["target_square_heatmap_patch"],
            "target_circle_heatmap_patch": sample["target_circle_heatmap_patch"],
        }

    # ---------------------------
    # Generation helpers
    # ---------------------------
    def _generate_video_random(self) -> Dict[str, Any]:
        T = self.n_frame
        size = self.image_size
        sq = self.square_size
        r = self.circle_radius

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
        R = (size / 2.0) - r - 8.0  # keep circle within borders
        # Random initial phase for the circular trajectory
        theta0 = float(np.random.uniform(0.0, 2 * np.pi))

        frames: list[Image.Image] = []
        # Accumulate absolute pixel centers for each object at every frame.
        # We use integer pixel indices in (x, y) order, within [0, size-1].
        square_centers = np.zeros((T, 2), dtype=np.int64)
        circle_centers = np.zeros((T, 2), dtype=np.int64)
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

            # Circle along circular trajectory
            theta = theta0 + 2 * np.pi * (t / T)
            cx = int(round(cx0 + R * np.cos(theta)))
            cy = int(round(cy0 + R * np.sin(theta)))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(50, 180, 250))
            # Circle center is directly (cx, cy) in absolute pixels
            cx = int(np.clip(cx, 0, size - 1))
            cy = int(np.clip(cy, 0, size - 1))
            circle_centers[t] = (cx, cy)

            frames.append(img)

        # Convert list of PIL images -> [T, H, W, C] uint8 -> [T, C, H, W]
        arr = np.stack([np.array(f, dtype=np.uint8) for f in frames], axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

        square_px = torch.from_numpy(square_centers)
        circle_px = torch.from_numpy(circle_centers)

        # Per-frame heatmaps marking the centers with value 1.0
        square_heatmap = torch.zeros((T, size, size, 1), dtype=torch.float32)
        circle_heatmap = torch.zeros((T, size, size, 1), dtype=torch.float32)
        patch_size = 16
        patch_h = max(size // patch_size, 1)
        patch_w = max(size // patch_size, 1)
        square_heatmap_patch = torch.zeros((T, patch_h, patch_w, 1), dtype=torch.float32)
        circle_heatmap_patch = torch.zeros((T, patch_h, patch_w, 1), dtype=torch.float32)
        for t_idx in range(T):
            sq_x, sq_y = square_centers[t_idx]
            square_heatmap[t_idx, sq_y, sq_x, 0] = 1.0
            sq_py = min(patch_h - 1, sq_y // patch_size)
            sq_px = min(patch_w - 1, sq_x // patch_size)
            square_heatmap_patch[t_idx, sq_py, sq_px, 0] = 1.0
            circ_x, circ_y = circle_centers[t_idx]
            circle_heatmap[t_idx, circ_y, circ_x, 0] = 1.0
            circ_py = min(patch_h - 1, circ_y // patch_size)
            circ_px = min(patch_w - 1, circ_x // patch_size)
            circle_heatmap_patch[t_idx, circ_py, circ_px, 0] = 1.0
        scale = 2.0 / float(self.image_size - 1)
        square_norm = square_px.float() * scale - 1.0
        circle_norm = circle_px.float() * scale - 1.0

        # Return a compact, well-structured dictionary
        return {
            "video": tensor,                                 # [T, C, H, W] uint8
            "target_square_position_pixel": square_px,       # [T, 2]
            "target_circle_position_pixel": circle_px,       # [T, 2]
            "target_square_position_normalized": square_norm,
            "target_circle_position_normalized": circle_norm,
            "target_square_heatmap": square_heatmap,         # [T, H, W, 1]
            "target_circle_heatmap": circle_heatmap,         # [T, H, W, 1]
            "target_square_heatmap_patch": square_heatmap_patch,
            "target_circle_heatmap_patch": circle_heatmap_patch,
        }


class SyntheticBouncingShapesImage(Dataset):
    """Dataset producing single-frame images with randomly placed shapes.

    The square and circle are positioned uniformly at random within valid
    borders so that the full shapes fit inside the image. Returned samples
    contain an ``image`` tensor (``[C, H, W]`` uint8) together with absolute
    pixel centers for each shape.

    Expected config keys (after uppercasing by the loader):
    - ``LENGTH``: number of samples to expose via ``__len__``
    - ``IMAGE_SIZE``: height/width in pixels
    - ``SQUARE_SIZE``: edge length of the square
    - ``DISK_RADIUS``: radius of the circle (kept for backward compatibility)
    """

    def __init__(self, config) -> None:
        self.config = config
        # Dataset length is configurable; default to 4 if unspecified
        self.length = int(getattr(config, "LENGTH", 4))
        self.image_size = int(config.IMAGE_SIZE)
        self.square_size = int(config.SQUARE_SIZE)
        self.circle_radius = int(config.DISK_RADIUS)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._generate_image_random()
        image = sample["image"]  # [C, H, W]
        c, h, w = image.shape
        return {
            "image": image,               # [C, H, W] uint8 image tensor
            "index": idx,
            "label": "synthetic_random_shapes_image",
            "image_path": "synthetic://random_square_circle_image",
            "channels": c,
            "height": h,
            "width": w,
            "padded": False,
            # Supervision targets: absolute pixel centers (x, y)
            "target_square_position_pixel": sample["target_square_position_pixel"],
            "target_circle_position_pixel": sample["target_circle_position_pixel"],
            "target_square_position_normalized": sample["target_square_position_normalized"],
            "target_circle_position_normalized": sample["target_circle_position_normalized"],
            "target_square_heatmap": sample["target_square_heatmap"],
            "target_circle_heatmap": sample["target_circle_heatmap"],
            "target_square_heatmap_patch": sample["target_square_heatmap_patch"],
            "target_circle_heatmap_patch": sample["target_circle_heatmap_patch"],
        }

    def _generate_image_random(self) -> Dict[str, Any]:
        size = self.image_size
        sq = self.square_size
        r = self.circle_radius

        # Sample square top-left uniformly where it fully fits
        xmin, ymin = 0.0, 0.0
        xmax, ymax = float(size - sq), float(size - sq)
        x = float(np.random.uniform(xmin, xmax))
        y = float(np.random.uniform(ymin, ymax))

        # Sample circle center uniformly where it fully fits
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

        # Circle
        cxi = int(round(cx))
        cyi = int(round(cy))
        draw.ellipse([cxi - r, cyi - r, cxi + r, cyi + r], fill=(50, 180, 250))
        cxi = int(np.clip(cxi, 0, size - 1))
        cyi = int(np.clip(cyi, 0, size - 1))

        # Convert to tensor [C, H, W]
        arr = np.array(img, dtype=np.uint8)  # [H, W, C]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]

        square_px = torch.tensor([square_cx, square_cy], dtype=torch.int64)
        circle_px = torch.tensor([cxi, cyi], dtype=torch.int64)

        # Heatmaps with a "1" at the respective centers
        square_heatmap = torch.zeros((size, size, 1), dtype=torch.float32)
        square_heatmap[square_cy, square_cx, 0] = 1.0
        circle_heatmap = torch.zeros((size, size, 1), dtype=torch.float32)
        circle_heatmap[cyi, cxi, 0] = 1.0
        patch_size = 16
        patch_h = max(size // patch_size, 1)
        patch_w = max(size // patch_size, 1)
        square_heatmap_patch = torch.zeros((patch_h, patch_w, 1), dtype=torch.float32)
        circle_heatmap_patch = torch.zeros((patch_h, patch_w, 1), dtype=torch.float32)
        square_heatmap_patch[min(patch_h - 1, square_cy // patch_size), min(patch_w - 1, square_cx // patch_size), 0] = 1.0
        circle_heatmap_patch[min(patch_h - 1, cyi // patch_size), min(patch_w - 1, cxi // patch_size), 0] = 1.0
        scale = 2.0 / float(self.image_size - 1)
        square_norm = square_px.float() * scale - 1.0
        circle_norm = circle_px.float() * scale - 1.0

        return {
            "image": tensor,           # [C, H, W] uint8
            "target_square_position_pixel": square_px,
            "target_circle_position_pixel": circle_px,
            "target_square_position_normalized": square_norm,
            "target_circle_position_normalized": circle_norm,
            "target_square_heatmap": square_heatmap,  # [H, W, 1]
            "target_circle_heatmap": circle_heatmap,  # [H, W, 1]
            "target_square_heatmap_patch": square_heatmap_patch,
            "target_circle_heatmap_patch": circle_heatmap_patch,
        }


__all__ = [
    "SyntheticBouncingShapesVideo",
    "SyntheticBouncingShapesImage",
]
