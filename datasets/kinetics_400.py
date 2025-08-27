"""PyTorch dataset for the Kinetics-400 annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from torch.utils.data import Dataset
from decord import VideoReader

def load_64_frames_with_padding(vr: VideoReader) -> Tuple[np.ndarray, bool]:
    """
    Returns exactly 64 frames as uint8 [T,H,W,C] and a 'padded' flag.
    Desired indices are 0..126 step 2. If shorter, pad by repeating last frame.
    """
    desired_idx = np.arange(0, 128, 2)  # 64 indices: 0,2,...,126
    n = len(vr)
    if n == 0:
        raise RuntimeError("Video has zero frames.")
    valid_idx = desired_idx[desired_idx < n]
    if valid_idx.size == 0:
        valid_idx = np.array([0], dtype=np.int64)

    frames = vr.get_batch(valid_idx).asnumpy()  # [T,H,W,C] uint8
    t = frames.shape[0]
    padded = False
    if t < 64:
        last = frames[-1:]                          # [1,H,W,C]
        pad = np.repeat(last, 64 - t, axis=0)       # [64-t,H,W,C]
        frames = np.concatenate([frames, pad], axis=0)
        padded = True
    elif t > 64:
        frames = frames[:64]  # safeguard
    return frames, padded

class Kinetics400(Dataset):
    """Dataset representing the Kinetics-400 annotation CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the dataset annotations.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.csv_path = str(config["datasets"]["kinetics_400"]["paths"]["csv"])
        self.dataframe = pd.read_csv(self.csv_path, header=None, names=["video_path", "index"], sep=" ")
        breakpoint()

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        """Return the number of samples in the dataset."""

        return len(self.dataframe)

    def __getitem__(self, k):
        video_path = self.dataframe.loc[k, "video_path"]
        index = int(self.dataframe.loc[k, "index"])

        padded = False

        vr = VideoReader(video_path)
        n_frames = len(vr)
        video, padded = load_64_frames_with_padding(vr)     # [64,H,W,C] uint8
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [64,C,H,W]

        return {
            "video": video_tensor,          # [64,C,H,W] uint8 tensor
            "index": index,
            "video_path": video_path,
            "n_frames": n_frames,
            "padded": padded,
        }
