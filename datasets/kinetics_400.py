"""PyTorch dataset for the Kinetics-400 annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader
from omegaconf import DictConfig

def load_n_frames_with_padding(vr: VideoReader, n_frame: int, stride: int) -> Tuple[np.ndarray, bool]:
    """
    Returns exactly ``n_frame`` frames as uint8 [T,H,W,C] and a 'padded' flag.
    Desired indices are 0..(n_frame * stride - 1) step ``stride``. If shorter,
    pad by repeating the last frame.
    """
    desired_idx = np.arange(0, n_frame * stride, stride)
    n = len(vr)  # number of frames in video
    if n == 0:
        raise RuntimeError("Video has zero frames.")
    valid_idx = desired_idx[desired_idx < n]
    if valid_idx.size == 0:
        valid_idx = np.array([0], dtype=np.int64)

    frames = vr.get_batch(valid_idx).asnumpy()  # [T,H,W,C] uint8
    t = frames.shape[0]  # number of valid frames
    padded = False
    if t < n_frame:
        last = frames[-1:]
        pad = np.repeat(last, n_frame - t, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
        padded = True
    elif t > n_frame:
        frames = frames[:n_frame]  # safeguard
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
        self.csv_path = str(config.PATHS.CSV)
        self.labels_path = str(config.PATHS.LABELS_TXT)
        self.n_frame = int(config.N_FRAME)
        self.stride = int(config.STRIDE)

        self.dataframe = pd.read_csv(self.csv_path, header=None, names=["video_path", "index"], sep=" ")
        with open(self.labels_path, "r") as f:
            lines = [line.strip().split(" ", 1) for line in f]
        df_labels = pd.DataFrame(lines, columns=["index", "label"])
        df_labels["index"] = df_labels["index"].astype("int64")
        self.dataframe = pd.merge(self.dataframe, df_labels, on="index")

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        """Return the number of samples in the dataset."""

        return len(self.dataframe)

    def __getitem__(self, k):
        video_path = self.dataframe.loc[k, "video_path"]
        index = int(self.dataframe.loc[k, "index"])
        label = self.dataframe.loc[k, "label"]

        padded = False

        vr = VideoReader(video_path)
        n_frames_original_video = len(vr)
        video, padded = load_n_frames_with_padding(vr, self.n_frame, self.stride)     # [n_frame,H,W,C] uint8, values are between 0 and 255
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [n_frame,C,H,W]
        
        n_frames, C, H, W = video_tensor.shape

        return {
            "video": video_tensor,          # [n_frame,C,H,W] uint8 tensor
            "index": index,
            "label": label,
            "video_path": video_path,
            "n_frames_original_video": n_frames_original_video,
            "n_frames": n_frames,
            "C_original": C,
            "H_original": H,
            "W_original": W,
            "padded": padded,
        }
