"""PyTorch dataset for cached Kinetics-400 embeddings."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class Kinetics400Cached(Dataset):
    """Dataset representing cached Kinetics-400 embeddings.

    Parameters
    ----------
    config:
        Global configuration containing the dataset specification.
    """

    def __init__(self, config) -> None:
        self.config = config
        cfg = config["datasets"]["kinetics_400_cached"]
        self.metadata_path = str(cfg["paths"]["metadata"])
        self.path_col = str(cfg.get("path_col", "out_path"))

        self.df_metadata = pd.read_csv(self.metadata_path)
        self.metadata_list_dict = [
            self.df_metadata.iloc[i].to_dict() for i in range(self.df_metadata.shape[0])
        ]
        if len(self.metadata_list_dict) == 0:
            raise ValueError(f"Empty dataset found in {self.metadata_path}")

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return len(self.metadata_list_dict)

    def _resolve_index(self, idx: int) -> int:
        return idx % len(self.metadata_list_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = self._resolve_index(idx)
        record = self.metadata_list_dict[base_idx]

        if self.path_col not in record:
            raise ValueError(f"Column {self.path_col} not found in metadata")

        embedding_path = record[self.path_col]
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        embedding = torch.load(embedding_path, weights_only=False, map_location="cpu")

        return {
            "embedding": embedding,
            "metadata": record,
            "index": base_idx,
        }
