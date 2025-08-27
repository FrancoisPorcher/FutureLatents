"""PyTorch dataset for the Kinetics-400 annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from torch.utils.data import Dataset


class Kinetics400(Dataset):
    """Dataset representing the Kinetics-400 annotation CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the dataset annotations.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = str(csv_path)
        self.dataframe = pd.read_csv(self.csv_path)

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        """Return the number of samples in the dataset."""

        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a sample from the dataset.

        Parameters
        ----------
        index:
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, Any]
            A dictionary representing the selected row from the CSV file.
        """

        row = self.dataframe.iloc[index]
        return row.to_dict()

