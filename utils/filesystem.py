from __future__ import annotations

from pathlib import Path


def make_experiment_dirs(root: Path) -> None:
    """Create the standard experiment directory structure under ``root``.

    Creates: ``checkpoints``, ``logs``, ``config``, ``slurm`` ''dump''
    """
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    (root / "dump").mkdir(parents=True, exist_ok=True)
