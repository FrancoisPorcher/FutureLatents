from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

@dataclass
class ExperimentDirs:
    """Container for standard experiment directories under a given ``root``."""
    root: Path
    checkpoint_dir: Path
    logs_dir: Path
    config_dir: Path
    dump_dir: Path


def make_experiment_dirs(root: Path) -> ExperimentDirs:
    """Create and return the standard experiment directory structure under ``root``.

    Creates: ``checkpoints``, ``logs``, ``config``, ``dump`` and returns
    their paths bundled in an ``ExperimentDirs`` instance.
    """
    checkpoint_dir = root / "checkpoints"
    logs_dir = root / "logs"
    config_dir = root / "config"
    dump_dir = root / "dump"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentDirs(
        root=root,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        dump_dir=dump_dir,
    )
