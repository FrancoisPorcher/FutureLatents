from __future__ import annotations

from pathlib import Path
import pathlib
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


def get_files_dir_from_example_dir(example_dir):
    example_dir = pathlib.Path(example_dir)


    dir_context_latents = example_dir / "context_latents.pt"
    dir_prediction_latents = example_dir / "prediction_latents.pt"
    dir_target_latents = example_dir / "target_latents.pt"
    dir_video_pt = example_dir / "video.pt"
    dir_video_mp4 = example_dir / "video.mp4"
    
    # check if all these files exist
    files_dir = {
        "context_latents": dir_context_latents,
        "prediction_latents": dir_prediction_latents,
        "target_latents": dir_target_latents,
        "video_pt": dir_video_pt,
        "video_mp4": dir_video_mp4,
    }
    for k, v in files_dir.items():
        if not v.exists():
            raise FileNotFoundError(f"File {v} does not exist")
    return files_dir
