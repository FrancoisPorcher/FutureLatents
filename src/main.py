from pathlib import Path

import yaml
from futurelatents.models import LatentVideoModel

from datasets.kinetics_400 import Kinetics400
from utils.parser import create_parser
import torch


def load_config(path: Path) -> dict:
    """Load configuration with simple inheritance support."""
    with open(path) as f:
        config = yaml.safe_load(f)
    defaults = config.pop("defaults", [])
    merged = {}
    for item in defaults:
        for key, value in item.items():
            sub_path = path.parent / key / f"{value}.yaml"
            with open(sub_path) as sf:
                sub_cfg = yaml.safe_load(sf)
            merged.update(sub_cfg)
    merged.update(config)
    return merged


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    dataset = Kinetics400(config)

    model = LatentVideoModel(config)

    # Build optimiser using only trainable parameters
    params = list(model.trainable_parameters())
    if not params:
        raise ValueError("No trainable parameters found for optimisation.")
    optim_cfg = config.get("optimizer", {})
    optim_cls = getattr(torch.optim, optim_cfg.get("name", "AdamW"))
    optimizer = optim_cls(params, **optim_cfg.get("params", {}))

    # Learning rate scheduler configuration
    sched_cfg = config.get("scheduler", {})
    sched_cls = getattr(torch.optim.lr_scheduler, sched_cfg.get("name", "ConstantLR"))
    scheduler = sched_cls(optimizer, **sched_cfg.get("params", {}))
    


if __name__ == "__main__":
    # run with "python -m src.main --config_path configs/vjepa2_kinetics_400.yaml"
    main()
