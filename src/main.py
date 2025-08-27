from pathlib import Path

import yaml
from transformers import AutoVideoProcessor

# Import the project package relative to this module so that running
# ``python -m src.main`` works without requiring ``src`` on the
# ``PYTHONPATH``.
from .futurelatents.models import LatentVideoModel
breakpoint()
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
    breakpoint()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    


if __name__ == "__main__":
    # run with "python -m src.main --config_path configs/vjepa2_kinetics_400.yaml"
    main()
