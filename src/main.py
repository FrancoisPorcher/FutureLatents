from pathlib import Path

import yaml
from transformers import AutoModel, AutoVideoProcessor

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
    breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    breakpoint()
    model = AutoModel.from_pretrained(config["backbone"]["hf_repo"]).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(config["backbone"]["hf_repo"])
    
    breakpoint()
    
    # try to get sample
    sample = dataset[10]
    breakpoint()



if __name__ == "__main__":
    # run with "python -m src.main --config_path configs/vjepa2_kinetics_400.yaml"
    main()
