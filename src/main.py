from pathlib import Path

import yaml

from datasets.kinetics_400 import Kinetics400
from utils.parser import create_parser


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()
    

    with open(Path(args.config_path)) as f:
        config = yaml.safe_load(f)
    dataset = Kinetics400(config)
    breakpoint()



if __name__ == "__main__":
    # run with "python -m src.main --config_path /private/home/francoisporcher/FutureLatents/configs/default.yaml"
    main()
