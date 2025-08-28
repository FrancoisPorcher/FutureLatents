from pathlib import Path

# Import the project package relative to this module so that running
# ``python -m src.main`` works without requiring ``src`` on the
# ``PYTHONPATH``.
from .futurelatents.models import LatentVideoModel
from datasets.kinetics_400 import Kinetics400
from training.trainer import Trainer
from utils.parser import create_parser
from utils.config import load_config, print_config
import torch


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    print_config(config)
    dataset = Kinetics400(config)

    model = LatentVideoModel(config)

    model.count_parameters()

    learning_rate = float(config["trainer"]["learning_rate"])
    weight_decay = float(config["trainer"]["weight_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    num_workers = int(config["trainer"]["num_workers"])
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=num_workers
    )

    trainer = Trainer(model, optimizer, scheduler)
    trainer.fit(dataloader)


if __name__ == "__main__":
    # run with "python -m src.main --config_path configs/vjepa2_kinetics_400.yaml"
    main()
