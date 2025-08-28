"""Entry point that wires together the model, data and trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .models import LatentVideoModel
from datasets.kinetics_400 import Kinetics400
from training.trainer import Trainer
from utils.config import load_config, print_config
from utils.parser import create_parser


def main() -> None:  # pragma: no cover - thin wrapper around training loop
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    print_config(config)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = Kinetics400(config)
    batch_size = int(config["trainer"].get("batch_size", 1))
    train_loader = DataLoader(dataset, batch_size=batch_size)

    # In lieu of a separate validation set we reuse the training loader.
    val_loader = train_loader

    # ------------------------------------------------------------------
    # Model and optimiser
    # ------------------------------------------------------------------
    model = LatentVideoModel(config)

    learning_rate = float(config["trainer"]["learning_rate"])
    weight_decay = float(config["trainer"]["weight_decay"])
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(model, optimizer, scheduler)
    epochs = int(config["trainer"].get("epochs", 1))
    trainer.fit(train_loader, val_loader=val_loader, epochs=epochs)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    # run with "python -m src.futurelatents.main --config_path configs/vjepa2_kinetics_400.yaml"
    main()

