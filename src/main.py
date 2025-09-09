from pathlib import Path
import logging
from copy import deepcopy

# Import the project package relative to this module so that running
# ``python -m src.main`` works without requiring ``src`` on the
# ``PYTHONPATH``.
from models import build_model
from datasets import build_dataset
from training.trainer import Trainer, DeterministicTrainer
from utils.parser import create_parser
from utils.config import load_config, print_config
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()

    config_path = Path(args.config_path)
    config = load_config(config_path)
    config_name = config_path.stem

    dataset_name = next(iter(config.DATASETS))
    dataset_cfg = getattr(config.DATASETS, dataset_name)

    train_cfg = deepcopy(config)
    getattr(train_cfg.DATASETS, dataset_name).PATHS.CSV = dataset_cfg.PATHS.TRAIN_CSV
    train_dataset = build_dataset(train_cfg)

    val_cfg = deepcopy(config)
    getattr(val_cfg.DATASETS, dataset_name).PATHS.CSV = dataset_cfg.PATHS.VAL_CSV
    val_dataset = build_dataset(val_cfg)

    model = build_model(config)

    learning_rate = float(config.TRAINER.TRAINING.LEARNING_RATE)
    weight_decay = float(config.TRAINER.TRAINING.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    num_workers = int(config.TRAINER.TRAINING.NUM_WORKERS)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, num_workers=num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, num_workers=num_workers
    )

    gradient_accumulation_steps = int(
        config.TRAINER.TRAINING.GRADIENT_ACCUMULATION_STEPS
    )
    find_unused_parameters = bool(
        config.TRAINER.TRAINING.FIND_UNUSED_PARAMETERS
    )
    mixed_precision = str(config.TRAINER.TRAINING.MIXED_PRECISION)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=find_unused_parameters
            )
        ],
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    use_wandb = config.WANDB and not getattr(args, "debug", False)
    print("use_wandb", use_wandb)

    if accelerator.is_main_process:
        model.count_parameters()
        print_config(config)
        if use_wandb:
            wandb.init(project="FutureLatent", name=config_name)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler)

    if accelerator.is_main_process and use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    trainer_cls = (
        DeterministicTrainer
        if str(config.MODEL.TYPE).lower() == "deterministic"
        else Trainer
    )
    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        config=config.TRAINER,
        scheduler=scheduler,
        accelerator=accelerator,
        logger=logger,
    )

    trainer.fit(train_dataloader, val_dataloader)

    if accelerator.is_main_process and use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    # run with "accelerate launch --num_processes 2 --num_machines 1 -m src.main \
    #          --config_path configs/vjepa2_kinetics_400.yaml"
    main()
