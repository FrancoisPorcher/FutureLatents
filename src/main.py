from pathlib import Path
import logging
import shutil

# Import the project package relative to this module so that running
# ``python -m src.main`` works without requiring ``src`` on the
# ``PYTHONPATH``.
from models import build_model
from datasets import build_dataset
from training import build_trainer
from utils.parser import create_parser
from utils.config import load_config, print_config
from utils.filesystem import make_experiment_dirs
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

    # ------------------------------------------------------------------
    # Accelerator initialisation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Experiment directories
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    experiment_root = project_root / "experiment" / config_name
    overwrite_experiment = experiment_root.exists()
    # When overwriting, remove the entire folder (no distinctions), unless debug.
    if overwrite_experiment and accelerator.is_main_process:
        print("Overwriting experiment", experiment_root)
        shutil.rmtree(experiment_root)
    accelerator.wait_for_everyone()

    # Define directory paths for later use
    checkpoint_dir = experiment_root / "checkpoints"
    logs_dir = experiment_root / "logs"
    config_dir = experiment_root / "config"
    slurm_dir = experiment_root / "slurm"
    dump_dir = experiment_root / "dump"

    # Create folder structure via helper, unless debug.
    if accelerator.is_main_process:
        make_experiment_dirs(experiment_root)
    accelerator.wait_for_everyone()

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    # In debug mode, limit dataset sizes to speed up iterations.
    if args.debug:
        DEBUG_TRAIN_STEPS = 50
        DEBUG_VAL_STEPS = 50
        train_dataset = torch.utils.data.Subset(train_dataset, range(DEBUG_TRAIN_STEPS))
        val_dataset = torch.utils.data.Subset(val_dataset, range(DEBUG_VAL_STEPS))

    model = build_model(config)

    learning_rate = float(config.TRAINER.TRAINING.LEARNING_RATE)
    weight_decay = float(config.TRAINER.TRAINING.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    num_workers = int(config.TRAINER.TRAINING.NUM_WORKERS)
    train_batch_size = int(config.TRAINER.TRAINING.BATCH_SIZE_PER_GPU)
    eval_batch_size = int(config.TRAINER.EVALUATION.BATCH_SIZE_PER_GPU)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=num_workers,
        batch_size=train_batch_size,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=eval_batch_size,
    )

    log_file = logs_dir / "train.log"
    if accelerator.is_main_process:
        if args.debug:
            logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
        else:
            logging.basicConfig(
                level=logging.INFO,
                handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
            )
    else:
        logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    if overwrite_experiment and accelerator.is_main_process:
        logger.info(
            "Existing experiment directory %s removed for a fresh run",
            experiment_root,
        )

    use_wandb = config.WANDB and not args.debug
    logger.info("use_wandb %s", use_wandb)

    if accelerator.is_main_process:
        model.count_parameters()
        resolved_config = print_config(config)
        logger.info(resolved_config)
        resolved_config_path = config_dir / "resolved_config.yaml"
        resolved_config_path.write_text(resolved_config)
        if use_wandb:
            wandb.init(project="FutureLatent", name=config_name)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler)

    if accelerator.is_main_process and use_wandb:
        wandb.watch(model, log="gradients", log_freq=500)

    trainer = build_trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        logger=logger,
        debug=args.debug,
        dump_dir=dump_dir if args.debug else None,
    )

    trainer.fit(
        train_dataloader,
        val_dataloader,
        epochs=1 if args.debug else None,
        checkpoint_dir=checkpoint_dir,
    )

    if accelerator.is_main_process and use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    # run with "accelerate launch --num_processes 2 --num_machines 1 -m src.main \
    #          --config_path configs/vjepa2_kinetics_400.yaml"
    main()
