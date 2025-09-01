from pathlib import Path
import logging

# Import the project package relative to this module so that running
# ``python -m src.main`` works without requiring ``src`` on the
# ``PYTHONPATH``.
from models.latent_video_model import LatentVideoModel
from datasets import build_dataset
from training.trainer import Trainer
from utils.parser import create_parser
from utils.config import load_config, print_config
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    
    dataset = build_dataset(config)

    model = LatentVideoModel(config)

    learning_rate = float(config.TRAINER.LEARNING_RATE)
    weight_decay = float(config.TRAINER.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    num_workers = int(config.TRAINER.NUM_WORKERS)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=num_workers
    )

    gradient_accumulation_steps = int(config.TRAINER.GRADIENT_ACCUMULATION_STEPS)
    find_unused_parameters = bool(config.TRAINER.FIND_UNUSED_PARAMETERS)
    mixed_precision = str(config.TRAINER.MIXED_PRECISION)

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
    
    if accelerator.is_main_process:
        model.count_parameters()
        print_config(config)
        
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)



    max_grad_norm = config.TRAINER.MAX_GRAD_NORM
    max_grad_value = config.TRAINER.MAX_GRAD_VALUE
    if max_grad_norm is None and max_grad_value is None:
        raise ValueError(
            "Either MAX_GRAD_NORM or MAX_GRAD_VALUE must be specified in the config"
        )
    if max_grad_norm is not None and max_grad_value is not None:
        raise ValueError(
            "Only one of MAX_GRAD_NORM or MAX_GRAD_VALUE may be specified"
        )
        

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        accelerator=accelerator,
        max_grad_norm=float(max_grad_norm) if max_grad_norm is not None else None,
        max_grad_value=float(max_grad_value) if max_grad_value is not None else None,
        logger=logger,
    )

    eval_every = int(config.EVALUATION.EVAL_EVERY)
    trainer.fit(dataloader, epochs=1, eval_every=eval_every)


if __name__ == "__main__":
    # run with "accelerate launch --num_processes 2 --num_machines 1 -m src.main \
    #          --config_path configs/vjepa2_kinetics_400.yaml"
    main()
