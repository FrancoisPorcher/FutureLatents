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
from diffusers import DDPMScheduler


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    print_config(config)
    dataset = build_dataset(config)
    breakpoint()

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

    num_train_timesteps = int(
        config['flow_matching'].get("num_train_timesteps", 1000)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    gradient_accumulation_steps = int(
        config["trainer"].get("gradient_accumulation_steps", 1)
    )
    find_unused_parameters = bool(
        config["trainer"].get("find_unused_parameters", False)
    )
    mixed_precision = str(config["trainer"].get("mixed_precision", "no"))

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=find_unused_parameters
            )
        ],
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        noise_scheduler=noise_scheduler,
        accelerator=accelerator,
        logger=logger,
    )

    eval_every = int(config.get("evaluation", {}).get("eval_every", 1))
    trainer.fit(dataloader, epochs=1, eval_every=eval_every)


if __name__ == "__main__":
    # run with "accelerate launch --num_processes 2 --num_machines 1 -m src.main \
    #          --config_path configs/vjepa2_kinetics_400.yaml"
    main()
