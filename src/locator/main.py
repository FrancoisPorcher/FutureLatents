from pathlib import Path
import logging
import shutil
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from models import build_model
from datasets import build_dataset
from training import build_trainer
from utils.parser import create_parser
from utils.config import load_config, print_config
from utils.filesystem import make_experiment_dirs



def main() -> None:
    """Entry point for the locator training application."""
    parser = create_parser()
    args = parser.parse_args()
    

    config_path = Path(args.config_path)
    config = load_config(config_path)
    config_name = config_path.stem
    
    breakpoint()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=find_unused_parameters
            )
        ],
    )
    if accelerator.device.type == "cuda":
        # Ensure a concrete CUDA device index is selected
        device_index = accelerator.device.index
        if device_index is None:
            device_index = 0
        torch.cuda.set_device(device_index)
    
    with accelerator.main_process_first():
        model = build_model(config)
    accelerator.wait_for_everyone()
    
    breakpoint()
    

if __name__ == "__main__":
    # run with "accelerate launch --num_processes 1 --num_machines 1 -m src.locator.main \
    #          --config_path configs/references/dinov3_locator_bouncing_shapes.yaml"
    main()
