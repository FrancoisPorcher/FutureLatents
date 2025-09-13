"""Utilities for training FutureLatents models.

This module defines a small ``Trainer`` class that provides convenience
wrappers around the typical PyTorch training loop.  It purposefully keeps the
implementation lightweight so that it can serve as a starting point for more
feature rich trainers in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Callable, Dict, Any
import contextlib
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm
import logging
import wandb


def get_criterion(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a loss function for ``name``."""

    loss_map = {"mse": F.mse_loss, "l1": F.l1_loss}
    criterion = loss_map.get(name.lower())
    if criterion is None:  # pragma: no cover - config error
        raise ValueError("LOSS must be 'mse' or 'l1'")
    return criterion


@dataclass
class TrainState:
    """Container for tracking training state."""

    # Progress
    epoch: int = 0                      # 0-based
    step: int = 0                       # global optimization steps (after grad accumulation)
    micro_step: int = 0                 # raw batches seen (before accumulation)

    # Timing
    wall_time_start: float = 0.0        # set at trainer init
    wall_time_total: float = 0.0        # cumulative seconds

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def increment_epoch(self) -> None:
        """Advance the epoch counter by one."""
        self.epoch += 1

    def increment_step(self) -> None:
        """Advance the optimisation step counter by one."""
        self.step += 1

    def increment_micro_step(self) -> None:
        """Advance the raw micro step counter by one."""
        self.micro_step += 1

    def start_timer(self) -> None:
        """Record the wall time start."""
        self.wall_time_start = time.time()

    def update_wall_time(self) -> None:
        """Update the cumulative wall time."""
        self.wall_time_total = time.time() - self.wall_time_start



class Trainer:
    """Basic PyTorch training helper.

    Parameters
    ----------
    model:
        The ``nn.Module`` to optimise.
    optimizer:
        Optimiser responsible for updating the model parameters.
    scheduler:
        Optional learning rate scheduler stepped after each optimisation step.
    max_grad_norm:
        If specified, clip gradients to this maximum L2 norm.
    max_grad_value:
        If specified, clip gradients to this maximum absolute value.
    config:
        Configuration object holding training and evaluation parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: object,
        train_dataloader: Optional[Iterable[dict]] = None,
        val_dataloader: Optional[Iterable[dict]] = None,
        checkpoint_dir: Optional[Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
        dump_dir: Optional[Path] = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = TrainState()
        self.state.start_timer()
        self.logger = logger or logging.getLogger(__name__)
        self.debug = debug
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        self.config = config
        # Persist common runtime objects
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        # Evaluation parameters
        self.eval_every = int(config.EVALUATION.EVAL_EVERY)
        self.eval_first = bool(config.EVALUATION.EVAL_FIRST)
        
        # Training parameters
        self.epochs = int(config.TRAINING.EPOCHS) if not self.debug else 1
        self.max_grad_norm = config.TRAINING.MAX_GRAD_NORM
        self.max_grad_value = config.TRAINING.MAX_GRAD_VALUE
        self.criterion_name = str(config.TRAINING.LOSS).lower()
        self.criterion = get_criterion(self.criterion_name)
        self.save_every = int(config.TRAINING.SAVE_EVERY)
        

        if self.max_grad_norm is None and self.max_grad_value is None:
            raise ValueError(
                "Either MAX_GRAD_NORM or MAX_GRAD_VALUE must be specified in the config"
            )
        if self.max_grad_norm is not None and self.max_grad_value is not None:
            raise ValueError(
                "Only one of MAX_GRAD_NORM or MAX_GRAD_VALUE may be specified"
        )

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> float:
        """Perform a single optimisation step."""

        self.model.train()
        if self.accelerator is None:
            raise NotImplementedError(
                "Trainer.train_step() requires an accelerator."
            )
        self.state.increment_micro_step()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(batch, return_norms=self.debug)
                if self.debug:
                    prediction, target, norms = outputs
                else:
                    prediction, target = outputs
                    norms = None
                loss = self.criterion(prediction, target)
            self.accelerator.backward(loss)
            # Only clip gradients on real optimisation steps (post accumulation)
            if self.accelerator.sync_gradients:
                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                if self.max_grad_value is not None:
                    self.accelerator.clip_grad_value_(
                        self.model.parameters(), self.max_grad_value
                    )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        loss_value = loss.detach()
        if self.accelerator.num_processes > 1:
            loss_value = self.accelerator.gather(loss_value).mean()
        # Log and advance the global step only on real optimisation steps
        if self.accelerator.sync_gradients:
            if wandb.run is not None and self.accelerator.is_main_process:
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {f"train/{self.criterion_name}_loss": loss_value.item(), "train/lr": lr},
                    step=self.state.step,
                )
            self.state.increment_step()
        self.state.update_wall_time()
        return loss_value.item()



    def train_epoch(
        self,
    ) -> float:
        """Iterate over ``dataloader`` once, log metrics and return the mean loss."""

        total_loss = 0.0
        disable = (
            self.accelerator is not None
            and not self.accelerator.is_main_process
        )
        progress_bar = tqdm(
            self.train_dataloader,
            disable=disable,
            desc=f"Epoch {self.state.epoch + 1}",
        )
        for batch in progress_bar:
            total_loss += self.train_step(batch)
            progress_bar.set_postfix(
                {f"{self.criterion_name}_loss": total_loss / max(progress_bar.n, 1)},
                refresh=False,
            )
            if self.debug and progress_bar.n >= 10:
                break
        mean_loss = total_loss / max(progress_bar.n, 1)
        if wandb.run is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            wandb.log({f"train/avg_{self.criterion_name}_loss": mean_loss}, step=self.state.step)
        return mean_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_evaluation():
        pass
    
    # ------------------------------------------------------------------
    # Visualisation utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualise(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------
    def save_checkpoint(self) -> None:
        pass


    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def fit(self):
        if self.eval_first:
            self.run_evaluation()
        for _ in range(self.state.epoch, self.epochs):
            self.train_epoch()
            self.state.increment_epoch()
            self.run_evaluation()
            self.save_checkpoint()
        self.run_evaluation()
            
class DeterministicTrainer(Trainer):
    """Trainer variant for deterministic models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: object,
        train_dataloader: Optional[Iterable[dict]] = None,
        val_dataloader: Optional[Iterable[dict]] = None,
        checkpoint_dir: Optional[Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
        dump_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler,
            accelerator=accelerator,
            logger=logger,
            debug=debug,
            dump_dir=dump_dir,
        )


__all__ = ["Trainer", "TrainState", "DeterministicTrainer", "get_criterion"]
