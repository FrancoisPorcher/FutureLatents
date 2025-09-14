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
from accelerate import Accelerator
from accelerate.utils import tqdm
import logging
import wandb
from .losses import get_criterion

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

        # Event counters (reset each epoch)
        self._epoch_loss_nan_inf_count: int = 0
        self._epoch_grad_nan_inf_count: int = 0
        self._epoch_grad_clip_norm_count: int = 0
        self._epoch_grad_clip_value_count: int = 0

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
    def check_loss_is_finite(self, loss: torch.Tensor) -> bool:
        """Check whether ``loss`` is finite and log if not.

        Increments ``_epoch_loss_nan_inf_count`` and logs an error on the main
        process when the loss contains NaN/Inf. Returns ``True`` when finite,
        otherwise ``False``.
        """
        if not torch.isfinite(loss):
            self._epoch_loss_nan_inf_count += 1
            if self.accelerator.is_main_process:
                self.logger.error(
                    "Step %d (micro %d): NaN/Inf loss detected: %s",
                    self.state.step,
                    self.state.micro_step,
                    str(loss.detach().item() if loss.numel() == 1 else "tensor"),
                )
            return False
        return True

    def train_step(self, batch: dict) -> float:
        """Perform a single optimisation step."""
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Begin train_step (epoch=%d, step=%d, micro_step=%d)",
                self.state.epoch,
                self.state.step,
                self.state.micro_step + 1,  # increment happens just below
            )
        self.model.train()
        self.state.increment_micro_step()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(batch, return_norms=self.debug)
                prediction, target, norms = outputs
                loss = self.criterion(prediction, target)
            self.check_loss_is_finite(loss)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                if self.max_grad_norm is not None:
                    total_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    # Increment only when clipping actually occurred
                    tn = torch.as_tensor(total_norm)
                    if not torch.isfinite(tn):
                        self._epoch_grad_nan_inf_count += 1
                    elif float(tn.item()) > float(self.max_grad_norm):
                        self._epoch_grad_clip_norm_count += 1
                if self.max_grad_value is not None:
                    self.accelerator.clip_grad_value_(self.model.parameters(), self.max_grad_value)
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
        if self.accelerator.is_main_process:
            self.logger.debug(
                "End train_step (epoch=%d, step=%d, micro_step=%d, loss=%.6f)",
                self.state.epoch,
                self.state.step,
                self.state.micro_step,
                float(loss_value.item()),
            )
        return loss_value.item()

    def reset_epoch_counters(self) -> None:
        """Reset per-epoch event counters."""
        self._epoch_loss_nan_inf_count = 0
        self._epoch_grad_nan_inf_count = 0
        self._epoch_grad_clip_norm_count = 0
        self._epoch_grad_clip_value_count = 0
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Reset epoch counters (epoch=%d)",
                self.state.epoch + 1,
            )

    def train_epoch(
        self,
    ) -> float:
        """Iterate over ``dataloader`` once, log metrics and return the mean loss."""

        # Reset epoch counters
        self.reset_epoch_counters()

        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin train_epoch (epoch=%d, start_step=%d)",
                self.state.epoch + 1,
                self.state.step,
            )

        total_loss = 0.0
        epoch_start_time = time.time()
        start_step = self.state.step
        disable = self.accelerator.is_main_process
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
        mean_loss = total_loss / max(progress_bar.n, 1)
        epoch_time = time.time() - epoch_start_time
        num_opt_steps = max(self.state.step - start_step, 1)
        avg_step_time = epoch_time / num_opt_steps
        if wandb.run is not None and self.accelerator.is_main_process:
            wandb.log({f"train/avg_{self.criterion_name}_loss": mean_loss}, step=self.state.step)

        # Aggregate event counters across processes (sum) and log
        counts_local = torch.tensor(
            [
                self._epoch_loss_nan_inf_count,
                self._epoch_grad_nan_inf_count,
                self._epoch_grad_clip_norm_count,
                self._epoch_grad_clip_value_count,
            ],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        if self.accelerator.num_processes > 1:
            gathered = self.accelerator.gather(counts_local)
            try:
                counts_all = gathered.view(-1, 4).sum(dim=0)
            except RuntimeError:
                counts_all = gathered.reshape(-1, 4).sum(dim=0)
        else:
            counts_all = counts_local

        if self.accelerator.is_main_process:
            loss_nan_inf_count = int(counts_all[0].item())
            grad_nan_inf_count = int(counts_all[1].item())
            grad_clip_norm_count = int(counts_all[2].item())
            grad_clip_value_count = int(counts_all[3].item())

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/events/loss_nan_or_inf_count": loss_nan_inf_count,
                        "train/events/grad_nan_or_inf_count": grad_nan_inf_count,
                        "train/events/grad_clip_norm_count": grad_clip_norm_count,
                        "train/events/grad_clip_value_count": grad_clip_value_count,
                    },
                    step=self.state.step,
                )

            self.logger.info(
                "Epoch %d | events: loss_nan_or_inf=%d, grad_nan_or_inf=%d, clip_norm=%d, clip_value=%d",
                self.state.epoch + 1,
                loss_nan_inf_count,
                grad_nan_inf_count,
                grad_clip_norm_count,
                grad_clip_value_count,
            )

        # Log final epoch training loss to the configured logger
        if self.accelerator.is_main_process:
            self.logger.info(
                "Epoch %d | train/avg_%s_loss=%.6f | avg_step_time=%.4fs | epoch_time=%.2fs | step=%d",
                self.state.epoch + 1,
                self.criterion_name,
                mean_loss,
                avg_step_time,
                epoch_time,
                self.state.step,
            )
            self.logger.info(
                "End train_epoch (epoch=%d)",
                self.state.epoch + 1,
            )
        self.state.increment_epoch()
        return mean_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_evaluation_step(self, batch: dict) -> float:
        """Compute loss for a single validation batch.

        Returns the scalar loss value (local process only; no gathering).
        """
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Begin run_evaluation_step (epoch=%d)",
                self.state.epoch,
            )
        self.model.eval()
        with self.accelerator.autocast():
            outputs = self.model(batch, return_norms=False)
            prediction, target = outputs
            loss = self.criterion(prediction, target)
        # Return plain float for lightweight local accumulation
        value = float(loss.detach().item())
        if self.accelerator.is_main_process:
            self.logger.debug(
                "End run_evaluation_step (epoch=%d, loss=%.6f)",
                self.state.epoch,
                value,
            )
        return value
    
    @torch.no_grad()
    def run_evaluation(self) -> Optional[float]:
        """Run evaluation over the validation dataloader and log mean loss.

        Efficiently aggregates loss across processes once at the end.
        Returns the global mean loss (or ``None`` if evaluation is skipped).
        """
        # Respect evaluation cadence if configured
        if self.eval_every is not None and self.eval_every > 1:
            if (self.state.epoch % self.eval_every) != 0:
                if self.accelerator.is_main_process:
                    self.logger.info(
                        "Skip run_evaluation (epoch=%d): eval_every=%d",
                        self.state.epoch,
                        self.eval_every,
                    )
                return None

        if self.val_dataloader is None:
            if self.accelerator.is_main_process:
                self.logger.info(
                    "Skip run_evaluation (epoch=%d): no val_dataloader",
                    self.state.epoch,
                )
            return None

        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin run_evaluation (epoch=%d, step=%d)",
                self.state.epoch,
                self.state.step,
            )

        self.model.eval()

        total_loss_local = 0.0
        num_batches_local = 0

        disable = self.accelerator.is_main_process
        progress_bar = tqdm(
            self.val_dataloader,
            disable=disable,
            desc=f"Eval {self.state.epoch}",
        )

        for batch in progress_bar:
            batch_loss = self.run_evaluation_step(batch)
            total_loss_local += batch_loss
            num_batches_local += 1
            # Lightweight live display on the progress bar
            mean_so_far = total_loss_local / max(num_batches_local, 1)
            progress_bar.set_postfix({f"{self.criterion_name}_loss": mean_so_far}, refresh=False)

        # Aggregate across processes once at the end for efficiency
        sum_tensor = torch.tensor([total_loss_local], device=self.accelerator.device, dtype=torch.float32)
        cnt_tensor = torch.tensor([num_batches_local], device=self.accelerator.device, dtype=torch.long)
        if self.accelerator.num_processes > 1:
            sum_all = self.accelerator.gather(sum_tensor).sum()
            cnt_all = self.accelerator.gather(cnt_tensor).sum()
        else:
            sum_all = sum_tensor
            cnt_all = cnt_tensor

        global_sum = float(sum_all.item())
        global_count = int(cnt_all.item()) if cnt_all.numel() == 1 else int(cnt_all.sum().item())
        mean_loss = global_sum / max(global_count, 1)

        if wandb.run is not None and self.accelerator.is_main_process:
            wandb.log({f"val/avg_{self.criterion_name}_loss": mean_loss}, step=self.state.step)

        if self.accelerator.is_main_process:
            self.logger.info(
                "Epoch %d | val/avg_%s_loss=%.6f | step=%d",
                self.state.epoch,
                self.criterion_name,
                mean_loss,
                self.state.step,
            )
            self.logger.info(
                "End run_evaluation (epoch=%d)",
                self.state.epoch,
            )

        return mean_loss
    
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
            self.run_evaluation()
            self.save_checkpoint()
        
            
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


__all__ = ["Trainer", "TrainState", "DeterministicTrainer"]
