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
from einops import rearrange
from .losses import get_criterion
from utils.video import (
    convert_video_tensor_to_mp4,
    save_mp4_video,
    save_tensor,
)

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
        """Advance the epoch counter by one.

        Definition: An "epoch" is one full pass over the training dataset
        (i.e., iterating through the entire train dataloader once). This
        counter is tracked internally as 0-based, while logs often display
        ``epoch + 1`` for human-friendly 1-based numbering. It is typically
        incremented after ``train_epoch`` completes.
        """
        self.epoch += 1

    def increment_step(self) -> None:
        """Advance the optimisation step counter by one.

        Definition: A "step" is a real optimisation update (a call to
        ``optimizer.step()``) that happens after gradient accumulation. In the
        training loop this is only incremented when
        ``accelerator.sync_gradients`` is ``True`` (i.e., once per accumulation
        cycle). This serves as the global training step used for logging and
        scheduler stepping.
        """
        self.step += 1

    def increment_micro_step(self) -> None:
        """Advance the raw micro step counter by one.

        Definition: A "micro step" counts every micro-batch processed (each
        dataloader iteration), regardless of whether gradients are synced. It
        increases on every batch seen before accumulation, so it can be larger
        than ``step`` when gradient accumulation is enabled.
        """
        self.micro_step += 1

    def start_timer(self) -> None:
        """Record the wall time start.

        Definition: ``wall_time_start`` stores the UNIX timestamp marking when
        timing began (initialised at trainer construction and can be reset).
        Used together with ``wall_time_total`` to measure elapsed real
        wall-clock time for the run.
        """
        self.wall_time_start = time.time()

    def update_wall_time(self) -> None:
        """Update the cumulative wall time.

        Definition: ``wall_time_total`` is the elapsed wall-clock time in
        seconds since ``wall_time_start``. This method refreshes that value
        using the current time. It is called after each ``train_step`` so that
        ``wall_time_total`` remains up-to-date for monitoring or logging.
        """
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
        visualisation_dataloader: Optional[Iterable[dict]] = None,
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
        self.visualisation_dataloader = visualisation_dataloader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        # Evaluation parameters
        self.eval_every = int(config.EVALUATION.EVAL_EVERY)
        self.eval_first = bool(config.EVALUATION.EVAL_FIRST)
        # Global configuration
        self.n_frames = int(config.N_FRAMES)

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

    def _record_grad_norm_event(self, total_norm: Any) -> None:
        """Update per-epoch gradient-related event counters from a norm value.

        Converts ``total_norm`` to a tensor ``gradient_norm`` and:
        - Increments ``_epoch_grad_nan_inf_count`` if non-finite.
        - Increments ``_epoch_grad_clip_norm_count`` if it exceeds ``max_grad_norm``.
        """
        gradient_norm = torch.as_tensor(total_norm)
        if not torch.isfinite(gradient_norm):
            self._epoch_grad_nan_inf_count += 1
        elif float(gradient_norm.item()) > float(self.max_grad_norm):
            # Increment only when clipping actually occurred
            self._epoch_grad_clip_norm_count += 1

    def _record_grad_value_event(self) -> None:
        """Update per-epoch gradient-related event counters from value clipping.

        Scans parameter gradients to:
        - Increment ``_epoch_grad_nan_inf_count`` if any gradient contains NaN/Inf.
        - Increment ``_epoch_grad_clip_value_count`` if any finite gradient value
          exceeds ``max_grad_value`` in absolute value (i.e., clipping would occur).
        """
        # First, detect any non-finite gradients
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                self._epoch_grad_nan_inf_count += 1
                return

        # If all gradients are finite, check whether value clipping will trigger
        threshold = float(self.max_grad_value)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if torch.any(torch.abs(p.grad) > threshold):
                self._epoch_grad_clip_value_count += 1
                break

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
                prediction, target, _, _ = self.model(batch)
                loss = self.criterion(prediction, target)
            self.check_loss_is_finite(loss)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                if self.max_grad_norm is not None:
                    total_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self._record_grad_norm_event(total_norm)
                if self.max_grad_value is not None:
                    # Record events analogous to norm-based clipping
                    self._record_grad_value_event()
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

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _aggregate_epoch_event_counts(self) -> tuple[int, int, int, int]:
        """Aggregate per-epoch event counters across processes.

        Returns a tuple of ints: (loss_nan_inf, grad_nan_inf, grad_clip_norm, grad_clip_value).
        """
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
            counts_all = gathered.view(-1, 4).sum(dim=0)
        else:
            counts_all = counts_local
        return (
            int(counts_all[0].item()),
            int(counts_all[1].item()),
            int(counts_all[2].item()),
            int(counts_all[3].item()),
        )

    def _log_epoch_event_counts(self, counts: tuple[int, int, int, int]) -> None:
        """Log aggregated per-epoch event counters to W&B and logger."""
        if not self.accelerator.is_main_process:
            return
        loss_nan_inf_count, grad_nan_inf_count, grad_clip_norm_count, grad_clip_value_count = counts

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

    def _log_epoch_train_summary(self, mean_loss: float, avg_step_time: float, epoch_time: float) -> None:
        """Log final epoch training summary to both W&B and the standard logger."""
        if not self.accelerator.is_main_process:
            return
        # W&B logging (mean loss + timing stats)
        if wandb.run is not None:
            wandb.log(
                {
                    f"train/avg_{self.criterion_name}_loss": mean_loss,
                    "train/avg_step_time": avg_step_time,
                    "train/epoch_time": epoch_time,
                },
                step=self.state.step,
            )
        # Classic logger summary
        self.logger.info(
            "Epoch %d | train/avg_%s_loss=%.6f | avg_step_time=%.4fs | epoch_time=%.2fs | step=%d",
            self.state.epoch + 1,
            self.criterion_name,
            mean_loss,
            avg_step_time,
            epoch_time,
            self.state.step,
        )
        self.logger.info("End train_epoch (epoch=%d)", self.state.epoch + 1)

    # -------------------------------
    # Validation logging helpers
    # -------------------------------
    def _aggregate_validation_mean(self, total_loss_local: float, num_batches_local: int) -> float:
        """Aggregate per-process validation loss to a global mean.

        Accepts the local process sum of losses and count of batches, reduces
        across processes, and returns the global mean loss.
        """
        sum_tensor = torch.tensor([total_loss_local], device=self.accelerator.device, dtype=torch.float32)
        cnt_tensor = torch.tensor([num_batches_local], device=self.accelerator.device, dtype=torch.long)
        if self.accelerator.num_processes > 1:
            sum_all = self.accelerator.gather(sum_tensor).sum()
            cnt_all = self.accelerator.gather(cnt_tensor).sum()
        else:
            sum_all = sum_tensor[0]
            cnt_all = cnt_tensor[0]
        global_sum = float(sum_all.item())
        global_count = max(int(cnt_all.item()), 1)
        return global_sum / global_count

    def _log_epoch_val_summary(self, mean_loss: float) -> None:
        """Log final validation summary to both W&B and the standard logger."""
        if not self.accelerator.is_main_process:
            return
        if wandb.run is not None:
            wandb.log({f"val/avg_{self.criterion_name}_loss": mean_loss}, step=self.state.step)
        self.logger.info(
            "Epoch %d | val/avg_%s_loss=%.6f | step=%d",
            self.state.epoch,
            self.criterion_name,
            mean_loss,
            self.state.step,
        )
        self.logger.info("End run_evaluation (epoch=%d)", self.state.epoch)

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
        # Show progress bar only on main process
        disable = not self.accelerator.is_main_process
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

        # Aggregate event counters across processes (sum) and log
        counts = self._aggregate_epoch_event_counts()
        self._log_epoch_event_counts(counts)

        # Log final epoch training loss to the configured logger
        self._log_epoch_train_summary(mean_loss, avg_step_time, epoch_time)
        self.state.increment_epoch()
        return mean_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_evaluation_step(self, batch: dict) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and gather latents for a single validation batch.

        Returns
        -------
        tuple
            (loss, pred_latents, target_latents, context_latents), where
            ``loss`` is returned as a plain float for lightweight accumulation.
        """
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Begin run_evaluation_step (epoch=%d)",
                self.state.epoch,
            )
        self.model.eval()
        with self.accelerator.autocast():
            pred_latents, target, context_latents, target_latents = self.model(batch)
            loss = self.criterion(pred_latents, target)
        value = float(loss.detach().item())
        if self.accelerator.is_main_process:
            self.logger.debug(
                "End run_evaluation_step (epoch=%d, loss=%.6f)",
                self.state.epoch,
                value,
            )
        return value, pred_latents, target_latents, context_latents
    
    @torch.inference_mode()
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

        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin run_evaluation (epoch=%d, step=%d)",
                self.state.epoch,
                self.state.step,
            )

        self.model.eval()

        total_loss_local = 0.0
        num_batches_local = 0

        # Show progress bar only on main process
        disable = not self.accelerator.is_main_process
        progress_bar = tqdm(
            self.val_dataloader,
            disable=disable,
            desc=f"Eval {self.state.epoch}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_loss, _, _, _ = self.run_evaluation_step(batch)
            total_loss_local += batch_loss
            num_batches_local += 1
            mean_so_far = total_loss_local / max(num_batches_local, 1)
            progress_bar.set_postfix({f"{self.criterion_name}_loss": mean_so_far}, refresh=False)

        # Aggregate across processes and log validation summary
        mean_loss = self._aggregate_validation_mean(total_loss_local, num_batches_local)
        self._log_epoch_val_summary(mean_loss)
        return mean_loss
    
    # ------------------------------------------------------------------
    # Visualisation utilities
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run_visualisation(self) -> None:
        """Export videos and latents for visual inspection."""
        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin run_visualisation (epoch=%d, step=%d)",
                self.state.epoch,
                self.state.step,
            )
        if not self.accelerator.is_main_process:
            return
        self.model.eval()
        # Silence Matplotlib animation INFO logs (only show our save messages)
        mpl_logger = logging.getLogger("matplotlib.animation")
        _prev_mpl_level = mpl_logger.level
        mpl_logger.setLevel(logging.WARNING)
        num_examples = 0
        try:
            example_idx = 0
            for batch in self.visualisation_dataloader:
                videos = batch["video"]
                with self.accelerator.autocast():
                    pred_latents, _, context_latents, target_latents = self.model(batch)

                for b in range(videos.shape[0]):
                    example_dir = self.dump_dir / f"example_{example_idx:02d}"
                    example_dir.mkdir(parents=True, exist_ok=True)

                    video = videos[b].detach().cpu()
                    save_tensor(video, example_dir / "video.pt", logger=self.logger)
                    frames, fps = convert_video_tensor_to_mp4(video)
                    save_mp4_video(frames, example_dir / "video", fps=fps, logger=self.logger)

                    save_tensor(context_latents[b], example_dir / "context_latents.pt", logger=self.logger)
                    save_tensor(target_latents[b], example_dir / "target_latents.pt", logger=self.logger)
                    save_tensor(pred_latents[b], example_dir / "prediction_latents.pt", logger=self.logger)

                    example_idx += 1
                    num_examples += 1
        finally:
            # Restore previous Matplotlib animation logger level
            mpl_logger.setLevel(_prev_mpl_level)
        if self.accelerator.is_main_process:
            self.logger.info(
                "End run_visualisation (epoch=%d, files=%d)",
                self.state.epoch,
                num_examples,
            )
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
            self.run_visualisation()
            self.run_evaluation()
        for _ in range(self.state.epoch, self.epochs):
            self.train_epoch()
            self.run_visualisation()
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
        visualisation_dataloader: Optional[Iterable[dict]] = None,
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
            visualisation_dataloader=visualisation_dataloader,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler,
            accelerator=accelerator,
            logger=logger,
            debug=debug,
            dump_dir=dump_dir,
        )


__all__ = ["Trainer", "TrainState", "DeterministicTrainer"]
