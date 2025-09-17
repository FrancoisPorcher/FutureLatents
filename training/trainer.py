"""Utilities for training FutureLatents models.

This module defines a small ``Trainer`` class that provides convenience
wrappers around the typical PyTorch training loop.  It purposefully keeps the
implementation lightweight so that it can serve as a starting point for more
feature rich trainers in the future.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from accelerate import Accelerator
from accelerate.utils import tqdm
from einops import rearrange
import torch
import wandb
from omegaconf import OmegaConf

from utils.video import (
    convert_video_tensor_to_mp4,
    save_batch,
    save_mp4_video,
    save_tensor,
    save_visualisation_tensors,
)
from utils.pca import pca_latents_to_video_tensors
from .losses import compute_locator_step_losses, get_criterion
from .trainer_logging import TrainerLoggingMixin
from utils.latents import infer_latent_dimensions
from utils.torch_utils import _move_batch_to_device

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



class Trainer(TrainerLoggingMixin):
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
        self.dump_dir = dump_dir
        self.config = config
        # Persist common runtime objects
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.visualisation_dataloader = visualisation_dataloader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        # Evaluation parameters
        self.eval_every = config.EVALUATION.EVAL_EVERY
        self.eval_first = config.EVALUATION.EVAL_FIRST
        # Global configuration
        self.n_frames = config.N_FRAMES
        # Latent/layout dimensions inferred from full model config
        self.n_ctx_lat, self.n_tgt_lat, self.spatial = infer_latent_dimensions(self.model.config)

        # Training parameters
        self.epochs = config.TRAINING.EPOCHS if not self.debug else 1
        self.max_grad_norm = config.TRAINING.MAX_GRAD_NORM
        self.max_grad_value = config.TRAINING.MAX_GRAD_VALUE
        self.criterion_name = str(config.TRAINING.LOSS).lower()
        self.criterion = get_criterion(self.criterion_name)
        self.save_every = config.TRAINING.SAVE_EVERY

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
        self._log_train_step_begin()
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

        return self._log_train_step_metrics(loss)

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

        start_step = self.state.step
        self._log_train_epoch_begin(start_step)

        total_loss = 0.0
        epoch_start_time = time.time()
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
        self._log_run_evaluation_step_begin()
        self.model.eval()
        with self.accelerator.autocast():
            pred_latents, target, context_latents, target_latents = self.model(batch)
            loss = self.criterion(pred_latents, target)
        value = float(loss.detach().item())
        self._log_run_evaluation_step_end({f"val/avg_{self.criterion_name}_loss": value})
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
                self._log_run_evaluation_skip(self.eval_every)
                return None

        self._log_run_evaluation_begin()

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
        self._log_visualisation_begin()
        if not self.accelerator.is_main_process:
            return
        self.model.eval()
        # Silence Matplotlib animation INFO logs (only show our save messages)
        mpl_logger = logging.getLogger("matplotlib.animation")
        _prev_mpl_level = mpl_logger.level
        mpl_logger.setLevel(logging.WARNING)
        num_examples = 0
        example_idx = 0
        for batch in self.visualisation_dataloader:
            videos = batch["video"] # (B, T, C, H, W)
            with self.accelerator.autocast():
                prediction_latents, _, context_latents, target_latents = self.model(batch)

            for b in range(videos.shape[0]):
                example_dir = self.dump_dir / f"epoch_{self.state.epoch:02d}" / f"example_{example_idx:02d}"
                example_dir.mkdir(parents=True, exist_ok=True)

                video = videos[b].detach().cpu()
                frames, fps = convert_video_tensor_to_mp4(video)
                save_mp4_video(frames, example_dir / "video", fps=fps, logger=self.logger)
                save_visualisation_tensors(
                    video,
                    context_latents[b],
                    target_latents[b],
                    prediction_latents[b],
                    example_dir,
                    logger=self.logger,
                )

                # PCA projections -> RGB video tensors using helper
                # Context latents: fit PCA on context and reshape with T=n_ctx_lat
                c_vid, t_vid, p_vid = pca_latents_to_video_tensors(
                    context_latents=context_latents[b],
                    target_latents=target_latents[b],
                    prediction_latents=prediction_latents[b],
                    n_ctx_lat=self.n_ctx_lat,
                    n_tgt_lat=self.n_tgt_lat,
                    H=self.spatial,
                    W=self.spatial,
                    fit_on="context",
                )

                # Save PCA tensors
                save_tensor(c_vid, example_dir / "context_latents_pca.pt", logger=self.logger)
                save_tensor(t_vid, example_dir / "target_latents_pca.pt", logger=self.logger)
                save_tensor(p_vid, example_dir / "prediction_latents_pca.pt", logger=self.logger)

                # Save PCA videos (MP4)
                frames_c, fps_c = convert_video_tensor_to_mp4(c_vid)
                frames_t, fps_t = convert_video_tensor_to_mp4(t_vid)
                frames_p, fps_p = convert_video_tensor_to_mp4(p_vid)
                save_mp4_video(frames_c, example_dir / "context_latents_pca", fps=fps_c, logger=self.logger)
                save_mp4_video(frames_t, example_dir / "target_latents_pca", fps=fps_t, logger=self.logger)
                save_mp4_video(frames_p, example_dir / "prediction_latents_pca", fps=fps_p, logger=self.logger)

                example_idx += 1
                num_examples += 1
        mpl_logger.setLevel(_prev_mpl_level)
        self._log_visualisation_end(num_examples)
    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------
    def _model_state_for_checkpoint(self) -> dict[str, torch.Tensor]:
        state = self.accelerator.get_state_dict(self.model)
        model = self.accelerator.unwrap_model(self.model)
        encoder = getattr(model, "encoder", None)
        if isinstance(encoder, torch.nn.Module):
            state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        return state

    def save_checkpoint(self) -> None:
        if self.save_every in (None, 0):
            return

        try:
            save_interval = int(self.save_every)
        except (TypeError, ValueError):
            return

        if save_interval <= 0:
            return

        if self.state.epoch == 0 or (self.state.epoch % save_interval) != 0:
            return

        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / "checkpoint.py"

        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "epoch": int(self.state.epoch),
                "checkpoint": self._model_state_for_checkpoint(),
                "optimizer": self.optimizer.state_dict(),
                "learning_rate_scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "config": OmegaConf.to_container(
                    self.config,
                    resolve=True,
                    enum_to_str=True,
                ),
            }
            torch.save(payload, checkpoint_path)
            self.logger.info("Saved checkpoint to %s", checkpoint_path)

        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None, strict: bool = False) -> Optional[int]:
        if checkpoint_path is None:
            if self.checkpoint_dir is None:
                return None
            checkpoint_path = self.checkpoint_dir / "checkpoint.py"

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model_state = checkpoint.get("checkpoint")
        if model_state is not None:
            missing, unexpected = self.accelerator.unwrap_model(self.model).load_state_dict(
                model_state,
                strict=strict,
            )
            if self.accelerator.is_main_process and (missing or unexpected):
                self.logger.warning(
                    "Checkpoint load with missing=%s unexpected=%s",
                    missing,
                    unexpected,
                )

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get("learning_rate_scheduler")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        config_data = checkpoint.get("config")
        if config_data is not None:
            config = OmegaConf.create(config_data)
            self.config = config
            self.eval_every = config.EVALUATION.EVAL_EVERY
            self.eval_first = config.EVALUATION.EVAL_FIRST
            self.n_frames = config.N_FRAMES
            self.epochs = config.TRAINING.EPOCHS if not self.debug else 1
            self.max_grad_norm = config.TRAINING.MAX_GRAD_NORM
            self.max_grad_value = config.TRAINING.MAX_GRAD_VALUE
            self.criterion_name = str(config.TRAINING.LOSS).lower()
            self.criterion = get_criterion(self.criterion_name)
            self.save_every = config.TRAINING.SAVE_EVERY

        loaded_epoch = int(checkpoint.get("epoch", 0))
        self.state.epoch = loaded_epoch
        self.state.step = 0
        self.state.micro_step = 0
        self.state.start_timer()

        self.accelerator.wait_for_everyone()

        return loaded_epoch

    def run_evaluation_from_checkpoint(
        self, checkpoint_path: Optional[Path] = None, strict: bool = False
    ) -> Optional[float]:
        self.load_checkpoint(checkpoint_path=checkpoint_path, strict=strict)
        return self.run_evaluation()


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
        
class LocatorTrainer(Trainer):
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
        self.heatmap_criterion_name = config.TRAINING.HEATMAP_LOSS
        self.heatmap_criterion = get_criterion(self.heatmap_criterion_name)

    @staticmethod
    def _reshape_heatmap_target(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if target.shape == reference.shape:
            return target
        return target.reshape(reference.shape)

    @staticmethod
    def _prepare_heatmap_cross_entropy_inputs(
        logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten logits/targets into ``[B, N]`` and derive class indices."""
        logits_flat = logits.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        indices = target_flat.argmax(dim=1).long()
        return logits_flat, indices

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> float:
        """Single optimisation step for the locator.

        Coordinates are supervised using normalised targets in ``[0, 1]``.
        """
        self._log_train_step_begin()

        self.model.train()
        self.state.increment_micro_step()

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                outputs = self.model(batch)
                loss, coord_loss, heatmap_loss = compute_locator_step_losses(
                    batch,
                    outputs,
                    self.criterion,
                    self.heatmap_criterion,
                    self._reshape_heatmap_target,
                    self._prepare_heatmap_cross_entropy_inputs,
                )

            self.check_loss_is_finite(loss)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.max_grad_norm is not None:
                    total_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self._record_grad_norm_event(total_norm)
                if self.max_grad_value is not None:
                    self._record_grad_value_event()
                    self.accelerator.clip_grad_value_(self.model.parameters(), self.max_grad_value)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return self._log_train_step_metrics(loss, coord_loss, heatmap_loss)

    def train_epoch(self) -> float:
        """One full pass over the training set for the locator."""
        # Reset counters and log start
        self.reset_epoch_counters()

        start_step = self.state.step
        self._log_train_epoch_begin(start_step)

        total_loss = 0.0
        epoch_start_time = time.time()

        disable = not self.accelerator.is_main_process
        progress_bar = tqdm(
            self.train_dataloader,
            disable=disable,
            desc=f"Epoch {self.state.epoch + 1}",
        )
        for batch in progress_bar:
            total_loss += self.train_step(batch)
            mean_so_far = total_loss / max(progress_bar.n, 1)
            progress_bar.set_postfix({f"{self.criterion_name}_loss": mean_so_far}, refresh=False)

        mean_loss = total_loss / max(progress_bar.n, 1)
        epoch_time = time.time() - epoch_start_time
        num_opt_steps = max(self.state.step - start_step, 1)
        avg_step_time = epoch_time / num_opt_steps

        counts = self._aggregate_epoch_event_counts()
        self._log_epoch_event_counts(counts)

        self._log_epoch_train_summary(mean_loss, avg_step_time, epoch_time)
        self.state.increment_epoch()
        return mean_loss

    # ------------------------------------------------------------------
    # Visualisation utilities
    # ------------------------------------------------------------------
    @torch.inference_mode()
    # Overwrite run_visualisation from parent Trainer class
    def run_visualisation(self) -> None:
        self._log_visualisation_begin()
        if not self.accelerator.is_main_process:
            return
        self.model.eval()
        epoch_dir = self.dump_dir / f"epoch_{self.state.epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        num_examples = 0
        for batch_idx, batch in enumerate(self.visualisation_dataloader):
            batch = _move_batch_to_device(batch=batch, device=self.accelerator.device)
            with self.accelerator.autocast():
                outputs = self.model(batch)
            save_batch(
                batch=batch,
                outputs=outputs,
                out_dir=epoch_dir,
                batch_idx=batch_idx,
                logger=self.logger,
            )
            num_examples += batch["image"].shape[0]
        self._log_visualisation_end(num_examples)

    # ------------------------------------------------------------------
    # Evaluation step override (loss on normalised positions)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_evaluation_step(self, batch: dict) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._log_run_evaluation_step_begin()
        self.model.eval()
        with self.accelerator.autocast():
            outputs = self.model(batch)
            square_pred = outputs["square_pred"]
            circle_pred = outputs["circle_pred"]
            square_tgt = batch["target_square_position_normalized"].to(square_pred.dtype)
            circle_tgt = batch["target_circle_position_normalized"].to(circle_pred.dtype)
            coord_loss = self.criterion(square_pred, square_tgt) + self.criterion(circle_pred, circle_tgt)
            square_pred_px = outputs["denormalized_square_pred"]
            circle_pred_px = outputs["denormalized_circle_pred"]
            square_tgt_px = batch["target_square_position_pixel"].to(square_pred_px.dtype)
            circle_tgt_px = batch["target_circle_position_pixel"].to(circle_pred_px.dtype)
            pixel_loss = self.criterion(square_pred_px, square_tgt_px) + self.criterion(circle_pred_px, circle_tgt_px)
            square_logits = outputs["square_heatmap_logits"]
            circle_logits = outputs["circle_heatmap_logits"]
            square_patch = self._reshape_heatmap_target(
                batch["target_square_heatmap_patch"].to(square_logits),
                square_logits,
            )
            circle_patch = self._reshape_heatmap_target(
                batch["target_circle_heatmap_patch"].to(circle_logits),
                circle_logits,
            )
            square_logits_flat, square_target_idx = self._prepare_heatmap_cross_entropy_inputs(
                square_logits,
                square_patch,
            )
            circle_logits_flat, circle_target_idx = self._prepare_heatmap_cross_entropy_inputs(
                circle_logits,
                circle_patch,
            )
            heatmap_loss = self.heatmap_criterion(square_logits_flat, square_target_idx) + self.heatmap_criterion(circle_logits_flat, circle_target_idx)

        pixel_loss = pixel_loss.detach()
        heatmap_loss = heatmap_loss.detach()
        value = float(coord_loss.detach().item())
        heatmap_value = float(heatmap_loss.item())
        pixel_value = float(pixel_loss.item())
        metric_norm = f"val/avg_{self.criterion_name}_loss_normalized_v1"
        metric_pixel = f"val/avg_{self.criterion_name}_loss_pixel_v1"
        metric_heatmap = f"val/avg_{self.heatmap_criterion_name}_loss_patch_v1"
        self._log_run_evaluation_step_end(
            {
                metric_norm: value,
                metric_pixel: pixel_value,
                metric_heatmap: heatmap_value,
            }
        )
        # Return placeholders for latents to match base signature
        dummy = torch.empty(0, device=self.accelerator.device)
        return value, pixel_loss, heatmap_loss, dummy

    @torch.inference_mode()
    def run_evaluation(self) -> Optional[float]:
        """Run evaluation and log both normalized and pixel-space losses."""
        if self.eval_every is not None and self.eval_every > 1:
            if (self.state.epoch % self.eval_every) != 0:
                self._log_run_evaluation_skip(self.eval_every)
                return None

        self._log_run_evaluation_begin()

        self.model.eval()

        total_loss_local = 0.0
        total_pixel_loss_local = 0.0
        total_heatmap_loss_local = 0.0
        num_batches_local = 0

        disable = not self.accelerator.is_main_process
        progress_bar = tqdm(
            self.val_dataloader,
            disable=disable,
            desc=f"Eval {self.state.epoch}",
        )

        for batch in progress_bar:
            batch_loss, pixel_loss_tensor, heatmap_loss_tensor, _ = self.run_evaluation_step(batch)
            total_loss_local += batch_loss
            total_pixel_loss_local += float(pixel_loss_tensor.item())
            total_heatmap_loss_local += float(heatmap_loss_tensor.item())
            num_batches_local += 1

            mean_norm = total_loss_local / max(num_batches_local, 1)
            mean_pixel = total_pixel_loss_local / max(num_batches_local, 1)
            mean_heatmap = total_heatmap_loss_local / max(num_batches_local, 1)
            progress_bar.set_postfix(
                {
                    f"{self.criterion_name}_loss_normalized_v1": mean_norm,
                    f"{self.criterion_name}_loss_pixel_v1": mean_pixel,
                    f"{self.heatmap_criterion_name}_loss_patch_v1": mean_heatmap,
                },
                refresh=False,
            )

        mean_loss = self._aggregate_validation_mean(total_loss_local, num_batches_local)
        mean_pixel_loss = self._aggregate_validation_mean(total_pixel_loss_local, num_batches_local)
        mean_heatmap_loss = self._aggregate_validation_mean(total_heatmap_loss_local, num_batches_local)
        self._log_locator_val_summary(mean_loss, mean_pixel_loss, mean_heatmap_loss)
        return mean_loss
__all__ = ["Trainer", "TrainState", "DeterministicTrainer", "LocatorTrainer"]
