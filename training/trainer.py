"""Utilities for training FutureLatents models.

This module defines a small ``Trainer`` class that provides convenience
wrappers around the typical PyTorch training loop.  It purposefully keeps the
implementation lightweight so that it can serve as a starting point for more
feature rich trainers in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm
import logging
import wandb


def get_criterion(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a loss function given its name.

    Parameters
    ----------
    name:
        Name of the loss. Currently supports ``"mse"`` and ``"l1"``.
    """

    loss_map = {"mse": F.mse_loss, "l1": F.l1_loss}
    try:
        return loss_map[name.lower()]
    except KeyError as exc:  # pragma: no cover - config error
        raise ValueError("LOSS must be 'mse' or 'l1'") from exc


@dataclass
class TrainState:
    """Simple container for tracking the state of the training process."""

    epoch: int = 0
    step: int = 0


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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = TrainState()
        self.logger = logger or logging.getLogger(__name__)
        # Evaluation parameters
        self.eval_every = int(config.EVALUATION.EVAL_EVERY)
        self.eval_first = bool(config.EVALUATION.EVAL_FIRST)
        
        # Training parameters
        self.epochs = int(config.TRAINING.EPOCHS)
        self.max_grad_norm = config.TRAINING.MAX_GRAD_NORM
        self.max_grad_value = config.TRAINING.MAX_GRAD_VALUE
        self.criterion = get_criterion(str(config.TRAINING.LOSS))
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

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                prediction, target = self.model(batch)
                loss = self.criterion(prediction, target)
            self.accelerator.backward(loss)
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
        if wandb.run is not None and self.accelerator.is_main_process:
            lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {"train/loss": loss_value.item(), "train/lr": lr},
                step=self.state.step,
            )
        self.state.step += 1
        return loss_value.item()

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        """Iterate over ``dataloader`` once and return the mean loss."""

        total_loss = 0.0
        disable = (
            self.accelerator is not None
            and not self.accelerator.is_local_main_process
        )
        progress_bar = tqdm(
            dataloader,
            disable=disable,
            desc=f"Epoch {self.state.epoch + 1}",
        )
        for batch in progress_bar:
            total_loss += self.train_step(batch)
            progress_bar.set_postfix(
                loss=total_loss / max(progress_bar.n, 1), refresh=False
            )
        mean_loss = total_loss / max(len(dataloader), 1)
        if wandb.run is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            wandb.log({"train/avg_loss": mean_loss}, step=self.state.step)
        return mean_loss

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def val(self, dataloader: Iterable[dict]) -> float:
        """Evaluate the model on ``dataloader`` and return the mean loss."""

        self.model.eval()
        total_loss = 0.0
        if self.accelerator is None:
            raise NotImplementedError(
                "Trainer.train_step() requires an accelerator."
            )
        
        disable = (
            self.accelerator is not None
            and not self.accelerator.is_local_main_process
        )
        progress_bar = tqdm(
            dataloader,
            disable=disable,
            desc=f"Eval {self.state.epoch + 1}",
        )
        for batch in progress_bar:
            with self.accelerator.autocast():
                prediction, target = self.model(batch)
            prediction, target = self.accelerator.gather_for_metrics(
                (prediction, target)
            )
            # loss computed in fp32
            loss = self.criterion(prediction.float(), target.float())
            loss_value = loss.detach()
            total_loss += loss_value.item()
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"eval/loss": loss_value.item()}, step=self.state.step)
            self.state.step += 1
            progress_bar.set_postfix(
                loss=total_loss / max(progress_bar.n, 1), refresh=False
            )
        mean_loss = total_loss / max(len(dataloader), 1)
        if wandb.run is not None and self.accelerator.is_main_process:
            wandb.log({"eval/avg_loss": mean_loss}, step=self.state.step)
        return mean_loss

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path) -> None:
        """Persist the training state to ``path``."""

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.state.epoch,
            "step": self.state.step,
        }
        torch.save(ckpt, path)
        if self.logger is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            self.logger.info(f"Saved checkpoint to {path}")

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict],
        val_loader: Optional[Iterable[dict]] = None,
        epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """Run the training loop for ``epochs`` epochs.

        Parameters
        ----------
        train_loader:
            Dataloader yielding training batches.
        val_loader:
            Optional dataloader used for evaluation.
        epochs:
            Total number of epochs to train for. If ``None``, the value from
            the configuration is used.
        checkpoint_dir:
            Directory to store checkpoints in.  If ``None`` no checkpoints are
            written.

        Notes
        -----
        Evaluation frequency is controlled by ``self.eval_every`` set during
        initialisation.  If ``self.eval_first`` is ``True`` evaluation is run
        once before any training.  Regardless of these settings, a final
        evaluation is always performed after training concludes.
        """

        ckpt_path: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_path is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            ckpt_path.mkdir(parents=True, exist_ok=True)
        if ckpt_path is not None and self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        if epochs is None:
            epochs = self.epochs

        if self.eval_first and val_loader is not None:
            val_loss = self.val(val_loader)
            msg = f"epoch 0/{epochs} - val_loss: {val_loss:.4f}"
            if self.logger is not None and (
                self.accelerator is None or self.accelerator.is_main_process
            ):
                self.logger.info(msg)

        for epoch in range(epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch(train_loader)
            msg = f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}"
            epoch_log = {
                "epoch": epoch + 1,
                "epoch/train_loss": train_loss,
            }
            if val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_loss = self.val(val_loader)
                msg += f", val_loss: {val_loss:.4f}"
                epoch_log["epoch/eval_loss"] = val_loss
            if self.logger is not None and (
                self.accelerator is None or self.accelerator.is_main_process
            ):
                self.logger.info(msg)
            if wandb.run is not None and (
                self.accelerator is None or self.accelerator.is_main_process
            ):
                wandb.log(epoch_log, step=self.state.step)

            if (
                ckpt_path is not None
                and (epoch + 1) % self.save_every == 0
                and (
                    self.accelerator is None or self.accelerator.is_main_process
                )
            ):
                filename = ckpt_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(filename)
            if ckpt_path is not None and self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        if val_loader is not None:
            val_loss = self.val(val_loader)
            msg = f"final val_loss: {val_loss:.4f}"
            if self.logger is not None and (
                self.accelerator is None or self.accelerator.is_main_process
            ):
                self.logger.info(msg)
            if wandb.run is not None and (
                self.accelerator is None or self.accelerator.is_main_process
            ):
                wandb.log({"epoch/final_eval_loss": val_loss}, step=self.state.step)

class DeterministicTrainer(Trainer):
    """Trainer variant for deterministic models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: object,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        accelerator: Optional[Accelerator] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            scheduler=scheduler,
            accelerator=accelerator,
            logger=logger,
        )
        # Deterministic models are evaluated before the first training epoch.
        self.eval_first = True


__all__ = ["Trainer", "TrainState", "DeterministicTrainer", "get_criterion"]

