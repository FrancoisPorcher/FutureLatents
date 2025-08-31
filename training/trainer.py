"""Utilities for training FutureLatents models.

This module defines a small ``Trainer`` class that provides convenience
wrappers around the typical PyTorch training loop.  It purposefully keeps the
implementation lightweight so that it can serve as a starting point for more
feature rich trainers in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm
import logging


@dataclass
class TrainState:
    """Simple container for tracking the state of the training process."""

    epoch: int = 0


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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
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

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> float:
        """Perform a single optimisation step using flow matching."""

        self.model.train()
        if self.accelerator is None:
            raise NotImplementedError(
                "Trainer.train_step() requires an accelerator."
            )

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                prediction, target = self.model(batch)
                loss = F.mse_loss(prediction, target)
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        return loss.item()

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
        return total_loss / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def val(self, dataloader: Iterable[dict]) -> float:
        """Evaluate the model on ``dataloader`` and return the mean loss."""

        self.model.eval()
        total_loss = 0.0
        for batch in dataloader:
            ctx = (
                self.accelerator.autocast()
                if self.accelerator is not None
                else nullcontext()
            )
            with ctx:
                latents = self.model.encode_inputs(batch)
            total_loss += latents.mean().item()
        return total_loss / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path) -> None:
        """Persist the training state to ``path``."""

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.state.epoch,
        }
        torch.save(ckpt, path)

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict],
        val_loader: Optional[Iterable[dict]] = None,
        epochs: int = 1,
        eval_every: int = 1,
        checkpoint_dir: Optional[str] = None,
        save_every: int = 1,
    ) -> None:
        """Run the training loop for ``epochs`` epochs.

        Parameters
        ----------
        train_loader:
            Dataloader yielding training batches.
        val_loader:
            Optional dataloader used for evaluation.
        epochs:
            Total number of epochs to train for.
        eval_every:
            Perform evaluation every ``eval_every`` epochs.
        checkpoint_dir:
            Directory to store checkpoints in.  If ``None`` no checkpoints are
            written.
        save_every:
            Save a checkpoint every ``save_every`` epochs.
        """

        ckpt_path: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
        if ckpt_path is not None:
            ckpt_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch(train_loader)
            msg = f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loader is not None and (epoch + 1) % eval_every == 0:
                val_loss = self.val(val_loader)
                msg += f", val_loss: {val_loss:.4f}"
            self.logger.info(msg)

            if ckpt_path is not None and (epoch + 1) % save_every == 0:
                filename = ckpt_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(filename)


__all__ = ["Trainer", "TrainState"]

