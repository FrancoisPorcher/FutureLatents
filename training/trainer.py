"""Utilities for training FutureLatents models.

This module defines a small ``Trainer`` class that provides convenience
wrappers around the typical PyTorch training loop.  It purposefully keeps the
implementation lightweight so that it can serve as a starting point for more
feature rich trainers in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


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
    device:
        Device on which the model and input batches should be placed.  If
        ``None`` the trainer will use ``"cuda"`` when available, otherwise
        ``"cpu"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = TrainState()

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> float:
        """Perform a single optimisation step.

        The example loss used here simply averages the encoder output of the
        provided video batch.  Real projects are expected to replace this with a
        task specific objective.
        """

        self.model.train()
        video = batch["video"].to(self.device)

        # Forward pass through the encoder and compute a dummy loss.
        features_video_encoded_with_backbone = self.model.encode_video_with_backbone(video)
        batch['features_video_encoded_with_backbone'] = features_video_encoded_with_backbone
        breakpoint()
        loss = features_video_encoded_with_backbone.mean()

        # Optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def train_epoch(self, dataloader: Iterable[dict]) -> float:
        """Iterate over ``dataloader`` once and return the mean loss."""

        total_loss = 0.0
        for batch in dataloader:
            total_loss += self.train_step(batch)
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
            video = batch["video"].to(self.device)
            outputs = self.model.encode_video(video)
            loss = outputs.last_hidden_state.mean()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict],
        val_loader: Optional[Iterable[dict]] = None,
        epochs: int = 1,
    ) -> None:
        """Run the training loop for ``epochs`` epochs."""

        for epoch in range(epochs):
            self.state.epoch = epoch
            train_loss = self.train_epoch(train_loader)
            msg = f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loader is not None:
                val_loss = self.val(val_loader)
                msg += f", val_loss: {val_loss:.4f}"
            print(msg)


__all__ = ["Trainer", "TrainState"]

