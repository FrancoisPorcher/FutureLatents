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
                {f"train/{self.criterion_name}_loss": loss_value.item(), "train/lr": lr},
                step=self.state.step,
            )
        self.state.step += 1
        if self.debug and self.dump_dir is not None:
            self._dump_norms(norms)
        return loss_value.item()

    def _dump_norms(self, norms: Optional[Dict[str, torch.Tensor]]) -> None:
        """Persist L1/L2 norm tensors and histogram plots with statistics."""
        if norms is None or self.dump_dir is None:
            return
        import matplotlib.pyplot as plt
        import numpy as np

        for name, tensor in norms.items():
            tensor_cpu = tensor.detach().cpu()
            torch.save(tensor_cpu, self.dump_dir / f"{name}_norms_step_{self.state.step}.pt")
            data = tensor_cpu.flatten().numpy()
            plt.figure()
            plt.hist(data, bins=30)
            mean = float(np.mean(data))
            median = float(np.median(data))
            q1, q3 = np.quantile(data, [0.25, 0.75])
            for val, label, style in [
                (mean, "mean", "--"),
                (median, "median", "-"),
                (q1, "25%", ":"),
                (q3, "75%", "-."),
            ]:
                plt.axvline(val, color="r", linestyle=style, label=label)
            plt.title(f"Token {name.upper()} Norms Distribution")
            plt.legend()
            plt.savefig(self.dump_dir / f"{name}_hist_step_{self.state.step}.png")
            plt.close()


    def _log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> None:
        """Log epoch-level metrics to the configured logger and ``wandb``."""

        msg = (
            f"epoch {epoch + 1}/{epochs} - train_{self.criterion_name}_loss: "
            f"{train_loss:.4f}"
        )
        epoch_log: Dict[str, Any] = {
            "epoch": epoch + 1,
            f"epoch/train_{self.criterion_name}_loss": train_loss,
        }
        if val_loss is not None:
            msg += f", val_{self.criterion_name}_loss: {val_loss:.4f}"
            epoch_log[f"epoch/eval_{self.criterion_name}_loss"] = val_loss
        if self.logger is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            self.logger.info(msg)
        if wandb.run is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            wandb.log(epoch_log, step=self.state.step)

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
    # Validation utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_evaluation(
        self,
        dataloader: Iterable[dict],
        epoch: Optional[int] = None,
        epochs: Optional[int] = None,
        log: bool = False,
    ) -> float:
        """Evaluate the model on ``dataloader`` and return the mean loss.

        Parameters
        ----------
        dataloader:
            Iterable yielding batches for evaluation.
        epoch, epochs:
            When ``log`` is ``True``, these specify the current and total
            epochs used in the log message.
        log:
            If ``True`` log the aggregated validation loss using
            ``self.logger``.
        """

        self.model.eval()
        total_loss = 0.0
        if self.accelerator is None:
            raise NotImplementedError(
                "Trainer.train_step() requires an accelerator."
            )
        
        disable = (
            self.accelerator is not None
            and not self.accelerator.is_main_process
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
                wandb.log(
                    {f"eval/{self.criterion_name}_loss": loss_value.item()},
                    step=self.state.step,
                )
            self.state.step += 1
            progress_bar.set_postfix(
                {f"{self.criterion_name}_loss": total_loss / max(progress_bar.n, 1)},
                refresh=False,
            )
        mean_loss = total_loss / max(len(dataloader), 1)
        if wandb.run is not None and self.accelerator.is_main_process:
            wandb.log(
                {f"eval/avg_{self.criterion_name}_loss": mean_loss},
                step=self.state.step,
            )
        if log and self.logger is not None and (
            self.accelerator is None or self.accelerator.is_main_process
        ):
            if epoch is not None and epochs is not None:
                msg = (
                    f"epoch {epoch}/{epochs} - val_{self.criterion_name}_loss: "
                    f"{mean_loss:.4f}"
                )
            else:
                msg = f"val_{self.criterion_name}_loss: {mean_loss:.4f}"
            self.logger.info(msg)
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
        for epoch in range(self.epochs):
            self.state.epoch = epoch
            self.train_epoch()
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
