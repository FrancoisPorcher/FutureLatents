"""Training utilities for FutureLatents.

This module exposes the high level :func:`build_trainer` factory alongside the
concrete trainer implementations.  The factory mirrors the style of the
``build_model`` and ``build_dataset`` helpers in the project and selects the
appropriate trainer class based on ``config.MODEL.TYPE``.
"""

from __future__ import annotations

import logging
from typing import Optional

from .trainer import (
    DeterministicTrainer,
    Trainer,
    TrainState,
    get_criterion,
)


def build_trainer(
    config,
    model,
    optimizer,
    scheduler: Optional[object] = None,
    accelerator: Optional[object] = None,
    logger: Optional[logging.Logger] = None,
):
    """Instantiate a trainer based on ``config.MODEL.TYPE``.

    Parameters
    ----------
    config:
        Full configuration object.  ``config.MODEL.TYPE`` controls which
        trainer implementation is used.
    model:
        The model to be optimised.
    optimizer:
        Optimiser responsible for updating the model parameters.
    scheduler:
        Optional learning rate scheduler.
    accelerator:
        Optional ``accelerate.Accelerator`` used for distributed training.
    logger:
        Optional ``logging.Logger`` instance for status messages.
    """

    trainer_type = str(config.MODEL.TYPE).lower()
    trainer_cls = DeterministicTrainer if trainer_type == "deterministic" else Trainer
    return trainer_cls(
        model=model,
        optimizer=optimizer,
        config=config.TRAINER,
        scheduler=scheduler,
        accelerator=accelerator,
        logger=logger,
    )


__all__ = [
    "Trainer",
    "TrainState",
    "DeterministicTrainer",
    "get_criterion",
    "build_trainer",
]

