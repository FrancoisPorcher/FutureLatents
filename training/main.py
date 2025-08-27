"""Entry point for model training."""

from __future__ import annotations

import torch

from futurelatents.models import LatentVideoModel


def main() -> None:  # pragma: no cover - simple example script
    """Run a minimal training pipeline.

    This script demonstrates how to construct an optimiser that only updates
    parameters marked as trainable within :class:`LatentVideoModel`. The model
    freezes its encoder weights by default, so the returned parameter iterator
    will be empty unless other modules with trainable parameters are added.
    """

    # Example configuration for the VJEPA 2 backbone; replace the repository
    # name with a local checkpoint as needed.
    config = {"backbone": {"hf_repo": "facebook/vjepa2-vitl-fpc64-256"}}
    model = LatentVideoModel(config)

    # Define the optimiser over trainable parameters only
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-4)
    print(optimizer)


if __name__ == "__main__":
    main()
