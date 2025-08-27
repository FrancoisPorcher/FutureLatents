# FutureLatents

World Model to predict future latents.

This repository now provides a simple training pipeline that freezes the
pretrained encoder weights by default and keeps the original Hugging Face
``AutoVideoProcessor`` alongside the model so preprocessing and encoding stay
coupled. Optimisers can be constructed by passing ``model.trainable_modules()``
which returns only the parameters that require gradients.

## Project Structure

- `data/` - datasets and related assets (not tracked in git)
- `models/` - trained model checkpoints (not tracked)
- `notebooks/` - exploratory notebooks
- `src/` - core library code
- `training/` - training scripts

See `requirements.txt` for dependencies.
