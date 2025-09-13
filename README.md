# FutureLatents

FutureLatents explores world models that predict the remainder of a video by
operating directly in latent space. Given a set of initial context frames, the
model generates the future latent representations in a purely conditioned
fashion.

It currently targets the VJEPA2 backbone and the Kinetics 400 dataset and
provides a `Kinetics400_cached` variant for training on pre-computed
embeddings.  Future work will extend support to additional backbones and
datasets.


## Installation

1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The dataset loader additionally requires the `decord` package.

## Project structure

```
configs/     YAML experiment configuration files
data/        Placeholder for dataset files (not tracked)
dataloader/  Utilities for constructing PyTorch dataloaders
datasets/    Dataset wrappers such as Kinetics‑400 and Kinetics‑400 cached
models/      Model components including flow and deterministic transformers
src/         Core library code and entry point
training/    Minimal training script
utils/       Miscellaneous helpers
notebooks/   Exploratory notebooks (empty placeholder)
```

## Configuration

Configurations can be composed using the `inherits` key. For example
`configs/references/vjepa2_kinetics_400.yaml` combines dataset, backbone, and training
settings (including the model configuration):

```yaml
inherits:
  - ../datasets/kinetics_400.yaml
  - ../backbones/vjepa2.yaml
  - ../training/trainer_flow_matching.yaml
```

The `trainer.training` section controls optimisation parameters such as loss
function selection, precision, gradient checkpointing, and gradient clipping.

### Flow matching

Flow matching follows the diffusive modelling paradigm where latent tokens are
incrementally noised and denoised.  The `model.flow_matching` section in
`training/trainer_flow_matching.yaml` provides the number of training timesteps and the
configuration for the diffusion transformer (DiT) used to predict the noise at
each step.  The `LatentVideoModel` reads these values to build its internal DiT
module.

### Deterministic prediction

Alternatively, FutureLatents offers a deterministic path that dispenses with
diffusion and directly regresses future latents from the context frames using a
`PredictorTransformer`.  To enable it, set `model.type` to `deterministic` and
provide the predictor configuration under `model.predictor`:

```yaml
model:
  type: deterministic
  num_context_latents: 16
  predictor:
    dit:
      input_dim: 1024
      hidden_dim: 1024
      depth: 12
      num_heads: 8
      mlp_ratio: 4.0
```

A reference configuration is available in `training/trainer_deterministic.yaml`.

This `DeterministicLatentVideoModel` predicts the remaining latents in a single
forward pass and optimises the reconstruction `loss` specified in the training
configuration.

## Usage

Print the resolved configuration, instantiate the dataset and model and set up the optimiser:

```bash
python -m src.main --config_path configs/references/vjepa2_kinetics_400.yaml
# or the deterministic variant
python -m src.main --config_path configs/references/vjepa2_kinetics_400_deterministic.yaml
```

A minimal training example that freezes the encoder and builds an `AdamW` optimiser over trainable
parameters is provided in `training/main.py` and can be run with:

```bash
python -m training.main
```

Both commands expect that the Kinetics‑400 annotation CSV path in
`configs/datasets/kinetics_400.yaml` points to a valid location. When using cached embeddings, make sure `configs/datasets/kinetics_400_cached.yaml` references a metadata CSV describing the saved embeddings.

### Pre-computed embeddings

If you have already encoded your videos, the repository can train directly on cached features using the `Kinetics400Cached` dataset. The `configs/datasets/kinetics_400_cached.yaml` file expects a metadata CSV where each row contains a path (in the `out_path` column by default) to a saved embedding tensor. Running with

```bash
python -m src.main --config_path configs/references/vjepa2_kinetics_400_cached.yaml
```

will load those embeddings with `torch.load` and bypass raw video decoding.
