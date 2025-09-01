# FutureLatents

FutureLatents explores world models that predict future visual representations directly in latent space, conditioned (or not), on previous video frames.

It currently supports VJEPA2 backbone and the Kinetics 400 dataset, and also provides a `Kinetics400_cached` variant for training on pre-computed embeddings. Will soon support DinoV3 and 4DS backbones, and Kinetics 700, WISA, Something Something V2 datasets.


## Installation

1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The dataset loader additionally requires the `decord` package.

## Project structure

```
configs/   YAML experiment configuration files
datasets/  Dataset wrappers such as Kinetics‑400 and Kinetics‑400 cached
models/    Model components including the flow transformer
src/       Core library code and entry point
training/  Minimal training script
notebooks/ Exploratory notebooks (empty placeholder)
```

## Configuration

Configurations can be composed using the `inherits` key. For example
`configs/vjepa2_kinetics_400.yaml` combines dataset, backbone, training, evaluation
and flow‑matching settings:

```yaml
inherits:
  - datasets/kinetics_400.yaml
  - backbones/vjepa2.yaml
  - training/trainer.yaml
  - training/evaluation.yaml
  - training/flow_matching.yaml
encoder_trainable: false
```

Run `utils.config.load_config` to resolve the hierarchy and `utils.config.print_config` to display it.

The training configuration additionally supports a `mixed_precision` field
(`"no"`, `"fp16"` or `"bf16"`) that is forwarded to the Hugging Face
`Accelerator` for mixed precision training. A `gradient_checkpointing` flag
enables PyTorch's gradient checkpointing for reduced memory usage. Optional
`max_grad_norm` and `max_grad_value` settings clip gradients via the
accelerator to stabilise training.

### Flow matching

Flow matching follows the diffusive modelling paradigm where latent tokens are
incrementally noised and denoised.  The `training/flow_matching.yaml` file
provides the number of training timesteps and the configuration for the
diffusion transformer (DiT) used to predict the noise at each step.  The
`LatentVideoModel` reads these values from the `flow_matching` section to build
its internal DiT module.

## Usage

Print the resolved configuration, instantiate the dataset and model and set up the optimiser:

```bash
python -m src.main --config_path configs/vjepa2_kinetics_400.yaml
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
python -m src.main --config_path configs/vjepa2_kinetics_400_cached.yaml
```

will load those embeddings with `torch.load` and bypass raw video decoding.
