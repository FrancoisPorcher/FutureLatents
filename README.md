# FutureLatents

FutureLatents explores world models that predict future visual representations directly in latent space.
It currently provides a minimal PyTorch code base built around pretrained video encoders from Hugging Face
and utilities for loading video datasets and composing experiment configurations.

## Features

- **Latent video model** – wraps a Hugging Face encoder and associated `AutoVideoProcessor` so that
  preprocessing and encoding remain coupled. The encoder is frozen by default but can be unfrozen to
  finetune specific modules.
- **Config system** – YAML configurations are loaded with [OmegaConf](https://omegaconf.readthedocs.io/en/latest/)
  and support simple inheritance via an `inherits` list. Later configs override earlier ones with a notice for
  conflicting keys.
- **Kinetics‑400 dataset loader** – reads annotation CSV files and returns videos as padded frame tensors using
  [decord](https://github.com/dmlc/decord) for efficient video access.
- **Training utilities** – example scripts show how to construct optimisers over only the trainable parameters
  and print parameter counts for inspection.

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
datasets/  Dataset wrappers such as Kinetics‑400
src/       Core library code and entry point
training/  Minimal training script
notebooks/ Exploratory notebooks (empty placeholder)
```

## Configuration

Configurations can be composed using the `inherits` key. For example
`configs/vjepa2_kinetics_400.yaml` combines dataset, backbone and training settings:

```yaml
inherits:
  - datasets/kinetics_400.yaml
  - backbones/vjepa2.yaml
  - training/trainer.yaml
encoder_trainable: false
```

Run `utils.config.load_config` to resolve the hierarchy and `utils.config.print_config` to display it.

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
`configs/datasets/kinetics_400.yaml` points to a valid location.

## Data

The repository does not track datasets or model checkpoints. Provide paths to your local copies in
the configuration files.
