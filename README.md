# FutureLatents

FutureLatents explores world models that predict future visual representations directly in latent space, conditioned (or not), on previous video frames.

It currently supports VJEPA2 backbone, and Kinetics 400 dataset. Will soon support DinoV3 and 4DS backbones, and Kinetics 700, WISA, Something Something V2 datasets.


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
