# FutureLatents

FutureLatents explores world models that predict the remainder of a video by
operating directly in latent space. Given a set of initial context frames, the
model generates the future latent representations in a purely conditioned
fashion.

It supports two backbones — VJEPA2 and DINOv3 — on Kinetics‑400. Three
prediction regimes are available:

- Deterministic (Predictor): direct regression of future latents
- Deterministic (Cross‑Attention): context→target cross‑attention variant
- Stochastic (Flow Matching): diffusive/flow‑matching training

There is also a `Kinetics400_cached` path to train on pre‑computed embeddings.
Support for 4DS is incoming alongside additional backbones and datasets.


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
# Stochastic (flow matching), VJEPA2 backbone
python -m src.main --config_path configs/references/vjepa2_kinetics_400.yaml

# Deterministic (predictor), VJEPA2 backbone
python -m src.main --config_path configs/references/vjepa2_kinetics_400_deterministic.yaml

# Deterministic (cross‑attention), VJEPA2 backbone
python -m src.main --config_path configs/references/vjepa2_kinetics_400_deterministic_cross_attention.yaml

# Deterministic (predictor), DINOv3 backbone
python -m src.main --config_path configs/references/dinov3_kinetics_400_deterministic.yaml

# Deterministic (cross‑attention), DINOv3 backbone
python -m src.main --config_path configs/references/dinov3_kinetics_400_deterministic_cross_attention.yaml

# Optional: cached embeddings variant (VJEPA2 + cached features)
python -m src.main --config_path configs/references/vjepa2_kinetics_400_cached.yaml

# Tip: append --debug to use tiny subsets and disable W&B
python -m src.main --config_path configs/references/dinov3_kinetics_400_deterministic.yaml --debug
```

A minimal training example that freezes the encoder and builds an `AdamW` optimiser over trainable
parameters is provided in `training/main.py` and can be run with:

```bash
python -m training.main
```

Both commands expect that the Kinetics‑400 annotation CSV path in
`configs/datasets/kinetics_400.yaml` points to a valid location. When using cached embeddings, make sure `configs/datasets/kinetics_400_cached.yaml` references a metadata CSV describing the saved embeddings.

### Synthetic Bouncing Shapes

You can train on a procedurally generated dataset (no downloads required). The
video configuration at `configs/datasets/synthetic_bouncing_shapes.yaml` defines train/val/visualisation
splits, while `configs/datasets/synthetic_bouncing_shapes_images.yaml` exposes the single-frame image
variant used by the cross-attention reference. Ready‑to‑run references:

```bash
# Deterministic predictor (VJEPA2 backbone)
python -m src.main --config_path configs/references/vjepa2_bouncing_shapes_deterministic.yaml

# Flow matching (VJEPA2 backbone)
python -m src.main --config_path configs/references/vjepa2_bouncing_shapes_flow_matching.yaml
```

The number of frames is inherited from `trainer.n_frames` and the resolution
from the selected backbone (e.g., VJEPA2).

### Experiments overview

- Backbones: `configs/backbones/vjepa2.yaml`, `configs/backbones/dinov3.yaml`.
- Deterministic variants: predictor and cross‑attention.
- Stochastic variant: flow matching (reference provided for VJEPA2).
- Reference configs: see `configs/references/` for ready‑to‑run combinations.

### Visualisation tools

During training, the trainer exports per‑epoch visualisations to
`experiment/<config_name>/dump/`. For each example, it saves:

- `video.mp4`: preview video (requires `ffmpeg` in PATH)
- `video.pt`: raw frames tensor (`[T,C,H,W]`)
- `context_latents.pt`: latents used as context
- `target_latents.pt`: ground‑truth future latents
- `prediction_latents.pt`: model predictions

You can control when these dumps happen via `trainer.evaluation.eval_every`
and `trainer.evaluation.eval_first` in the training configs. If `ffmpeg` is not
available, MP4 export is skipped but the `.pt` tensors are still saved.

Quick inspection in Python:

```python
import torch
from pathlib import Path

ex = Path("experiment/<config_name>/dump/example_00")
ctx = torch.load(ex / "context_latents.pt")
tgt = torch.load(ex / "target_latents.pt")
pred = torch.load(ex / "prediction_latents.pt")
vid = torch.load(ex / "video.pt")  # [T,C,H,W]
```

Programmatic export utilities live in `utils/video.py`:

- `convert_video_tensor_to_mp4(video_tensor)` → frames, fps
- `save_mp4_video(frames, out_base_path, fps)` → writes `*.mp4`
- `save_visualisation_tensors(video, context, target, prediction, out_dir)`

There is also an example notebook at `notebooks/visualise_dumps.ipynb`.

### Pre-computed embeddings

If you have already encoded your videos, the repository can train directly on cached features using the `Kinetics400Cached` dataset. The `configs/datasets/kinetics_400_cached.yaml` file expects a metadata CSV where each row contains a path (in the `out_path` column by default) to a saved embedding tensor. Running with

```bash
python -m src.main --config_path configs/references/vjepa2_kinetics_400_cached.yaml
```

will load those embeddings with `torch.load` and bypass raw video decoding.
