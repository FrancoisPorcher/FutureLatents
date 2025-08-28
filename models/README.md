# Models

This directory contains the model components used by FutureLatents.

- `latent_video_model.py` couples a pretrained video encoder with a
  flow matching transformer operating on latent tokens.
- `DiT.py` implements a minimal diffusion transformer (DiT) that predicts
  the noise added at each training timestep.

The architecture is configured through the `flow_matching` section in the
configuration files.  For instance, `configs/training/flow_matching.yaml`
provides both the number of diffusion steps and the DiT hyperparameters:

```yaml
flow_matching:
  num_train_timesteps: 500
  dit:
    input_dim: 1024
    hidden_dim: 1024
    depth: 6
    num_heads: 8
    mlp_ratio: 4.0
```

`LatentVideoModel` reads these values to construct the flow transformer.  If no
`flow_matching` configuration is supplied the transformer is omitted, allowing
experiments that focus solely on the backbone encoder.
