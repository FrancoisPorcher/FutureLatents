# Models

This directory contains the model components used by FutureLatents.

All architectures are implemented in a single file, `DiT.py`, which provides
the transformer backbone (`DiT`), a deterministic `PredictorTransformer` and
two wrapper models: `LatentVideoModel` for flow matching and
`DeterministicLatentVideoModel` for direct latent prediction.  Model behaviour
is configured through the `model` section of the YAML configuration files, for
example:

```yaml
model:
  type: flow_matching  # or deterministic
  num_context_latents: 16
  flow_matching:
    num_train_timesteps: 500
    dit:
      input_dim: 1024
      hidden_dim: 1024
      depth: 6
      num_heads: 8
      mlp_ratio: 4.0
```

`LatentVideoModel` reads these values to construct the flow transformer.
