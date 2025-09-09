# Models

This directory contains the model components used by FutureLatents.

The transformer backbone and related blocks live in `DiT.py`, while `models.py`
contains the wrapper models that assemble the backbone for flow-matching and
deterministic prediction. `DiT.py` provides the diffusion transformer (`DiT`)
and a deterministic `PredictorTransformer`, and `models.py` defines the
`LatentVideoModel` for flow matching and the `DeterministicLatentVideoModel`
for direct latent prediction.  Model behaviour is configured through the
`model` section of the YAML configuration files, for example:

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
