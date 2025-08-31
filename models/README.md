# Models

This directory contains the model components used by FutureLatents.

- `latent_video_model.py` couples a pretrained video encoder with a
  flow matching transformer operating on latent tokens.
- `DiT.py` implements a minimal diffusion transformer (DiT) that predicts
  the noise added at each training timestep.  It uses PyTorch's
  ``scaled_dot_product_attention`` with optional flash or efficient
  attention backends selected via the ``attn_backends`` argument.
  For example:

  ```python
  from torch.nn.functional import scaled_dot_product_attention
  from torch.nn.attention import SDPBackend, sdpa_kernel

  # Only enable flash attention backend
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
      scaled_dot_product_attention(...)

  # Enable the Math or Efficient attention backends
  with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
      scaled_dot_product_attention(...)
  ```

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
