"""Model package for FutureLatents."""

from .DiT import DiT, PredictorTransformer, PredictorTransformerCrossAttention
from .models import (
    FlowMatchingLatentVideoModel,
    DeterministicLatentVideoModel,
    DeterministicCrossAttentionLatentVideoModel,
)


def build_model(config):
    """Instantiate a model based on ``config.MODEL.TYPE``."""

    model_type = str(config.MODEL.TYPE).lower()
    models = {
        "flow_matching": FlowMatchingLatentVideoModel,
        "deterministic": DeterministicLatentVideoModel,
        "deterministic_cross_attention": DeterministicCrossAttentionLatentVideoModel,
    }
    try:
        model_cls = models[model_type]
    except KeyError as exc:  # pragma: no cover - simple error path
        raise ValueError(f"Unknown model type: {model_type}") from exc
    return model_cls(config)


__all__ = [
    "DiT",
    "PredictorTransformer",
    "PredictorTransformerCrossAttention",
    "FlowMatchingLatentVideoModel",
    "DeterministicLatentVideoModel",
    "DeterministicCrossAttentionLatentVideoModel",
    "build_model",
]
