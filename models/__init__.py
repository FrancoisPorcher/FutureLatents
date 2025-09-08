"""Model package for FutureLatents."""

from .DiT import (
    DiT,
    PredictorTransformer,
    LatentVideoModel,
    DeterministicLatentVideoModel,
)


def build_model(config):
    """Instantiate a model based on ``config.MODEL.TYPE``."""

    model_type = str(config.MODEL.TYPE).lower()
    models = {
        "flow_matching": LatentVideoModel,
        "deterministic": DeterministicLatentVideoModel,
    }
    try:
        model_cls = models[model_type]
    except KeyError as exc:  # pragma: no cover - simple error path
        raise ValueError(f"Unknown model type: {model_type}") from exc
    return model_cls(config)


__all__ = [
    "DiT",
    "PredictorTransformer",
    "LatentVideoModel",
    "DeterministicLatentVideoModel",
    "build_model",
]
