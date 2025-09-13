from typing import Any, Tuple
from transformers import AutoModel, AutoVideoProcessor


def build_backbone(backbone_cfg: Any) -> Tuple[Any, Any]:
    """Construct the optional backbone encoder and preprocessor.

    Returns a tuple ``(encoder, preprocessor)``. When no backbone is
    configured (e.g., missing or empty configuration), both are ``None``.
    """
    if not backbone_cfg:
        return None, None

    backbone_type = str(backbone_cfg.get("BACKBONE_TYPE", "") or "").lower()
    if backbone_type == "vjepa2":
        hf_repo = backbone_cfg.get("HF_REPO")
        if hf_repo:
            encoder = AutoModel.from_pretrained(hf_repo)
            # AutoVideoProcessor also covers many video/image processors on HF
            preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
            return encoder, preprocessor
    return None, None
