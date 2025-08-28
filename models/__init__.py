"""Model package for FutureLatents.

This exposes the main model components so they can be imported as
``from models import LatentVideoModel, DiT``.
"""

from .DiT import DiT
from .latent_video_model import LatentVideoModel

__all__ = ["DiT", "LatentVideoModel"]
