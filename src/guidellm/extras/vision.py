from __future__ import annotations

try:
    from PIL import Image as PILImage
    from PIL.Image import Image
except ImportError as e:
    raise AttributeError(
        "Please install guidellm[vision] to use image/video features"
    ) from e

__all__ = [
    "Image",
    "PILImage",
]
