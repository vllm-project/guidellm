from __future__ import annotations

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    attrs={
        "PILImage": ("PIL", "Image"),
        "Image": ("PIL.Image", "Image"),
    },
    error_message="Please install guidellm[vision] to use image/video features",
)
