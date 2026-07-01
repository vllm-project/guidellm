from __future__ import annotations

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    attrs={
        "PILImage": lazy.ExtraAttr("PIL", alias="Image"),
        "Image": lazy.ExtraAttr("PIL.Image", alias="Image"),
    },
    error_message="Please install guidellm[vision] to use image/video features",
)

iio = lazy.load("imageio.v3", suppress_warning=True)
__all__.append("iio")
