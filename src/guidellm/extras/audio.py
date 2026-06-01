from __future__ import annotations

import guidellm.utils.lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_extras(
    __name__,
    attrs={
        "AudioSamples": "torchcodec",
        "AudioDecoder": "torchcodec.decoders",
        "AudioEncoder": "torchcodec.encoders",
    },
    error_message="Please install guidellm[audio] to use audio features",
)
