from __future__ import annotations

try:
    from torchcodec import AudioSamples
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
except ImportError as e:
    raise AttributeError("Please install guidellm[audio] to use audio features") from e

__all__ = [
    "AudioSamples",
    "AudioDecoder",
    "AudioEncoder",
]
