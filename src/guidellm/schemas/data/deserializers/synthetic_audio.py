from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataArgs

__all__ = ["SyntheticAudioDataArgs"]


@DataArgs.register("synthetic_audio")
class SyntheticAudioDataArgs(DataArgs):
    """Arguments for synthetic audio dataset deserializer."""

    kind: Literal["synthetic_audio"] = Field(  # type: ignore[assignment]
        default="synthetic_audio",
        description="Type identifier for the synthetic audio dataset configuration.",
    )
    duration_seconds: float = Field(
        default=1.0,
        gt=0,
        description="Duration of each generated audio clip in seconds.",
    )
    sample_rate: int = Field(
        default=16000,
        gt=0,
        description="Sample rate of generated audio in hertz.",
    )
    content: Literal["sine", "noise", "silence"] = Field(
        default="sine",
        description="Waveform content to generate.",
    )
    format: Literal["wav", "mp3", "flac", "ogg", "aac"] = Field(
        default="wav",
        description="Encoded audio format.",
    )
    seed: int = Field(
        default=12,
        description="Base random seed for reproducible generated audio.",
    )
