from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataPreprocessorArgs


@DataPreprocessorArgs.register("encode_media")
class MediaEncoderArgs(DataPreprocessorArgs):
    """Model for media encoder preprocessor arguments."""

    kind: Literal["encode_media"] = Field(
        default="encode_media",
        description="Type identifier for the media encoder preprocessor.",
    )
    audio_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for audio encoding.",
        examples=[{"format": "mp3"}],
    )
    image_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for image encoding.",
        examples=[{"format": "jpg"}],
    )
    video_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for video encoding.",
        examples=[{"format": "mp4"}],
    )
