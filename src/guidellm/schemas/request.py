from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import Field, computed_field

from guidellm.utils import StandardBaseDict, StandardBaseModel

__all__ = [
    "GenerationRequest",
    "GenerationRequestArguments",
    "GenerativeRequestType",
    "UsageMetrics",
]


GenerativeRequestType = Literal[
    "text_completions",
    "chat_completions",
    "audio_transcriptions",
    "audio_translations",
]


class GenerationRequestArguments(StandardBaseDict):
    method: str | None = Field(
        default=None,
        description="The HTTP method to use for the request (e.g., 'POST', 'GET').",
    )
    stream: bool | None = Field(
        default=None,
        description="Whether to stream the response, if applicable.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers to include in the request, if applicable.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Query parameters to include in the request URL, if applicable.",
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description="Content to include in the main request body.",
    )
    files: dict[str, Any] | None = Field(
        default=None,
        description="Files to include in the request, if applicable.",
    )

    def model_combine(
        self, additional: GenerationRequestArguments | dict[str, Any]
    ) -> GenerationRequestArguments:
        additional_dict = (
            additional.model_dump()
            if isinstance(additional, GenerationRequestArguments)
            else additional
        )

        for overwrite in ("method", "stream"):
            if (val := additional_dict.get(overwrite)) is not None:
                setattr(self, overwrite, val)

        for combine in ("headers", "params", "json_body", "files"):
            if (val := additional_dict.get(combine)) is not None:
                setattr(self, combine, {**getattr(self, combine, {}), **val})

        return self


class UsageMetrics(StandardBaseDict):
    # Text stats
    text_tokens: int | None = Field(
        default=None, description="Number of text tokens processed/generated."
    )
    text_words: int | None = Field(
        default=None, description="Number of text words processed/generated."
    )
    text_characters: int | None = Field(
        default=None, description="Number of text characters processed/generated."
    )
    text_bytes: int | None = Field(
        default=None, description="Number of text bytes processed/generated."
    )

    # Vision image stats
    image_tokens: int | None = Field(
        default=None, description="Number of image tokens processed/generated."
    )
    image_count: int | None = Field(
        default=None, description="Number of images processed/generated."
    )
    image_pixels: int | None = Field(
        default=None, description="Number of image pixels processed/generated."
    )
    image_bytes: int | None = Field(
        default=None, description="Number of image bytes processed/generated."
    )

    # Vision video stats
    video_tokens: int | None = Field(
        default=None, description="Number of video tokens processed/generated."
    )
    video_frames: int | None = Field(
        default=None, description="Number of video frames processed/generated."
    )
    video_seconds: float | None = Field(
        default=None, description="Duration of video processed/generated in seconds."
    )
    video_bytes: int | None = Field(
        default=None, description="Number of video bytes processed/generated."
    )

    # Audio stats
    audio_tokens: int | None = Field(
        default=None, description="Number of audio tokens processed/generated."
    )
    audio_samples: int | None = Field(
        default=None, description="Number of audio samples processed/generated."
    )
    audio_seconds: float | None = Field(
        default=None, description="Duration of audio processed/generated in seconds."
    )
    audio_bytes: int | None = Field(
        default=None, description="Number of audio bytes processed/generated."
    )

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int | None:
        return (self.text_tokens or 0) + (self.image_tokens or 0) + (
            self.video_tokens or 0
        ) + (self.audio_tokens or 0) or None


class GenerationRequest(StandardBaseModel):
    """Request model for backend generation operations."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request.",
    )
    request_type: GenerativeRequestType | str = Field(
        description=(
            "Type of request. If url is not provided in arguments, "
            "this will be used to determine the request url."
        ),
    )
    arguments: GenerationRequestArguments = Field(
        description=(
            "Payload for the request, structured as a dictionary of arguments to pass "
            "to the respective backend method. For example, can contain "
            "'json', 'headers', 'files', etc."
        )
    )
    input_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Input statistics including token counts and audio duration.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Output statistics including token counts and audio duration.",
    )
