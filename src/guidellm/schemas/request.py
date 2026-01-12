"""
Request schema definitions for generation operations.

Contains request models and data structures used to define and execute generation
requests across different backend services. Provides standardized interfaces for
request arguments, usage metrics tracking, and request type definitions that enable
consistent interaction with various AI generation APIs.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import Field, computed_field

from guidellm.schemas.base import StandardBaseDict, StandardBaseModel

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
    """
    HTTP request arguments for generation operations.

    Encapsulates all necessary HTTP request components including method, headers,
    parameters, and payload data required to execute generation requests against
    backend services. Supports file uploads and streaming responses.
    """

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
        description="Any headers to include in the request, if applicable.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Query parameters to include in the request, if applicable.",
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
        """
        Merge additional request arguments into the current instance.

        Combines method and stream fields by overwriting, while merging collection
        fields like headers, params, body, and files by extending existing values.

        :param additional: Additional arguments to merge with current instance
        :return: Updated instance with merged arguments
        """
        additional_dict = (
            additional.model_dump()
            if isinstance(additional, GenerationRequestArguments)
            else additional
        )

        for overwrite in ("method", "stream"):
            if (val := additional_dict.get(overwrite)) is not None:
                setattr(self, overwrite, val)

        for combine in ("headers", "params", "body", "files"):
            if (val := additional_dict.get(combine)) is not None:
                current = getattr(self, combine, None) or {}
                setattr(self, combine, {**current, **val})

        return self


class UsageMetrics(StandardBaseDict):
    """
    Multimodal usage metrics for generation requests.

    Tracks resource consumption across different modalities including text, images,
    video, and audio. Provides granular metrics for tokens, bytes, duration, and
    format-specific measurements to enable comprehensive usage monitoring and billing.
    """

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
        """
        Calculate total tokens across all modalities.

        :return: Sum of text, image, video, and audio tokens, or None if all are None
        """
        token_metrics = [
            self.text_tokens,
            self.image_tokens,
            self.video_tokens,
            self.audio_tokens,
        ]
        # NOTE: None should indicate no data rather than zero usage
        if token_metrics.count(None) == len(token_metrics):
            return None
        else:
            return sum(token or 0 for token in token_metrics)

    def add_text_metrics(self, text):
        """
        Adds the metrics from the given text to the fields
        `text_characters` and `text_words`.

        :param text: Text to add metrics from
        """
        self.text_characters = (self.text_characters or 0) + len(text)
        self.text_words = (self.text_words or 0) + len(text.split())


class GenerationRequest(StandardBaseModel):
    """
    Complete request specification for backend generation operations.

    Encapsulates all components needed to execute a generation request including
    unique identification, request type specification, HTTP arguments, and input/output
    usage metrics. Serves as the primary interface between the scheduler and backend
    services for coordinating AI generation tasks.

    Example::
        request = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(
                method="POST",
                body={"prompt": "Hello world", "max_tokens": 100}
            )
        )
    """

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
        description="Input statistics including counts, sizes, and durations.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Output statistics including counts, sizes, and durations.",
    )
