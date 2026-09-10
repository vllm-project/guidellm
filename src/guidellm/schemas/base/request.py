"""
Request schema definitions for generation operations.

Contains request models and data structures used to define and execute generation
requests across different backend services. Provides standardized interfaces for
request arguments, usage metrics tracking, and request type definitions that enable
consistent interaction with various AI generation APIs.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError, computed_field
from pydantic_core import to_json

from guidellm.schemas.base.base import StandardBaseDict, StandardBaseModel
from guidellm.utils.dict import deep_update

TurnType = Literal[
    "standard",
    "client_tool_call",
    "tool_response_injection",
    "server_tool_call",
]

__all__ = [
    "GenerationRequest",
    "GenerationRequestArguments",
    "TurnType",
    "UsageMetrics",
]

_DEFAULT_BINARY_MIME_TYPE = "application/octet-stream"
_MULTIPART_PAYLOAD_INDEX = 1
_MULTIPART_MIME_INDEX = 2
_BINARY_TYPES = (bytes, bytearray, memoryview)
_MAX_BASE64_PADDING = 2
_BASE64_BODY_PATTERN = re.compile(r"[A-Za-z0-9+/]*")


def _binary_metadata(
    value: bytes | bytearray | memoryview,
    filename: str | None = None,
    mime_type: str | None = None,
) -> dict[str, int | str | None]:
    """Describe binary persistence data without retaining its content."""
    return {
        "filename": filename,
        "mime_type": mime_type or _DEFAULT_BINARY_MIME_TYPE,
        "byte_count": value.nbytes if isinstance(value, memoryview) else len(value),
    }


def _multipart_metadata(value: tuple[Any, ...]) -> dict[str, int | str | None] | None:
    """Return safe metadata for a standard multipart file tuple."""
    if (
        len(value) <= _MULTIPART_PAYLOAD_INDEX
        or not isinstance(value[0], str)
        or not isinstance(value[_MULTIPART_PAYLOAD_INDEX], _BINARY_TYPES)
    ):
        return None
    mime_type = (
        value[_MULTIPART_MIME_INDEX]
        if len(value) > _MULTIPART_MIME_INDEX
        and isinstance(value[_MULTIPART_MIME_INDEX], str)
        else None
    )
    return _binary_metadata(value[_MULTIPART_PAYLOAD_INDEX], value[0], mime_type)


def _base64_byte_count(value: str, start: int = 0) -> int | None:
    """Validate Base64 and calculate decoded length without allocating payload bytes."""
    payload_length = len(value) - start
    if payload_length <= 0 or payload_length % 4:
        return None
    padding = 1 if value[-1] == "=" else 0
    if payload_length > 1 and value[-2] == "=":
        padding += 1
    payload_end = len(value) - padding
    # Match in place with pos/endpos: no slice of the payload is materialized, and
    # the scan runs in C rather than as a per-character Python loop.
    contains_only_base64 = (
        _BASE64_BODY_PATTERN.fullmatch(value, start, payload_end) is not None
    )
    if padding > _MAX_BASE64_PADDING or not contains_only_base64:
        return None
    return 3 * (payload_length // 4) - padding


def _data_url_metadata(value: str) -> dict[str, int | str | None] | None:
    """Return safe metadata for a Base64 data URL, if applicable."""
    if not value.startswith("data:"):
        return None
    comma_index = value.find(",")
    if comma_index < 0:
        return None
    header = value[len("data:") : comma_index]
    header_parts = header.split(";")
    if not any(part.lower() == "base64" for part in header_parts[1:]):
        return None
    mime_type = header_parts[0] or _DEFAULT_BINARY_MIME_TYPE
    byte_count = _base64_byte_count(value, comma_index + 1)
    if byte_count is None:
        return {"filename": None, "mime_type": mime_type, "byte_count": None}
    return {"filename": None, "mime_type": mime_type, "byte_count": byte_count}


def _sanitize_persistence_value(  # noqa: PLR0911
    value: Any, known_inline_media: bool = False
) -> Any:
    """Replace persistence-only binary and inline media values recursively."""
    if isinstance(value, _BINARY_TYPES):
        return _binary_metadata(value)
    if isinstance(value, tuple):
        return _multipart_metadata(value) or [
            _sanitize_persistence_value(item, known_inline_media) for item in value
        ]
    if isinstance(value, list):
        return [_sanitize_persistence_value(item, known_inline_media) for item in value]
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_persistence_value(
                item,
                known_inline_media=(
                    item_key == "file_data"
                    or (item_key == "data" and "format" in value)
                ),
            )
            for item_key, item in value.items()
        }
    if isinstance(value, str):
        if metadata := _data_url_metadata(value):
            return metadata
        if known_inline_media and (byte_count := _base64_byte_count(value)) is not None:
            return {
                "filename": None,
                "mime_type": _DEFAULT_BINARY_MIME_TYPE,
                "byte_count": byte_count,
            }
    return value


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
        examples=[
            {
                "temperature": 0.5,
                "max_tokens": 100,
            }
        ],
    )
    files: dict[str, Any] | None = Field(
        default=None,
        description="Files to include in the request, if applicable.",
    )
    content: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional fields to include in generated text content objects, "
            "if applicable."
        ),
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
                deep_update(current, val)
                setattr(self, combine, current)

        return self

    def model_dump_persistence_json(self) -> str:
        """Serialize request arguments safely for persisted benchmark responses.

        The transport arguments remain untouched. Binary and inline media are
        replaced only in the persisted representation so result files do not
        unexpectedly contain request payloads.

        :return: JSON with binary payloads represented by bounded metadata.
        """
        sanitized = _sanitize_persistence_value(self.model_dump(mode="python"))

        try:
            revalidated = GenerationRequestArguments.model_validate(sanitized)
        except ValidationError:
            # Sanitizing can replace a string with a metadata mapping, which a
            # strictly typed field such as `headers: dict[str, str]` rejects.
            # Serialize the sanitized structure directly rather than raising on
            # the response-compile path; only the type-preserving round trip is
            # lost, and the payload stays redacted.
            return to_json(sanitized, fallback=str).decode()

        return revalidated.model_dump_json()


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
    cached_tokens: int | None = Field(
        default=None,
        description=(
            "Number of input tokens served from the prefix cache (KV cache hit)."
        ),
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

    # Tool call stats (subset of text_tokens)
    tool_call_tokens: int | None = Field(
        default=None,
        description=(
            "Output completion token total for tool-only turns (content null). "
            "Equal to text_tokens when the entire completion is tool output; "
            "None on mixed or text-only turns. Subset of text_tokens. "
            "See mixed_content_tool_tokens for the mixed case."
        ),
    )
    mixed_content_tool_tokens: int | None = Field(
        default=None,
        description=(
            "Output completion token total for mixed content + tool call turns. "
            "Equal to text_tokens when both natural language text and tool calls "
            "are present; None on text-only or tool-only turns. Subset of "
            "text_tokens."
        ),
    )
    tool_call_count: int | None = Field(
        default=None,
        description=(
            "Number of tool calls generated. Set whenever the response includes "
            "tool calls, regardless of whether content is also present."
        ),
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
    columns: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "Columnar data associated with the request, structured as a dictionary "
            "where keys are column names and values are lists of column entries."
        ),
    )
    turn_type: TurnType = Field(
        default="standard",
        description="Discriminator for the kind of turn this request represents. "
        "'standard' is a normal user turn. "
        "'client_tool_call' expects the server to produce tool calls that the "
        "client will handle by injecting tool responses in a follow-up turn. "
        "'tool_response_injection' sends tool output back to the server and "
        "expects a text response. "
        "'server_tool_call' is a turn where the server handles tool execution "
        "end-to-end; it prevents tool_choice='none' from being set so that "
        "server-configured tools remain usable.",
    )
    input_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Input statistics including counts, sizes, and durations.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Output statistics including counts, sizes, and durations.",
    )
