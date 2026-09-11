"""
Multimodal content accounting utilities for the mock server.

Provides validation and prompt token accounting for the multimodal chat
completion content parts that GuideLLM's OpenAI-compatible backend produces:
``image_url``, ``video_url``, and ``input_audio``. Token charges use simple,
configurable heuristics so benchmarks receive plausible per-modality usage
statistics without running real encoders. Audio duration is estimated from the
payload size using the byte rates of GuideLLM's synthetic audio encoding
defaults (PCM16 mono at 16 kHz for WAV, 64 kbit/s for lossy formats).
"""

from __future__ import annotations

import base64
import binascii
import math

from pydantic import BaseModel, Field

from guidellm.mock_server.models import ChatMessage
from guidellm.schemas.mock_server import MockServerConfig

__all__ = [
    "LOSSY_BYTES_PER_SECOND",
    "WAV_BYTES_PER_SECOND",
    "InvalidContentPartError",
    "MultimodalPromptStats",
    "accumulate_multimodal_content",
    "estimate_audio_seconds",
]

WAV_BYTES_PER_SECOND = 32000
"""Bytes per second for WAV payloads, assuming PCM16 mono at 16 kHz."""

LOSSY_BYTES_PER_SECOND = 8000
"""Bytes per second for lossy audio payloads, assuming 64 kbit/s encoding."""


class InvalidContentPartError(ValueError):
    """Raised when a multimodal content part fails shape validation."""


class MultimodalPromptStats(BaseModel):
    """
    Per-modality prompt token accounting for a multimodal chat request.

    Fields are ``None`` when the request contains no parts of that modality so
    that text-only requests keep their existing usage payloads and benchmark
    metrics do not record spurious zero values for absent modalities.
    """

    image_tokens: int | None = Field(
        default=None, description="Prompt tokens charged for image content parts"
    )
    video_tokens: int | None = Field(
        default=None, description="Prompt tokens charged for video content parts"
    )
    audio_tokens: int | None = Field(
        default=None, description="Prompt tokens charged for audio content parts"
    )
    audio_seconds: float | None = Field(
        default=None, description="Estimated total seconds of audio input"
    )

    @property
    def total_tokens(self) -> int:
        """
        :return: Total multimodal prompt tokens across all modalities
        """
        return (
            (self.image_tokens or 0)
            + (self.video_tokens or 0)
            + (self.audio_tokens or 0)
        )

    def prompt_tokens_details(self, text_tokens: int) -> dict[str, int | float] | None:
        """
        Build a vLLM-style ``prompt_tokens_details`` usage breakdown.

        :param text_tokens: Number of text-only prompt tokens for the request
        :return: Details dict with per-modality token counts, or None when the
            request contained no multimodal content
        """
        if (
            self.image_tokens is None
            and self.video_tokens is None
            and self.audio_tokens is None
        ):
            return None

        details: dict[str, int | float] = {"prompt_tokens": text_tokens}
        if self.image_tokens is not None:
            details["image_tokens"] = self.image_tokens
        if self.video_tokens is not None:
            details["video_tokens"] = self.video_tokens
        if self.audio_tokens is not None:
            details["audio_tokens"] = self.audio_tokens
        if self.audio_seconds is not None:
            details["seconds"] = self.audio_seconds
        return details


def estimate_audio_seconds(num_bytes: int, format_hint: str | None) -> float:
    """
    Estimate audio duration in seconds from payload size and format.

    Mirrors the encoding defaults of GuideLLM's synthetic audio pipeline: WAV
    (and raw PCM) payloads are assumed to be PCM16 mono at 16 kHz, while all
    other formats are assumed to be encoded at 64 kbit/s.

    :param num_bytes: Size of the decoded audio payload in bytes
    :param format_hint: Format name, file name, or MIME type of the audio
    :return: Estimated duration of the audio in seconds
    """
    hint = (format_hint or "").lower()
    bytes_per_second = (
        WAV_BYTES_PER_SECOND
        if "wav" in hint or "pcm" in hint
        else LOSSY_BYTES_PER_SECOND
    )
    return num_bytes / bytes_per_second


def _validate_text_part(part: dict) -> None:
    """Validate a ``text`` content part."""
    if not isinstance(part.get("text"), str):
        raise InvalidContentPartError(
            "Content part 'text' requires a string 'text' field"
        )


def _validate_url_part(part: dict, part_type: str) -> None:
    """Validate an ``image_url`` or ``video_url`` content part."""
    url_obj = part.get(part_type)
    if not isinstance(url_obj, dict) or not isinstance(url_obj.get("url"), str):
        raise InvalidContentPartError(
            f"Content part '{part_type}' requires an object '{part_type}' "
            "with a string 'url' field"
        )
    if not url_obj["url"]:
        raise InvalidContentPartError(
            f"Content part '{part_type}' requires a non-empty 'url'"
        )


def _decode_audio_part(part: dict) -> tuple[bytes, str | None]:
    """Validate an ``input_audio`` content part and decode its payload."""
    audio_obj = part.get("input_audio")
    if not isinstance(audio_obj, dict) or not isinstance(audio_obj.get("data"), str):
        raise InvalidContentPartError(
            "Content part 'input_audio' requires an object 'input_audio' "
            "with a base64 string 'data' field"
        )
    if not audio_obj["data"]:
        raise InvalidContentPartError(
            "Content part 'input_audio' requires non-empty base64 'data'"
        )
    audio_format = audio_obj.get("format")
    if audio_format is not None and not isinstance(audio_format, str):
        raise InvalidContentPartError(
            "Content part 'input_audio' has a non-string 'format' field"
        )
    try:
        audio_bytes = base64.b64decode(audio_obj["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidContentPartError(
            f"Content part 'input_audio' has invalid base64 data: {exc}"
        ) from exc
    return audio_bytes, audio_format


def accumulate_multimodal_content(
    messages: list[ChatMessage], config: MockServerConfig
) -> MultimodalPromptStats:
    """
    Validate multimodal content parts and accumulate prompt token charges.

    Walks every message's content list, validates the shape of each content
    part, and charges tokens per part using the configured heuristics: a flat
    per-image and per-video token count, and a per-second rate applied to the
    estimated duration of ``input_audio`` payloads.

    :param messages: Chat messages from a validated chat completions request
    :param config: Mock server configuration with multimodal token settings
    :return: Accumulated per-modality prompt token statistics
    :raises InvalidContentPartError: When a content part fails shape validation
    """
    images = 0
    videos = 0
    audio_parts = 0
    audio_seconds = 0.0

    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            part_type = part.get("type")
            if part_type == "text":
                _validate_text_part(part)
            elif part_type in ("image_url", "video_url"):
                _validate_url_part(part, part_type)
                if part_type == "image_url":
                    images += 1
                else:
                    videos += 1
            elif part_type == "input_audio":
                audio_bytes, audio_format = _decode_audio_part(part)
                audio_parts += 1
                audio_seconds += estimate_audio_seconds(len(audio_bytes), audio_format)
            else:
                raise InvalidContentPartError(
                    f"Unsupported content part type: {part_type!r}"
                )

    return MultimodalPromptStats(
        image_tokens=images * config.image_tokens if images else None,
        video_tokens=videos * config.video_tokens if videos else None,
        audio_tokens=(
            math.ceil(audio_seconds * config.audio_tokens_per_second)
            if audio_parts
            else None
        ),
        audio_seconds=round(audio_seconds, 3) if audio_parts else None,
    )
