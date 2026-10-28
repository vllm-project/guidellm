"""Audio output parser for OpenAI-compatible responses."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from .base import ModalityOutputParser, ModalityParserRegistry


@ModalityParserRegistry.register("audio")
class AudioOutputParser(ModalityOutputParser):
    """
    Audio-output parser: OpenAI-shaped message.audio / delta.audio (chat
    completions) and vLLM-Omni's speech.audio.delta SSE event (speech
    endpoint).
    """

    def parse_content_part(self, part: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Ignore content-list parts because audio uses a dedicated field.

        :param part: One item from a message content list.
        :return: None because audio is represented by a dedicated field.
        """
        return None

    def parse_dedicated_field(self, value: Any) -> dict[str, Any] | None:
        """
        Parse message.audio / delta.audio object containing data, expires_at,
        and transcript fields.

        :param value: Standalone audio field or None if absent.
        :return: A normalized output_media item or None.
        :raises ValueError: If value is not a well-formed object or data is not
            valid, non-empty base64-encoded audio.
        """
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] message.audio present but not "
                f"a dictionary object (got {type(value).__name__})"
            )

        raw_data = value.get("data")
        if not raw_data or not isinstance(raw_data, str):
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] message.audio present but "
                "missing a non-empty string 'data' field"
            )

        try:
            audio_bytes = base64.b64decode(raw_data, validate=True)
        except (binascii.Error, ValueError, TypeError) as e:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] message.audio.data is not "
                f"valid base64: {e}"
            ) from e

        if not audio_bytes:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] message.audio.data decoded to "
                "empty audio bytes"
            )

        return {
            "modality": "audio",
            "data": audio_bytes,
            "format": value.get("format"),
            "transcript": value.get("transcript"),
            "id": value.get("id"),
            "expires_at": value.get("expires_at"),
        }
