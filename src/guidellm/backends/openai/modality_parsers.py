"""
Registry of per-modality parsers for non-text output embedded in chat
completion responses (message.content list items, message.audio, etc.).

Each phase (audio, image, video) registers exactly one parser for its
modality here and never touches the other phases' registrations or the
dispatch logic below. This keeps "what modalities does this build of
GuideLLM understand" a pure function of which phases have landed, with zero
shared code to merge-conflict on.
"""

from __future__ import annotations

from typing import Any, Protocol

from guidellm.utils.registry import RegistryMixin

__all__ = ["ModalityOutputParser", "ModalityParserRegistry"]


class ModalityOutputParser(Protocol):
    """
    Protocol for a single-modality output parser.

    Unparseable/malformed item is treated as a normal
    "no output of this modality in this response" case, not an error
    """

    def parse_content_part(self, part: dict[str, Any]) -> dict[str, Any] | None:
        """
        Parse one item from a chat completion message's list-shaped
        ``content`` (e.g. an ``image_url`` part) into a normalized
        ``output_media`` item, or return None if ``part`` isn't this
        modality's shape.

        :param part: One dict from ``message["content"]`` when it's a list.
        :return: A dict with at least {"modality": ..., "data": ..., "format": ...},
            or None when ``part`` doesn't match this modality.
        """
        ...

    def parse_dedicated_field(self, value: Any) -> dict[str, Any] | None:
        """
        Parse a modality-specific top-level response field (e.g.
        ``message.audio``), or return None if absent/malformed.

        :param value: The raw field value (e.g. ``message.get("audio")``).
        :return: A normalized output_media item, or None.
        """
        ...


class ModalityParserRegistry(RegistryMixin["type[ModalityOutputParser]"]):
    """
    Registry mapping a short modality name ("audio", "image", "video", ...)
    to its parser implementation. Uses RegistryMixin pattern
    (see also OpenAIRequestHandlerFactory, DatasetDeserializerFactory).
    """
