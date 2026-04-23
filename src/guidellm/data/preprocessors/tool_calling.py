"""Preprocessor for extracting prompts from tool calling datasets.

Handles HuggingFace datasets where prompts are stored as OpenAI-format
``messages`` arrays rather than plain text columns.
"""

from __future__ import annotations

from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)

__all__ = ["ToolCallingMessageExtractor"]


@PreprocessorRegistry.register("tool_calling_message_extractor")
class ToolCallingMessageExtractor(DatasetPreprocessor):
    """Extract user prompts and system prompts from an OpenAI ``messages`` array.

    Many tool calling datasets (e.g. ``madroid/glaive-function-calling-openai``)
    store conversations as a ``messages`` column containing an array of
    ``{"role": ..., "content": ...}`` dicts.  This preprocessor replaces the
    ``text_column`` value with the extracted user content and populates
    ``prefix_column`` with the system prompt when present.

    Usage::

        guidellm benchmark run \\
            --data madroid/glaive-function-calling-openai \\
            --data-column-mapper \\
                '{"text_column": "messages", "tools_column": "tools"}' \\
            --data-preprocessors tool_calling_message_extractor,encode_media
    """

    def __init__(self, **_: Any) -> None:
        pass

    def __call__(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            text_values = item.get("text_column")
            if not text_values or not isinstance(text_values, list):
                continue

            new_texts: list[str] = []
            prefixes: list[str] = []

            for value in text_values:
                if isinstance(value, list):
                    user_parts, system_parts = _extract_from_messages(value)
                    if user_parts:
                        new_texts.append(" ".join(user_parts))
                    if system_parts:
                        prefixes.append(" ".join(system_parts))
                elif isinstance(value, str):
                    new_texts.append(value)

            if new_texts:
                item["text_column"] = new_texts
            if prefixes:
                item.setdefault("prefix_column", []).extend(prefixes)

        return items


def _extract_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Pull user and system content from an OpenAI messages array."""
    user_parts: list[str] = []
    system_parts: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue

        if role == "user":
            user_parts.append(content)
        elif role == "system":
            system_parts.append(content)

    return user_parts, system_parts
