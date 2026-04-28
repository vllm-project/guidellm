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
    """Extract user prompts, system prompts, and tool responses from messages.

    Many tool calling datasets (e.g. ``madroid/glaive-function-calling-openai``)
    store conversations as a ``messages`` column containing an array of
    ``{"role": ..., "content": ...}`` dicts.  This preprocessor replaces the
    ``text_column`` value with the extracted user content, populates
    ``prefix_column`` with the system prompt when present, and populates
    ``tool_response_column`` with ``role: "tool"`` response content.

    Usage::

        guidellm benchmark run \\
            --data madroid/glaive-function-calling-openai \\
            --data-column-mapper \\
                '{"text_column": "messages", "tools_column": "tools"}' \\
            --data-preprocessors tool_calling_message_extractor,encode_media
    """

    def __init__(self, **_: Any) -> None:
        pass

    def __call__(  # noqa: C901
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for item in items:
            text_values = item.get("text_column")
            if not text_values or not isinstance(text_values, list):
                continue

            new_texts: list[str] = []
            prefixes: list[str] = []
            tool_responses: list[str] = []

            for value in text_values:
                if isinstance(value, list):
                    user_parts, system_parts, tool_parts = (
                        _extract_from_messages(value)
                    )
                    if user_parts:
                        new_texts.append(" ".join(user_parts))
                    if system_parts:
                        prefixes.append(" ".join(system_parts))
                    tool_responses.extend(tool_parts)
                elif isinstance(value, str):
                    new_texts.append(value)

            if new_texts:
                item["text_column"] = new_texts
            if prefixes:
                item.setdefault("prefix_column", []).extend(prefixes)
            if tool_responses:
                item.setdefault("tool_response_column", []).extend(
                    tool_responses
                )

        return items


def _extract_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Pull user, system, and tool response content from an OpenAI messages array.

    :return: Tuple of (user_parts, system_parts, tool_response_parts).
    """
    user_parts: list[str] = []
    system_parts: list[str] = []
    tool_response_parts: list[str] = []

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
        elif role == "tool":
            tool_response_parts.append(content)

    return user_parts, system_parts, tool_response_parts
