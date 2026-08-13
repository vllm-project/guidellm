"""Preprocessor for extracting prompts from tool calling datasets.

Handles HuggingFace datasets where prompts are stored as OpenAI-format
``messages`` arrays rather than plain text columns.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from guidellm.data.preprocessors.preprocessor import (
    DataPreprocessorArgs,
    DatasetPreprocessor,
    PreprocessorRegistry,
)

__all__ = ["ToolCallingMessageExtractor", "ToolCallingMessageExtractorArgs"]


@DataPreprocessorArgs.register("tool_calling_message_extractor")
class ToolCallingMessageExtractorArgs(DataPreprocessorArgs):
    """Model for tool calling message extractor preprocessor arguments."""

    kind: Literal["tool_calling_message_extractor"] = Field(
        default="tool_calling_message_extractor",
        description="Type identifier for the preprocessor.",
    )


@PreprocessorRegistry.register("tool_calling_message_extractor")
class ToolCallingMessageExtractor(DatasetPreprocessor):
    """Extract user prompts, system prompts, and tool responses from messages.

    Many tool calling datasets (e.g. ``madroid/glaive-function-calling-openai``)
    store conversations as a ``messages`` column containing an array of
    message dicts. This preprocessor supports both OpenAI format
    (``role``/``content``) and ShareGPT format (``from``/``value``).

    It replaces the ``text_column`` value with the extracted user content,
    populates ``prefix_column`` with the system prompt when present, and
    populates ``tool_response_column`` with tool response content.

    Usage::

        guidellm benchmark run \\
            --data '{"kind": "hf", "source": "..."}' \\
            --data-column-mapper '{"kind": "generative_column_mapper",
                "column_mappings": {"text_column": "messages"}}' \\
            --data-preprocessor kind=tool_calling_message_extractor
    """

    def __init__(self, config: ToolCallingMessageExtractorArgs, **_: Any) -> None:
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
                    user_parts, system_parts, tool_parts = _extract_from_messages(value)
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
                item.setdefault("tool_response_column", []).extend(tool_responses)

        return items


def _normalize_message(msg: dict[str, Any]) -> tuple[str, str]:
    """Extract role and content from a message dict, handling multiple formats.

    Supports OpenAI format (role/content) and ShareGPT format (from/value).
    Normalizes role aliases: ``"human"`` -> ``"user"``.

    :param msg: A single message dict from a conversation array.
    :return: Tuple of (normalized_role, content).
    """
    role = msg.get("role") or msg.get("from") or ""
    content = msg.get("content") or msg.get("value") or ""

    # Normalize role aliases
    if role == "human":
        role = "user"

    return role, content


def _extract_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Pull user, system, and tool response content from a messages array.

    Supports both OpenAI format (role/content) and ShareGPT format (from/value).

    :return: Tuple of (user_parts, system_parts, tool_response_parts).
    """
    user_parts: list[str] = []
    system_parts: list[str] = []
    tool_response_parts: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role, content = _normalize_message(msg)
        if not content or not isinstance(content, str):
            continue

        if role == "user":
            user_parts.append(content)
        elif role == "system":
            system_parts.append(content)
        elif role == "tool":
            tool_response_parts.append(content)

    return user_parts, system_parts, tool_response_parts
