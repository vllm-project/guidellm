from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataPreprocessorArgs


@DataPreprocessorArgs.register("tool_calling_message_extractor")
class ToolCallingMessageExtractorArgs(DataPreprocessorArgs):
    """Model for tool calling message extractor preprocessor arguments."""

    kind: Literal["tool_calling_message_extractor"] = Field(
        default="tool_calling_message_extractor",
        description="Type identifier for the preprocessor.",
    )
