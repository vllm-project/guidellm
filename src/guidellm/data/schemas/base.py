from __future__ import annotations

from typing import Literal

__all__ = [
    "DataNotSupportedError",
    "GenerativeDatasetColumnType",
]


GenerativeDatasetColumnType = Literal[
    "prompt_tokens_count_column",
    "output_tokens_count_column",
    "prefix_column",
    "text_column",
    "image_column",
    "video_column",
    "audio_column",
    "tools_column",
    "tool_response_column",
]


class DataNotSupportedError(Exception):
    """
    Exception raised when the data format is not supported by deserializer or config.
    """
