from __future__ import annotations

from typing import Literal, TypeAlias

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

__all__ = [
    "DataNotSupportedError",
    "DatasetType",
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
    "turn_type_column",
    "relative_timestamp_column",
]

DatasetType: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


class DataNotSupportedError(Exception):
    """
    Exception raised when the data format is not supported by deserializer or config.
    """
