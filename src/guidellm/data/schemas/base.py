from __future__ import annotations

from typing import Literal, TypeAlias

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

__all__ = [
    "DataNotSupportedError",
    "DatasetDictType",
    "DatasetType",
    "GenerativeDatasetColumnType",
    "IndefiniteDataset",
    "InvalidRowError",
]


GenerativeDatasetColumnType = Literal[
    "prompt_tokens_count_column",
    "output_tokens_count_column",
    "prefix_column",
    "text_column",
    "raw_messages_column",
    "image_column",
    "video_column",
    "audio_column",
    "tools_column",
    "tool_response_column",
    "tool_choice_column",
    "turn_type_column",
    "relative_timestamp_column",
    "requeue_delay_column",
    "request_duration_column",
    "conversation_turns_column",
]

DatasetType: TypeAlias = Dataset | IterableDataset


DatasetDictType: TypeAlias = DatasetType | DatasetDict | IterableDatasetDict


class IndefiniteDataset:
    """
    Marker for a dataset whose iteration never raises ``StopIteration``.

    Synthetic generators use this so a full-dataset prefetch can fail before
    the scheduler waits for an end that will not arrive. Finite file, trace,
    and Hugging Face datasets do not inherit it.
    """


class DataNotSupportedError(Exception):
    """
    Exception raised when the data format is not supported by deserializer or config.
    """


class InvalidRowError(Exception):
    """A single dataset row failed validation and should be skipped."""
