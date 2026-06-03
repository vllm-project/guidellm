"""
Trace deserializer for Mooncake formatted files that generates synthetic prompts per row.

Reads a trace file (timestamp, input_length, output_length, hash_ids) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from datasets.exceptions import DatasetGenerationError
from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.trace_synthetic import TraceSyntheticDataArgs
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import load_trace_rows

__all__ = ["TraceMooncakeDataArgs", "TraceMooncakeDatasetDeserializer"]


@DataArgs.register("trace_mooncake")
class TraceMooncakeDataArgs(TraceSyntheticDataArgs):
    hash_ids_column: str = Field(
        default = "hash_ids",
        description = "Column name for lists of hash IDs in the trace file."
    )
    hash_id_block_size: int = Field(
        gt = 0,
        default = 512,  # Default used in Mooncake's paper https://arxiv.org/pdf/2407.00079
        description = "Amount of tokens represented by one hash ID."
    )


@DatasetDeserializerFactory.register("trace_mooncake")
class TraceMooncakeDatasetDeserializer(DatasetDeserializer):
    """Mooncake trace format requires a column for timestamps, prompt token counts,
    ouput token counts and lists of hash IDs.

    Hash IDs are globally unique identifiers based on the current and previous token
    blocks in a prompt. The relationships of IDs forms a tree, where every first ID
    in a prompt has a parent node of `None`. Parent nodes can have an unbounded
    number of children. Two hash IDs can represent identical blocks of tokens so long
    as they do not share the same parent (previous ID). For more details, see section 4 of
    https://arxiv.org/pdf/2407.00079.

    Generated prompts match the prompt token count of the row."""

    def __call__(
        self,
        config: TraceMooncakeDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int
    ) -> Dataset:
        ...
