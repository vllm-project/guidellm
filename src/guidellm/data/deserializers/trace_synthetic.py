"""
Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.exceptions import DatasetGenerationError
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field, field_serializer
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import TraceColumn, load_trace_rows

__all__ = ["TraceSyntheticDataArgs", "TraceSyntheticDatasetDeserializer"]


def _encode_prompt(
    processor: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Encode text with the configured tokenizer defaults."""
    return processor.encode(text)


def _decode_prompt(
    processor: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    """Decode token ids into a prompt string."""
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


def _create_base_prompt_token_ids(
    processor: PreTrainedTokenizerBase,
    faker: Faker,
    token_count: int,
) -> list[int]:
    """Generate reusable synthetic token ids for trace prompt construction."""
    if token_count <= 0:
        return []

    token_text = (faker.word() or "x")[0]
    text = token_text
    token_ids = _encode_prompt(processor, text)
    max_attempts = 8
    attempts = 0

    while len(token_ids) < token_count and attempts < max_attempts:
        attempts += 1
        missing_tokens = token_count - len(token_ids)
        text = f"{text} {' '.join([token_text] * missing_tokens)}"
        token_ids = _encode_prompt(processor, text)

    if len(token_ids) < token_count:
        raise DataNotSupportedError(
            "Could not generate enough synthetic prompt tokens for "
            f"{token_count} tokens after {max_attempts} attempts"
        )

    return token_ids


def _create_prompt(
    processor: PreTrainedTokenizerBase,
    prompt_tokens_count: int,
    base_prompt_token_ids: list[int],
    request_index: int,
) -> str:
    """
    Build a prompt from unique prefix tokens and reusable base prompt tokens.

    For very small prompt lengths (roughly under 15 tokens, depending on the
    tokenizer), the target slice can truncate the per-row unique prefix before
    it includes the request index, so prompts may become similar across rows and
    less cache-resistant.
    """
    if prompt_tokens_count <= 0:
        return ""

    unique_prefix = f"guidellm-trace-request-{request_index}: "
    prefix_token_ids = _encode_prompt(processor, unique_prefix)
    prompt_token_ids = (prefix_token_ids + base_prompt_token_ids)[:prompt_tokens_count]
    if len(prompt_token_ids) < prompt_tokens_count:
        raise DataNotSupportedError(
            "Could not build a synthetic prompt with "
            f"{prompt_tokens_count} tokens from generated base tokens"
        )

    return _decode_prompt(processor, prompt_token_ids)


def _validate_row(row: dict, config: TraceSyntheticDataArgs) -> None:
    n_in = row[config.prompt_tokens_column]
    n_out = row[config.output_tokens_column]
    if n_in < 0 or n_out < 0:
        raise DataNotSupportedError(
            f"Trace token counts must be non-negative, got "
            f"input_length={n_in}, output_length={n_out}"
        )


@DataArgs.register("trace_synthetic")
class TraceSyntheticDataArgs(DataArgs):
    """Model for synthetic trace dataset deserializer arguments."""

    kind: Literal["trace_synthetic"] = Field(
        default="trace_synthetic",
        description="Type identifier for the trace synthetic dataset deserializer.",
    )
    path: Path = Field(description="Path to the trace file.")
    timestamp_column: str = Field(
        default="timestamp",
        description="Column name for timestamps in the trace file.",
    )
    prompt_tokens_column: str = Field(
        default="input_length",
        description="Column name for prompt token counts in the trace file.",
    )
    output_tokens_column: str = Field(
        default="output_length",
        description="Column name for output token counts in the trace file.",
    )

    @field_serializer("path")
    @classmethod
    def serialize_path(cls, path: Path) -> str:
        """Serialize path as a string because Path is not JSON serializable."""
        return str(path)


class _TraceSyntheticExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic prompt generation. Used to avoid
    pre-generating a prompt for every row in the dataset on load."""

    def __init__(
        self,
        config: TraceSyntheticDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        try:
            self.trace_rows = load_trace_rows(
                config.path,
                TraceColumn(config.timestamp_column, Value("float")),
                required_columns=[
                    TraceColumn(config.prompt_tokens_column, Value("int32")),
                    TraceColumn(config.output_tokens_column, Value("int32")),
                ],
            )
        except (DatasetGenerationError, KeyError, ValueError) as e:
            raise DataNotSupportedError(str(e)) from e
        
        max_prompt_tokens = max(row[config.prompt_tokens_column] for row in self.trace_rows)
        self.base_prompt_token_ids = _create_base_prompt_token_ids(
            self.processor, self.faker, max_prompt_tokens
        )
        for row in self.trace_rows:
            _validate_row(row, self.config)
        self.iteration_count = 0

    def __iter__(self) -> Iterable[tuple[int, dict[str, Any]]]:
        self.iteration_count += 1
        row_idx = 0
        while True:
            try:
                row = self.trace_rows[row_idx]
            except IndexError:
                break

            n_in = row[self.config.prompt_tokens_column]
            prompt = _create_prompt(
                self.processor, n_in, self.base_prompt_token_ids, request_index=row_idx
            )
            timestamps = self.trace_rows[self.config.timestamp_column]
            relative_timestamp = timestamps[row_idx] - timestamps[0]
            yield (
                row_idx,
                {
                    "prompt_tokens_count": row[self.config.prompt_tokens_column],
                    "output_tokens_count": row[self.config.output_tokens_column],
                    "prompt": prompt,
                    "relative_timestamp": relative_timestamp,
                },
            )
            row_idx += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features(
            {
                "prompt": Value("string"),
                "prompt_tokens_count": Value("int32"),
                "output_tokens_count": Value("int32"),
                "relative_timestamp": Value("float"),
            }
        )

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _TraceSyntheticExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _TraceSyntheticExamplesIterable:
        """Returns self as sharding is not implemented yet."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self):
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class _TraceSyntheticDataset(IterableDataset):
    def __init__(
        self,
        config: TraceSyntheticDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        ex_iterable = _TraceSyntheticExamplesIterable(config, processor, random_seed)
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Mooncake trace dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if isinstance(self._ex_iterable, _TraceSyntheticExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("trace_synthetic")
class TraceSyntheticDatasetDeserializer(DatasetDeserializer):
    """
    Load a trace file and generate a synthetic prompt per row.

    Trace file must have timestamp, and columns for prompt and output token counts
    (default: input_length, output_length). Each row becomes one request with
    a synthetic prompt of the requested input length.
    """

    def __call__(
        self,
        config: TraceSyntheticDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        if not config.path.is_file():
            raise DataNotSupportedError(
                f"{type(self).__name__} expects a path to a trace file, "
                f"got {config.path}"
            )
        return _TraceSyntheticDataset(config, processor_factory(), random_seed)
