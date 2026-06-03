"""
Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from datasets.exceptions import DatasetGenerationError
from faker import Faker
from pydantic import Field, field_serializer
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.trace_io import load_trace_rows

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


def _load_trace_rows(
    path: Path,
    timestamp_column: str,
    prompt_tokens_column: str,
    output_tokens_column: str,
) -> list[dict[str, Any]]:
    """Load trace file into list of dicts with timestamp, prompt_tokens,
    output_tokens."""
    try:
        raw = load_trace_rows(
            path,
            required_columns=[
                prompt_tokens_column,
                output_tokens_column,
            ],
            timestamp_column=timestamp_column,
        )
    except (DatasetGenerationError, KeyError, ValueError) as e:
        raise DataNotSupportedError(str(e)) from e
    try:
        return [
            {
                "timestamp": float(row[timestamp_column]),
                "prompt_tokens": int(row[prompt_tokens_column]),
                "output_tokens": int(row[output_tokens_column]),
            }
            for row in raw
        ]
    except (TypeError, ValueError) as e:
        raise DataNotSupportedError(str(e)) from e


@DataArgs.register("trace_synthetic")
class TraceSyntheticDataArgs(DataArgs):
    """
    DataArgs for TraceSyntheticDatasetDeserializer.
    """

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
    ) -> Dataset:
        if not (path := Path(config.path)).exists() or not path.is_file():
            raise DataNotSupportedError(
                "TraceSyntheticDatasetDeserializer expects a path to a trace file, "
                f"got {path}"
            )
        rows = _load_trace_rows(
            path,
            config.timestamp_column,
            config.prompt_tokens_column,
            config.output_tokens_column,
        )
        if not rows:
            raise DataNotSupportedError("Trace file is empty")

        processor = processor_factory()
        faker = Faker()
        faker.seed_instance(random_seed)
        max_prompt_tokens = max(row["prompt_tokens"] for row in rows)
        base_prompt_token_ids = _create_base_prompt_token_ids(
            processor, faker, max_prompt_tokens
        )

        prompts: list[str] = []
        prompt_tokens_counts: list[int] = []
        output_tokens_counts: list[int] = []
        for i, row in enumerate(rows):
            n_in = row["prompt_tokens"]
            n_out = row["output_tokens"]
            if n_in < 0 or n_out < 0:
                raise DataNotSupportedError(
                    "Trace token counts must be non-negative, got "
                    f"input_length={n_in}, output_length={n_out}"
                )
            prompt = _create_prompt(
                processor, n_in, base_prompt_token_ids, request_index=i
            )
            prompts.append(prompt)
            prompt_tokens_counts.append(n_in)
            output_tokens_counts.append(n_out)

        return Dataset.from_dict(
            {
                "prompt": prompts,
                "prompt_tokens_count": prompt_tokens_counts,
                "output_tokens_count": output_tokens_counts,
            },
            **config.load_kwargs,
        )
