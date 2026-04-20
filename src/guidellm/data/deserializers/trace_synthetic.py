"""
Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.utils.trace_io import load_trace_rows

__all__ = ["TraceSyntheticDatasetDeserializer"]


def _create_prompt(
    processor: PreTrainedTokenizerBase,
    prompt_tokens_count: int,
    faker: Faker,
    unique: str = "",
) -> str:
    """Generate text that tokenizes to exactly prompt_tokens_count tokens."""
    prompt_token_ids: list[int] = []
    avg_chars_per_token = 5
    margin_of_safety = 1.5
    attempts = 0

    while len(prompt_token_ids) < prompt_tokens_count:
        attempts += 1
        num_chars = int(
            prompt_tokens_count * avg_chars_per_token * margin_of_safety * attempts
        )
        text = unique + faker.text(max_nb_chars=num_chars)
        prompt_token_ids = processor.encode(text)

    decoded = processor.decode(
        prompt_token_ids[:prompt_tokens_count], skip_special_tokens=True
    )
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


def _load_trace_rows(
    path: Path,
    timestamp_column: str,
    prompt_tokens_column: str,
    output_tokens_column: str,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Load trace file into list of dicts with timestamp, prompt_tokens,
    output_tokens."""
    try:
        raw = load_trace_rows(
            path,
            required_columns=[
                timestamp_column,
                prompt_tokens_column,
                output_tokens_column,
            ],
            max_rows=max_rows,
        )
    except (KeyError, ValueError) as e:
        raise DataNotSupportedError(str(e)) from e
    return [
        {
            "timestamp": float(row[timestamp_column]),
            "prompt_tokens": int(row[prompt_tokens_column]),
            "output_tokens": int(row[output_tokens_column]),
        }
        for row in raw
    ]


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
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> Dataset:
        if (
            not isinstance(data, str | Path)
            or not (path := Path(data)).exists()
            or not path.is_file()
        ):
            raise DataNotSupportedError(
                "TraceSyntheticDatasetDeserializer expects a path to a trace file, "
                f"got {data}"
            )
        timestamp_column = str(data_kwargs.pop("timestamp_column", "timestamp"))
        prompt_tokens_column = str(
            data_kwargs.pop("prompt_tokens_column", "input_length")
        )
        output_tokens_column = str(
            data_kwargs.pop("output_tokens_column", "output_length")
        )
        max_rows_val = data_kwargs.pop("max_rows", None)
        max_rows: int | None = None
        if max_rows_val is not None:
            if isinstance(max_rows_val, int):
                max_rows = max_rows_val
            elif isinstance(max_rows_val, str):
                max_rows = int(max_rows_val)

        rows = _load_trace_rows(
            path, timestamp_column, prompt_tokens_column, output_tokens_column, max_rows
        )
        if not rows:
            raise DataNotSupportedError("Trace file is empty")

        processor = processor_factory()
        faker = Faker()
        faker.seed_instance(random_seed)

        prompts: list[str] = []
        prompt_tokens_counts: list[int] = []
        output_tokens_counts: list[int] = []
        for i, row in enumerate(rows):
            n_in = row["prompt_tokens"]
            n_out = row["output_tokens"]
            prompt = _create_prompt(processor, n_in, faker, unique=f"{i} ")
            prompts.append(prompt)
            prompt_tokens_counts.append(n_in)
            output_tokens_counts.append(n_out)

        # Avoid passing deserializer-only keys to Dataset.from_dict
        data_kwargs.pop("type_", None)

        return Dataset.from_dict(
            {
                "prompt": prompts,
                "prompt_tokens_count": prompt_tokens_counts,
                "output_tokens_count": output_tokens_counts,
            },
            **data_kwargs,
        )
