"""
OUTDATED: Trace file deserializer that generates synthetic prompts per row.

Reads a trace file (timestamp, input_length, output_length) and yields one row per
line with a synthetic prompt matching the requested input_length for replay benchmarks.
"""

from __future__ import annotations

from typing import Literal

from faker import Faker
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.trace_common import (
    TraceDataArgs,
    TraceFormatArgs,
)

__all__ = ["MinimalTraceFormatArgs"]


def _decode_prompt(
    processor: PreTrainedTokenizerBase,
    token_ids: list[int],
) -> str:
    """Decode token ids into a prompt string."""
    decoded = processor.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return decoded


def _generate_token_ids(
    token_count: int,
    processor: PreTrainedTokenizerBase,
    faker: Faker,
) -> list[int]:
    """Generate `token_count` synthetic token ids for trace prompt construction."""
    # Ideally, `margin_of_safety` should be set to slighty more than
    # the average number of characters used by tokenizers to form one token.
    margin_of_safety = 8
    attempt = 0
    while True:
        attempt += 1
        # The Faker.text() can only generate text of at least 5 characters.
        num_chars = max(token_count * margin_of_safety * attempt, 5)
        text = faker.text(max_nb_chars=num_chars)
        token_ids = processor.encode(text)
        if len(token_ids) >= token_count:
            return token_ids[:token_count]


@TraceFormatArgs.register("minimal")
class MinimalTraceFormatArgs(TraceFormatArgs):
    """TODO"""

    kind: Literal["minimal"] = Field(
        default="minimal",
        description="Type identifier for the minimal trace format.",
    )

    @classmethod
    def create_prompt(
        cls,
        row: dict,  # noqa: ARG002
        config: TraceDataArgs,  # noqa: ARG002
        processor: PreTrainedTokenizerBase,  # noqa: ARG002
        faker: Faker,  # noqa: ARG002
    ) -> str:
        token_ids = _generate_token_ids(
            row[config.prompt_tokens_column], processor, faker
        )
        return _decode_prompt(processor, token_ids)
