"""Preprocess strategy argument models and short-prompt handlers."""

from __future__ import annotations

import codecs
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

from loguru import logger
from pydantic import Field, field_validator
from transformers import PreTrainedTokenizerBase

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = [
    "ConcatenatePreprocessStrategyArgs",
    "ErrorPreprocessStrategyArgs",
    "IgnorePreprocessStrategyArgs",
    "PadPreprocessStrategyArgs",
    "PreprocessStrategyArgs",
    "PromptTooShortError",
]


class PromptTooShortError(Exception):
    """Raised when a prompt is shorter than the required token length."""


class PreprocessStrategyArgs(
    PydanticClassRegistryMixin["PreprocessStrategyArgs"],
    ABC,
):
    """
    Base class for dataset preprocessing strategy configurations.

    Combines target token-count settings with a short-prompt handling strategy
    selected by ``kind``. Subclasses encapsulate strategy-specific options and
    short-prompt handling logic.
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[PreprocessStrategyArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base PreprocessStrategyArgs class for schema validation
        """
        if cls.__name__ == "PreprocessStrategyArgs":
            return cls

        return PreprocessStrategyArgs

    kind: str = Field(
        description="Short-prompt strategy kind for dataset preprocessing.",
        examples=["ignore", "concatenate", "pad", "error"],
    )
    prompt_tokens: int = Field(
        description="The average number of text tokens retained or added to prompts.",
        examples=[100],
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description=(
            "The standard deviation of the number of tokens retained in or "
            "added to prompts."
        ),
        examples=[10],
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to prompts.",
        examples=[100],
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to prompts.",
        examples=[100],
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens retained or added to outputs.",
        examples=[100],
        gt=0,
    )
    output_tokens_stdev: int | None = Field(
        description=(
            "The standard deviation of the number of tokens retained or "
            "added to outputs."
        ),
        examples=[10],
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to outputs.",
        examples=[100],
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to outputs.",
        examples=[100],
        gt=0,
        default=None,
    )
    prefix_tokens_max: int | None = Field(
        description="The maximum number of text tokens left in the prefixes.",
        gt=0,
        examples=[100],
        default=None,
    )
    count_prefix: bool = Field(
        default=False,
        description=(
            "When True, include prefix tokens in the prompt token budget. "
            "Prefix trimming via prefix_tokens_max still applies when set."
        ),
    )

    @abstractmethod
    def handle_short_prompt(
        self,
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_iterator: Iterator[dict[str, Any]] | None = None,
        prompt_column: str | None = None,
    ) -> str | None:
        """
        Apply this strategy when a prompt is shorter than the target length.

        :param current_prompt: The input prompt string.
        :param min_prompt_tokens: Minimum required token count.
        :param tokenizer: Tokenizer used to count tokens.
        :param dataset_iterator: Optional iterator for strategies that consume
            additional rows (e.g. concatenate).
        :param prompt_column: Optional prompt column name for row extraction.
        :return: Adjusted prompt text, or None to skip the row.
        """


@PreprocessStrategyArgs.register("ignore")
class IgnorePreprocessStrategyArgs(PreprocessStrategyArgs):
    """Skip prompts that are shorter than the target token length."""

    kind: Literal["ignore"] = Field(
        default="ignore",
        description="Skip short prompts instead of extending them.",
    )

    def handle_short_prompt(
        self,
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_iterator: Iterator[dict[str, Any]] | None = None,
        prompt_column: str | None = None,
    ) -> str | None:
        _ = (dataset_iterator, prompt_column)
        if len(tokenizer.encode(current_prompt)) < min_prompt_tokens:
            logger.warning("Prompt too short, ignoring")
            return None
        return current_prompt


@PreprocessStrategyArgs.register("concatenate")
class ConcatenatePreprocessStrategyArgs(PreprocessStrategyArgs):
    """Concatenate successive short prompts until the target length is reached."""

    kind: Literal["concatenate"] = Field(
        default="concatenate",
        description="Concatenate short prompts until the target length is met.",
    )
    delimiter: str = Field(
        default="",
        description="Delimiter inserted between concatenated prompts.",
        examples=["\\n\\n", " "],
    )

    @field_validator("delimiter", mode="before")
    @classmethod
    def _decode_delimiter_escapes(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return codecs.decode(value, "unicode_escape")

    def handle_short_prompt(
        self,
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_iterator: Iterator[dict[str, Any]] | None = None,
        prompt_column: str | None = None,
    ) -> str | None:
        if dataset_iterator is None or prompt_column is None:
            raise ValueError(
                "concatenate strategy requires dataset_iterator and prompt_column"
            )

        tokens_len = len(tokenizer.encode(current_prompt))
        while tokens_len < min_prompt_tokens:
            try:
                next_row = next(dataset_iterator)
            except StopIteration:
                logger.warning(
                    "Could not concatenate enough prompts to reach minimum "
                    "length, ignoring"
                )
                return None
            current_prompt += self.delimiter + next_row[prompt_column]
            tokens_len = len(tokenizer.encode(current_prompt))
        return current_prompt


@PreprocessStrategyArgs.register("pad")
class PadPreprocessStrategyArgs(PreprocessStrategyArgs):
    """Pad short prompts with a character until the target length is reached."""

    kind: Literal["pad"] = Field(
        default="pad",
        description="Pad short prompts with a character until the target length.",
    )
    pad: str = Field(
        default="",
        description="Character or string used to pad short prompts.",
        examples=[" ", "X"],
    )

    @field_validator("pad", mode="before")
    @classmethod
    def _decode_pad_escapes(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return codecs.decode(value, "unicode_escape")

    def handle_short_prompt(
        self,
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_iterator: Iterator[dict[str, Any]] | None = None,
        prompt_column: str | None = None,
    ) -> str | None:
        _ = (dataset_iterator, prompt_column)
        tokens = tokenizer.encode(current_prompt)
        pad_count = 1
        pad_multiplier = 2
        prompt = current_prompt
        while len(tokens) < min_prompt_tokens:
            prompt += self.pad * pad_count
            tokens = tokenizer.encode(prompt)
            pad_count *= pad_multiplier
        return prompt


@PreprocessStrategyArgs.register("error")
class ErrorPreprocessStrategyArgs(PreprocessStrategyArgs):
    """Raise an error when a prompt is shorter than the target length."""

    kind: Literal["error"] = Field(
        default="error",
        description="Raise an error when a prompt is shorter than the target length.",
    )

    def handle_short_prompt(
        self,
        current_prompt: str,
        min_prompt_tokens: int,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_iterator: Iterator[dict[str, Any]] | None = None,
        prompt_column: str | None = None,
    ) -> str | None:
        _ = (dataset_iterator, prompt_column)
        prompt_len = len(tokenizer.encode(current_prompt))
        if prompt_len < min_prompt_tokens:
            raise PromptTooShortError(
                f"Found too short prompt: {current_prompt}, with length: {prompt_len}. "
                f"Minimum length required: {min_prompt_tokens}.",
            )
        return current_prompt
