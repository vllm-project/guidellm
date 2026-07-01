from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from random import Random
from typing import Any, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import BaseModel, Field, field_validator, model_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.settings import settings
from guidellm.utils.imports import json
from guidellm.utils.random import FloatRangeSampler, IntegerRangeSampler

__all__ = [
    "SyntheticTextDataArgs",
    "SyntheticTextDataset",
    "SyntheticTextDatasetDeserializer",
    "SyntheticTextPrefixBucketConfig",
]

# Placeholder tool definition used when the user doesn't supply their own
# tools but configures tool_call_turns with at least one turn.
DEFAULT_SYNTHETIC_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Retrieve data from the system",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The query"}},
                "required": ["query"],
            },
        },
    }
]


class SyntheticTextPrefixBucketConfig(BaseModel):
    bucket_weight: int = Field(
        description="Weight of this bucket in the overall distribution.",
        gt=0,
        default=100,
    )
    prefix_count: int = Field(
        description="The number of unique prefixes to generate for this bucket.",
        ge=1,
        default=1,
    )
    prefix_tokens: int = Field(
        description="The number of prefix tokens per-prompt for this bucket.",
        ge=0,
        default=0,
    )


@DataArgs.register("synthetic_text")
class SyntheticTextDataArgs(DataArgs):
    """Model for synthetic text dataset deserializer arguments."""

    kind: Literal["synthetic_text"] = Field(  # type: ignore[assignment]
        default="synthetic_text",
        description="Type identifier for the synthetic text dataset configuration.",
    )
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts.",
        gt=0,
        examples=[30],
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for prompts.",
        gt=0,
        default=None,
        examples=[3],
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for prompts.",
        gt=0,
        default=None,
        examples=[10],
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for prompts.",
        gt=0,
        default=None,
        examples=[30],
    )
    output_tokens: int | None = Field(
        description=(
            "The average number of text tokens generated for outputs. "
            "When omitted, output tokens are not sampled and ``max_tokens`` is left "
            "to the backend default. Useful for endpoints that do not produce "
            "output tokens (e.g. embeddings)."
        ),
        gt=0,
        default=None,
        examples=[10],
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for outputs.",
        gt=0,
        default=None,
        examples=[3],
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for outputs.",
        gt=0,
        default=None,
        examples=[10],
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for outputs.",
        gt=0,
        default=None,
        examples=[30],
    )
    delay: float | None = Field(
        description='The average requeue delay, or "think time" for prompts.',
        gt=0,
        default=None,
        examples=[10.0],
    )
    delay_stdev: float | None = Field(
        description=(
            'The standard deviation of requeue delays, or "think time" for prompts.'
        ),
        gt=0,
        default=None,
        examples=[1.0],
    )
    delay_min: float | None = Field(
        description='The minimum requeue delay, or "think time" for prompts.',
        gt=0,
        default=None,
        examples=[0.5],
    )
    delay_max: float | None = Field(
        description='The maximum requeue delay, or "think time" for prompts.',
        gt=0,
        default=None,
        examples=[5.0],
    )
    turns: int = Field(
        description="The number of turns in the conversation.",
        gt=0,
        default=1,
    )
    tool_call_turns: list[int] = Field(
        description="Which turns should include tool definitions and expect "
        "tool-call responses. An int N means 'the first N turns'; a list "
        "of ints specifies explicit 0-based turn indices (e.g. [0, 2]). "
        "Normalized to a sorted list after validation. "
        "When 0 or [] (default), no tool calling is configured.",
        default_factory=list,
        examples=[1, [0, 1]],
    )
    tools: list[dict[str, Any]] | None = Field(
        description="Tool definitions in OpenAI format. When tool_call_turns is "
        "non-empty and this is None, a static placeholder tool definition is used.",
        default=None,
        examples=[
            {
                "type": "function",
                "function": {
                    "name": "get_data",
                    "description": "Retrieve data from the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query"}
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    )
    tool_response_tokens: int | None = Field(
        description="Average number of tokens for synthetic tool call responses. "
        "When None (default), a short placeholder response is used.",
        gt=0,
        default=None,
        examples=[10],
    )
    tool_response_tokens_stdev: int | None = Field(
        description="Standard deviation for tool response token count.",
        gt=0,
        default=None,
        examples=[1],
    )
    tool_response_tokens_min: int | None = Field(
        description="Minimum number of tokens for tool response.",
        gt=0,
        default=None,
        examples=[5],
    )
    tool_response_tokens_max: int | None = Field(
        description="Maximum number of tokens for tool response.",
        gt=0,
        default=None,
        examples=[20],
    )

    prefix_buckets: list[SyntheticTextPrefixBucketConfig] | None = Field(
        description="Buckets for the prefix tokens distribution.",
        default=None,
        examples=[
            {"bucket_weight": 100, "prefix_count": 1, "prefix_tokens": 0},
        ],
    )

    @model_validator(mode="after")
    def check_prefix_options(self) -> SyntheticTextDataArgs:
        if self.__pydantic_extra__ is not None:
            prefix_count = self.__pydantic_extra__.get("prefix_count", None)  # type: ignore[attr-defined]
            prefix_tokens = self.__pydantic_extra__.get("prefix_tokens", None)  # type: ignore[attr-defined]

            if prefix_count is not None or prefix_tokens is not None:
                if self.prefix_buckets:
                    raise ValueError(
                        "prefix_buckets is mutually exclusive"
                        " with prefix_count and prefix_tokens"
                    )

                self.prefix_buckets = [
                    SyntheticTextPrefixBucketConfig(
                        prefix_count=prefix_count or 1,
                        prefix_tokens=prefix_tokens or 0,
                    )
                ]

        return self

    @field_validator("tool_call_turns", mode="before")
    @classmethod
    def _coerce_tool_call_turns(cls, v: int | list[int]) -> list[int]:
        """Convert an int N to [0, ..., N-1]; pass lists through sorted."""
        if isinstance(v, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                v = json.loads(v)
        if isinstance(v, int):
            if v < 0:
                raise ValueError("tool_call_turns int must be >= 0")
            return list(range(v))
        if len(v) != len(set(v)):
            raise ValueError("tool_call_turns list must not contain duplicates")
        return sorted(v)

    @model_validator(mode="after")
    def _validate_tool_call_turn_indices(self) -> SyntheticTextDataArgs:
        """Ensure all tool_call_turns indices are within [0, turns)."""
        for idx in self.tool_call_turns:
            if idx < 0 or idx >= self.turns:
                raise ValueError(
                    f"tool_call_turns index {idx} out of range [0, {self.turns})"
                )
        return self


class _SyntheticTextExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic text generation."""

    def __init__(
        self,
        config: SyntheticTextDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        self.iteration_count = 0

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        iter_random_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_random_seed)
        prompt_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.prompt_tokens,
                variance=self.config.prompt_tokens_stdev,
                min_value=self.config.prompt_tokens_min,
                max_value=self.config.prompt_tokens_max,
                random_seed=iter_random_seed,
            )
        )
        output_tokens_sampler = (
            iter(
                IntegerRangeSampler(
                    average=self.config.output_tokens,
                    variance=self.config.output_tokens_stdev,
                    min_value=self.config.output_tokens_min,
                    max_value=self.config.output_tokens_max,
                    random_seed=iter_random_seed + 1,  # ensure diff dist from prompts
                )
            )
            if self.config.output_tokens is not None
            else None
        )
        delay_sampler = (
            iter(
                FloatRangeSampler(
                    average=self.config.delay,
                    variance=self.config.delay_stdev,
                    min_value=self.config.delay_min,
                    max_value=self.config.delay_max,
                    # ensure diff dist from prompts and outputs
                    random_seed=iter_random_seed + 2,
                )
            )
            if self.config.delay is not None
            else None
        )

        # Create a shared prefix if specified
        rand = Random(iter_random_seed + 3)
        prefix_iter = self._create_prefix_iter(faker, rand)
        samples_count = 0

        # Resolve tool definitions for tool-call turns
        tool_call_turns_set = set(self.config.tool_call_turns)
        tools_defs: list[dict[str, Any]] | None = None
        if tool_call_turns_set:
            tools_defs = self.config.tools or DEFAULT_SYNTHETIC_TOOLS

        # Optional sampler for variable-length tool responses
        tool_response_sampler: Iterator[int] | None = None
        if self.config.tool_response_tokens is not None:
            tool_response_sampler = iter(
                IntegerRangeSampler(
                    average=self.config.tool_response_tokens,
                    variance=self.config.tool_response_tokens_stdev,
                    min_value=self.config.tool_response_tokens_min,
                    max_value=self.config.tool_response_tokens_max,
                    random_seed=iter_random_seed + 2,
                )
            )

        while True:
            prompt_tokens_count = next(prompt_tokens_sampler)
            output_tokens_count = (
                next(output_tokens_sampler)
                if output_tokens_sampler is not None
                else None
            )
            delay = next(delay_sampler) if delay_sampler is not None else None

            row: dict[str, Any] = {"prefix": next(prefix_iter)}
            for turn in range(self.config.turns):
                row[f"prompt_{turn}"] = self._create_prompt(
                    prompt_tokens_count,
                    faker,
                    f"{self.iteration_count} {samples_count} ",
                )
                row[f"prompt_tokens_count_{turn}"] = prompt_tokens_count
                if output_tokens_count is not None:
                    row[f"output_tokens_count_{turn}"] = output_tokens_count
                if delay is not None:
                    row[f"delay_{turn}"] = delay

                if tools_defs is not None and turn in tool_call_turns_set:
                    row[f"tools_{turn}"] = json.dumps(tools_defs)

                    if tool_response_sampler is not None:
                        tr_tokens = next(tool_response_sampler)
                        body = self._create_prompt(tr_tokens, faker)
                        row[f"tool_response_{turn}"] = json.dumps({"result": body})
                    else:
                        row[f"tool_response_{turn}"] = (
                            settings.default_synthetic_tool_response
                        )

                samples_count += 1

            yield samples_count, row

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        features: dict[str, Any] = {"prefix": Value("string")}
        for i in range(self.config.turns):
            features[f"prompt_{i}"] = Value("string")
            features[f"prompt_tokens_count_{i}"] = Value("int32")
            if self.config.output_tokens is not None:
                features[f"output_tokens_count_{i}"] = Value("int32")
            if self.config.delay is not None:
                features[f"delay_{i}"] = Value("float")

            if i in set(self.config.tool_call_turns):
                # Tools column is a JSON-serialised list; store as string
                # to keep the HuggingFace Features schema simple.
                features[f"tools_{i}"] = Value("large_string")
                features[f"tool_response_{i}"] = Value("large_string")

        return Features(features)

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data doesn't have fixed sources to shuffle."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data generation is infinite and stateless."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict

    def _create_prompt(
        self, prompt_tokens_count: int, faker: Faker, unique: str = ""
    ) -> str:
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
            prompt_token_ids = self.processor.encode(text)

        return self.processor.decode(  # type: ignore[return-value]
            prompt_token_ids[:prompt_tokens_count], skip_special_tokens=True
        )

    def _create_prefix_iter(self, faker: Faker, rand: Random) -> Iterator[str]:
        if not self.config.prefix_buckets:
            while True:
                yield ""

        # Increase weights to ensure an integer number of samples per per-prefix
        least_common_prefix_count = math.lcm(
            *(bucket.prefix_count for bucket in self.config.prefix_buckets)
        )
        unnorm_weights = [
            least_common_prefix_count * bucket.bucket_weight // bucket.prefix_count
            for bucket in self.config.prefix_buckets
        ]
        # Use GCD to reduce the weights to smallest integer ratio
        common_divisor = math.gcd(*unnorm_weights)

        # Create prefix list maintaining the correct distribution
        prefixes = []
        for bucket, weight in zip(
            self.config.prefix_buckets, unnorm_weights, strict=False
        ):
            bucket_prefixes = [
                self._create_prompt(bucket.prefix_tokens, faker)
                for _ in range(bucket.prefix_count)
            ]
            sample_count = weight // common_divisor
            prefixes.extend(bucket_prefixes * sample_count)

        while True:
            yield rand.choice(prefixes)


class SyntheticTextDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticTextDataArgs,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

        # Create the examples iterable
        ex_iterable = _SyntheticTextExamplesIterable(
            config=config,
            processor=processor,
            random_seed=random_seed,
        )

        # Initialize parent with proper ex_iterable
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic text dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if isinstance(self._ex_iterable, _SyntheticTextExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("synthetic_text")
class SyntheticTextDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticTextDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        return SyntheticTextDataset(
            config=config,
            processor=processor_factory(),
            random_seed=random_seed,
        )
