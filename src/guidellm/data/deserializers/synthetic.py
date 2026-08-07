from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from random import Random
from typing import Any, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationParentRef,
    ConversationTurnData,
)
from guidellm.schemas import RequestSettings, StandardBaseModel
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


def _integer_range_sampler(
    average: int | None,
    variance: int | None,
    min_value: int | None,
    max_value: int | None,
    random_seed: int,
) -> Iterator[int] | None:
    """Build an ``IntegerRangeSampler`` iterator, or ``None`` if average is unset."""
    if average is None:
        return None
    return iter(
        IntegerRangeSampler(
            average=average,
            variance=variance,
            min_value=min_value,
            max_value=max_value,
            random_seed=random_seed,
        )
    )


def _require_mean_if_distribution_knobs(
    mean: int | None,
    stdev: int | None,
    min_value: int | None,
    max_value: int | None,
    mean_name: str,
) -> None:
    """Reject stdev/min/max when the corresponding mean field is unset."""
    if mean is None and (
        stdev is not None or min_value is not None or max_value is not None
    ):
        raise ValueError(
            f"{mean_name} must be set when {mean_name}_stdev, "
            f"{mean_name}_min, or {mean_name}_max are provided"
        )


class BranchSpec(StandardBaseModel):
    """
    Specifies a sub-agent branch spawned from the main conversation.

    Each branch spawns at ``at_turn`` in the main chain and merges
    back at ``at_turn + merge_after`` via a ``last`` edge. The branch
    runs for ``turns`` turns with an independent context (``new`` edge
    from the spawn point). Token sizes for branch turns are sampled from
    the parent conversation's ``prompt_tokens`` / ``output_tokens``
    distribution. Optional ``first_*`` fields override only the branch's
    first turn; when unset they inherit the parent's ``first_*`` settings.

    :param at_turn: Main conversation turn index where the branch spawns.
    :param turns: Number of turns in this branch.
    :param agent_id: Agent identity for branch nodes.
    :param merge_after: How many main (parent) conversation turns after ``at_turn``
        the branch merges back. Default 1 merges at ``at_turn + 1``.
    :param first_prompt_tokens: Optional average prompt tokens for this
        branch's first turn. If None, inherits the parent's
        ``first_prompt_tokens`` (or the main ``prompt_tokens`` distribution).
    :param first_output_tokens: Optional average output tokens for this
        branch's first turn. If None, inherits the parent's
        ``first_output_tokens`` (or the main ``output_tokens`` distribution).
    """

    at_turn: int = Field(
        description="Main chain turn index where this branch spawns.",
        ge=0,
    )
    turns: int = Field(
        description="Number of turns in this branch.",
        gt=0,
    )
    agent_id: str = Field(
        description="Agent identity for branch nodes.",
        default="worker",
    )
    merge_after: int = Field(
        description=(
            "How many main (parent) conversation turns after at_turn the branch "
            "merges back. Default 1 merges at at_turn + 1."
        ),
        default=1,
        ge=1,
    )
    first_prompt_tokens: int | None = Field(
        description=(
            "Average prompt tokens for this branch's first turn only. "
            "If None, inherits the parent conversation's first_prompt_tokens "
            "(or the main prompt_tokens distribution when that is also unset)."
        ),
        default=None,
        gt=0,
    )
    first_prompt_tokens_stdev: int | None = Field(
        description="Standard deviation for this branch's first-turn prompt tokens.",
        gt=0,
        default=None,
    )
    first_prompt_tokens_min: int | None = Field(
        description="Minimum prompt tokens for this branch's first turn.",
        gt=0,
        default=None,
    )
    first_prompt_tokens_max: int | None = Field(
        description="Maximum prompt tokens for this branch's first turn.",
        gt=0,
        default=None,
    )
    first_output_tokens: int | None = Field(
        description=(
            "Average output tokens for this branch's first turn only. "
            "If None, inherits the parent conversation's first_output_tokens "
            "(or the main output_tokens distribution when that is also unset)."
        ),
        default=None,
        gt=0,
    )
    first_output_tokens_stdev: int | None = Field(
        description="Standard deviation for this branch's first-turn output tokens.",
        gt=0,
        default=None,
    )
    first_output_tokens_min: int | None = Field(
        description="Minimum output tokens for this branch's first turn.",
        gt=0,
        default=None,
    )
    first_output_tokens_max: int | None = Field(
        description="Maximum output tokens for this branch's first turn.",
        gt=0,
        default=None,
    )

    @model_validator(mode="after")
    def _validate_first_token_means(self) -> BranchSpec:
        _require_mean_if_distribution_knobs(
            self.first_prompt_tokens,
            self.first_prompt_tokens_stdev,
            self.first_prompt_tokens_min,
            self.first_prompt_tokens_max,
            "first_prompt_tokens",
        )
        _require_mean_if_distribution_knobs(
            self.first_output_tokens,
            self.first_output_tokens_stdev,
            self.first_output_tokens_min,
            self.first_output_tokens_max,
            "first_output_tokens",
        )
        return self


@DataArgs.register("synthetic_text")
class SyntheticTextDataArgs(DataArgs):
    """Model for synthetic text dataset deserializer arguments."""

    kind: Literal["synthetic_text"] = Field(  # type: ignore[assignment]
        default="synthetic_text",
        description="Type identifier for the synthetic text dataset configuration.",
    )
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for each prompt.",
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
            "The average number of text tokens generated for each output. "
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
    first_prompt_tokens: int | None = Field(
        description=(
            "Optional average prompt tokens for the first turn of a multiturn "
            "conversation. When unset, turn 0 uses prompt_tokens like later turns. "
            "Sub-agent branches inherit this setting for their first turn unless "
            "they override it on BranchSpec."
        ),
        gt=0,
        default=None,
        examples=[512],
    )
    first_prompt_tokens_stdev: int | None = Field(
        description=(
            "Standard deviation for first-turn prompt tokens (multiturn only)."
        ),
        gt=0,
        default=None,
    )
    first_prompt_tokens_min: int | None = Field(
        description="Minimum prompt tokens for the first multiturn turn.",
        gt=0,
        default=None,
    )
    first_prompt_tokens_max: int | None = Field(
        description="Maximum prompt tokens for the first multiturn turn.",
        gt=0,
        default=None,
    )
    first_output_tokens: int | None = Field(
        description=(
            "Optional average output tokens for the first turn of a multiturn "
            "conversation. When unset, turn 0 uses output_tokens like later turns. "
            "Sub-agent branches inherit this setting for their first turn unless "
            "they override it on BranchSpec."
        ),
        gt=0,
        default=None,
        examples=[128],
    )
    first_output_tokens_stdev: int | None = Field(
        description=(
            "Standard deviation for first-turn output tokens (multiturn only)."
        ),
        gt=0,
        default=None,
    )
    first_output_tokens_min: int | None = Field(
        description="Minimum output tokens for the first multiturn turn.",
        gt=0,
        default=None,
    )
    first_output_tokens_max: int | None = Field(
        description="Maximum output tokens for the first multiturn turn.",
        gt=0,
        default=None,
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
        ge=0,
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
        description=(
            "The number of user turns in the conversation. "
            "Each tool-calling user turn automatically generates an additional "
            "tool_response_injection request, so the total request count per "
            "conversation is turns + len(tool_call_turns)."
        ),
        gt=0,
        default=1,
    )
    tool_call_turns: list[int] = Field(
        description=(
            "Which user turns should include tool definitions and expect "
            "tool-call responses. Indices are 0-based into the user turns "
            "(not the expanded request list). An int N means 'the first "
            "N user turns'; a list of ints specifies explicit indices "
            "(e.g. [0, 2]); -1 means all turns. Normalized to a sorted "
            "list after validation. "
            "When 0 or [] (default), no tool calling is configured."
        ),
        default_factory=list,
        examples=[1, [0, 1]],
    )
    tools: list[dict[str, Any]] | None = Field(
        description=(
            "Tool definitions in OpenAI format. When tool_call_turns is non-empty "
            "and this is None, a static placeholder tool definition is used."
        ),
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
        description=(
            "Average number of tokens for synthetic tool call responses. "
            "When None (default), a short placeholder response is used."
        ),
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
    server_tool_call_turns: list[int] = Field(
        description=(
            "Which user turns use server-side tool calling. "
            "These turns are marked as server_tool_call so tool_choice='none' "
            "is not applied, letting the server use its configured tools. "
            "No injection turn is created. Must not overlap with "
            "tool_call_turns. Indices are 0-based into user turns. "
            "An int N means 'the first N user turns'; a list of ints "
            "specifies explicit indices (e.g. [0, 2]); -1 means all turns."
        ),
        default_factory=list,
    )

    branches: list[BranchSpec] = Field(
        description=(
            "Sub-agent branches spawned from the main conversation. "
            "Each branch spawns at a specified main-chain turn and merges "
            "back at at_turn + merge_after (default 1). Multiple branches "
            "at the same turn are supported and may have different lengths."
        ),
        default_factory=list,
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

    @field_validator("tool_call_turns", "server_tool_call_turns", mode="before")
    @classmethod
    def _coerce_tool_call_turns(
        cls, v: int | str | list[int], info: ValidationInfo
    ) -> list[int]:
        """Convert an int N to [0, ..., N-1]; pass lists through sorted.

        Strings are parsed as JSON to support CLI/env-var coercion.
        The value ``-1`` is converted to the sentinel ``[-1]`` which is
        expanded to all turn indices by :meth:`_validate_tool_call_turn_indices`
        once ``self.turns`` is available.
        """
        field = info.field_name
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError) as err:
                raise ValueError(
                    f"{field} string must be a JSON int or list of ints, got {v!r}"
                ) from err
        if isinstance(v, int):
            if v == -1:
                return [-1]
            if v < 0:
                raise ValueError(f"{field} int must be >= 0 or -1 for all")
            return list(range(v))
        if not isinstance(v, list):
            raise ValueError(
                f"{field} must be int, list[int], or a JSON representation"
                f" of either, got {type(v)}"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"{field} list must not contain duplicates")
        return sorted(v)

    @model_validator(mode="after")
    def _validate_first_token_means(self) -> SyntheticTextDataArgs:
        """Require first_* means when their distribution knobs are set."""
        _require_mean_if_distribution_knobs(
            self.first_prompt_tokens,
            self.first_prompt_tokens_stdev,
            self.first_prompt_tokens_min,
            self.first_prompt_tokens_max,
            "first_prompt_tokens",
        )
        _require_mean_if_distribution_knobs(
            self.first_output_tokens,
            self.first_output_tokens_stdev,
            self.first_output_tokens_min,
            self.first_output_tokens_max,
            "first_output_tokens",
        )
        return self

    @model_validator(mode="after")
    def _validate_tool_call_turn_indices(self) -> SyntheticTextDataArgs:
        """Ensure all tool call turn indices are within [0, turns) and don't overlap.

        The sentinel ``[-1]`` is expanded to ``list(range(self.turns))``
        before validation.
        """
        # Expand -1 sentinel ("all turns") for both fields
        if self.tool_call_turns == [-1]:
            self.tool_call_turns = list(range(self.turns))
        if self.server_tool_call_turns == [-1]:
            self.server_tool_call_turns = list(range(self.turns))

        for idx in self.tool_call_turns:
            if idx < 0 or idx >= self.turns:
                raise ValueError(
                    f"tool_call_turns index {idx} out of range [0, {self.turns})"
                )
        for idx in self.server_tool_call_turns:
            if idx < 0 or idx >= self.turns:
                raise ValueError(
                    f"server_tool_call_turns index {idx} out of range [0, {self.turns})"
                )
        overlap = set(self.tool_call_turns) & set(self.server_tool_call_turns)
        if overlap:
            raise ValueError(
                f"tool_call_turns and server_tool_call_turns must not overlap; "
                f"overlapping indices: {sorted(overlap)}"
            )

        # Validate branch specs: merge_turn = at_turn + merge_after must
        # exist on the main chain
        for i, branch in enumerate(self.branches):
            merge_turn = branch.at_turn + branch.merge_after
            if merge_turn >= self.turns:
                raise ValueError(
                    f"branches[{i}].at_turn={branch.at_turn} + "
                    f"merge_after={branch.merge_after} = {merge_turn} must be "
                    f"less than turns={self.turns} (merge point must exist)"
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

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:  # noqa: C901, PLR0915
        iter_random_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_random_seed)
        prompt_tokens_sampler: Iterator[int] = iter(
            IntegerRangeSampler(
                average=self.config.prompt_tokens,
                variance=self.config.prompt_tokens_stdev,
                min_value=self.config.prompt_tokens_min,
                max_value=self.config.prompt_tokens_max,
                random_seed=iter_random_seed,
            )
        )
        output_tokens_sampler = _integer_range_sampler(
            average=self.config.output_tokens,
            variance=self.config.output_tokens_stdev,
            min_value=self.config.output_tokens_min,
            max_value=self.config.output_tokens_max,
            random_seed=iter_random_seed + 1,  # ensure diff dist from prompts
        )
        first_prompt_tokens_sampler = _integer_range_sampler(
            average=self.config.first_prompt_tokens,
            variance=self.config.first_prompt_tokens_stdev,
            min_value=self.config.first_prompt_tokens_min,
            max_value=self.config.first_prompt_tokens_max,
            random_seed=iter_random_seed + 4,
        )
        first_output_tokens_sampler = _integer_range_sampler(
            average=self.config.first_output_tokens,
            variance=self.config.first_output_tokens_stdev,
            min_value=self.config.first_output_tokens_min,
            max_value=self.config.first_output_tokens_max,
            random_seed=iter_random_seed + 5,
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

        # Resolve tool definitions for client-side tool-call turns
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
            delay = next(delay_sampler) if delay_sampler is not None else None
            row = self._create_conversation_row(
                faker=faker,
                samples_count=samples_count,
                prompt_tokens_sampler=prompt_tokens_sampler,
                output_tokens_sampler=output_tokens_sampler,
                first_prompt_tokens_sampler=first_prompt_tokens_sampler,
                first_output_tokens_sampler=first_output_tokens_sampler,
                delay=delay,
                prefix=next(prefix_iter),
                tools_defs=tools_defs,
                tool_response_sampler=tool_response_sampler,
                iter_random_seed=iter_random_seed,
            )
            # Count logical main turns, client-tool injection nodes (added by
            # the shared finalizer expander), and branch turns.
            client_tool_extras = sum(
                1 for i in tool_call_turns_set if i < self.config.turns
            )
            samples_count += (
                self.config.turns
                + client_tool_extras
                + sum(b.turns for b in self.config.branches)
            )
            yield samples_count, row

    @staticmethod
    def _sample_turn_tokens(
        turn_index: int,
        main_sampler: Iterator[int] | None,
        first_sampler: Iterator[int] | None,
    ) -> int | None:
        """Sample token count for a turn, using first_* on turn 0 when configured.

        Returns ``None`` when neither a first-turn nor main sampler applies
        (e.g. ``output_tokens`` omitted and no ``first_output_tokens``).
        """
        if turn_index == 0 and first_sampler is not None:
            return next(first_sampler)
        if main_sampler is not None:
            return next(main_sampler)
        return None

    @staticmethod
    def _sample_required_turn_tokens(
        turn_index: int,
        main_sampler: Iterator[int],
        first_sampler: Iterator[int] | None,
    ) -> int:
        """Sample a required prompt token count for a turn."""
        count = _SyntheticTextExamplesIterable._sample_turn_tokens(
            turn_index=turn_index,
            main_sampler=main_sampler,
            first_sampler=first_sampler,
        )
        if count is None:
            raise ValueError("prompt token sampler produced no value")
        return count

    def _create_conversation_row(  # noqa: C901 PLR0912 PLR0915
        self,
        faker: Faker,
        samples_count: int,
        prompt_tokens_sampler: Iterator[int],
        output_tokens_sampler: Iterator[int] | None,
        first_prompt_tokens_sampler: Iterator[int] | None,
        first_output_tokens_sampler: Iterator[int] | None,
        delay: float | None,
        prefix: str,
        tools_defs: list[dict[str, Any]] | None,
        tool_response_sampler: Iterator[int] | None,
        iter_random_seed: int,
    ) -> dict[str, Any]:
        """
        Build a ``conversation_turns`` payload for linear or branched graphs.

        Client ``tool_call_turns`` are emitted as a single logical turn that
        still carries ``tools_column`` and ``tool_response_column``. The shared
        finalizer expander splits those into tool-call + injection nodes and
        rewrites parent refs. ``BranchSpec.at_turn`` remains a logical
        conversation index.

        Token sizes are sampled independently per turn from the main
        distribution. Turn 0 of the main chain and of each branch may use
        ``first_*`` overrides (branch first inherits parent first when unset).

        :param faker: Seeded Faker for prompt text.
        :param samples_count: Counter used to uniquify prompts.
        :param prompt_tokens_sampler: Main prompt token sampler.
        :param output_tokens_sampler: Main output token sampler, if configured.
        :param first_prompt_tokens_sampler: Optional parent first-turn prompt sampler.
        :param first_output_tokens_sampler: Optional parent first-turn output sampler.
        :param delay: Optional requeue delay for main turns.
        :param prefix: Optional system prefix applied to the first main turn.
        :param tools_defs: Tool definitions for client tool-call turns.
        :param tool_response_sampler: Optional sampler for tool response size.
        :param iter_random_seed: Seed base for per-branch first-turn samplers.
        :return: A dataset row with a JSON ``conversation_turns`` column.
        """
        tool_call_turns = set(self.config.tool_call_turns)
        server_tool_call_turns = set(self.config.server_tool_call_turns)
        turn_settings = (
            RequestSettings(requeue_delay=delay) if delay is not None else None
        )

        turns: list[ConversationTurnData] = []

        for turn_idx in range(self.config.turns):
            parents: list[ConversationParentRef] = []
            if turn_idx > 0:
                parents.append(
                    ConversationParentRef(
                        parent_node_id=f"main_{turn_idx - 1}",
                        history_context="full",
                    )
                )
            for b_idx, branch in enumerate(self.config.branches):
                if branch.at_turn + branch.merge_after == turn_idx:
                    parents.append(
                        ConversationParentRef(
                            parent_node_id=f"branch_{b_idx}_{branch.turns - 1}",
                            history_context="last",
                        )
                    )

            prompt_tokens_count = self._sample_required_turn_tokens(
                turn_index=turn_idx,
                main_sampler=prompt_tokens_sampler,
                first_sampler=first_prompt_tokens_sampler,
            )
            output_tokens_count = self._sample_turn_tokens(
                turn_index=turn_idx,
                main_sampler=output_tokens_sampler,
                first_sampler=first_output_tokens_sampler,
            )
            text = self._create_prompt(
                prompt_tokens_count,
                faker,
                f"{self.iteration_count} {samples_count} m{turn_idx} ",
            )
            columns: dict[str, list[Any]] = {
                "text_column": [text],
                "prompt_tokens_count_column": [prompt_tokens_count],
            }
            if turn_idx == 0 and prefix:
                columns["prefix_column"] = [prefix]
            if output_tokens_count is not None:
                columns["output_tokens_count_column"] = [output_tokens_count]
            if turn_idx in server_tool_call_turns:
                columns["turn_type_column"] = ["server_tool_call"]

            if turn_idx in tool_call_turns:
                tools_raw = json.dumps(tools_defs or DEFAULT_SYNTHETIC_TOOLS)
                columns["tools_column"] = [
                    tools_raw.decode() if isinstance(tools_raw, bytes) else tools_raw
                ]
                if tool_response_sampler is not None:
                    tr_tokens = next(tool_response_sampler)
                    body = self._create_prompt(tr_tokens, faker)
                    response_raw = json.dumps({"result": body})
                    tool_response = (
                        response_raw.decode()
                        if isinstance(response_raw, bytes)
                        else response_raw
                    )
                else:
                    tool_response = settings.default_synthetic_tool_response
                columns["tool_response_column"] = [tool_response]

            turns.append(
                ConversationTurnData(
                    node_id=f"main_{turn_idx}",
                    agent_id="default",
                    parents=parents,
                    columns=columns,
                    settings=turn_settings,
                )
            )

        for b_idx, branch in enumerate(self.config.branches):
            # Branch-local first_* overrides; otherwise inherit parent first_* samplers
            branch_first_prompt = _integer_range_sampler(
                average=branch.first_prompt_tokens,
                variance=branch.first_prompt_tokens_stdev,
                min_value=branch.first_prompt_tokens_min,
                max_value=branch.first_prompt_tokens_max,
                random_seed=iter_random_seed + 10 + b_idx * 2,
            )
            branch_first_output = _integer_range_sampler(
                average=branch.first_output_tokens,
                variance=branch.first_output_tokens_stdev,
                min_value=branch.first_output_tokens_min,
                max_value=branch.first_output_tokens_max,
                random_seed=iter_random_seed + 11 + b_idx * 2,
            )
            resolved_first_prompt = (
                branch_first_prompt
                if branch_first_prompt is not None
                else first_prompt_tokens_sampler
            )
            resolved_first_output = (
                branch_first_output
                if branch_first_output is not None
                else first_output_tokens_sampler
            )

            for t in range(branch.turns):
                if t == 0:
                    parents = [
                        ConversationParentRef(
                            parent_node_id=f"main_{branch.at_turn}",
                            history_context="new",
                        )
                    ]
                else:
                    parents = [
                        ConversationParentRef(
                            parent_node_id=f"branch_{b_idx}_{t - 1}",
                            history_context="full",
                        )
                    ]

                branch_prompt_tokens = self._sample_required_turn_tokens(
                    turn_index=t,
                    main_sampler=prompt_tokens_sampler,
                    first_sampler=resolved_first_prompt,
                )
                branch_output_tokens = self._sample_turn_tokens(
                    turn_index=t,
                    main_sampler=output_tokens_sampler,
                    first_sampler=resolved_first_output,
                )

                branch_columns: dict[str, list[Any]] = {
                    "text_column": [
                        self._create_prompt(
                            branch_prompt_tokens,
                            faker,
                            f"{self.iteration_count} {samples_count} b{b_idx}_{t} ",
                        )
                    ],
                    "prompt_tokens_count_column": [branch_prompt_tokens],
                }
                if branch_output_tokens is not None:
                    branch_columns["output_tokens_count_column"] = [
                        branch_output_tokens
                    ]

                turns.append(
                    ConversationTurnData(
                        node_id=f"branch_{b_idx}_{t}",
                        agent_id=branch.agent_id,
                        parents=parents,
                        columns=branch_columns,
                    )
                )

        graph_data = ConversationGraphData(turns=turns)
        payload = json.dumps(graph_data.model_dump(mode="json"))
        return {
            "conversation_turns": (
                payload.decode() if isinstance(payload, bytes) else payload
            )
        }

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features({"conversation_turns": Value("large_string")})

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
