from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from guidellm.schemas import StandardBaseModel
from guidellm.schemas.data.entrypoints import DataArgs
from guidellm.utils.imports import json

__all__ = [
    "DEFAULT_SYNTHETIC_TOOLS",
    "BranchSpec",
    "SyntheticTextDataArgs",
    "SyntheticTextPrefixBucketConfig",
    "_require_mean_if_distribution_knobs",
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
