from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from guidellm.schemas import StandardBaseModel

__all__ = [
    "DataConfig",
    "DataNotSupportedError",
    "GenerativeDatasetColumnType",
    "SyntheticTextDatasetConfig",
    "SyntheticTextPrefixBucketConfig",
]


GenerativeDatasetColumnType = Literal[
    "prompt_tokens_count_column",
    "output_tokens_count_column",
    "prefix_column",
    "text_column",
    "image_column",
    "video_column",
    "audio_column",
    "tools_column",
    "tool_response_column",
]


class DataNotSupportedError(Exception):
    """
    Exception raised when the data format is not supported by deserializer or config.
    """


class DataConfig(StandardBaseModel):
    """
    A generic parent class for various configs for the data package
    that can be passed in as key-value pairs or JSON.
    """


class PreprocessDatasetConfig(DataConfig):
    prompt_tokens: int = Field(
        description="The average number of text tokens retained or added to prompts.",
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the number of tokens retained in or "
        "added to prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens retained or added to outputs.",
        gt=0,
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the number of tokens retained or "
        "added to outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens retained or added to outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens retained or added to outputs.",
        gt=0,
        default=None,
    )
    prefix_tokens_max: int | None = Field(
        description="The maximum number of text tokens left in the prefixes.",
        gt=0,
        default=None,
    )


class SyntheticTextPrefixBucketConfig(StandardBaseModel):
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


class SyntheticTextDatasetConfig(DataConfig):
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts.",
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens generated for outputs.",
        gt=0,
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    turns: int = Field(
        description="The number of turns in the conversation.",
        gt=0,
        default=1,
    )
    tool_call_turns: int = Field(
        description="Number of turns (from the start) that should include tool "
        "definitions and expect tool-call responses. Must be <= turns. "
        "When equal to turns, every turn is a tool-call turn and no "
        "final plain-text response is produced. "
        "When 0 (default), no tool calling is configured.",
        ge=0,
        default=0,
    )
    tools: list[dict[str, Any]] | None = Field(
        description="Tool definitions in OpenAI format. When tool_call_turns > 0 "
        "and this is None, a static placeholder tool definition is used.",
        default=None,
    )
    tool_response_tokens: int | None = Field(
        description="Average number of tokens for synthetic tool call responses. "
        "When None (default), a short placeholder response is used.",
        gt=0,
        default=None,
    )
    tool_response_tokens_stdev: int | None = Field(
        description="Standard deviation for tool response token count.",
        gt=0,
        default=None,
    )
    tool_response_tokens_min: int | None = Field(
        description="Minimum number of tokens for tool response.",
        gt=0,
        default=None,
    )
    tool_response_tokens_max: int | None = Field(
        description="Maximum number of tokens for tool response.",
        gt=0,
        default=None,
    )

    model_config = ConfigDict(
        extra="allow",
    )

    prefix_buckets: list[SyntheticTextPrefixBucketConfig] | None = Field(
        description="Buckets for the prefix tokens distribution.",
        default=None,
    )

    @model_validator(mode="after")
    def check_tool_call_options(self) -> SyntheticTextDatasetConfig:
        if self.tool_call_turns > self.turns:
            raise ValueError(
                f"tool_call_turns ({self.tool_call_turns}) must be <= "
                f"turns ({self.turns})."
            )

        if self.tools is not None and self.tool_call_turns == 0:
            raise ValueError(
                "tools were provided but tool_call_turns is 0. "
                "Set tool_call_turns > 0 to enable tool calling."
            )

        if self.tool_response_tokens is not None and self.tool_call_turns == 0:
            raise ValueError(
                "tool_response_tokens was set but tool_call_turns is 0. "
                "Set tool_call_turns > 0 to enable tool calling."
            )

        return self

    @model_validator(mode="after")
    def check_prefix_options(self) -> SyntheticTextDatasetConfig:
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
