from __future__ import annotations

from pydantic import Field

from guidellm.schemas import StandardBaseModel

__all__ = ["PreprocessDatasetConfig"]


class PreprocessDatasetConfig(StandardBaseModel):
    """Model for preprocess dataset configuration."""

    prompt_tokens: int = Field(
        description="The average number of text tokens retained or added to prompts.",
        examples=[100],
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the number of tokens retained in or "
        "added to prompts.",
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
        description="The standard deviation of the number of tokens retained or "
        "added to outputs.",
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
