from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from guidellm.schemas.data.entrypoints import DataArgs
from guidellm.utils.dict import deep_update

__all__ = ["WeightedSamplerDataArgs"]


@DataArgs.register("weighted_sampler")
class WeightedSamplerDataArgs(DataArgs):
    """Arguments for infinite weighted sampling from a finite source dataset."""

    kind: Literal["weighted_sampler"] = Field(  # type: ignore[assignment]
        default="weighted_sampler",
        description="Type identifier for the weighted sampler deserializer.",
    )
    source: DataArgs = Field(
        description="Nested source configuration for a finite, indexable dataset.",
    )
    weight_column: str = Field(
        default="weights",
        min_length=1,
        description="Source column containing non-negative integer-valued weights.",
    )
    replacement: bool = Field(
        default=True,
        description=(
            "Sample expanded pool entries independently with replacement. If false, "
            "shuffle and exhaust the reduced weighted pool before starting each cycle."
        ),
    )

    @model_validator(mode="after")
    def merge_kwargs(self) -> WeightedSamplerDataArgs:
        """Forward outer loading options, overriding nested options.

        :return: The configuration with merged source loading options.
        """
        if self.load_kwargs:
            deep_update(self.source.load_kwargs, self.load_kwargs)
        return self
