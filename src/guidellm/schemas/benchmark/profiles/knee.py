"""Configuration for throughput knee detection and adaptive refinement."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from guidellm.schemas.benchmark.profiles.profile import ProfileArgs
from guidellm.utils.imports import json

__all__ = ["KneeProfileArgs"]


@ProfileArgs.register("knee")
class KneeProfileArgs(ProfileArgs):
    """Arguments for a concurrent sweep with optional knee refinement."""

    kind: Literal["knee"] = Field(
        default="knee",
        description="Profile type discriminator for throughput knee detection",
    )
    streams: list[Annotated[int, Field(gt=0, strict=True)]] = Field(
        min_length=1,
        description=(
            "Distinct initial concurrency points, executed in the supplied order. "
            "At least five measured points are needed to fit a throughput knee"
        ),
        examples=[[1, 5, 10, 20, 40, 80, 160]],
    )
    adaptive: bool = Field(
        default=False,
        description="Run one additional pass around the initial saturation estimate",
    )
    points_each_side: int = Field(
        default=5,
        gt=0,
        strict=True,
        description="Maximum number of adaptive grid points on each side of the anchor",
    )
    max_step: int = Field(
        default=5,
        gt=0,
        strict=True,
        description="Largest integer spacing considered for the adaptive grid",
    )

    @field_validator("streams", mode="before")
    @classmethod
    def _coerce_streams_to_list(cls, value: Any) -> Any:
        """Accept a single stream count or a JSON list from CLI configuration."""
        if isinstance(value, str):
            value = json.loads(value)
        return [value] if isinstance(value, int) else value

    @field_validator("streams")
    @classmethod
    def _require_distinct_streams(cls, value: list[int]) -> list[int]:
        """Reject repeated points whose saturation decisions could be ambiguous."""
        if len(value) != len(set(value)):
            raise ValueError("Knee detection requires distinct concurrency points")
        return value
