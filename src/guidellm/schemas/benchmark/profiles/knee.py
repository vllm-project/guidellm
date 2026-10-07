"""Configuration for throughput knee detection and adaptive refinement."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

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
    initial_streams: list[Annotated[int, Field(gt=0, strict=True)]] | None = Field(
        default=None,
        min_length=5,
        description=(
            "At least five distinct initial concurrency points in strictly "
            "increasing order; alternatively set min_streams, max_streams, and count"
        ),
        examples=[[1, 2, 5, 25, 50, 100, 200, 300]],
    )
    min_streams: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="Lowest initial concurrency when generating evenly spaced points",
    )
    max_streams: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="Highest initial concurrency when generating evenly spaced points",
    )
    count: int | None = Field(
        default=None,
        ge=5,
        strict=True,
        description="Number of evenly spaced integer concurrency points to generate",
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

    @field_validator("initial_streams", mode="before")
    @classmethod
    def _parse_initial_streams(cls, value: Any) -> Any:
        """Parse a JSON list of concurrencies from CLI configuration."""
        if isinstance(value, str):
            value = json.loads(value)
        return value

    @field_validator("initial_streams")
    @classmethod
    def _require_distinct_initial_streams(
        cls, value: list[int] | None
    ) -> list[int] | None:
        """Reject repeated or out-of-order explicit concurrency points."""
        if value is None:
            return None
        if len(value) != len(set(value)):
            raise ValueError("Knee detection requires distinct concurrency points")
        if any(previous > current for previous, current in pairwise(value)):
            raise ValueError("Knee detection requires increasing concurrency points")
        return value

    @model_validator(mode="after")
    def _validate_initial_points(self) -> KneeProfileArgs:
        """Require one complete source of at least five initial points."""
        bounds = (self.min_streams, self.max_streams, self.count)
        if self.initial_streams is not None:
            if any(value is not None for value in bounds):
                raise ValueError(
                    "Choose either initial_streams or min_streams, max_streams, "
                    "and count"
                )
            return self

        if self.min_streams is None or self.max_streams is None or self.count is None:
            raise ValueError(
                "Provide initial_streams or all of min_streams, max_streams, and count"
            )
        if self.min_streams >= self.max_streams:
            raise ValueError("max_streams must be greater than min_streams")
        if self.max_streams - self.min_streams < self.count - 1:
            raise ValueError(
                "The min_streams to max_streams range cannot provide count "
                "distinct integer concurrency points"
            )
        return self

    def resolved_initial_streams(self) -> list[int]:
        """Return the explicit or generated initial concurrency points.

        :return: At least five strictly increasing integer concurrency points
        """
        if self.initial_streams is not None:
            return list(self.initial_streams)
        if self.min_streams is None or self.max_streams is None or self.count is None:
            raise ValueError("The knee profile has no initial concurrency points")

        steps = self.count - 1
        span = self.max_streams - self.min_streams
        return [
            self.min_streams + (index * span + steps // 2) // steps
            for index in range(self.count)
        ]
