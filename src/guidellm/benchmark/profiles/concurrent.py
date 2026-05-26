"""Concurrent benchmark profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import AliasChoices, Field, PositiveInt, field_validator

from guidellm.scheduler import ConcurrentStrategy, SchedulingStrategy

from .profile import Profile, ProfileArgs

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("concurrent")
class ConcurrentProfileArgs(ProfileArgs):
    """Pydantic model for concurrent profile creation arguments."""

    kind: Literal["concurrent"] = Field(
        default="concurrent",
        description="Profile type discriminator for polymorphic serialization",
    )
    streams: list[PositiveInt] = Field(
        validation_alias=AliasChoices("streams", "rate"),
        description="Concurrent stream counts to execute",
    )

    @field_validator("streams", mode="before")
    @classmethod
    def _coerce_streams_to_list(cls, value: Any) -> Any:
        """
        Convert streams to a list of integers from either a single number or a list of
        numbers.
        """
        if not value:
            raise ValueError("streams (rate) requires at least one value")
        if isinstance(value, list | tuple):
            return [int(stream) for stream in value]
        if isinstance(value, int | float):
            return [int(value)]
        raise ValueError(
            "streams (rate) must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )


@Profile.register("concurrent")
class ConcurrentProfile(Profile):
    """
    Execute strategies with fixed concurrency levels for performance testing.

    Executes requests with a fixed number of concurrent streams, useful for
    testing system performance under specific concurrency levels.
    """

    kind: Literal["concurrent"] = Field(
        default="concurrent",
        description="Profile type discriminator for polymorphic serialization",
    )
    streams: list[PositiveInt] = Field(
        description="Concurrent stream counts to execute",
    )

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Concurrent strategy types for each configured stream count
        """
        return [self.kind] * len(self.streams)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ConcurrentStrategy | None:
        """
        Generate concurrent strategy for next stream count.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ConcurrentStrategy with next stream count, or None if complete
        """
        _ = (prev_strategy, prev_benchmark)  # unused

        if len(self.completed_strategies) >= len(self.streams):
            return None

        return ConcurrentStrategy(
            streams=self.streams[len(self.completed_strategies)],
            rampup_duration=self.rampup_duration,
        )
