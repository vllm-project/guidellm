"""Throughput benchmark profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import AliasChoices, Field, PositiveInt, field_validator

from guidellm.scheduler import (
    SchedulingStrategy,
    ThroughputStrategy,
)

from .profile import Profile, ProfileArgs, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("throughput")
class ThroughputProfileArgs(ProfileArgs):
    """Pydantic model for throughput profile creation arguments."""

    kind: Literal["throughput"] = Field(
        default="throughput",
        description="Profile type discriminator for polymorphic serialization",
    )
    max_concurrency: PositiveInt | None = Field(
        validation_alias=AliasChoices("max_concurrency", "rate"),
        description="Maximum concurrent requests to schedule",
    )

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _coerce_max_concurrency_from_rate(cls, value: Any) -> Any:
        """
        Accept global ``--rate`` list values as max concurrency.

        The CLI passes ``rate`` as a list of floats; throughput uses the first entry
        as the maximum concurrency.
        """
        if not value:
            raise ValueError("max_concurrency (rate) requires at least one value")
        if isinstance(value, list | tuple):
            return int(value[0])
        if isinstance(value, int | float) and not isinstance(value, bool):
            return int(value)
        raise ValueError(
            "max_concurrency (rate) must be an integer or a list of numeric values, "
            f"got {type(value).__name__}"
        )


@ProfileFactory.register("throughput")
class ThroughputProfile(Profile):
    """
    Maximize system throughput with optional concurrency constraints.

    Maximizes system throughput by maintaining maximum concurrent requests,
    optionally constrained by a concurrency limit.
    """

    args: ThroughputProfileArgs

    def __init__(
        self,
        args: ThroughputProfileArgs,
        constraints: dict[str, Any] | None,
    ):
        super().__init__(args, constraints)
        self.args = args

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Single throughput strategy type
        """
        return [self.kind]

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> ThroughputStrategy | None:
        """
        Generate throughput strategy for first execution only.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: ThroughputStrategy for first execution, None afterward
        """
        _ = (prev_strategy, prev_benchmark)  # unused
        if len(self.completed_strategies) >= 1:
            return None

        return ThroughputStrategy(
            max_concurrency=self.args.max_concurrency,
            rampup_duration=self.args.rampup_duration,
        )
