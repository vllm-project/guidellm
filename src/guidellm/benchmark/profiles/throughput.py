"""Throughput benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveInt

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.scheduler import (
    ConstraintInitializer,
    SchedulingStrategy,
    ThroughputStrategy,
)

from .profile import Profile, ProfileFactory

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
        description="Maximum concurrent requests to schedule",
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
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
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
