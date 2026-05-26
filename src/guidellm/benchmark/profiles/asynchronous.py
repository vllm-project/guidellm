"""Async (constant/poisson rate) benchmark profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    SchedulingStrategy,
)

from .profile import Profile, ProfileArgs

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register(["async", "constant", "poisson"])
class AsyncProfileArgs(ProfileArgs):
    """Pydantic model for Async profile creation arguments."""

    kind: Literal["async", "constant", "poisson"] = Field(
        default="async",
        description="Profile type discriminator for polymorphic serialization",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rates in requests per second",
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )

    @field_validator("rate", mode="before")
    @classmethod
    def _coerce_rate_to_list(
        cls, value: list[PositiveFloat] | PositiveFloat
    ) -> list[PositiveFloat]:
        """
        Convert rates to a list from either a single number or a list of
        numbers.
        """
        if not value:
            raise ValueError("rate requires at least one value")
        if isinstance(value, list | tuple):
            return value
        if isinstance(value, int | float):
            return [value]
        raise ValueError(
            "rate must be a number or a list of numeric values, "
            f"got {type(value).__name__}"
        )


@Profile.register(["async", "constant", "poisson"])
class AsyncProfile(Profile):
    """
    Schedule requests at specified rates using constant or Poisson patterns.

    Schedules requests at specified rates using either constant interval or
    Poisson distribution patterns for realistic load simulation.
    """

    kind: Literal["async", "constant", "poisson"] = Field(
        default="async",
        description="Profile type discriminator for polymorphic serialization",
    )
    strategy_type: Literal["constant", "poisson"] = Field(
        description="Asynchronous strategy pattern type derived from profile kind",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rates in requests per second",
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution strategy",
    )

    @model_validator(mode="before")
    @classmethod
    def derive_strategy_type_from_kind(cls, value: Any) -> Any:
        """
        Map profile kind to the scheduling strategy implementation type.

        ``async`` and ``constant`` kinds both use constant-interval scheduling;
        ``poisson`` uses Poisson-distributed scheduling.
        """
        if not isinstance(value, dict):
            return value

        kind = value.get("kind", "async")
        if kind in ("async", "constant"):
            strategy_type = "constant"
        elif kind == "poisson":
            strategy_type = "poisson"
        else:
            raise ValueError(f"Invalid profile kind: {kind}")

        return {**value, "strategy_type": strategy_type}

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Async strategy types for each configured rate
        """
        return [self.strategy_type] * len(self.rate)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> AsyncConstantStrategy | AsyncPoissonStrategy | None:
        """
        Generate async strategy for next configured rate.

        :param prev_strategy: Previously completed strategy (unused)
        :param prev_benchmark: Benchmark results from previous execution (unused)
        :return: AsyncConstantStrategy or AsyncPoissonStrategy for next rate,
            or None if all rates completed
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        _ = (prev_strategy, prev_benchmark)  # unused

        if len(self.completed_strategies) >= len(self.rate):
            return None

        current_rate = self.rate[len(self.completed_strategies)]

        if self.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=current_rate,
                max_concurrency=self.max_concurrency,
                rampup_duration=self.rampup_duration,
            )
        if self.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=current_rate,
                max_concurrency=self.max_concurrency,
                random_seed=self.random_seed,
            )
        raise ValueError(f"Invalid strategy type: {self.strategy_type}")
