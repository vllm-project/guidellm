"""Async (constant/poisson rate) benchmark profile."""

from __future__ import annotations

import contextlib
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
)
from guidellm.utils.imports import json

from .profile import Profile, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register(["async", "constant", "poisson"])
class AsyncProfileArgs(ProfileArgs):
    """Pydantic model for asynchronous profile creation arguments."""

    kind: Literal["async", "constant", "poisson"] = Field(
        default="async",
        description="Profile type discriminator for asynchronous scheduling",
    )
    rate: list[PositiveFloat] = Field(
        description="Request scheduling rates in requests per second",
        examples=[1.0, [1.0, 2.0, 3.0]],
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum concurrent requests to schedule",
        examples=[10],
    )

    @field_validator("rate", mode="before")
    @classmethod
    def _coerce_rate_to_list(
        cls, value: list[PositiveFloat] | PositiveFloat
    ) -> list[PositiveFloat]:
        """Normalize rate to a list of integers.

        Allow single integer or list of integers.
        """
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                value = json.loads(value)
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


@ProfileFactory.register(["async", "constant", "poisson"])
class AsyncProfile(Profile):
    """
    Schedule requests at specified rates using constant or Poisson patterns.

    Schedules requests at specified rates using either constant interval or
    Poisson distribution patterns for realistic load simulation.
    """

    args: AsyncProfileArgs

    def __init__(
        self,
        args: AsyncProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args
        if args.kind in ("async", "constant"):
            self._strategy_type: Literal["constant", "poisson"] = "constant"
        elif args.kind == "poisson":
            self._strategy_type = "poisson"
        else:
            raise ValueError(f"Invalid profile kind: {args.kind}")

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Async strategy types for each configured rate
        """
        return [self._strategy_type] * len(self.args.rate)

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> AsyncConstantStrategy | AsyncPoissonStrategy | None:
        """
        Generate async strategy for next configured rate.

        If a previous rate was terminated by a constraint with
        stopping_scope='all', remaining rates are skipped.

        :param prev_strategy: Previously completed strategy
        :param prev_benchmark: Benchmark results from previous execution
        :return: AsyncConstantStrategy or AsyncPoissonStrategy for next rate,
            or None if all rates completed or escalation halted
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        _ = prev_strategy

        if len(self.completed_strategies) >= len(self.args.rate):
            return None

        if prev_benchmark is not None and self._should_stop_escalating(prev_benchmark):
            return None

        current_rate = self.args.rate[len(self.completed_strategies)]

        if self._strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=current_rate,
                max_concurrency=self.args.max_concurrency,
                rampup_duration=self.args.rampup_duration,
            )
        if self._strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=current_rate,
                max_concurrency=self.args.max_concurrency,
                random_seed=self.random_seed,
            )
        raise ValueError(f"Invalid strategy type: {self._strategy_type}")
