"""Adaptive sweep benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, PositiveInt

from guidellm.benchmark.schemas import ProfileArgs
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConstraintInitializer,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)

from .profile import Profile, ProfileFactory

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark


@ProfileArgs.register("sweep")
class SweepProfileArgs(ProfileArgs):
    """Pydantic model for sweep profile creation arguments."""

    kind: Literal["sweep"] = Field(
        default="sweep",
        description="Profile type discriminator for sweep scheduling",
    )
    sweep_size: int = Field(
        default=10,
        description="Number of strategies to generate for the sweep",
        ge=2,
    )
    strategy_type: Literal["constant", "poisson"] = Field(
        default="constant",
        description="Type of strategy to use for the asynchronous sweep",
    )
    max_concurrency: PositiveInt | None = Field(
        default=512,
        description="Maximum concurrent requests to schedule",
    )


@ProfileFactory.register("sweep")
class SweepProfile(Profile):
    """
    Discover optimal rate range through adaptive multi-strategy execution.

    Automatically discovers optimal rate range by executing synchronous and
    throughput strategies first, then interpolating rates for async strategies
    to comprehensively sweep the performance space.
    """

    args: SweepProfileArgs

    def __init__(
        self,
        args: SweepProfileArgs,
        random_seed: int,
        constraints: MutableMapping[str, ConstraintInitializer | Any] | None,
        **kwargs: Any,
    ):
        super().__init__(args, random_seed, constraints, **kwargs)
        self.args = args
        self.synchronous_rate = -1.0
        self.throughput_rate = -1.0
        self.async_rates: list[float] = []
        self.measured_rates: list[float] = []

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types for the complete sweep sequence
        """
        types = ["synchronous", "throughput"]
        types += [self.args.strategy_type] * (self.args.sweep_size - len(types))
        return types

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> (
        AsyncConstantStrategy
        | AsyncPoissonStrategy
        | SynchronousStrategy
        | ThroughputStrategy
        | None
    ):
        """
        Generate next strategy in adaptive sweep sequence.

        Executes synchronous and throughput strategies first to measure baseline
        rates, then generates interpolated rates for async strategies. If a
        failure constraint is triggered during the async phase, all remaining
        higher rates are skipped.

        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Next strategy in sweep sequence, or None if complete
        :raises ValueError: If strategy_type is neither 'constant' nor 'poisson'
        """
        if prev_strategy is None:
            return SynchronousStrategy()

        if prev_strategy.type_ == "synchronous":
            self.synchronous_rate = prev_benchmark.request_throughput.successful.mean

            return ThroughputStrategy(
                max_concurrency=self.args.max_concurrency,
                rampup_duration=self.args.rampup_duration,
            )

        if prev_strategy.type_ == "throughput":
            self.throughput_rate = prev_benchmark.request_throughput.successful.mean
            if self.synchronous_rate <= 0 and self.throughput_rate <= 0:
                raise RuntimeError(
                    "Invalid rates in sweep; aborting. "
                    "Were there any successful requests?"
                )
            self.measured_rates = list(
                np.linspace(
                    self.synchronous_rate,
                    self.throughput_rate,
                    self.args.sweep_size - 1,
                )
            )[1:]  # don't rerun synchronous
            # After throughput, fall through to async rate logic below.
            # Don't check escalation since throughput is designed to push
            # beyond sustainable load (over-saturation is expected).

        # Stop escalation if a failure constraint was triggered.
        # The throughput guard above skips this via the != "throughput" check.
        # Synchronous never reaches here (returns ThroughputStrategy above).
        if (
            prev_strategy.type_ != "throughput"
            and self._should_stop_escalating(prev_benchmark)  # type: ignore[arg-type]
        ):
            return None

        next_index = (
            len(self.completed_strategies) - 1 - 1
        )  # subtract synchronous and throughput
        next_rate = (
            self.measured_rates[next_index]
            if next_index < len(self.measured_rates)
            else None
        )

        if next_rate is None or next_rate <= 0:
            # Stop if we don't have another valid rate to run
            return None

        if self.args.strategy_type == "constant":
            return AsyncConstantStrategy(
                rate=next_rate, max_concurrency=self.args.max_concurrency
            )
        if self.args.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=next_rate,
                max_concurrency=self.args.max_concurrency,
                random_seed=self.random_seed,
            )
        raise ValueError(f"Invalid strategy type: {self.args.strategy_type}")
