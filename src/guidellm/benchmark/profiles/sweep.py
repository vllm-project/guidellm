"""Adaptive sweep benchmark profile."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import numpy as np

from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.logger import logger
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    AsyncRampStrategy,
    Constraint,
    ConstraintInitializer,
    RampSaturationConstraint,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.schemas.benchmark.profiles import SweepProfileArgs

from .profile import Profile, ProfileFactory

__all__ = ["SweepProfile"]

if TYPE_CHECKING:
    from guidellm.benchmark.schemas import Benchmark

# Ramp sizing, relative to the synchronous step's request latency.
RAMP_DOUBLING_LATENCIES = 4.0
RAMP_MIN_DOUBLING_SECONDS = 2.0
RAMP_CONSTRAINT_KEY = "ramp_saturation"
RAMP_NOT_SATURATED_WARNING = (
    "The sweep ramp stopped before the server saturated, so its peak of "
    "{:.2f} req/s is a lower bound and the constant steps may stop short of "
    "capacity. Give each step more time, for example with a higher --max-seconds."
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
        self._ramp_doubling_interval: float | None = None
        self._ramp_peak_source: str | None = None

    @property
    def strategy_types(self) -> list[str]:
        """
        :return: Strategy types for the complete sweep sequence
        """
        types = ["synchronous", self._peak_strategy_type]
        types += [self.args.strategy_type] * (self.args.sweep_size - len(types))
        return types

    @property
    def conclusion(self) -> dict[str, Any] | None:
        """
        :return: The peak the ramp found, or None when not using a ramp
        """
        if self.args.peak_detection != "ramp" or self._ramp_peak_source is None:
            return None

        return {
            "kind": "sweep_ramp",
            "synchronous_rate": self.synchronous_rate,
            "peak_rate": self.throughput_rate,
            "peak_source": self._ramp_peak_source,
            "doubling_interval": self._ramp_doubling_interval,
        }

    @property
    def _peak_strategy_type(self) -> str:
        return "ramp" if self.args.peak_detection == "ramp" else "throughput"

    @property
    def _max_concurrency(self) -> int | None:
        # The 512 default is for the throughput step; a ramp only uses an explicit cap.
        if (
            self.args.peak_detection == "ramp"
            and "max_concurrency" not in self.args.model_fields_set
        ):
            return None

        return self.args.max_concurrency

    def next_strategy(
        self,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> (
        AsyncConstantStrategy
        | AsyncPoissonStrategy
        | AsyncRampStrategy
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

            return (
                self._ramp_strategy(prev_benchmark)  # type: ignore[arg-type]
                if self.args.peak_detection == "ramp"
                else ThroughputStrategy(
                    max_concurrency=self.args.max_concurrency,
                    rampup_duration=self.args.rampup_duration,
                )
            )

        if prev_strategy.type_ in ("throughput", "ramp"):
            self.throughput_rate = self._peak_rate(prev_benchmark)  # type: ignore[arg-type]
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

        # Stop escalation if a constraint with stopping_scope='all' triggered
        # during the async phase. Throughput and ramp are excluded because they
        # intentionally push beyond sustainable load. Synchronous never reaches here.
        if prev_strategy.type_ not in (
            "throughput",
            "ramp",
        ) and self._should_stop_escalating(prev_benchmark):  # type: ignore[arg-type]
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
                rate=next_rate, max_concurrency=self._max_concurrency
            )
        if self.args.strategy_type == "poisson":
            return AsyncPoissonStrategy(
                rate=next_rate,
                max_concurrency=self._max_concurrency,
                random_seed=self.random_seed,
            )
        raise ValueError(f"Invalid strategy type: {self.args.strategy_type}")

    def next_strategy_constraints(
        self,
        next_strategy: SchedulingStrategy | None,
        prev_strategy: SchedulingStrategy | None,
        prev_benchmark: Benchmark | None,
    ) -> dict[str, Constraint] | None:
        """
        Add saturation detection to the ramp step.

        The ramp has no end of its own. It stops sending once completions fall
        short of what was sent one latency earlier, then measures the peak while
        the requests already sent complete. User constraints still apply
        alongside it.

        :param next_strategy: Strategy to be executed next
        :param prev_strategy: Previously completed strategy instance
        :param prev_benchmark: Benchmark results from previous strategy execution
        :return: Constraints dictionary for next strategy, or None
        """
        constraints = super().next_strategy_constraints(
            next_strategy, prev_strategy, prev_benchmark
        )
        if not isinstance(next_strategy, AsyncRampStrategy):
            return constraints

        constraints = dict(constraints or {})
        constraints[RAMP_CONSTRAINT_KEY] = RampSaturationConstraint(
            window_seconds=next_strategy.doubling_interval
        )
        return constraints

    def _ramp_strategy(self, synchronous: Benchmark) -> AsyncRampStrategy:
        latency = synchronous.request_latency.successful.mean
        self._ramp_doubling_interval = max(
            RAMP_DOUBLING_LATENCIES * latency, RAMP_MIN_DOUBLING_SECONDS
        )

        if self.synchronous_rate <= 0:
            raise RuntimeError(
                "Invalid synchronous rate in sweep; cannot start the ramp. "
                "Were there any successful requests?"
            )

        return AsyncRampStrategy(
            start_rate=self.synchronous_rate,
            doubling_interval=self._ramp_doubling_interval,
            max_concurrency=self._max_concurrency,
        )

    def _peak_rate(self, benchmark: Benchmark) -> float:
        if self.args.peak_detection == "ramp" and isinstance(
            benchmark, GenerativeBenchmark
        ):
            action = benchmark.scheduler_state.scheduler_constraints.get(
                RAMP_CONSTRAINT_KEY
            )
            peak = action.metadata.get("peak_rate") if action else None
            if peak:
                # A duration limit can end the ramp before it saturates.
                self._ramp_peak_source = (
                    "saturated"
                    if action and action.metadata.get("saturated_at") is not None
                    else "not_saturated"
                )
                if self._ramp_peak_source == "not_saturated":
                    logger.warning(RAMP_NOT_SATURATED_WARNING, peak)
                return peak

        mean_rate = benchmark.request_throughput.successful.mean
        if self.args.peak_detection == "ramp":
            # No full window completed, so use the mean like the throughput step.
            self._ramp_peak_source = "mean_rate"
            logger.warning(RAMP_NOT_SATURATED_WARNING, mean_rate)

        return mean_rate
