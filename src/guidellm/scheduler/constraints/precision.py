"""
Target margin of error constraint implementation.

Stops a benchmark once a request-level statistic has been measured to a requested
relative precision. The margin of error is computed with the same estimators the
report uses for its confidence intervals, so the precision a run stops at is the
precision its report states.

The check is made on a sparse schedule after a minimum number of samples rather
than after every request. Stopping the first time an interval looks narrow enough
favours moments when the sample spread happens to be low, so checking every
request would make the final interval cover less often than its nominal level.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field

import guidellm.extras.numpy as np
from guidellm.scheduler.constraints.constraint import (
    Constraint,
    PydanticConstraintInitializer,
    constraint_stop_time,
)
from guidellm.scheduler.constraints.factory import ConstraintsInitializerFactory
from guidellm.scheduler.schemas import (
    SchedulerProgress,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo
from guidellm.schemas.base.statistics import PERCENTILE_PROBABILITIES, Percentiles
from guidellm.schemas.scheduler.constraints import (
    TargetMoeConstraintArgs,
    TargetMoeMetric,
    TargetMoeStatistic,
)
from guidellm.utils.statistics import (
    mean_confidence_interval,
    quantile_confidence_interval,
)

__all__ = [
    "TargetMoeConstraint",
    "TargetMoeConstraintInitializer",
]


def _time_to_first_token_ms(info: RequestInfo) -> float | None:
    start = info.timings.request_start
    first_token = info.timings.first_token_iteration
    if start is None or first_token is None:
        return None

    return 1000 * (first_token - start)


def _request_latency(info: RequestInfo) -> float | None:
    start = info.timings.request_start
    end = info.timings.request_end
    if start is None or end is None:
        return None

    return end - start


# Mirrors the definitions in ``GenerativeRequestStats`` so the constraint measures
# the same quantity the report summarizes.
_METRIC_FUNCTIONS: dict[str, Callable[[RequestInfo], float | None]] = {
    "time_to_first_token_ms": _time_to_first_token_ms,
    "request_latency": _request_latency,
}


class TargetMoeConstraint(Constraint):
    """
    Constraint that stops once a statistic reaches a target margin of error.

    Collects one value per successfully completed request for the configured
    metric and, every ``check_interval`` new samples after ``min_samples``,
    computes a confidence interval for the configured statistic. Once the larger
    distance from the estimate to either bound falls to ``moe`` times the
    estimate, request queuing and local processing stop. The decision is kept
    for the rest of the run. Each check also estimates the samples and seconds
    still needed to reach the target.

    The interval treats requests as independent draws. Requests issued under load
    share queue state, so it describes how precisely this run located its own
    statistic, not how much the statistic would move across repeated runs.

    Example:
    ::
        constraint = TargetMoeConstraint(
            metric="time_to_first_token_ms", statistic="p95", moe=0.05
        )
        action = constraint(state, request_info)
    """

    def __init__(  # noqa: PLR0913
        self,
        moe: float,
        metric: TargetMoeMetric = "time_to_first_token_ms",
        statistic: TargetMoeStatistic = "mean",
        confidence: float = 0.95,
        min_samples: int = 30,
        check_interval: int = 10,
        stopping_scope: Literal["current", "all"] = "current",
    ):
        """
        Initialize the target margin of error constraint.

        :param moe: Target relative margin of error, as a fraction of the estimate
        :param metric: Request-level metric to measure, named as in the report
        :param statistic: ``mean`` or a percentile key such as ``p95``
        :param confidence: Two-sided confidence level of the interval
        :param min_samples: Minimum successful requests before the first check
        :param check_interval: New successful requests between checks
        :param stopping_scope: Whether stopping halts only the current benchmark
            or also escalation to subsequent strategies
        """
        self.moe = moe
        self.metric = metric
        self.statistic = statistic
        self.confidence = confidence
        self.min_samples = min_samples
        self.check_interval = check_interval
        self.stopping_scope = stopping_scope
        self.reset()

    @property
    def info(self) -> dict[str, Any]:
        """
        Get current constraint configuration.

        :return: Dictionary with the constraint configuration
        """
        return {
            "type_": "target_moe",
            "metric": self.metric,
            "statistic": self.statistic,
            "moe": self.moe,
            "confidence": self.confidence,
            "min_samples": self.min_samples,
            "check_interval": self.check_interval,
            "stopping_scope": self.stopping_scope,
        }

    def reset(self) -> None:
        """
        Reset collected samples and the stopping decision.
        """
        self.sorted_values: list[float] = []
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.last_checked_count = 0
        self.estimate: float | None = None
        self.lower: float | None = None
        self.upper: float | None = None
        self.relative_moe: float | None = None
        self.required_samples: int | None = None
        self.estimated_remaining_seconds: float | None = None
        self.first_sample_time: float | None = None
        self.last_sample_time: float | None = None
        self.target_reached = False
        self._recorded_request_ids: set[str] = set()

    def __call__(
        self, state: SchedulerState, request_info: RequestInfo | None
    ) -> SchedulerUpdateAction:
        """
        Record a completed request and evaluate the margin of error when due.

        :param state: Current scheduler state (unused)
        :param request_info: Individual request information, or ``None`` on poll
        :return: Action indicating whether to continue or stop operations
        """
        _ = state  # Unused parameter
        if not self.target_reached and request_info is not None:
            self._record(request_info)
            if self._check_due():
                self._evaluate()

        stop_time = constraint_stop_time(request_info, stopped=self.target_reached)

        return SchedulerUpdateAction(
            request_queuing="stop" if self.target_reached else "continue",
            request_processing="stop_local" if self.target_reached else "continue",
            stopping_scope=self.stopping_scope,
            metadata={
                "metric": self.metric,
                "statistic": self.statistic,
                "target_moe": self.moe,
                "confidence": self.confidence,
                "samples": self.count,
                "estimate": self.estimate,
                "lower": self.lower,
                "upper": self.upper,
                "relative_moe": self.relative_moe,
                "required_samples": self.required_samples,
                "estimated_remaining_seconds": self.estimated_remaining_seconds,
                "target_moe_reached": self.target_reached,
                "stop_time": stop_time,
            },
            progress=self._progress(stop_time),
        )

    def _record(self, request_info: RequestInfo) -> None:
        if (
            request_info.status != "completed"
            or request_info.request_id in self._recorded_request_ids
        ):
            return

        value = _METRIC_FUNCTIONS[self.metric](request_info)
        if value is None:
            return

        self._recorded_request_ids.add(request_info.request_id)
        bisect.insort(self.sorted_values, value)

        # Welford's online update keeps the mean and spread exact without
        # rescanning the samples at each check.
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_squares += delta * (value - self.mean)

        if (completed_at := request_info.completed_at) is not None:
            if self.first_sample_time is None:
                self.first_sample_time = completed_at
            self.last_sample_time = completed_at

    def _check_due(self) -> bool:
        return (
            self.count >= self.min_samples
            and self.count - self.last_checked_count >= self.check_interval
        )

    def _evaluate(self) -> None:
        self.last_checked_count = self.count
        self._update_margin()
        self.estimated_remaining_seconds = self._remaining_seconds()

    def _update_margin(self) -> None:
        estimate, interval, minimum_count = self._interval()
        self.estimate = estimate

        if interval is None:
            self.lower = self.upper = self.relative_moe = None
            self.required_samples = minimum_count
            return

        self.lower, self.upper = interval
        half_width = max(estimate - self.lower, self.upper - estimate)
        if estimate <= 0:
            # A relative margin is undefined around a zero estimate.
            self.relative_moe = None
            self.required_samples = None
            return

        self.relative_moe = half_width / estimate
        self.target_reached = self.relative_moe <= self.moe
        # The interval width shrinks with the square root of the sample count,
        # which extrapolates the samples still needed from the current spread.
        scaled = self.count * (self.relative_moe / self.moe) ** 2
        self.required_samples = max(self.count, minimum_count, math.ceil(scaled))

    def _remaining_seconds(self) -> float | None:
        if self.required_samples is None:
            return None

        remaining = max(0, self.required_samples - self.count)
        if remaining == 0:
            return 0.0

        if (
            self.first_sample_time is None
            or self.last_sample_time is None
            or self.last_sample_time <= self.first_sample_time
        ):
            return None

        # Extrapolate at the rate samples have been arriving so far.
        rate = (self.count - 1) / (self.last_sample_time - self.first_sample_time)
        return remaining / rate

    def _interval(self) -> tuple[float, tuple[float, float] | None, int]:
        if self.statistic == "mean":
            std_dev = math.sqrt(self.sum_squares / self.count)
            interval = mean_confidence_interval(
                count=self.count,
                mean=self.mean,
                std_dev=std_dev,
                confidence=self.confidence,
            )
            return self.mean, interval, self.min_samples

        quantile = PERCENTILE_PROBABILITIES[self.statistic]
        # Use the report's percentile definition so the estimate matches it.
        pdf = np.column_stack(
            (
                np.asarray(self.sorted_values),
                np.full(self.count, 1.0 / self.count),
            )
        )
        estimate = Percentiles.from_pdf(pdf, validate=False).model_dump()[
            self.statistic
        ]
        interval = quantile_confidence_interval(
            self.sorted_values, quantile, self.confidence
        )
        tail = (1.0 - self.confidence) / 2.0
        minimum_count = max(
            self.min_samples, math.ceil(math.log(tail) / math.log(quantile))
        )
        return estimate, interval, minimum_count

    def _progress(self, stop_time: float | None) -> SchedulerProgress:
        if self.required_samples is None:
            return SchedulerProgress(stop_time=stop_time)

        return SchedulerProgress(
            remaining_requests=max(0, self.required_samples - self.count),
            total_requests=self.required_samples,
            stop_time=stop_time,
        )


@ConstraintsInitializerFactory.register("target_moe")
class TargetMoeConstraintInitializer(PydanticConstraintInitializer):
    """
    Factory for creating TargetMoeConstraint instances from configuration.

    Example:
    ::

        from guidellm.schemas.scheduler import TargetMoeConstraintArgs

        args = TargetMoeConstraintArgs(statistic="p95", moe=0.05)
        initializer = TargetMoeConstraintInitializer(args=args)
        constraint = initializer.create_constraint()
    """

    type_: Literal["target_moe"] = "target_moe"  # type: ignore[assignment]
    args: TargetMoeConstraintArgs = Field(
        description="Configuration arguments for target margin of error stopping",
    )

    def create_constraint(self, **_kwargs) -> Constraint:
        """
        Create a TargetMoeConstraint instance from stored args.

        :param _kwargs: Additional keyword arguments (unused)
        :return: Configured TargetMoeConstraint instance with fresh samples
        """
        return TargetMoeConstraint(
            moe=self.args.moe,
            metric=self.args.metric,
            statistic=self.args.statistic,
            confidence=self.args.confidence,
            min_samples=self.args.min_samples,
            check_interval=self.args.check_interval,
            stopping_scope=self.args.stopping_scope,
        )
