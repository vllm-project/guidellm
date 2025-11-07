"""
Base schemas for benchmark execution, metric accumulation, and result compilation.

Defines abstract interfaces and configuration models for coordinating benchmark
execution with schedulers. The module centers around three key abstractions:
BenchmarkConfig encapsulates execution parameters and constraints; BenchmarkAccumulator
tracks incremental metrics during scheduler runs; and Benchmark compiles final results
with comprehensive latency, throughput, and concurrency distributions. Supports
configurable warmup/cooldown phases, transient period handling, and flexible metric
sampling strategies.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from guidellm.benchmark.profiles import Profile
from guidellm.scheduler import (
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.schemas import (
    RequestInfo,
    StandardBaseDict,
    StandardBaseModel,
    StatusDistributionSummary,
)

__all__ = [
    "Benchmark",
    "BenchmarkAccumulator",
    "BenchmarkAccumulatorT",
    "BenchmarkConfig",
    "BenchmarkT",
]

BenchmarkAccumulatorT = TypeVar(
    "BenchmarkAccumulatorT", bound="BenchmarkAccumulator[Any, Any]"
)
"Generic type variable for benchmark accumulator implementations"

BenchmarkT = TypeVar("BenchmarkT", bound="Benchmark")
"Generic type variable for benchmark result implementations"


class TransientPhaseConfig(StandardBaseModel):
    """
    Configure warmup and cooldown phases for benchmark execution.

    Supports flexible phase definition through percentage or absolute value
    specifications with multiple interpretation modes. Phases can be bounded
    by duration, request count, or both, enabling precise control over transient
    periods that should be excluded from final benchmark metrics.
    """

    @classmethod
    def create_from_value(
        cls, value: int | float | dict | TransientPhaseConfig | None
    ) -> TransientPhaseConfig:
        """
        Create configuration from flexible input formats.

        :param value: Configuration as int/float (percent if <1.0, absolute
            otherwise), dict (validated to model), TransientPhaseConfig instance,
            or None for defaults
        :return: Configured TransientPhaseConfig instance
        :raises ValueError: If value type is unsupported
        """
        if value is None:
            return TransientPhaseConfig()

        if isinstance(value, TransientPhaseConfig):
            return value

        if isinstance(value, dict):
            return TransientPhaseConfig.model_validate(value)

        if isinstance(value, int | float):
            kwargs = {
                "percent": value if value < 1.0 else None,
                "value": value if value >= 1.0 else None,
            }
            return TransientPhaseConfig.model_validate(kwargs)

        raise ValueError(f"Unsupported type for TransientPhaseConfig: {type(value)}")

    percent: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Phase size as percentage (0.0-1.0) of total duration/requests; "
            "interpretation depends on mode. Takes precedence over value when target "
            "mode is available, otherwise falls back to value"
        ),
        lt=1.0,
    )
    value: NonNegativeInt | NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Phase size as absolute duration (seconds) or request count; "
            "interpretation depends on mode. Used when percent is unset or "
            "target mode unavailable"
        ),
    )
    mode: Literal[
        "duration", "requests", "prefer_duration", "prefer_requests", "both"
    ] = Field(
        default="prefer_duration",
        description=(
            "Interpretation mode: 'duration' for time-based phases, 'requests' for "
            "count-based phases, 'prefer_duration'/'prefer_requests' for fallback "
            "behavior, 'both' requires satisfying both conditions"
        ),
    )

    def compute_limits(
        self, max_requests: int | None, max_seconds: float | None
    ) -> tuple[float | None, int | None]:
        """
        Calculate phase boundaries from benchmark constraints.

        :param max_requests: Total request budget for benchmark execution
        :param max_seconds: Total duration budget for benchmark execution
        :return: Tuple of (phase duration in seconds, phase request count)
        """
        duration: float | None = None
        requests: int | None = None

        if self.mode in ["duration", "prefer_duration", "both"]:
            if self.percent is not None and max_seconds is not None:
                duration = self.percent * max_seconds
            elif self.value is not None:
                duration = float(self.value)

        if self.mode in ["requests", "prefer_requests", "both"]:
            if self.percent is not None and max_requests is not None:
                requests = int(self.percent * max_requests)
            elif self.value is not None:
                requests = int(self.value)

        return duration, requests

    def compute_transition_time(
        self,
        start_time: float,
        request_start: float | None,
        request_end: float | None,
        current_requests: int,
        current_duration: float,
        remaining_requests: int | None,
        remaining_duration: float | None,
        period: Literal["start", "end"],
    ) -> tuple[bool, float | None]:
        """
        Determine transition timestamp for entering or exiting phase.

        :param start_time: Benchmark start timestamp in seconds since epoch
        :param request_start: Current request start timestamp
        :param request_end: Current request end timestamp
        :param current_requests: Requests completed at transition point
        :param current_duration: Elapsed duration at transition point
        :param remaining_requests: Requests remaining in benchmark budget
        :param remaining_duration: Duration remaining in benchmark budget
        :param period: Phase period ('start' for warmup, 'end' for cooldown)
        :return: Tuple of (phase active flag, transition timestamp if applicable)
        """
        max_duration = (
            current_duration + remaining_duration
            if remaining_duration is not None
            else None
        )
        max_requests = (
            current_requests + remaining_requests
            if remaining_requests is not None
            else None
        )
        target_duration, target_requests = self.compute_limits(
            max_requests=max_requests, max_seconds=max_duration
        )

        if target_duration is None and target_requests is None:
            return False, None

        duration_transition_time: float | None = None
        request_transition_time: float | None = None
        phase_active: bool = False

        if (
            target_duration is not None
            and max_duration is not None
            and remaining_duration is not None
        ):
            duration_transition_time = (
                start_time + target_duration
                if period == "start"
                else start_time + max_duration - target_duration
            )
            phase_active = True
        if (
            target_requests is not None
            and max_requests is not None
            and remaining_requests is not None
        ):
            request_transition_time = (
                request_start
                if period == "start" and current_requests > target_requests
                else request_end
                if period == "end" and remaining_requests < target_requests + 1
                else -1.0
            )
            phase_active = True

        transition_time: float | None = None

        match self.mode:
            case "duration":
                transition_time = duration_transition_time
            case "requests":
                transition_time = request_transition_time
            case "prefer_duration":
                transition_time = duration_transition_time or request_transition_time
            case "prefer_requests":
                transition_time = request_transition_time or duration_transition_time
            case "both":
                transition_time = (
                    -1.0
                    if request_transition_time == -1.0
                    else request_transition_time
                    if duration_transition_time is None
                    else duration_transition_time
                    if request_transition_time is None
                    else min(duration_transition_time, request_transition_time)
                    if period == "start"
                    else max(duration_transition_time, request_transition_time)
                )

        return phase_active, transition_time if transition_time != -1.0 else None


class BenchmarkConfig(StandardBaseDict):
    """
    Encapsulate execution parameters and constraints for benchmark runs.

    Defines comprehensive configuration including scheduler strategy, constraint
    sets, transient phase handling, metric sampling preferences, and execution
    metadata. Coordinates profile, request, backend, and environment configurations
    to enable reproducible benchmark execution with precise control over metric
    collection.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this benchmark execution",
    )
    run_id: str = Field(
        description="Identifier grouping related benchmark runs in a series",
    )
    run_index: int = Field(
        description="Zero-based index of this run within the benchmark series",
    )
    strategy: SchedulingStrategy = Field(
        description="Scheduler strategy controlling request execution patterns",
    )
    constraints: dict[str, dict[str, Any]] = Field(
        description="Constraint definitions applied to scheduler strategy execution",
    )
    sample_requests: int | None = Field(
        default=20,
        description="Request count for statistical sampling in final metrics",
    )
    warmup: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Warmup phase configuration excluding initial transient period",
    )
    cooldown: TransientPhaseConfig = Field(
        default_factory=TransientPhaseConfig,
        description="Cooldown phase configuration excluding final transient period",
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description="Prioritize response-based metrics over request-based metrics",
    )
    profile: Profile = Field(
        description="Profile instance coordinating multi-strategy execution",
    )
    requests: dict[str, Any] = Field(
        description="Request generation configuration and dataset metadata",
    )
    backend: dict[str, Any] = Field(
        description="Backend connection parameters and service configuration",
    )
    environment: dict[str, Any] = Field(
        description="Execution environment details and system metadata",
    )


class BenchmarkAccumulator(StandardBaseDict, ABC, Generic[RequestT, ResponseT]):
    """
    Track and accumulate benchmark metrics during scheduler execution.

    Maintains incremental metric estimates as requests are processed, enabling
    real-time progress monitoring and efficient metric compilation. Subclasses
    implement specific metric calculation strategies based on request/response
    characteristics and scheduler state evolution.
    """

    config: BenchmarkConfig = Field(
        description="Benchmark execution configuration and constraints",
    )

    @abstractmethod
    def update_estimate(
        self,
        response: ResponseT | None,
        request: RequestT | MultiTurnRequestT[RequestT],
        info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Incrementally update metrics with completed request data.

        :param response: Backend response data if request succeeded
        :param request: Request instance submitted to backend
        :param info: Request timing, status, and execution metadata
        :param scheduler_state: Current scheduler state with queue and concurrency info
        """
        ...


class Benchmark(StandardBaseDict, ABC, Generic[BenchmarkAccumulatorT]):
    """
    Compile and expose final benchmark execution metrics.

    Defines the interface for benchmark result implementations capturing
    comprehensive performance metrics including latency distributions, throughput
    measurements, and concurrency patterns. Subclasses implement compilation
    logic to transform accumulated metrics and scheduler state into structured
    results with statistical summaries.
    """

    @property
    @abstractmethod
    def start_time(self) -> float:
        """
        :return: Benchmark start timestamp in seconds since epoch
        """

    @property
    @abstractmethod
    def end_time(self) -> float:
        """
        :return: Benchmark completion timestamp in seconds since epoch
        """

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        :return: Benchmark execution duration in seconds
        """

    @property
    @abstractmethod
    def request_latency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of request latencies
        """

    @property
    @abstractmethod
    def request_throughput(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of throughput measurements
        """

    @property
    @abstractmethod
    def request_concurrency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of concurrent request counts
        """

    @classmethod
    @abstractmethod
    def compile(
        cls, accumulator: BenchmarkAccumulatorT, scheduler_state: SchedulerState
    ) -> Any:
        """
        Transform accumulated metrics into final benchmark results.

        :param accumulator: Accumulator instance with collected metrics and state
        :param scheduler_state: Scheduler's final state after execution completion
        :return: Compiled benchmark instance with complete statistical results
        """
