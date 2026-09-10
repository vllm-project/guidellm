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
from typing import Any, Generic, TypeVar

from pydantic import Field

from guidellm.scheduler import (
    RequestT,
    ResponseT,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.schemas import (
    RequestInfo,
    StandardBaseDict,
    StatusDistributionSummary,
)
from guidellm.schemas.benchmark.goodput import GoodputSLO
from guidellm.schemas.benchmark.transient import TransientPhaseConfig

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
    sample_size: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests per status group to retain full data for. "
            "None keeps all, 0 strips all, N > 0 uses reservoir sampling."
        ),
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
    slo: GoodputSLO | None = Field(
        default=None,
        description=(
            "Per-request latency objectives defining which requests count "
            "toward goodput. None disables goodput measurement"
        ),
    )
    profile: dict[str, Any] = Field(
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
        request: RequestT,
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
