"""
Core benchmark schemas for performance measurement and result analysis.

Provides base classes and configuration for benchmark execution, including
accumulation of metrics during scheduler runs and compilation of final results.
Supports configurable scheduling strategies with comprehensive metric collection
for latency, throughput, and concurrency analysis.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import Field

from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.schemas import RequestInfo, StandardBaseDict, StatusDistributionSummary

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

BenchmarkT = TypeVar("BenchmarkT", bound="Benchmark")


class BenchmarkConfig(StandardBaseDict):
    """
    Configuration parameters for benchmark execution.

    Encapsulates scheduler strategy, request sampling, warmup/cooldown phases,
    and metric collection preferences for controlled benchmark runs.
    """

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this benchmark execution",
    )
    run_id: str = Field(
        description="Unique identifier for the benchmark run",
    )
    run_index: int = Field(
        description="Sequential index of this run within a benchmark series",
    )
    strategy: SchedulingStrategy = Field(
        description="Scheduling strategy for request execution",
    )
    constraints: dict[str, dict[str, Any]] = Field(
        description="Constraints applied to the scheduling strategy",
    )
    sample_requests: int | None = Field(
        default=20,
        description="Number of requests to sample for final benchmark metrics",
    )
    warmup: int | float | None = Field(
        default=None,
        description="Warmup period in seconds before benchmarking starts",
    )
    cooldown: int | float | None = Field(
        default=None,
        description="Cooldown period in seconds after benchmarking ends",
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description="Whether to prioritize response metrics over request metrics",
    )
    profile: Profile = Field(
        description="Benchmark profile defining execution parameters",
    )
    requests: dict[str, Any] = Field(
        description="Request configuration and dataset information",
    )
    backend: dict[str, Any] = Field(
        description="Backend configuration and connection details",
    )
    environment: dict[str, Any] = Field(
        description="Execution environment configuration and metadata",
    )


class BenchmarkAccumulator(StandardBaseDict, ABC, Generic[RequestT, ResponseT]):
    """
    Accumulates metrics and state during benchmark execution.

    Tracks benchmark progress by updating estimates as requests are processed,
    enabling incremental metric collection during scheduler runs.
    """

    config: BenchmarkConfig = Field(
        description="Configuration parameters for this benchmark execution",
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
        Update benchmark estimates with new request/response data.

        :param response: Response from the backend, if available
        :param request: Request submitted to the backend
        :param info: Metadata about request execution timing and status
        :param scheduler_state: Current state of the scheduler
        """
        ...


class Benchmark(StandardBaseDict, ABC, Generic[BenchmarkAccumulatorT]):
    """
    Abstract base class for benchmark result implementations.

    Defines the interface for capturing execution metrics and compiling final results
    from scheduler-driven workload executions, including request latency, throughput,
    and concurrency distributions.
    """

    @property
    @abstractmethod
    def start_time(self) -> float:
        """
        :return: Benchmark start time in seconds since epoch
        """

    @property
    @abstractmethod
    def end_time(self) -> float:
        """
        :return: Benchmark end time in seconds since epoch
        """

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        :return: Total benchmark execution duration in seconds
        """

    @property
    @abstractmethod
    def request_latency(self) -> StatusDistributionSummary:
        """
        :return: Distribution of request latencies across all processed requests
        """

    @property
    @abstractmethod
    def request_throughput(self) -> StatusDistributionSummary:
        """
        :return: Distribution of request throughput across benchmark duration
        """

    @property
    @abstractmethod
    def request_concurrency(self) -> StatusDistributionSummary:
        """
        :return: Distribution of concurrent requests across benchmark duration
        """

    @classmethod
    @abstractmethod
    def compile(
        cls, accumulator: BenchmarkAccumulatorT, scheduler_state: SchedulerState
    ) -> Any:
        """
        Compile final benchmark results from accumulated metrics.

        :param accumulator: Accumulated benchmark state with request statistics
        :param scheduler_state: Final state of the scheduler after execution
        :return: Compiled benchmark instance with complete results
        """
