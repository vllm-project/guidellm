"""
Benchmark data models and metrics for embeddings performance measurement.

Provides comprehensive data structures for capturing, storing, and analyzing
benchmark results from scheduler-driven embeddings workload executions. Core
abstractions include embeddings-specific metrics without output tokens or streaming
behavior, request-level statistics tracking, and multi-benchmark reporting capabilities.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from guidellm.benchmark.schemas.base import Benchmark, BenchmarkConfig
from guidellm.benchmark.schemas.embeddings.accumulator import (
    EmbeddingsBenchmarkAccumulator,
)
from guidellm.benchmark.schemas.embeddings.metrics import (
    EmbeddingsMetrics,
    SchedulerMetrics,
)
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    EmbeddingsRequestStats,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = ["EmbeddingsBenchmark"]


class EmbeddingsBenchmark(Benchmark[EmbeddingsBenchmarkAccumulator]):
    """
    Complete embeddings benchmark results with specialized metrics.

    Encapsulates comprehensive performance data from scheduler-driven embeddings
    workload executions including request-level statistics, input token metrics,
    latency distributions, and optional quality validation metrics. Unlike generative
    benchmarks, does not track output tokens or streaming behavior.
    """

    type_: Literal["embeddings_benchmark"] = "embeddings_benchmark"  # type: ignore[assignment]

    config: BenchmarkConfig = Field(
        description="Configuration parameters for this benchmark execution",
    )
    scheduler_state: SchedulerState = Field(
        description="Final state of the scheduler after benchmark completion",
    )
    scheduler_metrics: SchedulerMetrics = Field(
        description="Scheduler timing and performance statistics",
    )
    metrics: EmbeddingsMetrics = Field(
        description="Performance metrics and statistical distributions",
    )
    requests: StatusBreakdown[
        list[EmbeddingsRequestStats],
        list[EmbeddingsRequestStats],
        list[EmbeddingsRequestStats],
        None,
    ] = Field(
        description=(
            "Request details grouped by status: successful, incomplete, errored"
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def start_time(self) -> float:
        """
        :return: Benchmark start time in seconds since epoch
        """
        return self.scheduler_metrics.measure_start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def end_time(self) -> float:
        """
        :return: Benchmark end time in seconds since epoch
        """
        return self.scheduler_metrics.measure_end_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        """
        :return: Total benchmark execution duration in seconds
        """
        return self.end_time - self.start_time

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warmup_duration(self) -> float:
        """
        :return: Warmup phase duration in seconds
        """
        return (
            self.scheduler_metrics.measure_start_time
            - self.scheduler_metrics.request_start_time
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cooldown_duration(self) -> float:
        """
        :return: Cooldown phase duration in seconds
        """
        return (
            self.scheduler_metrics.request_end_time
            - self.scheduler_metrics.measure_end_time
        )

    @property
    def request_latency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of request latencies across all requests
        """
        return self.metrics.request_latency

    @property
    def request_throughput(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of throughput measured in requests per second
        """
        return self.metrics.requests_per_second

    @property
    def request_concurrency(self) -> StatusDistributionSummary:
        """
        :return: Statistical distribution of concurrent requests throughout execution
        """
        return self.metrics.request_concurrency

    @classmethod
    def compile(
        cls,
        accumulator: EmbeddingsBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> EmbeddingsBenchmark:
        """
        Compile final benchmark results from accumulated execution state.

        :param accumulator: Accumulated benchmark state with request statistics
        :param scheduler_state: Final scheduler state after execution completion
        :return: Compiled embeddings benchmark instance with complete metrics
        """
        return EmbeddingsBenchmark(
            config=accumulator.config,
            scheduler_state=scheduler_state,
            scheduler_metrics=SchedulerMetrics.compile(accumulator, scheduler_state),
            metrics=EmbeddingsMetrics.compile(accumulator, scheduler_state),
            requests=StatusBreakdown(
                successful=accumulator.requests.successful,
                incomplete=accumulator.requests.incomplete,
                errored=accumulator.requests.errored,
                total=None,
            ),
        )
