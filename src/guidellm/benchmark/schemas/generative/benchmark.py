"""
Benchmark data models and metrics for generative AI performance measurement.

Provides comprehensive data structures for capturing, storing, and analyzing
benchmark results from scheduler-driven generative AI workload executions.
Core abstractions include base benchmark interfaces, generative-specific
metrics with token/latency distributions, request-level statistics tracking,
and multi-benchmark reporting capabilities. These models enable detailed
performance analysis including throughput, latency, concurrency patterns, and
domain-specific metrics for text, image, video, and audio generation tasks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from guidellm.benchmark.schemas.base import Benchmark, BenchmarkConfig
from guidellm.benchmark.schemas.generative.accumulator import (
    GenerativeBenchmarkAccumulator,
)
from guidellm.benchmark.schemas.generative.metrics import (
    GenerativeMetrics,
    SchedulerMetrics,
)
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    GenerativeRequestStats,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = ["GenerativeBenchmark"]


class GenerativeBenchmark(Benchmark[GenerativeBenchmarkAccumulator]):
    """
    Complete generative AI benchmark results with specialized metrics.

    Encapsulates comprehensive performance data from scheduler-driven generative
    workload executions including request-level statistics, token/latency distributions,
    throughput analysis, and concurrency patterns. Provides computed fields for temporal
    analysis and status-grouped request details for detailed post-execution reporting.
    """

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]

    config: BenchmarkConfig = Field(
        description="Configuration parameters for this benchmark execution",
    )
    scheduler_state: SchedulerState = Field(
        description="Final state of the scheduler after benchmark completion",
    )
    scheduler_metrics: SchedulerMetrics = Field(
        description="Scheduler timing and performance statistics",
    )
    metrics: GenerativeMetrics = Field(
        description="Performance metrics and statistical distributions",
    )
    requests: StatusBreakdown[
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
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
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> GenerativeBenchmark:
        """
        Compile final benchmark results from accumulated execution state.

        :param accumulator: Accumulated benchmark state with request statistics
        :param scheduler_state: Final scheduler state after execution completion
        :return: Compiled generative benchmark instance with complete metrics
        """
        return GenerativeBenchmark(
            config=accumulator.config,
            scheduler_state=scheduler_state,
            scheduler_metrics=SchedulerMetrics.compile(accumulator, scheduler_state),
            metrics=GenerativeMetrics.compile(accumulator),
            requests=StatusBreakdown(
                successful=accumulator.completed.get_sampled(),
                incomplete=accumulator.incomplete.get_sampled(),
                errored=accumulator.errored.get_sampled(),
                total=None,
            ),
        )
