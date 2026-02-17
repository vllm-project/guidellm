"""
Metrics schemas for embeddings benchmark results and performance analysis.

This module defines comprehensive metric structures for tracking and analyzing
embeddings benchmark performance including request statistics, input token metrics,
and optional quality validation metrics such as cosine similarity and MTEB scores.
It provides statistical summaries with distribution analysis across successful,
incomplete, and errored requests, along with scheduler-level performance metrics
for request processing and queueing behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas.embeddings.accumulator import (
    EmbeddingsBenchmarkAccumulator,
)
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    StandardBaseDict,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = [
    "EmbeddingsMetrics",
    "EmbeddingsQualityMetrics",
    "SchedulerMetrics",
    "StatusTypes",
    "TimedMetricTypeAlias",
]


TimedMetricTypeAlias = (
    tuple[float, float, int | float | None, int | float | None] | None
)
"""Timed metric tuple containing start_time, end_time, input_value, and output_value."""

StatusTypes = Literal["successful", "incomplete", "errored"]
"""Request status category for metric compilation."""

# Constants for tuple indexing
_TIMED_METRIC_START_TIME_INDEX = 0
_TIMED_METRIC_END_TIME_INDEX = 1
_TIMED_METRIC_INPUT_VALUE_INDEX = 2
_TIMED_METRIC_OUTPUT_VALUE_INDEX = 3


class SchedulerMetrics(StandardBaseDict):
    """
    Scheduler timing and performance statistics.

    Tracks overall benchmark timing, request counts by status, and detailed internal
    scheduler performance metrics including queue times, processing delays, and
    request execution statistics. Used to analyze scheduler efficiency and identify
    bottlenecks in request processing pipelines.
    """

    # Overall timings for the scheduler
    start_time: float = Field(
        description="Unix timestamp when the benchmark run started"
    )
    request_start_time: float = Field(
        description="Unix timestamp when first request was made"
    )
    measure_start_time: float = Field(
        description="Unix timestamp when measurement period started"
    )
    measure_end_time: float = Field(
        description="Unix timestamp when measurement period ended"
    )
    request_end_time: float = Field(
        description="Unix timestamp when last request completed"
    )
    end_time: float = Field(description="Unix timestamp when the benchmark run ended")

    # Request details tracked by the scheduler
    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )

    # Scheduler internal performance timings
    queued_time_avg: float = Field(
        description="Avg time requests spent in the queue (seconds)"
    )
    resolve_start_delay_avg: float = Field(
        description="Avg delay before worker begins resolving req after dequeue (sec)"
    )
    resolve_targeted_start_delay_avg: float = Field(
        description="Avg delay to targeted resolve start time (seconds)"
    )
    request_start_delay_avg: float = Field(
        description="Avg delay from resolve start to actual request start (seconds)"
    )
    resolve_time_avg: float = Field(
        description="Avg total resolution time per request (seconds)"
    )

    @classmethod
    def compile(
        cls,
        accumulator: EmbeddingsBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> SchedulerMetrics:
        """
        Compile scheduler metrics from accumulator and scheduler state.

        :param accumulator: Accumulator containing scheduler timing and request data
        :param scheduler_state: Scheduler state with execution timing information
        :return: Compiled SchedulerMetrics instance with timing statistics
        """
        num_requests = accumulator.scheduler_metrics.requests_made.total

        # Avoid division by zero - use -1.0 to indicate no requests processed
        if num_requests is None or num_requests == 0:
            queued_time_avg = -1.0
            resolve_start_delay_avg = -1.0
            resolve_targeted_start_delay_avg = -1.0
            request_start_delay_avg = -1.0
            resolve_time_avg = -1.0
        else:
            queued_time_avg = (
                accumulator.scheduler_metrics.queued_time_sum / num_requests
            )
            resolve_start_delay_avg = (
                accumulator.scheduler_metrics.resolve_start_delay_sum
                / num_requests
            )
            resolve_targeted_start_delay_avg = (
                accumulator.scheduler_metrics
                .resolve_targeted_start_delay_sum
                / num_requests
            )
            request_start_delay_avg = (
                accumulator.scheduler_metrics.request_start_delay_sum
                / num_requests
            )
            resolve_time_avg = (
                accumulator.scheduler_metrics.resolve_time_sum / num_requests
            )

        return SchedulerMetrics(
            start_time=scheduler_state.start_time,
            request_start_time=accumulator.timings.finalized_request_start,
            measure_start_time=accumulator.timings.finalized_measure_start,
            measure_end_time=accumulator.timings.finalized_measure_end,
            request_end_time=accumulator.timings.finalized_request_end,
            end_time=scheduler_state.end_time or -1.0,
            requests_made=accumulator.scheduler_metrics.requests_made,
            queued_time_avg=queued_time_avg,
            resolve_start_delay_avg=resolve_start_delay_avg,
            resolve_targeted_start_delay_avg=resolve_targeted_start_delay_avg,
            request_start_delay_avg=request_start_delay_avg,
            resolve_time_avg=resolve_time_avg,
        )


class EmbeddingsQualityMetrics(StandardBaseDict):
    """
    Quality validation metrics for embeddings.

    Tracks cosine similarity scores against baseline models and MTEB benchmark
    performance. These metrics provide insights into embedding quality beyond
    raw performance measurements.
    """

    baseline_cosine_similarity: StatusDistributionSummary | None = Field(
        default=None,
        description="Cosine similarity distribution against baseline model (0.0-1.0)",
    )
    self_consistency_score: StatusDistributionSummary | None = Field(
        default=None,
        description="Self-consistency scores (same input → same embedding)",
    )
    mteb_main_score: float | None = Field(
        default=None,
        description="MTEB benchmark main score (average across tasks)",
    )
    mteb_task_scores: dict[str, float] | None = Field(
        default=None,
        description="Individual MTEB task scores (e.g., STS12, STS13)",
    )


class EmbeddingsMetrics(StandardBaseDict):
    """
    Performance and quality metrics for embeddings benchmarks.

    Encapsulates comprehensive performance data from embeddings workload executions
    including request-level statistics, input token metrics, and optional quality
    validation metrics. Unlike generative metrics, embeddings metrics do not track
    output tokens or streaming behavior (TTFT, ITL).
    """

    # Request statistics
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description="Total requests by status: successful, incomplete, errored, total"
    )
    requests_per_second: StatusDistributionSummary = Field(
        description=(
            "Requests per second distribution across measurement period"
        )
    )
    request_concurrency: StatusDistributionSummary = Field(
        description=(
            "Concurrent requests distribution throughout execution"
        )
    )
    request_latency: StatusDistributionSummary = Field(
        description="Request latency distribution (seconds)"
    )

    # Input token metrics (no output tokens for embeddings)
    input_tokens_count: StatusBreakdown[int, int, int, int] = Field(
        description=(
            "Total input tokens by status: successful, incomplete, "
            "errored, total"
        )
    )
    input_tokens_per_second: StatusDistributionSummary = Field(
        description="Input tokens per second distribution"
    )

    # Dummy output token fields for progress tracker compatibility (always zero)
    output_token_count: StatusBreakdown[int, int, int, int] = Field(
        default_factory=lambda: StatusBreakdown[int, int, int, int](
            successful=0, incomplete=0, errored=0, total=0
        ),
        description="Output tokens (always 0 for embeddings)",
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        default_factory=StatusDistributionSummary,
        description="Output tokens per second (always 0 for embeddings)",
    )
    prompt_token_count: StatusBreakdown[int, int, int, int] | None = Field(
        default=None,
        description="Same as input_tokens_count (for compatibility)",
    )
    tokens_per_second: StatusDistributionSummary | None = Field(
        default=None,
        description="Same as input_tokens_per_second (for compatibility)",
    )

    # Quality validation metrics (optional)
    quality: EmbeddingsQualityMetrics | None = Field(
        default=None,
        description="Quality validation metrics (when enabled)",
    )

    # Encoding format breakdown
    encoding_format_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Request count by encoding format (e.g., "
            "{'float': 50, 'base64': 0})"
        ),
    )

    @classmethod
    def compile(
        cls,
        accumulator: EmbeddingsBenchmarkAccumulator,
        _scheduler_state: SchedulerState,
    ) -> EmbeddingsMetrics:
        """
        Compile final embeddings metrics from accumulated execution state.

        :param accumulator: Accumulated benchmark state with request statistics
        :param scheduler_state: Final scheduler state after execution completion
        :return: Compiled embeddings metrics instance with complete statistics
        """
        # Compile request counts
        request_totals = StatusBreakdown[int, int, int, int](
            successful=len(accumulator.requests.successful),
            incomplete=len(accumulator.requests.incomplete),
            errored=len(accumulator.requests.errored),
            total=(
                len(accumulator.requests.successful)
                + len(accumulator.requests.incomplete)
                + len(accumulator.requests.errored)
            ),
        )

        # Compile input token counts
        input_tokens_count = StatusBreakdown[int, int, int, int](
            successful=sum(
                req.input_metrics.total_tokens or 0
                for req in accumulator.requests.successful
            ),
            incomplete=sum(
                req.input_metrics.total_tokens or 0
                for req in accumulator.requests.incomplete
            ),
            errored=sum(
                req.input_metrics.total_tokens or 0
                for req in accumulator.requests.errored
            ),
            total=0,  # Will be computed
        )
        input_tokens_count.total = (
            (input_tokens_count.successful or 0)
            + (input_tokens_count.incomplete or 0)
            + (input_tokens_count.errored or 0)
        )

        # Compile distribution metrics from request statistics
        start_time = accumulator.timings.finalized_measure_start
        end_time = accumulator.timings.finalized_measure_end

        # Filter requests within measurement period
        # If no valid measurement window (both -1.0), use all requests
        if start_time == -1.0 or end_time == -1.0:
            successful = accumulator.requests.successful
            incomplete = accumulator.requests.incomplete
            errored = accumulator.requests.errored
        else:
            successful = [
                req for req in accumulator.requests.successful
                if start_time <= req.request_end_time <= end_time
            ]
            incomplete = [
                req for req in accumulator.requests.incomplete
                if start_time <= req.request_end_time <= end_time
            ]
            errored = [
                req for req in accumulator.requests.errored
                if start_time <= req.request_end_time <= end_time
            ]

        # Compile distribution summaries
        requests_per_second = (
            StatusDistributionSummary
            .rate_distribution_from_timings_function(
                function=lambda req: req.request_end_time,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
                start_time=start_time,
                end_time=end_time,
            )
        )

        request_concurrency = (
            StatusDistributionSummary
            .concurrency_distribution_from_timings_function(
                function=lambda req: (
                    (req.request_start_time, req.request_end_time)
                    if req.request_start_time is not None
                    and req.request_end_time is not None
                    else None
                ),
                successful=successful,
                incomplete=incomplete,
                errored=errored,
                start_time=start_time,
                end_time=end_time,
            )
        )

        request_latency = StatusDistributionSummary.from_values(
            successful=[
                req.request_latency
                for req in successful
                if req.request_latency is not None
            ],
            incomplete=[
                req.request_latency
                for req in incomplete
                if req.request_latency is not None
            ],
            errored=[
                req.request_latency
                for req in errored
                if req.request_latency is not None
            ],
        )

        input_tokens_per_second = (
            StatusDistributionSummary
            .rate_distribution_from_timings_function(
                function=lambda req: req.input_tokens_timing,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            )
        )

        # Compile quality metrics if available
        quality_metrics = None
        if accumulator.quality_enabled and accumulator.quality is not None:
            quality_metrics = EmbeddingsQualityMetrics(
                baseline_cosine_similarity=accumulator.quality.baseline_cosine_similarity,
                self_consistency_score=accumulator.quality.self_consistency_score,
                mteb_main_score=accumulator.quality.mteb_main_score,
                mteb_task_scores=accumulator.quality.mteb_task_scores,
            )

        return EmbeddingsMetrics(
            request_totals=request_totals,
            requests_per_second=requests_per_second,
            request_concurrency=request_concurrency,
            request_latency=request_latency,
            input_tokens_count=input_tokens_count,
            input_tokens_per_second=input_tokens_per_second,
            prompt_token_count=input_tokens_count,  # Alias for compatibility
            tokens_per_second=input_tokens_per_second,  # Alias for compatibility
            quality=quality_metrics,
            encoding_format_breakdown=accumulator.encoding_format_breakdown,
        )
