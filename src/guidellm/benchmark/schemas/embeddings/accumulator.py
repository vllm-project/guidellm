"""
Real-time metric accumulation for embeddings benchmark execution.

Captures and computes performance metrics during embeddings benchmark runs, tracking
timing phases, request statistics, input token throughput, and latency distributions.
Unlike generative workloads, embeddings do not have output tokens or streaming behavior,
so this accumulator focuses on input processing metrics and optional quality validation
metrics like cosine similarity.
"""

from __future__ import annotations

import random
import time
from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas.base import BenchmarkAccumulator, BenchmarkConfig
from guidellm.scheduler import MultiTurnRequestT, SchedulerState
from guidellm.schemas import (
    EmbeddingsRequestStats,
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = [
    "EmbeddingsBenchmarkAccumulator",
    "EmbeddingsBenchmarkTimings",
    "EmbeddingsMetricsAccumulator",
    "EmbeddingsQualityMetricsAccumulator",
    "EmbeddingsRequestsAccumulator",
    "RunningMetricStats",
    "SchedulerMetricsAccumulator",
]


class EmbeddingsBenchmarkTimings(StandardBaseModel):
    """
    Tracks timing phases and transitions during embeddings benchmark execution.

    Monitors timestamps throughout benchmark execution including request submission,
    measurement period boundaries (warmup/active/cooldown), and completion events.
    """

    request_start: float | None = Field(
        description="Timestamp when the first request was sent", default=None
    )
    measure_start: float | None = Field(
        description="Timestamp when measurement period started", default=None
    )
    measure_end: float | None = Field(
        description="Timestamp when measurement period ended", default=None
    )
    request_end: float | None = Field(
        description="Timestamp when the last request was completed", default=None
    )
    current_update: float | None = Field(
        description="Most recent timestamp observed during execution", default=None
    )
    current_request: float | None = Field(
        description="Most recent request completion timestamp observed", default=None
    )
    last_update: float | None = Field(
        description="Previous timestamp observed before the current one", default=None
    )
    last_request: float | None = Field(
        description="Previous request completion timestamp before the current one",
        default=None,
    )

    @property
    def status(self) -> Literal["pending", "warmup", "active", "cooldown"]:
        """
        :return: Current execution phase based on timing thresholds
        """
        if self.request_start is None or self.current_update is None:
            return "pending"

        if self.measure_start is None or self.current_update <= self.measure_start:
            return "warmup"

        if self.measure_end is not None and self.current_update >= self.measure_end:
            return "cooldown"

        return "active"

    @property
    def duration(self) -> float:
        """
        :return: Elapsed time since measurement or request start in seconds
        """
        if self.request_start is None or self.current_update is None:
            return 0.0

        return self.current_update - self.request_start

    @property
    def elapsed_time_last_update(self) -> float:
        """
        :return: Time elapsed since last update
        """
        if self.current_update is None or self.last_update is None:
            return 0.0

        return self.current_update - self.last_update

    @property
    def finalized_request_start(self) -> float:
        """
        :return: Finalized timestamp for when requests started
        """
        return self.request_start or -1.0

    @property
    def finalized_measure_start(self) -> float:
        """
        :return: Finalized timestamp for when measurement started
        """
        return self.measure_start or self.finalized_request_start

    @property
    def finalized_measure_end(self) -> float:
        """
        :return: Finalized timestamp for when measurement ended
        """
        return self.measure_end or self.finalized_request_end

    @property
    def finalized_request_end(self) -> float:
        """
        :return: Finalized timestamp for when requests ended
        """
        return self.request_end or self.current_request or -1.0

    def update_estimate(
        self,
        info: RequestInfo,
        scheduler_state: SchedulerState,
        config: BenchmarkConfig,
    ):
        """
        Update timing estimates based on request info and scheduler state.

        :param info: Request information containing timing data
        :param scheduler_state: Current scheduler state with progress metrics
        :param config: Benchmark configuration with warmup/cooldown settings
        """
        # Update non-terminal timestamps
        self.request_start = scheduler_state.start_requests_time
        self.last_update = self.current_update
        if (current_time := info.timings.last_reported) is not None:
            self.current_update = (
                current_time
                if self.current_update is None
                else max(self.current_update, current_time)
            )

        # Update measurement period timestamps
        warmup_active, measure_start = config.warmup.compute_transition_time(
            info=info, state=scheduler_state, period="start"
        )
        if not warmup_active:
            self.measure_start = self.request_start
        elif measure_start is not None:
            self.measure_start = measure_start

        cooldown_active, measure_end = config.cooldown.compute_transition_time(
            info=info, state=scheduler_state, period="end"
        )
        if cooldown_active and measure_end is not None:
            self.measure_end = measure_end

        # Update terminal timestamps for completed requests
        if info.status in {"completed", "errored", "cancelled"}:
            self.last_request = self.current_request
            if info.completed_at is not None and (
                self.current_request is None or info.completed_at > self.current_request
            ):
                self.current_request = info.completed_at

        # Update request stop timestamps
        if scheduler_state.end_processing_time is not None and self.request_end is None:
            self.request_end = (
                scheduler_state.progress.stop_time
                or self.current_request
                or scheduler_state.end_processing_time
            )
            if self.measure_end is None:
                self.measure_end = self.request_end


class RunningMetricStats(StandardBaseModel):
    """
    Maintains running statistics for a metric stream without storing all samples.

    Accumulates count, sum, time-weighted sum, and duration for efficient
    real-time metric tracking during long-running benchmarks.
    """

    count: int = Field(description="Number of samples accumulated", default=0)
    value_sum: float = Field(description="Total sum of accumulated values", default=0.0)
    time_weighted_sum: float = Field(
        description="Time-weighted sum of accumulated values", default=0.0
    )
    duration: float = Field(
        description="Total duration over which values were accumulated", default=0.0
    )
    last_value: float | None = Field(
        description="Most recent value added to the accumulator", default=None
    )

    @property
    def mean(self) -> float | None:
        """
        :return: Arithmetic mean of accumulated values, or None if no samples
        """
        if self.count <= 0:
            return None
        return self.value_sum / self.count

    @property
    def time_weighted_mean(self) -> float | None:
        """
        :return: Time-weighted mean considering duration between samples, or None
        """
        if self.duration <= 0.0:
            return None
        return self.time_weighted_sum / self.duration

    @property
    def rate_per_item(self) -> float | None:
        """
        :return: Average value per accumulated item, or None if no samples
        """
        if self.count <= 0:
            return None
        return self.value_sum / self.count

    @property
    def rate_per_second(self) -> float | None:
        """
        :return: Average value per second of duration, or None if no duration
        """
        if self.duration <= 0.0:
            return None
        return self.value_sum / self.duration

    def update_estimate(
        self,
        value: float | None,
        count: int = 1,
        duration: float | None = None,
        elapsed: float | None = None,
    ):
        """
        Incorporate a new metric value into running statistics.

        Updates count, sum, and time-weighted statistics using the new value and timing
        information. Time-weighted calculations use the previous value over the elapsed
        interval to capture sustained metric behavior.

        :param value: New metric value to accumulate
        :param count: Number of occurrences this value represents
        :param duration: Total duration to set, overriding incremental elapsed updates
        :param elapsed: Time elapsed since last update for time-weighted calculations
        """
        self.count += count
        self.value_sum += (value or 0.0) * count

        if elapsed is not None:
            self.time_weighted_sum += (self.last_value or 0.0) * elapsed

        self.duration = (
            duration if duration is not None else (self.duration + (elapsed or 0.0))
        )
        self.last_value = value


class SchedulerMetricsAccumulator(StandardBaseModel):
    """
    Tracks scheduler-level timing and overhead metrics during execution.
    """

    start_time: float = Field(description="Scheduler start timestamp", default=0.0)
    request_start_time: float = Field(
        description="First request timestamp", default=0.0
    )
    measure_start_time: float = Field(
        description="Measurement start timestamp", default=0.0
    )
    measure_end_time: float = Field(description="Measurement end timestamp", default=0.0)
    request_end_time: float = Field(description="Last request timestamp", default=0.0)
    end_time: float = Field(description="Scheduler end timestamp", default=0.0)

    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status",
        default_factory=lambda: StatusBreakdown[int, int, int, int](
            successful=0, errored=0, incomplete=0, total=0
        ),
    )

    # Running metrics for progress tracking (compatible with generative)
    queued_time: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for time requests spent in the queue",
    )
    resolve_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay before worker starts resolving",
    )
    resolve_targeted_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay to targeted resolve start",
    )
    request_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay from resolve to request start",
    )
    request_targeted_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay to targeted request start",
    )
    resolve_end_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay after request end till worker resolves",
    )

    # Sum fields for final compilation
    queued_time_sum: float = Field(
        description="Total time requests spent in queue", default=0.0
    )
    resolve_start_delay_sum: float = Field(
        description="Total delay before worker starts resolving", default=0.0
    )
    resolve_targeted_start_delay_sum: float = Field(
        description="Total delay to targeted resolve start", default=0.0
    )
    request_start_delay_sum: float = Field(
        description="Total delay from resolve to request start", default=0.0
    )
    resolve_time_sum: float = Field(
        description="Total resolution time", default=0.0
    )

    def update_estimate(
        self, scheduler_state: SchedulerState, stats: EmbeddingsRequestStats
    ):
        """
        Update scheduler metrics with completed request timing data.

        :param scheduler_state: Current scheduler state
        :param stats: Completed request statistics
        """
        # Update request counts
        self.requests_made.successful = scheduler_state.successful_requests
        self.requests_made.errored = scheduler_state.errored_requests
        self.requests_made.incomplete = scheduler_state.cancelled_requests
        self.requests_made.total = (
            scheduler_state.successful_requests
            + scheduler_state.errored_requests
            + scheduler_state.cancelled_requests
        )

        # Update timing sums and running stats
        timings = stats.info.timings
        if timings.queued is not None and timings.dequeued is not None:
            queued_time_val = timings.dequeued - timings.queued
            self.queued_time_sum += queued_time_val
            self.queued_time.update_estimate(value=queued_time_val)

        if timings.dequeued is not None and timings.resolve_start is not None:
            resolve_start_delay_val = timings.resolve_start - timings.dequeued
            self.resolve_start_delay_sum += resolve_start_delay_val
            self.resolve_start_delay.update_estimate(value=resolve_start_delay_val)

        if timings.targeted_start is not None and timings.resolve_start is not None:
            resolve_targeted_delay_val = timings.resolve_start - timings.targeted_start
            self.resolve_targeted_start_delay_sum += resolve_targeted_delay_val
            self.resolve_targeted_start_delay.update_estimate(
                value=resolve_targeted_delay_val
            )

        if timings.resolve_start is not None and timings.request_start is not None:
            request_start_delay_val = timings.request_start - timings.resolve_start
            self.request_start_delay_sum += request_start_delay_val
            self.request_start_delay.update_estimate(value=request_start_delay_val)

        if timings.targeted_start is not None and timings.request_start is not None:
            request_targeted_delay_val = (
                timings.request_start - timings.targeted_start
            )
            self.request_targeted_start_delay.update_estimate(
                value=request_targeted_delay_val
            )

        if timings.request_end is not None and timings.resolve_end is not None:
            resolve_end_delay_val = timings.resolve_end - timings.request_end
            self.resolve_end_delay.update_estimate(value=resolve_end_delay_val)

        if timings.resolve_start is not None and timings.resolve_end is not None:
            resolve_time_val = timings.resolve_end - timings.resolve_start
            self.resolve_time_sum += resolve_time_val


class EmbeddingsQualityMetricsAccumulator(StandardBaseModel):
    """
    Accumulates quality validation metrics for embeddings.

    Tracks cosine similarity scores and MTEB benchmark results when quality
    validation is enabled.
    """

    cosine_similarities: list[float] = Field(
        default_factory=list,
        description="Cosine similarity scores against baseline",
    )
    baseline_cosine_similarity: StatusDistributionSummary | None = Field(
        default=None,
        description="Compiled cosine similarity distribution",
    )
    self_consistency_score: StatusDistributionSummary | None = Field(
        default=None,
        description="Compiled self-consistency scores",
    )
    mteb_main_score: float | None = Field(
        default=None,
        description="MTEB main score (if evaluated)",
    )
    mteb_task_scores: dict[str, float] | None = Field(
        default=None,
        description="Individual MTEB task scores",
    )


class EmbeddingsCompletedMetricsAccumulator(StandardBaseModel):
    """
    Tracks real-time metrics for completed embeddings requests.

    Used for progress tracking during benchmark execution.
    """

    requests: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Requests completion metrics",
    )
    request_latency: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Request latency running stats",
    )
    prompt_tokens: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Input tokens running stats",
    )
    total_tokens: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Total tokens (same as prompt for embeddings)",
    )


class EmbeddingsMetricsAccumulator(StandardBaseModel):
    """
    Accumulates performance metrics during embeddings benchmark execution.

    Tracks request latency, throughput, and input token metrics. Does not track
    output tokens or streaming metrics (no TTFT/ITL for embeddings).
    """

    requests_per_second: StatusDistributionSummary = Field(
        default_factory=StatusDistributionSummary,
        description="Requests per second distribution",
    )
    request_concurrency: StatusDistributionSummary = Field(
        default_factory=StatusDistributionSummary,
        description="Request concurrency distribution",
    )
    request_latency: StatusDistributionSummary = Field(
        default_factory=StatusDistributionSummary,
        description="Request latency distribution",
    )
    input_tokens_per_second: StatusDistributionSummary = Field(
        default_factory=StatusDistributionSummary,
        description="Input tokens per second distribution",
    )


class EmbeddingsRequestsAccumulator(StandardBaseModel):
    """
    Accumulates embeddings request statistics during benchmark execution.

    Uses reservoir sampling to maintain a representative sample of requests
    across different status categories.
    """

    successful: list[EmbeddingsRequestStats] = Field(
        default_factory=list,
        description="Sample of successful embeddings requests",
    )
    incomplete: list[EmbeddingsRequestStats] = Field(
        default_factory=list,
        description="Sample of incomplete embeddings requests",
    )
    errored: list[EmbeddingsRequestStats] = Field(
        default_factory=list,
        description="Sample of errored embeddings requests",
    )


class EmbeddingsBenchmarkAccumulator(
    BenchmarkAccumulator[GenerationRequest, GenerationResponse]
):
    """
    Accumulates metrics during embeddings benchmark execution.

    Extends BenchmarkAccumulator with embeddings-specific metric tracking including
    input token processing, request latency, and optional quality validation metrics.
    Does not track output tokens or streaming behavior.
    """

    type_: Literal["embeddings_benchmark_accumulator"] = (
        "embeddings_benchmark_accumulator"
    )

    # Core accumulators
    timings: EmbeddingsBenchmarkTimings = Field(
        default_factory=EmbeddingsBenchmarkTimings,
        description="Timing phase tracking",
    )
    scheduler_metrics: SchedulerMetricsAccumulator = Field(
        default_factory=SchedulerMetricsAccumulator,
        description="Scheduler metrics accumulation",
    )
    concurrency_metric: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Time-weighted concurrency statistics",
    )
    completed_metrics: EmbeddingsCompletedMetricsAccumulator = Field(
        default_factory=EmbeddingsCompletedMetricsAccumulator,
        description="Real-time metrics for completed requests",
    )
    metrics: EmbeddingsMetricsAccumulator = Field(
        default_factory=EmbeddingsMetricsAccumulator,
        description="Performance metrics accumulation",
    )
    requests: EmbeddingsRequestsAccumulator = Field(
        default_factory=EmbeddingsRequestsAccumulator,
        description="Request statistics accumulation",
    )

    # Quality validation (optional)
    quality_enabled: bool = Field(
        default=False,
        description="Whether quality validation is enabled",
    )
    quality: EmbeddingsQualityMetricsAccumulator | None = Field(
        default=None,
        description="Quality metrics accumulation (when enabled)",
    )

    # Encoding format tracking
    encoding_format_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Request count by encoding format",
    )

    # Reservoir sampling parameters
    _sampling_counts: dict[str, int] = {}
    _max_samples: int = 1000

    def update_estimate(
        self,
        response: GenerationResponse | None,
        request: GenerationRequest | MultiTurnRequestT[GenerationRequest],
        info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Update accumulated metrics with a new request completion.

        :param response: Response from the backend (if successful)
        :param request: Original generation request
        :param info: Request metadata and timing information
        :param scheduler_state: Current scheduler state
        """
        # Update timing state
        self.timings.update_estimate(info, scheduler_state, self.config)
        duration = self.timings.duration
        self.concurrency_metric.update_estimate(
            value=scheduler_state.processing_requests,
            duration=duration,
        )

        # Determine request status and target accumulator
        if info.status == "completed":
            status_key = "completed"
            status_list = self.requests.successful
        elif info.status == "errored":
            status_key = "errored"
            status_list = self.requests.errored
        elif info.status == "cancelled" and info.timings.resolve_start is not None:
            status_key = "incomplete"
            status_list = self.requests.incomplete
        else:
            # Not a terminal status or cancelled before starting
            # Do not include in requests or metrics
            return

        # Build request stats
        # Use response metrics if available (has actual token counts from server),
        # otherwise fall back to request metrics (word/char counts only)
        input_metrics = (
            response.input_metrics if response is not None else request.input_metrics
        )
        stats = EmbeddingsRequestStats(
            request_id=info.request_id,
            info=info,
            input_metrics=input_metrics,
        )

        # Track encoding format if available
        if hasattr(request, "encoding_format"):
            format_key = request.encoding_format or "float"
            self.encoding_format_breakdown[format_key] = (
                self.encoding_format_breakdown.get(format_key, 0) + 1
            )

        # Update scheduler metrics
        self.scheduler_metrics.update_estimate(scheduler_state, stats)

        # Update completed metrics for progress tracking (only for completed requests)
        if status_key == "completed":
            self.completed_metrics.requests.update_estimate(
                value=1.0,
                count=1,
                duration=self.timings.duration,
            )
            if stats.request_latency is not None:
                self.completed_metrics.request_latency.update_estimate(
                    value=stats.request_latency,
                    count=1,
                )
            if stats.prompt_tokens is not None:
                self.completed_metrics.prompt_tokens.update_estimate(
                    value=float(stats.prompt_tokens),
                    count=1,
                )
                self.completed_metrics.total_tokens.update_estimate(
                    value=float(stats.prompt_tokens),
                    count=1,
                )

        # Reservoir sampling
        sample_count = self._sampling_counts.get(status_key, 0)
        if len(status_list) < self._max_samples:
            status_list.append(stats)
        else:
            # Replace with decreasing probability
            j = random.randint(0, sample_count)
            if j < self._max_samples:
                status_list[j] = stats
        self._sampling_counts[status_key] = sample_count + 1
