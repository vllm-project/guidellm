"""
Real-time metric accumulation for generative benchmark execution.

Captures and computes performance metrics during benchmark runs, tracking timing phases,
request statistics, token throughput, and latency distributions. Components include
timing trackers for warmup/cooldown phases, running statistical accumulators for
throughput and latency metrics, and reservoir sampling for request data. Enables
comprehensive performance measurement including scheduler overhead, time-to-first-token,
inter-token latency, and token generation rates across completed, errored, and
incomplete requests.
"""

from __future__ import annotations

import random
import time
from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas.base import BenchmarkAccumulator, BenchmarkConfig
from guidellm.scheduler import MultiTurnRequestT, SchedulerState
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    StandardBaseModel,
    StatusBreakdown,
)

__all__ = [
    "GenerativeBenchmarkAccumulator",
    "GenerativeBenchmarkTimings",
    "GenerativeMetricsAccumulator",
    "GenerativeRequestsAccumulator",
    "RunningMetricStats",
    "SchedulerMetricsAccumulator",
]


class GenerativeBenchmarkTimings(StandardBaseModel):
    """
    Tracks timing phases and transitions during benchmark execution.

    Monitors timestamps throughout benchmark execution including request submission,
    measurement period boundaries (warmup/active/cooldown), and completion events.
    Provides duration calculations and phase status determination based on configured
    warmup and cooldown periods.
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
        :return: Time elapsed between the last two update timestamps in seconds
        """
        if self.current_update is None or self.last_update is None:
            return 0.0

        return self.current_update - self.last_update

    @property
    def elapsed_time_last_request(self) -> float:
        """
        :return: Time elapsed between the last two request completions in seconds
        """
        if self.current_request is None or self.last_request is None:
            return 0.0

        return self.current_request - self.last_request

    @property
    def finalized_request_start(self) -> float:
        """
        :return: Finalized timestamp from the current state for when requests started
        """
        return self.request_start or -1.0

    @property
    def finalized_measure_start(self) -> float:
        """
        :return: Finalized timestamp from the current state for when measurement started
        """
        return self.measure_start or self.finalized_request_start

    @property
    def finalized_measure_end(self) -> float:
        """
        :return: Finalized timestamp from the current state for when measurement ended
        """
        return self.measure_end or self.finalized_request_end

    @property
    def finalized_request_end(self) -> float:
        """
        :return: Finalized timestamp from the current state for when requests ended
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

        Advances timing markers through benchmark phases (warmup to active to cooldown)
        based on configured thresholds. Updates current/last timestamps for updates and
        request completions, determining measurement period boundaries.

        :param info: Request information containing timing data
        :param scheduler_state: Current scheduler state with progress metrics
        :param config: Benchmark configuration with warmup/cooldown settings
        """
        # First update non terminal timestamps
        self.request_start = scheduler_state.start_requests_time
        self.last_update = self.current_update
        if (current_time := info.timings.last_reported) is not None:
            self.current_update = (
                current_time
                if self.current_update is None
                else max(self.current_update, current_time)
            )

        # Next update measurement period timestamps, if available and possible
        warmup_active, measure_start = config.warmup.compute_transition_time(
            info=info, state=scheduler_state, period="start"
        )
        if not warmup_active:
            # No warmup, set measure_start to first request start
            self.measure_start = self.request_start
        elif measure_start is not None:
            self.measure_start = measure_start
        cooldown_active, measure_end = config.cooldown.compute_transition_time(
            info=info, state=scheduler_state, period="end"
        )
        if cooldown_active and measure_end is not None:
            self.measure_end = measure_end

        # Update last request terminal timestamps, if request is terminal
        if info.status in {"completed", "errored", "cancelled"}:
            self.last_request = self.current_request
            if info.completed_at is not None and (
                self.current_request is None or info.completed_at > self.current_request
            ):
                self.current_request = info.completed_at

        # Finally, update request stop timestamps, if at that stage and available
        if scheduler_state.end_processing_time is not None and self.request_end is None:
            self.request_end = (
                scheduler_state.progress.stop_time
                or self.current_request
                or scheduler_state.end_processing_time
            )
            if self.measure_end is None:
                # No cooldown triggered, set measure_end to request_end
                self.measure_end = self.request_end


class RunningMetricStats(StandardBaseModel):
    """
    Maintains running statistics for a metric stream without storing all samples.

    Accumulates count, sum, time-weighted sum, and duration to compute mean, rate,
    and time-weighted statistics incrementally. Efficient for real-time metric tracking
    during long-running benchmarks where storing individual samples is impractical.
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

    Monitors request lifecycle timing from queuing through completion, capturing delays
    at each stage: queue time, worker start delays, request processing time, and
    finalization overhead. Provides insight into scheduler efficiency and bottleneck
    identification in request orchestration.
    """

    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total",
        default_factory=lambda: StatusBreakdown[int, int, int, int](
            successful=0, errored=0, incomplete=0, total=0
        ),
    )
    # Timings flow:
    # Request scheduling: queued->dequeued->scheduled_at->resolve_start->
    # Request processing: request_start->*_iteration->request_end->
    # Request finalizing: resolve_end->finalized->accumulation update processed
    queued_time: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for time requests spent in the queue",
    )
    resolve_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description=(
            "Running stats for delay before worker begins resolving req after dequeue"
        ),
    )
    resolve_targeted_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description=(
            "Running stats for delay from targeted start to actual worker start"
        ),
    )
    request_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay after resolve til request start",
    )
    request_targeted_start_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description=(
            "Running stats for delay from targeted start to actual request start"
        ),
    )
    request_time: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for request processing time",
    )
    resolve_end_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay after request end till worker resolves",
    )
    resolve_time: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for time for worker to resolve requests",
    )
    finalized_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Running stats for delay after resolve til finalized in scheduler",
    )
    processed_delay: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description=(
            "Running stats for delay from finalized til request being "
            "processed by accumulation"
        ),
    )

    def update_estimate(
        self, scheduler_state: SchedulerState, stats: GenerativeRequestStats
    ):
        """
        Update scheduler metrics with completed request timing data.

        Extracts timing information from request statistics to update running metrics
        for each scheduler lifecycle stage. Validates that required timing markers are
        present before processing.

        :param scheduler_state: Current scheduler state with request counts
        :param stats: Completed request statistics with detailed timing information
        :raises ValueError: If required timing markers are missing
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

        # All requests must have queued, dequeued, resolve_end, and finalized timings
        timings: RequestTimings = stats.info.timings
        if any(
            timing is None
            for timing in [
                timings.queued,
                timings.dequeued,
                timings.resolve_end,
                timings.finalized,
            ]
        ):
            raise ValueError(
                "Required timings 'queued', 'dequeued', 'resolve_end', and "
                "'finalized' must not be None"
            )

        # Store validated non-None timings for type safety
        queued: float = timings.queued  # type: ignore[assignment]
        dequeued: float = timings.dequeued  # type: ignore[assignment]
        resolve_end: float = timings.resolve_end  # type: ignore[assignment]
        finalized: float = timings.finalized  # type: ignore[assignment]

        # Update timing metrics in occurrence order
        self.queued_time.update_estimate(value=dequeued - queued)

        if timings.scheduled_at is not None and timings.resolve_start is not None:
            self.resolve_start_delay.update_estimate(
                value=timings.resolve_start - timings.scheduled_at
            )

        if timings.targeted_start is not None and timings.resolve_start is not None:
            self.resolve_targeted_start_delay.update_estimate(
                value=timings.resolve_start - timings.targeted_start
            )

        if timings.resolve_start is not None and timings.request_start is not None:
            self.request_start_delay.update_estimate(
                value=timings.request_start - timings.resolve_start
            )

        if timings.targeted_start is not None and timings.request_start is not None:
            self.request_targeted_start_delay.update_estimate(
                value=timings.request_start - timings.targeted_start
            )

        if timings.request_start is not None and timings.request_end is not None:
            self.request_time.update_estimate(
                value=timings.request_end - timings.request_start
            )

        if timings.request_end is not None:
            self.resolve_end_delay.update_estimate(
                value=resolve_end - timings.request_end
            )

        if timings.resolve_start is not None:
            self.resolve_time.update_estimate(value=resolve_end - timings.resolve_start)

        self.finalized_delay.update_estimate(value=finalized - resolve_end)
        self.processed_delay.update_estimate(value=time.time() - finalized)


class GenerativeMetricsAccumulator(StandardBaseModel):
    """
    Accumulates generative model performance metrics during execution.

    Tracks token throughput, latency characteristics, and request timing for generative
    workloads. Maintains running statistics for input/output tokens,
    time-to-first-token, inter-token latency, and streaming patterns for comprehensive
    performance analysis.
    """

    requests: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated request count statistics",
    )
    request_latency: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated request latency statistics",
    )
    prompt_tokens: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated input token count statistics",
    )
    output_tokens: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated output token count statistics",
    )
    total_tokens: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated total token count statistics",
    )
    time_to_first_token_ms: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated time to first token statistics in milliseconds",
    )
    time_per_output_token_ms: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated time per output token statistics in milliseconds",
    )
    inter_token_latency_ms: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated inter-token latency statistics in milliseconds",
    )
    streaming_iterations: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated streaming iteration count statistics",
    )
    output_tokens_by_iteration: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated output tokens per iteration statistics",
    )
    iter_tokens_by_iteration: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated iteration tokens per iteration statistics",
    )

    def update_estimate(self, stats: GenerativeRequestStats, duration: float):
        """
        Update generative metrics with completed request statistics.

        Incorporates token counts, latency measurements, and streaming characteristics
        from a completed request into running metric accumulators with time-weighted
        calculations.

        :param stats: Request statistics containing token and latency measurements
        :param duration: Current benchmark duration for time-weighted metrics
        """
        self.requests.update_estimate(1.0, duration=duration)
        self.prompt_tokens.update_estimate(stats.prompt_tokens, duration=duration)
        self.output_tokens.update_estimate(stats.output_tokens, duration=duration)
        self.total_tokens.update_estimate(stats.total_tokens, duration=duration)
        self.request_latency.update_estimate(stats.request_latency, duration=duration)
        self.time_to_first_token_ms.update_estimate(
            stats.time_to_first_token_ms, duration=duration
        )
        self.time_per_output_token_ms.update_estimate(
            stats.time_per_output_token_ms,
            count=int(stats.output_tokens or 0),
            duration=duration,
        )
        self.inter_token_latency_ms.update_estimate(
            stats.inter_token_latency_ms,
            count=int((stats.output_tokens or 1) - 1),
            duration=duration,
        )
        self.streaming_iterations.update_estimate(
            stats.token_iterations, duration=duration
        )
        self.output_tokens_by_iteration.update_estimate(
            stats.output_tokens_per_iteration,
            count=int(stats.token_iterations or 0),
            duration=duration,
        )
        self.iter_tokens_by_iteration.update_estimate(
            stats.iter_tokens_per_iteration,
            count=int((stats.token_iterations or 1) - 1),
            duration=duration,
        )


class GenerativeRequestsAccumulator(StandardBaseModel):
    """
    Manages request statistics collection with optional reservoir sampling.

    Collects detailed request statistics while optionally sampling to limit memory usage
    in long-running benchmarks. Supports configurable sampling rates and selective data
    retention (clearing request arguments and/or outputs for non-sampled requests).
    """

    sample_requests: int | None = Field(
        default=None,
        description=(
            "Number of requests to sample and keep in the final benchmark for metrics"
        ),
    )
    requests_stats: list[GenerativeRequestStats] = Field(
        description="List of generative request statistics", default_factory=list
    )
    samples: list[int] | None = Field(
        description="Indices of sampled generative requests", default=None
    )
    clear_nonsampled_request_args: bool = Field(
        default=True,
        description=(
            "Whether to clear request arguments and outputs for non-sampled requests"
        ),
    )
    clear_nonsampled_outputs: bool = Field(
        default=True,
        description=(
            "Whether to clear outputs for non-sampled requests while keeping args"
        ),
    )

    def get_sampled(self) -> list[GenerativeRequestStats]:
        """
        Retrieve the list of sampled request statistics.

        :return: List of sampled generative request statistics
        """
        if self.samples is None:
            return self.requests_stats

        return [self.requests_stats[ind] for ind in self.samples]

    def get_within_range(
        self, start_time: float, end_time: float
    ) -> list[GenerativeRequestStats]:
        """
        Retrieve request statistics within a specified time range.

        :param start_time: Start timestamp for filtering (requests must end after this)
        :param end_time: End timestamp for filtering (requests must start before this)
        :return: List of request statistics within the time range
        """
        return [
            stats
            for stats in self.requests_stats
            if (stats.request_end_time >= start_time)
            and (
                (
                    stats.request_start_time is not None
                    and stats.request_start_time <= end_time
                )
                or (
                    stats.request_start_time is None
                    and stats.request_end_time <= end_time
                )
            )
        ]

    def update_estimate(
        self,
        response: GenerationResponse | None,
        request: GenerationRequest | MultiTurnRequestT[GenerationRequest],
        info: RequestInfo,
        prefer_response_metrics: bool,
    ) -> GenerativeRequestStats:
        """
        Record request statistics and apply reservoir sampling if configured.

        Compiles statistics from the completed request and adds to the collection.
        Uses reservoir sampling algorithm to maintain uniform sample distribution when
        enabled, clearing non-sampled request data to manage memory.

        :param response: Generation response containing output and metrics
        :param request: Original generation request with input data
        :param info: Request execution information and timing
        :param prefer_response_metrics: Whether to prefer metrics from response
        :return: Compiled request statistics
        """
        stats = self.compile_stats(response, request, info, prefer_response_metrics)

        current_index = len(self.requests_stats)
        self.requests_stats.append(stats)

        if self.sample_requests is None:
            # Keeping all requests, don't need to sample
            self.samples = None
        elif self.sample_requests <= 0:
            # Not keeping any requests, clear out unnecessary memory usage for current
            self.clear_stats_data(stats)
        elif self.sample_requests >= len(self.requests_stats):
            # Add directly to samples, haven't filled yet
            if self.samples is None:
                self.samples = []
            self.samples.append(current_index)
        elif self.sample_requests / len(self.requests_stats) >= random.random():
            # Sampling logic: choose to replace with decreasing probability s / n
            # where s is sample size, n is current number of requests.
            # If chosen, choose random existing sample to replace.
            # P(new item in samples)  = s / n
            # P(prev item in samples) = P(item was in samples) * P(not replaced)
            # P(prev item in samples) =
            #    P(before replacement) * P(new item selected) * P(chosen from samples)
            # P(prev item in samples) = (s / (n - 1)) * (s / n) * (1 / s) = s / n
            # P(prev item in samples) = P(new item in samples)
            if self.samples is None:
                self.samples = []
            replace_index = random.randrange(len(self.samples))
            self.clear_stats_data(self.samples[replace_index])
            self.samples[replace_index] = current_index

        return stats

    def clear_stats_data(self, stats: GenerativeRequestStats | int):
        if isinstance(stats, int):
            stats = self.requests_stats[stats]

        if self.clear_nonsampled_request_args:
            stats.request_args = None
        if self.clear_nonsampled_outputs:
            stats.output = None

    @classmethod
    def compile_stats(
        cls,
        response: GenerationResponse | None,
        request: GenerationRequest | MultiTurnRequestT[GenerationRequest],
        info: RequestInfo,
        prefer_response_metrics: bool,
    ) -> GenerativeRequestStats:
        """
        Compile statistics from request, response, and execution info.

        :param response: Generation response with output and metrics, or None
        :param request: Original generation request with input data
        :param info: Request execution information and timing
        :param prefer_response_metrics: Whether to prefer metrics from response
        :return: Compiled generative request statistics
        """
        # Extract the first request for arguments if multi-turn
        first_request: GenerationRequest
        if isinstance(request, GenerationRequest):
            first_request = request
        else:
            # Multi-turn request: extract first item
            first_item = request[0]
            first_request = (
                first_item[0] if isinstance(first_item, tuple) else first_item
            )

        if response is None:
            response = GenerationResponse(
                request_id=info.request_id, request_args=str(first_request.arguments)
            )

        return response.compile_stats(
            request=first_request,
            info=info,
            prefer_response=prefer_response_metrics,
        )


class GenerativeBenchmarkAccumulator(
    BenchmarkAccumulator[GenerationRequest, GenerationResponse]
):
    """
    Primary accumulator for generative benchmark execution metrics and statistics.

    Orchestrates real-time metric collection across timing, scheduler, concurrency, and
    generative performance dimensions. Maintains separate accumulators for completed,
    errored, and incomplete requests while tracking overall metrics. Integrates with
    scheduler state to monitor warmup/cooldown phases and compute time-weighted
    statistics for throughput and latency analysis.
    """

    timings: GenerativeBenchmarkTimings = Field(
        default_factory=GenerativeBenchmarkTimings,
        description="Timing phases and transitions during benchmark execution",
    )
    completed: GenerativeRequestsAccumulator = Field(
        default_factory=GenerativeRequestsAccumulator,
        description="Accumulator for completed requests",
    )
    errored: GenerativeRequestsAccumulator = Field(
        default_factory=GenerativeRequestsAccumulator,
        description="Accumulator for errored requests",
    )
    incomplete: GenerativeRequestsAccumulator = Field(
        default_factory=GenerativeRequestsAccumulator,
        description="Accumulator for incomplete requests",
    )
    scheduler_metrics: SchedulerMetricsAccumulator = Field(
        default_factory=SchedulerMetricsAccumulator,
        description="Running metrics for scheduler state",
    )
    concurrency_metric: RunningMetricStats = Field(
        default_factory=RunningMetricStats,
        description="Accumulated request concurrency statistics",
    )
    total_metrics: GenerativeMetricsAccumulator = Field(
        default_factory=GenerativeMetricsAccumulator,
        description="Running metrics for all requests",
    )
    completed_metrics: GenerativeMetricsAccumulator = Field(
        default_factory=GenerativeMetricsAccumulator,
        description="Running metrics for completed requests",
    )
    errored_metrics: GenerativeMetricsAccumulator = Field(
        default_factory=GenerativeMetricsAccumulator,
        description="Running metrics for errored requests",
    )
    incomplete_metrics: GenerativeMetricsAccumulator = Field(
        default_factory=GenerativeMetricsAccumulator,
        description="Running metrics for incomplete requests",
    )

    def update_estimate(
        self,
        response: GenerationResponse | None,
        request: GenerationRequest | MultiTurnRequestT[GenerationRequest],
        info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Update all benchmark metrics with a completed request.

        Processes request completion by updating timing phases, concurrency metrics,
        scheduler statistics, and generative performance metrics. Routes request to
        appropriate status-specific accumulator (completed/errored/incomplete) and
        updates aggregate totals. Cancelled requests that never started are ignored.

        :param response: Generation response with output and metrics, or None
        :param request: Original generation request with input data
        :param info: Request execution information and timing
        :param scheduler_state: Current scheduler state for phase tracking
        """
        self.timings.update_estimate(info, scheduler_state, self.config)
        duration = self.timings.duration
        elapsed_time_last_update = self.timings.elapsed_time_last_update
        self.concurrency_metric.update_estimate(
            value=scheduler_state.processing_requests,
            duration=duration,
            elapsed=elapsed_time_last_update,
        )

        requests_accumulator: GenerativeRequestsAccumulator
        metrics_accumulator: GenerativeMetricsAccumulator

        if info.status == "completed":
            requests_accumulator = self.completed
            metrics_accumulator = self.completed_metrics
        elif info.status == "errored":
            requests_accumulator = self.errored
            metrics_accumulator = self.errored_metrics
        elif info.status == "cancelled" and info.timings.resolve_start is not None:
            requests_accumulator = self.incomplete
            metrics_accumulator = self.incomplete_metrics
        else:
            # Not a terminal status or cancelled before starting
            # Do not include in requests or metrics
            return

        stats = requests_accumulator.update_estimate(
            response, request, info, self.config.prefer_response_metrics
        )
        metrics_accumulator.update_estimate(stats, duration)
        self.total_metrics.update_estimate(stats, duration)
        self.scheduler_metrics.update_estimate(scheduler_state, stats)
