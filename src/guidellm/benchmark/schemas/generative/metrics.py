"""
Metrics schemas for generative AI benchmark results and performance analysis.

This module defines comprehensive metric structures for tracking and analyzing
generative AI benchmark performance across multiple dimensions including request
statistics, token metrics, and domain-specific measurements for text, image, video,
and audio generation. It provides statistical summaries with distribution analysis
across successful, incomplete, and errored requests, along with scheduler-level
performance metrics for request processing and queueing behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.benchmark.schemas.generative.accumulator import (
    GenerativeBenchmarkAccumulator,
)
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    GenerativeRequestStats,
    StandardBaseDict,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = [
    "GenerativeAudioMetricsSummary",
    "GenerativeImageMetricsSummary",
    "GenerativeMetrics",
    "GenerativeMetricsSummary",
    "GenerativeTextMetricsSummary",
    "GenerativeVideoMetricsSummary",
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
        description="Avg delay before request starts after resolve (seconds)"
    )
    request_targeted_start_delay_avg: float = Field(
        description="Avg delay to targeted request start time (seconds)"
    )
    request_time_avg: float = Field(description="Avg request execution time (seconds)")
    resolve_end_delay_avg: float = Field(
        description="Avg delay after request completes before resolve ends (seconds)"
    )
    resolve_time_avg: float = Field(
        description="Avg total resolve time including request (seconds)"
    )
    finalized_delay_avg: float = Field(
        description="Avg delay from resolve end to request finalization (seconds)"
    )
    processed_delay_avg: float = Field(
        description="Avg delay from finalization to processing completion (seconds)"
    )

    @classmethod
    def compile(
        cls,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ) -> SchedulerMetrics:
        """
        Compile scheduler metrics from accumulator and scheduler state.

        :param accumulator: Benchmark accumulator containing timing and metric data
        :param scheduler_state: Scheduler state with execution timing information
        :return: Compiled scheduler metrics with performance statistics
        """
        return SchedulerMetrics(
            # Overall timings for the scheduler
            start_time=scheduler_state.start_time,
            request_start_time=accumulator.timings.finalized_request_start,
            measure_start_time=accumulator.timings.finalized_measure_start,
            measure_end_time=accumulator.timings.finalized_measure_end,
            request_end_time=accumulator.timings.finalized_request_end,
            end_time=scheduler_state.end_time or -1.0,
            # Request details tracked by the scheduler
            requests_made=accumulator.scheduler_metrics.requests_made,
            # Scheduler internal performance timings
            queued_time_avg=accumulator.scheduler_metrics.queued_time.mean or -1.0,
            resolve_start_delay_avg=(
                accumulator.scheduler_metrics.resolve_start_delay.mean or -1.0
            ),
            resolve_targeted_start_delay_avg=(
                accumulator.scheduler_metrics.resolve_targeted_start_delay.mean or -1.0
            ),
            request_start_delay_avg=(
                accumulator.scheduler_metrics.request_start_delay.mean or -1.0
            ),
            request_targeted_start_delay_avg=(
                accumulator.scheduler_metrics.request_targeted_start_delay.mean or -1.0
            ),
            request_time_avg=accumulator.scheduler_metrics.request_time.mean or -1.0,
            resolve_end_delay_avg=(
                accumulator.scheduler_metrics.resolve_end_delay.mean or -1.0
            ),
            resolve_time_avg=accumulator.scheduler_metrics.resolve_time.mean or -1.0,
            finalized_delay_avg=(
                accumulator.scheduler_metrics.finalized_delay.mean or -1.0
            ),
            processed_delay_avg=(
                accumulator.scheduler_metrics.processed_delay.mean or -1.0
            ),
        )


class GenerativeMetricsSummary(StandardBaseDict):
    """
    Statistical summaries for input, output, and total metrics.

    Provides distribution summaries across successful, incomplete, and errored
    requests for absolute values, per-second rates, and concurrency levels.
    """

    input: StatusDistributionSummary | None = Field(
        description="Distribution of input metric values"
    )
    input_per_second: StatusDistributionSummary | None = Field(
        description="Distribution of input metric rates per second"
    )
    input_concurrency: StatusDistributionSummary | None = Field(
        description="Distribution of concurrent input metric values"
    )

    output: StatusDistributionSummary | None = Field(
        description="Distribution of output metric values"
    )
    output_per_second: StatusDistributionSummary | None = Field(
        description="Distribution of output metric rates per second"
    )
    output_concurrency: StatusDistributionSummary | None = Field(
        description="Distribution of concurrent output metric values"
    )

    total: StatusDistributionSummary | None = Field(
        description="Distribution of total metric values (input + output)"
    )
    total_per_second: StatusDistributionSummary | None = Field(
        description="Distribution of total metric rates per second"
    )
    total_concurrency: StatusDistributionSummary | None = Field(
        description="Distribution of concurrent total metric values"
    )

    @classmethod
    def compile(
        cls,
        property_name: str,
        successful: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
    ) -> GenerativeMetricsSummary | None:
        """
        Compile metrics summary from request statistics for a specific property.

        :param property_name: Name of the property to extract from request metrics
        :param successful: Successfully completed request statistics
        :param incomplete: Incomplete or cancelled request statistics
        :param errored: Failed request statistics
        :return: Compiled metrics summary or None if no data available
        """
        successful_metrics = cls.extract_property_metrics_for_summary(
            successful, property_name
        )
        incomplete_metrics = cls.extract_property_metrics_for_summary(
            incomplete, property_name
        )
        errored_metrics = cls.extract_property_metrics_for_summary(
            errored, property_name
        )

        return cls.compile_timed_metrics(
            successful=successful_metrics,
            incomplete=incomplete_metrics,
            errored=errored_metrics,
        )

    @classmethod
    def compile_timed_metrics(
        cls,
        successful: list[TimedMetricTypeAlias],
        incomplete: list[TimedMetricTypeAlias],
        errored: list[TimedMetricTypeAlias],
    ) -> GenerativeMetricsSummary | None:
        """
        Compile metrics summary from timed metric tuples.

        :param successful: Timed metrics from successful requests
        :param incomplete: Timed metrics from incomplete requests
        :param errored: Timed metrics from errored requests
        :return: Compiled metrics summary or None if no data available
        """

        def _compile_metric_distributions(
            metrics_by_status: dict[StatusTypes, list[TimedMetricTypeAlias]],
            value_index: int,
        ) -> tuple[
            StatusDistributionSummary | None,
            StatusDistributionSummary | None,
            StatusDistributionSummary | None,
            dict[StatusTypes, list[float]],
            dict[StatusTypes, list[tuple[float, float]]],
            dict[StatusTypes, list[tuple[float, float, float]]],
        ]:
            """Helper to compile value, rate, and concurrency distributions."""
            value_lists: dict[StatusTypes, list[float]] = {
                status: [
                    float(metric[value_index] or 0.0)
                    for metric in metrics
                    if metric is not None
                ]
                for status, metrics in metrics_by_status.items()
            }
            value_dist = StatusDistributionSummary.from_values(
                successful=value_lists["successful"],
                incomplete=value_lists["incomplete"],
                errored=value_lists["errored"],
            )

            if value_dist.total_sum == 0.0:
                return None, None, None, value_lists, {}, {}

            rate_lists: dict[StatusTypes, list[tuple[float, float]]] = {
                status: [
                    (  # type: ignore[misc]
                        metric[_TIMED_METRIC_END_TIME_INDEX],
                        float(metric[value_index] or 0.0),
                    )
                    for metric in metrics
                    if metric is not None
                ]
                for status, metrics in metrics_by_status.items()
            }
            rate_dist = StatusDistributionSummary.rate_distribution_from_timings(
                successful=rate_lists["successful"],
                incomplete=rate_lists["incomplete"],
                errored=rate_lists["errored"],
            )

            concurrency_lists: dict[StatusTypes, list[tuple[float, float, float]]] = {
                status: [
                    (  # type: ignore[misc]
                        metric[_TIMED_METRIC_START_TIME_INDEX],
                        metric[_TIMED_METRIC_END_TIME_INDEX],
                        float(metric[value_index] or 0.0),
                    )
                    for metric in metrics
                    if metric is not None
                ]
                for status, metrics in metrics_by_status.items()
            }
            concurrency_dist = (
                StatusDistributionSummary.concurrency_distribution_from_timings(
                    successful=concurrency_lists["successful"],
                    incomplete=concurrency_lists["incomplete"],
                    errored=concurrency_lists["errored"],
                )
            )

            return (
                value_dist,
                rate_dist,
                concurrency_dist,
                value_lists,
                rate_lists,
                concurrency_lists,
            )

        metrics_by_status: dict[StatusTypes, list[TimedMetricTypeAlias]] = {
            "successful": successful,
            "incomplete": incomplete,
            "errored": errored,
        }

        # Calculate input distributions
        (
            input_value_dist,
            input_rate_dist,
            input_concurrency_dist,
            input_value_lists,
            input_rate_lists,
            input_concurrency_lists,
        ) = _compile_metric_distributions(
            metrics_by_status, _TIMED_METRIC_INPUT_VALUE_INDEX
        )

        # Calculate output distributions
        (
            output_value_dist,
            output_rate_dist,
            output_concurrency_dist,
            output_value_lists,
            output_rate_lists,
            output_concurrency_lists,
        ) = _compile_metric_distributions(
            metrics_by_status, _TIMED_METRIC_OUTPUT_VALUE_INDEX
        )

        # Calculate total distributions if both input and output have data
        if input_value_dist is not None and output_value_dist is not None:
            total_value_dist = StatusDistributionSummary.from_values(
                successful=(
                    input_value_lists["successful"] + output_value_lists["successful"]
                ),
                incomplete=(
                    input_value_lists["incomplete"] + output_value_lists["incomplete"]
                ),
                errored=input_value_lists["errored"] + output_value_lists["errored"],
            )
            total_rate_dist = StatusDistributionSummary.rate_distribution_from_timings(
                successful=(
                    input_rate_lists["successful"] + output_rate_lists["successful"]
                ),
                incomplete=(
                    input_rate_lists["incomplete"] + output_rate_lists["incomplete"]
                ),
                errored=input_rate_lists["errored"] + output_rate_lists["errored"],
            )
            total_concurrency_dist = (
                StatusDistributionSummary.concurrency_distribution_from_timings(
                    successful=(
                        input_concurrency_lists["successful"]
                        + output_concurrency_lists["successful"]
                    ),
                    incomplete=(
                        input_concurrency_lists["incomplete"]
                        + output_concurrency_lists["incomplete"]
                    ),
                    errored=(
                        input_concurrency_lists["errored"]
                        + output_concurrency_lists["errored"]
                    ),
                )
            )
        else:
            total_value_dist = None
            total_rate_dist = None
            total_concurrency_dist = None

        return GenerativeMetricsSummary(
            input=input_value_dist,
            input_per_second=input_rate_dist,
            input_concurrency=input_concurrency_dist,
            output=output_value_dist,
            output_per_second=output_rate_dist,
            output_concurrency=output_concurrency_dist,
            total=total_value_dist,
            total_per_second=total_rate_dist,
            total_concurrency=total_concurrency_dist,
        )

    @classmethod
    def extract_property_metrics_for_summary(
        cls, stats_list: list[GenerativeRequestStats], property_name: str
    ) -> list[TimedMetricTypeAlias]:
        """
        Extract timed metrics for a specific property from request statistics.

        :param stats_list: List of request statistics to extract from
        :param property_name: Name of the property to extract from metrics
        :return: List of tuples containing
            (start_time, end_time, input_value, output_value)
        """
        return [
            (
                stats.request_start_time,
                stats.request_end_time,
                getattr(stats.input_metrics, property_name),
                getattr(stats.output_metrics, property_name),
            )
            for stats in stats_list
            if (
                stats.request_start_time
                and stats.request_end_time
                and (
                    getattr(stats.input_metrics, property_name) is not None
                    or getattr(stats.output_metrics, property_name) is not None
                )
            )
        ]


class GenerativeTextMetricsSummary(StandardBaseDict):
    """
    Text-specific metric summaries for generative benchmarks.

    Tracks token, word, and character-level metrics across input, output, and
    total usage for text generation workloads.
    """

    tokens: GenerativeMetricsSummary | None = Field(
        description="Token count metrics and distributions"
    )
    words: GenerativeMetricsSummary | None = Field(
        description="Word count metrics and distributions"
    )
    characters: GenerativeMetricsSummary | None = Field(
        description="Character count metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        successful: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
    ) -> GenerativeTextMetricsSummary:
        """
        Compile text metrics summary from request statistics.

        :param successful: Successfully completed request statistics
        :param incomplete: Incomplete/cancelled request statistics
        :param errored: Failed request statistics
        :return: Compiled text metrics summary
        """
        return GenerativeTextMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                property_name="text_tokens",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            words=GenerativeMetricsSummary.compile(
                property_name="text_words",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            characters=GenerativeMetricsSummary.compile(
                property_name="text_characters",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
        )


class GenerativeImageMetricsSummary(StandardBaseDict):
    """
    Image-specific metric summaries for generative benchmarks.

    Tracks token, image count, pixel, and byte-level metrics across input, output,
    and total usage for image generation workloads.
    """

    tokens: GenerativeMetricsSummary | None = Field(
        description="Image token count metrics and distributions"
    )
    images: GenerativeMetricsSummary | None = Field(
        description="Image count metrics and distributions"
    )
    pixels: GenerativeMetricsSummary | None = Field(
        description="Pixel count metrics and distributions"
    )
    bytes: GenerativeMetricsSummary | None = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        successful: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
    ) -> GenerativeImageMetricsSummary:
        """
        Compile image metrics summary from request statistics.

        :param successful: Successfully completed request statistics
        :param incomplete: Incomplete/cancelled request statistics
        :param errored: Failed request statistics
        :return: Compiled image metrics summary
        """
        return GenerativeImageMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                property_name="image_tokens",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            images=GenerativeMetricsSummary.compile(
                property_name="image_count",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            pixels=GenerativeMetricsSummary.compile(
                property_name="image_pixels",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            bytes=GenerativeMetricsSummary.compile(
                property_name="image_bytes",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
        )


class GenerativeVideoMetricsSummary(StandardBaseDict):
    """
    Video-specific metric summaries for generative benchmarks.

    Tracks token, frame count, duration, and byte-level metrics across input,
    output, and total usage for video generation workloads.
    """

    tokens: GenerativeMetricsSummary | None = Field(
        description="Video token count metrics and distributions"
    )
    frames: GenerativeMetricsSummary | None = Field(
        description="Frame count metrics and distributions"
    )
    seconds: GenerativeMetricsSummary | None = Field(
        description="Duration metrics in seconds and distributions"
    )
    bytes: GenerativeMetricsSummary | None = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        successful: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
    ) -> GenerativeVideoMetricsSummary:
        """
        Compile video metrics summary from request statistics.

        :param successful: Successfully completed request statistics
        :param incomplete: Incomplete/cancelled request statistics
        :param errored: Failed request statistics
        :return: Compiled video metrics summary
        """
        return GenerativeVideoMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                property_name="video_tokens",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            frames=GenerativeMetricsSummary.compile(
                property_name="video_frames",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            seconds=GenerativeMetricsSummary.compile(
                property_name="video_seconds",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            bytes=GenerativeMetricsSummary.compile(
                property_name="video_bytes",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
        )


class GenerativeAudioMetricsSummary(StandardBaseDict):
    """
    Audio-specific metric summaries for generative benchmarks.

    Tracks token, sample count, duration, and byte-level metrics across input,
    output, and total usage for audio generation workloads.
    """

    tokens: GenerativeMetricsSummary | None = Field(
        description="Audio token count metrics and distributions"
    )
    samples: GenerativeMetricsSummary | None = Field(
        description="Sample count metrics and distributions"
    )
    seconds: GenerativeMetricsSummary | None = Field(
        description="Duration metrics in seconds and distributions"
    )
    bytes: GenerativeMetricsSummary | None = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        successful: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
    ) -> GenerativeAudioMetricsSummary:
        """
        Compile audio metrics summary from request statistics.

        :param successful: Successfully completed request statistics
        :param incomplete: Incomplete/cancelled request statistics
        :param errored: Failed request statistics
        :return: Compiled audio metrics summary
        """
        return GenerativeAudioMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                property_name="audio_tokens",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            samples=GenerativeMetricsSummary.compile(
                property_name="audio_samples",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            seconds=GenerativeMetricsSummary.compile(
                property_name="audio_seconds",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            bytes=GenerativeMetricsSummary.compile(
                property_name="audio_bytes",
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
        )


class GenerativeMetrics(StandardBaseDict):
    """
    Comprehensive metrics for generative AI benchmarks.

    Aggregates request statistics, token metrics, timing distributions, and
    domain-specific measurements across text, image, video, and audio modalities.
    Provides detailed statistical summaries including distribution analysis for
    throughput, latency, concurrency, and resource utilization metrics across
    successful, incomplete, and errored requests.
    """

    # Request stats
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    requests_per_second: StatusDistributionSummary = Field(
        description="Distribution of requests per second across benchmark execution"
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent request counts during execution"
    )
    request_latency: StatusDistributionSummary = Field(
        description="Distribution of request latencies for completed requests"
    )
    request_streaming_iterations_count: StatusDistributionSummary = Field(
        description="Distribution of stream iterations for completed requests"
    )

    # General token stats
    prompt_token_count: StatusDistributionSummary = Field(
        description="Distribution of prompt token counts by request status"
    )
    output_token_count: StatusDistributionSummary = Field(
        description="Distribution of output token counts by request status"
    )
    total_token_count: StatusDistributionSummary = Field(
        description="Distribution of total token counts by request status"
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description="Distribution of first token latencies in milliseconds"
    )
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description="Distribution of average time per output token in milliseconds"
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description="Distribution of inter-token latencies in milliseconds"
    )
    prompt_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of prompt token processing rates"
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of output token generation rates"
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of total token throughput including prompt and output"
    )
    output_tokens_per_iteration: StatusDistributionSummary = Field(
        description="Distribution of output tokens generated per streaming iteration"
    )
    iter_tokens_per_iteration: StatusDistributionSummary = Field(
        description=(
            "Distribution of output tokens (without first) generated per "
            "streaming iteration"
        )
    )

    # Domain specific stats
    text: GenerativeTextMetricsSummary = Field(
        description="Text-specific metrics for tokens, words, and characters"
    )
    image: GenerativeImageMetricsSummary = Field(
        description="Image-specific metrics for tokens, images, pixels, and bytes"
    )
    video: GenerativeVideoMetricsSummary = Field(
        description="Video-specific metrics for tokens, frames, duration, and bytes"
    )
    audio: GenerativeAudioMetricsSummary = Field(
        description="Audio-specific metrics for tokens, samples, duration, and bytes"
    )

    @classmethod
    def compile(cls, accumulator: GenerativeBenchmarkAccumulator) -> GenerativeMetrics:
        """
        Compile comprehensive generative metrics from benchmark accumulator.

        :param accumulator: Benchmark accumulator with completed request statistics
        :return: Compiled generative metrics with all distributions and summaries
        :raises ValueError: If measure_start and measure_end/request_end are not set
        """
        start_time = accumulator.timings.finalized_measure_start
        end_time = accumulator.timings.finalized_measure_end

        if start_time == -1.0 or end_time == -1.0:
            raise ValueError(
                "Cannot compile GenerativeMetrics: "
                "No measurement start or end times available."
            )

        successful = accumulator.completed.get_within_range(start_time, end_time)
        incomplete = accumulator.incomplete.get_within_range(start_time, end_time)
        errored = accumulator.errored.get_within_range(start_time, end_time)

        return GenerativeMetrics(
            # Request stats
            request_totals=StatusBreakdown(
                successful=len(successful),
                incomplete=len(incomplete),
                errored=len(errored),
                total=(len(successful) + len(incomplete) + len(errored)),
            ),
            requests_per_second=StatusDistributionSummary.rate_distribution_from_timings_function(
                function=lambda req: req.request_end_time,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
                start_time=start_time,
                end_time=end_time,
            ),
            request_concurrency=StatusDistributionSummary.concurrency_distribution_from_timings_function(
                function=(
                    lambda req: (req.request_start_time, req.request_end_time)
                    if req.request_start_time is not None
                    and req.request_end_time is not None
                    else None
                ),
                successful=successful,
                incomplete=incomplete,
                errored=errored,
                start_time=start_time,
                end_time=end_time,
            ),
            request_latency=StatusDistributionSummary.from_values_function(
                function=lambda req: req.request_latency or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            request_streaming_iterations_count=StatusDistributionSummary.from_values_function(
                function=lambda req: req.info.timings.request_iterations or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            # General token stats
            prompt_token_count=StatusDistributionSummary.from_values_function(
                function=lambda req: req.prompt_tokens or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            output_token_count=StatusDistributionSummary.from_values_function(
                function=lambda req: req.output_tokens or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            total_token_count=StatusDistributionSummary.from_values_function(
                function=lambda req: req.total_tokens or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            time_to_first_token_ms=StatusDistributionSummary.from_values_function(
                function=lambda req: req.time_to_first_token_ms or 0.0,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            time_per_output_token_ms=StatusDistributionSummary.from_values_function(
                function=lambda req: (
                    req.time_per_output_token_ms or 0.0,
                    req.output_tokens or 0.0,
                ),
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            inter_token_latency_ms=StatusDistributionSummary.from_values_function(
                function=lambda req: (
                    req.inter_token_latency_ms or 0.0,
                    (req.output_tokens or 1.0) - 1.0,
                ),
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            prompt_tokens_per_second=StatusDistributionSummary.rate_distribution_from_timings_function(
                function=lambda req: req.prompt_tokens_timing,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            output_tokens_per_second=StatusDistributionSummary.rate_distribution_from_timings_function(
                function=lambda req: req.output_tokens_timings,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            tokens_per_second=StatusDistributionSummary.rate_distribution_from_timings_function(
                function=lambda req: req.total_tokens_timings,
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            output_tokens_per_iteration=StatusDistributionSummary.from_values_function(
                function=lambda req: [
                    tokens for (_timing, tokens) in req.output_tokens_timings
                ],
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            iter_tokens_per_iteration=StatusDistributionSummary.from_values_function(
                function=lambda req: [
                    tokens for (_timing, tokens) in req.iter_tokens_timings
                ],
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
            # Domain-specific stats
            text=GenerativeTextMetricsSummary.compile(
                successful=successful, incomplete=incomplete, errored=errored
            ),
            image=GenerativeImageMetricsSummary.compile(
                successful=successful, incomplete=incomplete, errored=errored
            ),
            video=GenerativeVideoMetricsSummary.compile(
                successful=successful, incomplete=incomplete, errored=errored
            ),
            audio=GenerativeAudioMetricsSummary.compile(
                successful=successful, incomplete=incomplete, errored=errored
            ),
        )
