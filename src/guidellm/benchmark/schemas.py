"""
Benchmark data models and metrics for performance measurement and analysis.

Provides comprehensive data structures for capturing, storing, and analyzing
benchmark results from scheduler executions. Includes timing measurements,
token statistics, and performance metrics for generative AI workloads.

Classes:
    BenchmarkSchedulerStats: Scheduler timing and performance statistics.
    BenchmarkMetrics: Core benchmark metrics and distributions.
    BenchmarkRequestStats: Individual request processing statistics.
    Benchmark: Base benchmark result container with generic metrics.
    GenerativeRequestStats: Request statistics for generative AI workloads.
    GenerativeMetrics: Comprehensive metrics for generative benchmarks.
    GenerativeBenchmark: Complete generative benchmark results and analysis.
    GenerativeBenchmarksReport: Container for multiple benchmark results.

Type Variables:
    BenchmarkMetricsT: Generic benchmark metrics type.
    BenchmarkRequestStatsT: Generic request statistics type.
    BenchmarkT: Generic benchmark container type.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast

import yaml
from pydantic import Field, computed_field

from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    BackendInterface,
    Environment,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
)
from guidellm.schemas.request import UsageMetrics
from guidellm.utils import (
    InfoMixin,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = [
    "Benchmark",
    "BenchmarkArgs",
    "BenchmarkSchedulerStats",
    "BenchmarkT",
    "BenchmarkerDict",
    "EstimatedBenchmarkState",
    "GenerativeAudioMetricsSummary",
    "GenerativeBenchmark",
    "GenerativeBenchmarksReport",
    "GenerativeImageMetricsSummary",
    "GenerativeMetrics",
    "GenerativeMetricsSummary",
    "GenerativeTextMetricsSummary",
    "GenerativeVideoMetricsSummary",
    "SchedulerDict",
]


class EstimatedBenchmarkState(dict[str, Any]):
    benchmark_state_group: ClassVar[Literal["benchmark_state"]] = "benchmark_state"
    benchmark_metrics_group: ClassVar[Literal["benchmark_metrics"]] = (
        "benchmark_metrics"
    )
    scheduler_state_group: ClassVar[Literal["scheduler_state"]] = "scheduler_state"

    def get_metric(
        self,
        group: str,
        key: str,
        default: int | float | None = None,
    ) -> int | float | None:
        return self.get(f"{group}_{key}", default)

    def set_metric(
        self,
        group: str,
        key: str,
        value: bool | int | float | None,
        start_val: bool | int | float | None = None,
    ) -> bool | int | float | None:
        if value is None:
            return None

        if start_val is not None:
            value -= start_val
        self[f"{group}_{key}"] = value

        return value

    def add_avg_metric(
        self,
        group: str,
        key: str,
        value: bool | int | float | None,
        start_val: bool | int | float | None = 0.0,
        count: int | None = 1,
    ):
        if value is None or count is None:
            return

        if start_val is not None:
            value -= start_val

        total_key = f"{group}_{key}_total"
        count_key = f"{group}_{key}_count"
        self[total_key] = self.get(total_key, 0) + value
        self[count_key] = self.get(count_key, 0) + count

        average = self[total_key] / self[count_key] if self[count_key] > 0 else 0.0
        self.set_metric(
            group=group,
            key=key,
            value=average,
        )

    def add_avg_rate_metric(
        self,
        group: str,
        key: str,
        value: bool | int | float | None,
        start_val: bool | int | float | None = 0.0,
        start_time: float | None = None,
        end_time: float | None = None,
        numerator_type: Literal["avg", "total", "count"] = "total",
    ):
        if value is None:
            return

        self.add_avg_metric(
            group=group,
            key=key,
            value=value,
            start_val=start_val,
        )
        start_time_key = f"{group}_{key}_start_time"
        if self.get(start_time_key) is None:
            if start_time is None:
                start_time = time.time()
            self[start_time_key] = start_time
        else:
            self[start_time_key] = start_time or self[start_time_key]

        end_time = end_time or time.time()
        elapsed_time = end_time - self[start_time_key]

        if elapsed_time > 0:
            numerator_key = (
                f"{group}_{key}_{numerator_type}"
                if numerator_type != "avg"
                else f"{group}_{key}"
            )
            rate = self[numerator_key] / elapsed_time
            self.set_metric(
                group=group,
                key=f"{key}_per_second",
                value=rate,
            )

    def add_time_averaged_metric(
        self,
        group: str,
        key: str,
        value: bool | int | float | None,
        recorded_time: float | None = None,
    ):
        if value is None:
            return

        if recorded_time is None:
            recorded_time = time.time()

        time_avg_numerator_key = f"{group}_{key}_time_avg_numerator"
        time_avg_denominator_key = f"{group}_{key}_time_avg_denominator"
        last_recorded_time_key = f"{group}_{key}_last_recorded_time"
        last_recorded_value_key = f"{group}_{key}_last_recorded_value"

        if last_recorded_time_key not in self:
            self[last_recorded_time_key] = recorded_time
            self[last_recorded_value_key] = value
            self[time_avg_numerator_key] = value
            self[time_avg_denominator_key] = 0.0
        else:
            time_delta = recorded_time - self[last_recorded_time_key]
            self[time_avg_numerator_key] += self[last_recorded_value_key] * time_delta
            self[time_avg_denominator_key] += time_delta
            self[last_recorded_time_key] = recorded_time
            self[last_recorded_value_key] = value

        if self[time_avg_denominator_key] > 0:
            average = self[time_avg_numerator_key] / self[time_avg_denominator_key]
        else:
            average = value

        self.set_metric(
            group=group,
            key=key,
            value=average,
        )


class BenchmarkArgs(StandardBaseDict):
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the benchmark run",
    )
    run_index: int = Field(default=0, description="Index of the benchmark run")
    sample_requests: int | None = Field(
        default=20,
        description="Number of requests to sample and keep in the final benchmark for metrics",
    )
    warmup: int | float | None = Field(
        default=None, description="Warmup time before benchmarking starts"
    )
    cooldown: int | float | None = Field(
        default=None, description="Cooldown time after benchmarking ends"
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description="Whether to prefer response metrics over request metrics",
    )

    def is_in_warmup(
        self, request_info: RequestInfo, scheduler_state: SchedulerState
    ) -> bool:
        if self.warmup is not None and 0 < self.warmup < 1:
            # Percentage-based warmup
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction > (1 - self.warmup)
            )

        if self.warmup is not None and self.warmup > 1:
            # Count/time-based warmup
            if scheduler_state.processed_requests < self.warmup:
                return True

            current_time = request_info.timings.targeted_start
            return (
                current_time is not None
                and (current_time - scheduler_state.start_time) < self.warmup
            )

        return False

    def is_in_cooldown(
        self, request_info: RequestInfo, scheduler_state: SchedulerState
    ) -> bool:
        if self.cooldown is not None and 0 < self.cooldown < 1:
            # Percentage-based cooldown
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction < self.cooldown
            )

        if self.cooldown is not None and self.cooldown > 1:
            # Count/time-based cooldown
            if (
                scheduler_state.remaining_requests is not None
                and scheduler_state.remaining_requests <= self.cooldown
            ):
                return True

            current_time = (
                request_info.timings.resolve_end or request_info.timings.targeted_start
            )
            return (
                current_time is not None
                and scheduler_state.remaining_duration is not None
                and scheduler_state.remaining_duration < self.cooldown
            )

        return False


class Benchmark(ABC):
    @abstractmethod
    def get_run_metrics_sample(
        self,
    ) -> dict[Literal["start_time", "end_time", "duration"], float]: ...

    @abstractmethod
    def get_request_metrics_sample(
        self,
    ) -> dict[
        Literal[
            "request_count",
            "request_latency",
            "request_throughput",
            "request_concurrency",
        ],
        float,
    ]: ...

    @classmethod
    @abstractmethod
    def update_estimate(
        cls,
        args: BenchmarkArgs,
        state: EstimatedBenchmarkState,
        response: Any,
        request: Any,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ): ...

    @classmethod
    @abstractmethod
    def compile(
        cls,
        args: BenchmarkArgs,
        estimated_state: EstimatedBenchmarkState,
        scheduler_state: SchedulerState,
        profile: Profile,
        requests: Iterable,
        backend: BackendInterface,
        environment: Environment,
        strategy: SchedulingStrategy,
        constraints: dict[str, dict[str, Any]],
    ) -> Any: ...


BenchmarkT = TypeVar("BenchmarkT", bound=Benchmark)


class BenchmarkSchedulerStats(StandardBaseDict):
    """Scheduler timing and performance statistics."""

    group_name: ClassVar[Literal["scheduler_stats"]] = "scheduler_stats"

    start_time: float = Field(
        description="Unix timestamp when the benchmark run started"
    )
    end_time: float = Field(description="Unix timestamp when the benchmark run ended")
    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    queued_time_avg: float = Field(
        description="Avg time requests spent in the queue (seconds)"
    )
    worker_resolve_start_delay_avg: float = Field(
        description="Avg delay before worker begins resolving req after dequeue (sec)"
    )
    worker_resolve_time_avg: float = Field(
        description="Avg time for worker to resolve requests (seconds)"
    )
    worker_resolve_end_delay_avg: float = Field(
        description="Avg delay after request end till worker resolves (seconds)"
    )
    finalized_delay_avg: float = Field(
        description="Avg delay after resolve til finalized with in scheduler (sec)"
    )
    worker_targeted_start_delay_avg: float = Field(
        description="Avg delay from targeted start to actual worker start (seconds)"
    )
    request_start_delay_avg: float = Field(
        description="Avg delay after resolve til request start (seconds)"
    )
    request_time_avg: float = Field(description="Avg request processing time (seconds)")
    request_targeted_start_delay_avg: float = Field(
        description="Avg delay from targeted start to actual request start"
    )

    @classmethod
    def update_estimate(cls, state: EstimatedBenchmarkState, request_info: RequestInfo):
        state.set_metric(group=cls.group_name, key="updated", value=True)
        state.add_avg_metric(
            group=cls.group_name,
            key="queued_time",
            value=request_info.timings.dequeued,
            start_val=request_info.timings.queued,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="worker_resolve_start_delay",
            value=request_info.timings.resolve_start,
            start_val=request_info.timings.scheduled_at,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="worker_resolve_time",
            value=request_info.timings.resolve_end,
            start_val=request_info.timings.resolve_start,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="worker_resolve_end_delay",
            value=request_info.timings.request_end,
            start_val=request_info.timings.resolve_end,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="finalized_delay",
            value=request_info.timings.finalized,
            start_val=request_info.timings.resolve_end,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="worker_targeted_start_delay",
            value=request_info.timings.resolve_start,
            start_val=request_info.timings.targeted_start,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="request_start_delay",
            value=request_info.timings.request_start,
            start_val=request_info.timings.resolve_start,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="request_time",
            value=request_info.timings.request_end,
            start_val=request_info.timings.request_start,
        )
        state.add_avg_metric(
            group=cls.group_name,
            key="request_targeted_start_delay",
            value=request_info.timings.request_start,
            start_val=request_info.timings.targeted_start,
        )

    @classmethod
    def compile(
        cls, estimated_state: EstimatedBenchmarkState, scheduler_state: SchedulerState
    ) -> BenchmarkSchedulerStats:
        return BenchmarkSchedulerStats(
            start_time=scheduler_state.start_time,
            end_time=scheduler_state.end_time or scheduler_state.start_time,
            requests_made=StatusBreakdown[int, int, int, int](
                successful=scheduler_state.successful_requests,
                incomplete=scheduler_state.cancelled_requests,
                errored=scheduler_state.errored_requests,
                total=(
                    scheduler_state.successful_requests
                    + scheduler_state.cancelled_requests
                    + scheduler_state.errored_requests
                ),
            ),
            queued_time_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="queued_time", default=-1.0
                ),
            ),
            worker_resolve_start_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="worker_resolve_start_delay", default=-1.0
                ),
            ),
            worker_resolve_time_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="worker_resolve_time", default=-1.0
                ),
            ),
            worker_resolve_end_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="worker_resolve_end_delay", default=-1.0
                ),
            ),
            finalized_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="finalized_delay", default=-1.0
                ),
            ),
            worker_targeted_start_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name,
                    key="worker_targeted_start_delay",
                    default=-1.0,
                ),
            ),
            request_start_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="request_start_delay", default=-1.0
                ),
            ),
            request_time_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name, key="request_time", default=-1.0
                ),
            ),
            request_targeted_start_delay_avg=cast(
                "float",
                estimated_state.get_metric(
                    group=cls.group_name,
                    key="request_targeted_start_delay",
                    default=-1.0,
                ),
            ),
        )


class GenerativeMetricsSummary(StandardBaseDict):
    input: StatusDistributionSummary = Field(description="")
    input_per_second: StatusDistributionSummary = Field(description="")
    input_concurrency: StatusDistributionSummary = Field(description="")

    output: StatusDistributionSummary = Field(description="")
    output_per_second: StatusDistributionSummary = Field(description="")
    output_concurrency: StatusDistributionSummary = Field(description="")

    total: StatusDistributionSummary = Field(description="")
    total_per_second: StatusDistributionSummary = Field(description="")
    total_concurrency: StatusDistributionSummary = Field(description="")

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_values: list[int | float],
        output_values: list[int | float],
    ) -> GenerativeMetricsSummary:
        total_values = [
            input_val + output_val
            for input_val, output_val in zip(input_values, output_values, strict=False)
        ]

        return GenerativeMetricsSummary(
            input=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=input_values,
            ),
            input_per_second=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="rate",
                weights=input_values,
            ),
            input_concurrency=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="concurrency",
                weights=input_values,
            ),
            output=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=output_values,
            ),
            output_per_second=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="rate",
                weights=output_values,
            ),
            output_concurrency=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="concurrency",
                weights=output_values,
            ),
            total=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=total_values,
            ),
            total_per_second=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="rate",
                weights=total_values,
            ),
            total_concurrency=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="concurrency",
                weights=total_values,
            ),
        )


class GenerativeTextMetricsSummary(StandardBaseDict):
    tokens: GenerativeMetricsSummary = Field(description="")
    words: GenerativeMetricsSummary = Field(description="")
    characters: GenerativeMetricsSummary = Field(description="")

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeTextMetricsSummary:
        return GenerativeTextMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.text_tokens or 0 for metrics in input_metrics],
                output_values=[metrics.text_tokens or 0 for metrics in output_metrics],
            ),
            words=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.text_words or 0 for metrics in input_metrics],
                output_values=[metrics.text_words or 0 for metrics in output_metrics],
            ),
            characters=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[
                    metrics.text_characters or 0 for metrics in input_metrics
                ],
                output_values=[
                    metrics.text_characters or 0 for metrics in output_metrics
                ],
            ),
        )


class GenerativeImageMetricsSummary(StandardBaseDict):
    tokens: GenerativeMetricsSummary = Field(description="")
    images: GenerativeMetricsSummary = Field(description="")
    pixels: GenerativeMetricsSummary = Field(description="")
    bytes: GenerativeMetricsSummary = Field(description="")

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeImageMetricsSummary:
        return GenerativeImageMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.image_tokens or 0 for metrics in input_metrics],
                output_values=[metrics.image_tokens or 0 for metrics in output_metrics],
            ),
            images=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.image_count or 0 for metrics in input_metrics],
                output_values=[metrics.image_count or 0 for metrics in output_metrics],
            ),
            pixels=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.image_pixels or 0 for metrics in input_metrics],
                output_values=[metrics.image_pixels or 0 for metrics in output_metrics],
            ),
            bytes=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.image_bytes or 0 for metrics in input_metrics],
                output_values=[metrics.image_bytes or 0 for metrics in output_metrics],
            ),
        )


class GenerativeVideoMetricsSummary(StandardBaseDict):
    tokens: GenerativeMetricsSummary = Field(description="")
    frames: GenerativeMetricsSummary = Field(description="")
    seconds: GenerativeMetricsSummary = Field(description="")
    bytes: GenerativeMetricsSummary = Field(description="")

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeVideoMetricsSummary:
        return GenerativeVideoMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.video_tokens or 0 for metrics in input_metrics],
                output_values=[metrics.video_tokens or 0 for metrics in output_metrics],
            ),
            frames=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.video_frames or 0 for metrics in input_metrics],
                output_values=[metrics.video_frames or 0 for metrics in output_metrics],
            ),
            seconds=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.video_seconds or 0 for metrics in input_metrics],
                output_values=[
                    metrics.video_seconds or 0 for metrics in output_metrics
                ],
            ),
            bytes=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.video_bytes or 0 for metrics in input_metrics],
                output_values=[metrics.video_bytes or 0 for metrics in output_metrics],
            ),
        )


class GenerativeAudioMetricsSummary(StandardBaseDict):
    tokens: GenerativeMetricsSummary = Field(description="")
    samples: GenerativeMetricsSummary = Field(description="")
    seconds: GenerativeMetricsSummary = Field(description="")
    bytes: GenerativeMetricsSummary = Field(description="")

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeAudioMetricsSummary:
        return GenerativeAudioMetricsSummary(
            tokens=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.audio_tokens or 0 for metrics in input_metrics],
                output_values=[metrics.audio_tokens or 0 for metrics in output_metrics],
            ),
            samples=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.audio_samples or 0 for metrics in input_metrics],
                output_values=[
                    metrics.audio_samples or 0 for metrics in output_metrics
                ],
            ),
            seconds=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.audio_seconds or 0 for metrics in input_metrics],
                output_values=[
                    metrics.audio_seconds or 0 for metrics in output_metrics
                ],
            ),
            bytes=GenerativeMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_values=[metrics.audio_bytes or 0 for metrics in input_metrics],
                output_values=[metrics.audio_bytes or 0 for metrics in output_metrics],
            ),
        )


class GenerativeMetrics(StandardBaseDict):
    """Comprehensive metrics for generative AI benchmarks."""

    # Request stats
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
    output_tokens_wo_first_per_iteration: StatusDistributionSummary = Field(
        description="Distribution of output tokens (without first) generated per streaming iteration"
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of output token generation rates"
    )
    output_tokens_per_iteration: StatusDistributionSummary = Field(
        description="Distribution of output tokens generated per streaming iteration"
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of total token throughput including prompt and output"
    )

    # Domain specific stats
    text: GenerativeTextMetricsSummary = Field(description="")
    image: GenerativeImageMetricsSummary = Field(description="")
    video: GenerativeVideoMetricsSummary = Field(description="")
    audio: GenerativeAudioMetricsSummary = Field(description="")

    @classmethod
    def update_estimate(
        cls,
        state: EstimatedBenchmarkState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        benchmark_start_time = scheduler_state.start_time
        request_start_time = (
            request_info.timings.request_start or request_info.timings.resolve_start
        )
        request_end_time = (
            request_info.timings.request_end or request_info.timings.resolve_end
        )
        event_occurence_time = (
            request_info.timings.queued
            if request_info.status == "queued"
            else (
                request_info.timings.dequeued
                if request_info.status == "pending"
                else request_start_time
                if request_info.status == "in_progress"
                else request_end_time
            )
        )
        benchmark_duration = (
            event_occurence_time - benchmark_start_time
            if event_occurence_time
            else None
        )
        request_duration = (
            request_end_time - request_start_time if request_end_time else None
        )

        # Always track concurrency
        if event_occurence_time is not None:
            state.add_time_averaged_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key="concurrency_requests",
                value=scheduler_state.processing_requests,
                recorded_time=event_occurence_time,
            )

        if request_info.status not in {"completed", "errored", "cancelled"}:
            return

        state.set_metric(
            group=EstimatedBenchmarkState.benchmark_metrics_group,
            key="updated",
            value=True,
        )

        for prefix in (request_info.status, "total"):
            requests_count = (
                scheduler_state.successful_requests
                if prefix == "completed"
                else scheduler_state.errored_requests
                if prefix == "errored"
                else scheduler_state.cancelled_requests
                if prefix == "cancelled"
                else scheduler_state.processed_requests
            )
            input_tokens = (
                (response.input_metrics.total_tokens if response else None)
                or request.input_metrics.total_tokens
                or 0
            )
            output_tokens = (
                (response.output_metrics.total_tokens if response else None)
                or request.output_metrics.total_tokens
                or 0
            )

            # Request distribution stats
            state.set_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_requests",
                value=requests_count,
            )
            state.set_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_requests_per_second",
                value=(
                    requests_count / benchmark_duration if benchmark_duration else None
                ),
            )
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_request_latency",
                value=request_duration,
            )
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_request_streaming_iterations",
                value=request_info.timings.iterations or 0,
            )

            # Token iteration stats
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key="output_tokens_iterations",
                value=output_tokens,
                count=request_info.timings.iterations or 1,
            )
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key="output_tokens_wo_first_iterations",
                value=output_tokens - 1 if output_tokens > 1 else 0,
                count=request_info.timings.iterations or 1,
            )

            # Token metrics stats
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_time_to_first_token",
                value=request_info.timings.first_iteration,
                start_val=request_start_time,
            )
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_inter_token_latency",
                value=request_info.timings.last_iteration,
                start_val=request_info.timings.first_iteration,
                count=(output_tokens or 1) - 1,
            )
            state.add_avg_metric(
                group=EstimatedBenchmarkState.benchmark_metrics_group,
                key=f"{prefix}_time_per_output_token",
                value=request_duration,
                count=output_tokens or 0,
            )

            # Input/output throughput stats
            if event_occurence_time is not None:
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="input_tokens",
                    value=input_tokens,
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="output_tokens",
                    value=output_tokens,
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="total_tokens",
                    value=input_tokens + output_tokens,
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="input_text_tokens",
                    value=(
                        (response.input_metrics.text_tokens if response else None)
                        or request.input_metrics.text_tokens
                        or 0
                    ),
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="input_images",
                    value=(
                        (response.input_metrics.image_count if response else None)
                        or request.input_metrics.image_count
                        or 0
                    ),
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="input_video_frames",
                    value=(
                        (response.input_metrics.video_frames if response else None)
                        or request.input_metrics.video_frames
                        or 0
                    ),
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )
                state.add_avg_rate_metric(
                    group=EstimatedBenchmarkState.benchmark_metrics_group,
                    key="input_audio_seconds",
                    value=request.input_metrics.audio_seconds or 0,
                    start_time=benchmark_start_time,
                    end_time=event_occurence_time,
                )

    @classmethod
    def compile(
        cls,
        completed: list[GenerativeRequestStats],
        errored: list[GenerativeRequestStats],
        incomplete: list[GenerativeRequestStats],
    ) -> GenerativeMetrics:
        requests = completed + errored + incomplete
        request_types = cast(
            "list[Literal['successful', 'error', 'incomplete']]",
            ["successful"] * len(completed)
            + ["error"] * len(errored)
            + ["incomplete"] * len(incomplete),
        )
        request_times = [
            (
                req.info.timings.request_start or req.info.timings.resolve_start or 0,
                req.info.timings.request_end or req.info.timings.resolve_end or 0,
            )
            for req in requests
        ]
        input_metrics = [req.input_metrics for req in requests]
        output_metrics = [req.output_metrics for req in requests]

        return GenerativeMetrics(
            # Request stats
            requests_per_second=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="rate",
            ),
            request_concurrency=StatusDistributionSummary.from_request_times(
                request_types=request_types,
                requests=request_times,
                distribution_type="concurrency",
            ),
            request_latency=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.request_latency or 0.0 for req in requests],
            ),
            request_streaming_iterations_count=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[float(req.info.timings.iterations or 0) for req in requests],
            ),
            # General token stats
            prompt_token_count=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[float(req.prompt_tokens or 0) for req in requests],
            ),
            output_token_count=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[float(req.output_tokens or 0) for req in requests],
            ),
            total_token_count=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[float(req.total_tokens or 0) for req in requests],
            ),
            time_to_first_token_ms=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.time_to_first_token_ms or 0.0 for req in requests],
            ),
            time_per_output_token_ms=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.time_per_output_token_ms or 0.0 for req in requests],
            ),
            inter_token_latency_ms=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.inter_token_latency_ms or 0.0 for req in requests],
            ),
            output_tokens_wo_first_per_iteration=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[
                    max(0.0, (req.output_metrics.total_tokens or 1.0) - 1.0)
                    for req in requests
                ],
                weights=[req.info.timings.iterations or 1 for req in requests],
            ),
            output_tokens_per_second=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.output_tokens_per_second or 0.0 for req in requests],
            ),
            output_tokens_per_iteration=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.output_tokens_per_iteration or 0.0 for req in requests],
                weights=[req.info.timings.iterations or 1 for req in requests],
            ),
            tokens_per_second=StatusDistributionSummary.from_values(
                value_types=request_types,
                values=[req.tokens_per_second or 0.0 for req in requests],
            ),
            # Domain-specific stats
            text=GenerativeTextMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_metrics=input_metrics,
                output_metrics=output_metrics,
            ),
            image=GenerativeImageMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_metrics=input_metrics,
                output_metrics=output_metrics,
            ),
            video=GenerativeVideoMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_metrics=input_metrics,
                output_metrics=output_metrics,
            ),
            audio=GenerativeAudioMetricsSummary.compile(
                request_types=request_types,
                request_times=request_times,
                input_metrics=input_metrics,
                output_metrics=output_metrics,
            ),
        )


class SchedulerDict(StandardBaseDict):
    """Scheduler configuration and execution state dictionary."""

    strategy: SchedulingStrategy
    constraints: dict[str, dict[str, Any]]
    state: SchedulerState


class BenchmarkerDict(StandardBaseDict):
    """Benchmarker configuration and component settings dictionary."""

    args: BenchmarkArgs
    profile: Profile
    requests: dict[str, Any]
    backend: dict[str, Any]
    environment: dict[str, Any]


class GenerativeBenchmark(Benchmark, StandardBaseDict):
    """Complete generative AI benchmark results with specialized metrics."""

    group_name: ClassVar[Literal["generative_benchmark"]] = "generative_benchmark"

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this benchmark execution",
    )
    run_id: str = Field(
        description="Identifier for the benchmarker run containing this benchmark"
    )
    run_index: int = Field(
        description="Sequential index of this benchmark within the benchmarker run"
    )
    scheduler: SchedulerDict = Field(
        description="Scheduler configuration and execution state"
    )
    benchmarker: BenchmarkerDict = Field(
        description="Benchmarker configuration and component settings"
    )
    run_stats: BenchmarkSchedulerStats = Field(
        description="Scheduler timing and performance statistics"
    )
    start_time: float = Field(
        default=-1.0, description="Unix timestamp when the first request was initiated"
    )
    end_time: float = Field(
        default=-1.0, description="Unix timestamp when the last request completed"
    )

    def get_run_metrics_sample(
        self,
    ) -> dict[Literal["start_time", "end_time", "duration"], float]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def get_request_metrics_sample(
        self,
    ) -> dict[
        Literal[
            "request_count",
            "request_latency",
            "request_throughput",
            "request_concurrency",
        ],
        float,
    ]:
        return {
            "request_count": self.request_totals.successful,
            "request_latency": self.metrics.request_latency.successful.mean,
            "request_throughput": self.metrics.requests_per_second.successful.mean,
            "request_concurrency": self.metrics.request_concurrency.successful.mean,
        }

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        Benchmark execution duration in seconds.

        :return: Time elapsed from first request start to last request completion.
        """
        return self.end_time - self.start_time

    metrics: GenerativeMetrics = Field(
        description="Performance metrics and statistical distributions"
    )
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    requests: StatusBreakdown[
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        None,
    ] = Field(
        description="Request details grouped by status: successful, incomplete, errored"
    )

    @classmethod
    def update_estimate(
        cls,
        args: BenchmarkArgs,
        state: EstimatedBenchmarkState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        if (
            request_info.status == "cancelled"
            and request_info.timings.resolve_start is None
        ):
            # Cancelled requests that never started should be ignored
            return

        # Update child metric groups
        BenchmarkSchedulerStats.update_estimate(state, request_info)
        GenerativeMetrics.update_estimate(
            state, response, request, request_info, scheduler_state
        )

        # Store requests and sampling info, update counts
        if "requests_completed" not in state:
            state["requests_completed"] = []
            state["samples_completed"] = []
            state["requests_errored"] = []
            state["samples_errored"] = []
            state["requests_incomplete"] = []
            state["samples_incomplete"] = []
        in_warmup = state.set_metric(
            group=EstimatedBenchmarkState.benchmark_state_group,
            key="in_warmup",
            value=args.is_in_warmup(request_info, scheduler_state),
        )
        in_cooldown = state.set_metric(
            group=EstimatedBenchmarkState.benchmark_state_group,
            key="in_cooldown",
            value=args.is_in_cooldown(request_info, scheduler_state),
        )
        state[f"{EstimatedBenchmarkState.benchmark_state_group}_status"] = (
            "in_cooldown"
            if in_cooldown
            else "in_warmup"
            if in_warmup
            else "in_progress"
        )

        if (
            request_info.status not in {"completed", "errored", "cancelled"}
            or in_warmup
            or in_cooldown
        ):
            # Must be fully resolved to be added
            return

        state.set_metric(
            group=EstimatedBenchmarkState.benchmark_state_group,
            key="updated",
            value=True,
        )

        if response is None:
            response = GenerationResponse(
                request_id=request.request_id, request_args=str(request.arguments)
            )

        stats = response.compile_stats(
            request, request_info, args.prefer_response_metrics
        )

        # Determine status and get corresponding lists
        if request_info.status == "completed":
            requests_list = state["requests_completed"]
            samples_list = state["samples_completed"]
        elif request_info.status == "errored":
            requests_list = state["requests_errored"]
            samples_list = state["samples_errored"]
        else:  # cancelled (incomplete)
            requests_list = state["requests_incomplete"]
            samples_list = state["samples_incomplete"]

        # Add to requests list
        requests_list.append(stats)
        current_index = len(requests_list) - 1

        # Handle request sampling logic
        if args.sample_requests is None:
            # No sampling, add index to samples list
            samples_list.append(current_index)
        elif args.sample_requests > 0 and len(samples_list) < args.sample_requests:
            # Space in samples list, add index
            samples_list.append(current_index)
        elif (
            args.sample_requests > 0
            and (replace_index := random.randrange(len(requests_list)))
            < args.sample_requests
        ):
            # No space, adding based on reservoir sampling
            samples_list[replace_index] = current_index
        # Sampling set to 0, don't keep any requests

    @classmethod
    def compile(
        cls,
        args: BenchmarkArgs,
        estimated_state: EstimatedBenchmarkState,
        scheduler_state: SchedulerState,
        profile: Profile,
        requests: Iterable,
        backend: BackendInterface,
        environment: Environment,
        strategy: SchedulingStrategy,
        constraints: dict[str, dict[str, Any]],
    ) -> GenerativeBenchmark:
        return GenerativeBenchmark(
            run_id=args.run_id,
            run_index=args.run_index,
            scheduler=SchedulerDict(
                strategy=strategy,
                constraints={
                    key: InfoMixin.extract_from_obj(val)
                    for key, val in constraints.items()
                },
                state=scheduler_state,
            ),
            benchmarker=BenchmarkerDict(
                args=args,
                profile=profile,
                requests=InfoMixin.extract_from_obj(requests),
                backend=backend.info,
                environment=environment.info,
            ),
            run_stats=BenchmarkSchedulerStats.compile(estimated_state, scheduler_state),
            start_time=scheduler_state.start_time or -1.0,
            end_time=scheduler_state.end_time or -1.0,
            metrics=GenerativeMetrics.compile(
                completed=estimated_state.get("requests_completed", []),
                errored=estimated_state.get("requests_errored", []),
                incomplete=estimated_state.get("requests_incomplete", []),
            ),
            request_totals=StatusBreakdown[int, int, int, int](
                successful=len(estimated_state.get("requests_completed", [])),
                incomplete=len(estimated_state.get("requests_incomplete", [])),
                errored=len(estimated_state.get("requests_errored", [])),
                total=(
                    len(estimated_state.get("requests_completed", []))
                    + len(estimated_state.get("requests_incomplete", []))
                    + len(estimated_state.get("requests_errored", []))
                ),
            ),
            requests=StatusBreakdown[
                list[GenerativeRequestStats],
                list[GenerativeRequestStats],
                list[GenerativeRequestStats],
                None,
            ](
                successful=estimated_state.get("requests_completed", []),
                incomplete=estimated_state.get("requests_incomplete", []),
                errored=estimated_state.get("requests_errored", []),
                total=None,
            ),
        )


class GenerativeBenchmarksReport(StandardBaseModel):
    """Container for multiple benchmark results with load/save functionality."""

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"

    @staticmethod
    def load_file(
        path: str | Path, type_: Literal["json", "yaml"] | None = None
    ) -> GenerativeBenchmarksReport:
        """
        Load a report from a file.

        :param path: The path to load the report from.
        :param type_: File type override, auto-detected from extension if None.
        :return: The loaded report.
        :raises ValueError: If file type is unsupported.
        """
        path = Path(path) if not isinstance(path, Path) else path

        if path.is_dir():
            path = path / GenerativeBenchmarksReport.DEFAULT_FILE

        path.parent.mkdir(parents=True, exist_ok=True)
        path_suffix = path.suffix.lower()[1:]

        with path.open("r") as file:
            if (type_ or path_suffix) == "json":
                model_dict = json.loads(file.read())
            elif (type_ or path_suffix) in ["yaml", "yml"]:
                model_dict = yaml.safe_load(file)
            else:
                raise ValueError(f"Unsupported file type: {type_} for {path}.")

        return GenerativeBenchmarksReport.model_validate(model_dict)

    benchmarks: list[GenerativeBenchmark] = Field(
        description="The list of completed benchmarks contained within the report.",
        default_factory=list,
    )

    def save_file(
        self, path: str | Path | None, type_: Literal["json", "yaml"] | None = None
    ) -> Path:
        """
        Save the report to a file.

        :param path: The path to save the report to.
        :param type_: File type override, auto-detected from extension if None.
        :return: The path to the saved report.
        :raises ValueError: If file type is unsupported.
        """
        if path is None:
            path = Path.cwd()
        elif not isinstance(path, Path):
            path = Path(path)

        if path.is_dir():
            path = path / GenerativeBenchmarksReport.DEFAULT_FILE

        path.parent.mkdir(parents=True, exist_ok=True)
        path_suffix = path.suffix.lower()[1:]
        model_dict = self.model_dump()

        if (type_ or path_suffix) == "json":
            save_str = json.dumps(model_dict)
        elif (type_ or path_suffix) in ["yaml", "yml"]:
            save_str = yaml.dump(model_dict)
        else:
            raise ValueError(f"Unsupported file type: {type_} for {path}.")

        with path.open("w") as file:
            file.write(save_str)

        return path
