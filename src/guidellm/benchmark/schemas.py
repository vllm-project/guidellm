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

import inspect
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast

import yaml
from pydantic import ConfigDict, Field, computed_field, model_serializer
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.scenarios import get_builtin_scenarios
from guidellm.data import DatasetPreprocessor
from guidellm.scheduler import (
    BackendInterface,
    Environment,
    SchedulerState,
    SchedulingStrategy,
    StrategyType,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    UsageMetrics,
)
from guidellm.utils import (
    InfoMixin,
    StandardBaseDict,
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)

__all__ = [
    "Benchmark",
    "BenchmarkGenerativeTextArgs",
    "BenchmarkSchedulerStats",
    "BenchmarkT",
    "BenchmarkerArgs",
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
    """
    Accumulator for real-time benchmark metrics during scheduler execution.

    Tracks incremental metrics, running averages, and time-based statistics as
    requests are processed. Maintains grouped metrics for benchmark state,
    benchmark-level metrics, and scheduler-level metrics with support for
    average, rate, and time-averaged metric calculations.

    :cvar benchmark_state_group: Metric group key for benchmark state tracking
    :cvar benchmark_metrics_group: Metric group key for benchmark-level metrics
    :cvar scheduler_state_group: Metric group key for scheduler-level metrics
    """

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
        """
        Retrieve a grouped metric value by group and key.

        :param group: Metric group identifier
        :param key: Metric key within the group
        :param default: Value returned if metric doesn't exist
        :return: The metric value or default if not found
        """
        return self.get(f"{group}_{key}", default)

    def set_metric(
        self,
        group: str,
        key: str,
        value: bool | int | float | None,
        start_val: bool | int | float | None = None,
    ) -> bool | int | float | None:
        """
        Set a grouped metric value, optionally adjusting by a starting value.

        :param group: Metric group identifier
        :param key: Metric key within the group
        :param value: Metric value to set
        :param start_val: Optional starting value to subtract from the metric value
        :return: The adjusted metric value or None if value is None
        """
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
        """
        Add a value to a running average metric calculation.

        :param group: Metric group identifier
        :param key: Metric key within the group
        :param value: Value to add to the average
        :param start_val: Optional starting value to subtract before adding
        :param count: Number of observations this value represents
        """
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
        """
        Add a value to a rate-based average metric calculation.

        :param group: Metric group identifier
        :param key: Metric key within the group
        :param value: Value to add to the average
        :param start_val: Optional starting value to subtract before adding
        :param start_time: Start time for rate calculation, defaults to current time
        :param end_time: End time for rate calculation, defaults to current time
        :param numerator_type: Type of numerator for rate calculation
        """
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
        """
        Add a value to a time-weighted average metric calculation.

        :param group: Metric group identifier
        :param key: Metric key within the group
        :param value: Value to add to the time-weighted average
        :param recorded_time: Time of the observation, defaults to current time
        """
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


class BenchmarkerArgs(StandardBaseDict):
    """
    Configuration parameters for benchmark execution and request sampling.

    Defines run identification, request sampling strategy, warmup/cooldown phases,
    and metric preferences for benchmark executions. Provides methods to determine
    whether a request falls within warmup or cooldown periods based on time,
    request count, or percentage-based thresholds.
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the benchmark run",
    )
    run_index: int = Field(default=0, description="Index of the benchmark run")
    sample_requests: int | None = Field(
        default=20,
        description=(
            "Number of requests to sample and keep in the final benchmark for metrics"
        ),
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
        """
        Check if a request is in the warmup phase.

        :param request_info: Information about the current request
        :param scheduler_state: Current state of the scheduler
        :return: True if the request is in warmup phase, False otherwise
        """
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
        """
        Check if a request is in the cooldown phase.

        :param request_info: Information about the current request
        :param scheduler_state: Current state of the scheduler
        :return: True if the request is in cooldown phase, False otherwise
        """
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
    """
    Abstract base interface for benchmark result implementations.

    Defines the contract for benchmark classes to provide run metrics sampling,
    request metrics sampling, real-time estimate updates, and final compilation
    of benchmark results from scheduler execution data.
    """

    @abstractmethod
    def get_run_metrics_sample(
        self,
    ) -> dict[Literal["start_time", "end_time", "duration"], float]:
        """
        Get a sample of run-level timing metrics.

        :return: Dictionary containing start_time, end_time, and duration metrics
        """
        ...

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
    ]:
        """
        Get a sample of request-level performance metrics.

        :return: Dictionary containing request count, latency, throughput, and
            concurrency metrics
        """
        ...

    @classmethod
    @abstractmethod
    def update_estimate(
        cls,
        args: BenchmarkerArgs,
        state: EstimatedBenchmarkState,
        response: Any,
        request: Any,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Update real-time benchmark estimates with new request data.

        :param args: Benchmark configuration arguments
        :param state: Current estimated benchmark state to update
        :param response: Response received from the backend
        :param request: Original request sent to the backend
        :param request_info: Metadata about the request execution
        :param scheduler_state: Current state of the scheduler
        """
        ...

    @classmethod
    @abstractmethod
    def compile(
        cls,
        args: BenchmarkerArgs,
        estimated_state: EstimatedBenchmarkState,
        scheduler_state: SchedulerState,
        profile: Profile,
        requests: Iterable,
        backend: BackendInterface,
        environment: Environment,
        strategy: SchedulingStrategy,
        constraints: dict[str, dict[str, Any]],
    ) -> Any:
        """
        Compile final benchmark results from accumulated state.

        :param args: Benchmark configuration arguments
        :param estimated_state: Accumulated benchmark state from execution
        :param scheduler_state: Final state of the scheduler
        :param profile: Benchmark profile configuration
        :param requests: Collection of requests executed
        :param backend: Backend interface used for execution
        :param environment: Execution environment configuration
        :param strategy: Scheduling strategy used
        :param constraints: Execution constraints applied
        :return: Compiled benchmark results instance
        """
        ...


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
        """
        Update estimated scheduler statistics with request timing information.

        :param state: Current estimated benchmark state to update
        :param request_info: Metadata about the request execution with timing data
        """
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
        """
        Compile final scheduler statistics from accumulated state.

        :param estimated_state: Accumulated benchmark state with scheduler metrics
        :param scheduler_state: Final state of the scheduler
        :return: Compiled scheduler statistics instance
        """
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
    """
    Statistical summaries for input, output, and total metrics.

    Provides distribution summaries across successful, incomplete, and errored
    requests for absolute values, per-second rates, and concurrency levels.
    """

    input: StatusDistributionSummary = Field(
        description="Distribution of input metric values"
    )
    input_per_second: StatusDistributionSummary = Field(
        description="Distribution of input metric rates per second"
    )
    input_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent input metric values"
    )

    output: StatusDistributionSummary = Field(
        description="Distribution of output metric values"
    )
    output_per_second: StatusDistributionSummary = Field(
        description="Distribution of output metric rates per second"
    )
    output_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent output metric values"
    )

    total: StatusDistributionSummary = Field(
        description="Distribution of total metric values (input + output)"
    )
    total_per_second: StatusDistributionSummary = Field(
        description="Distribution of total metric rates per second"
    )
    total_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent total metric values"
    )

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_values: list[int | float],
        output_values: list[int | float],
    ) -> GenerativeMetricsSummary:
        """
        Compile generative metrics summary from request data.

        :param request_types: Status types for each request
        :param request_times: Start and end times for each request
        :param input_values: Input metric values for each request
        :param output_values: Output metric values for each request
        :return: Compiled generative metrics summary
        """
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
    """
    Text-specific metric summaries for generative benchmarks.

    Tracks token, word, and character-level metrics across input, output, and
    total usage for text generation workloads.
    """

    tokens: GenerativeMetricsSummary = Field(
        description="Token count metrics and distributions"
    )
    words: GenerativeMetricsSummary = Field(
        description="Word count metrics and distributions"
    )
    characters: GenerativeMetricsSummary = Field(
        description="Character count metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeTextMetricsSummary:
        """
        Compile text metrics summary from request usage data.

        :param request_types: Status types for each request
        :param request_times: Start and end times for each request
        :param input_metrics: Input usage metrics for each request
        :param output_metrics: Output usage metrics for each request
        :return: Compiled text metrics summary
        """
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
    """
    Image-specific metric summaries for generative benchmarks.

    Tracks token, image count, pixel, and byte-level metrics across input, output,
    and total usage for image generation workloads.
    """

    tokens: GenerativeMetricsSummary = Field(
        description="Image token count metrics and distributions"
    )
    images: GenerativeMetricsSummary = Field(
        description="Image count metrics and distributions"
    )
    pixels: GenerativeMetricsSummary = Field(
        description="Pixel count metrics and distributions"
    )
    bytes: GenerativeMetricsSummary = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeImageMetricsSummary:
        """
        Compile image metrics summary from request usage data.

        :param request_types: Status types for each request
        :param request_times: Start and end times for each request
        :param input_metrics: Input usage metrics for each request
        :param output_metrics: Output usage metrics for each request
        :return: Compiled image metrics summary
        """
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
    """
    Video-specific metric summaries for generative benchmarks.

    Tracks token, frame count, duration, and byte-level metrics across input,
    output, and total usage for video generation workloads.
    """

    tokens: GenerativeMetricsSummary = Field(
        description="Video token count metrics and distributions"
    )
    frames: GenerativeMetricsSummary = Field(
        description="Frame count metrics and distributions"
    )
    seconds: GenerativeMetricsSummary = Field(
        description="Duration metrics in seconds and distributions"
    )
    bytes: GenerativeMetricsSummary = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeVideoMetricsSummary:
        """
        Compile video metrics summary from request usage data.

        :param request_types: Status types for each request
        :param request_times: Start and end times for each request
        :param input_metrics: Input usage metrics for each request
        :param output_metrics: Output usage metrics for each request
        :return: Compiled video metrics summary
        """
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
    """
    Audio-specific metric summaries for generative benchmarks.

    Tracks token, sample count, duration, and byte-level metrics across input,
    output, and total usage for audio generation workloads.
    """

    tokens: GenerativeMetricsSummary = Field(
        description="Audio token count metrics and distributions"
    )
    samples: GenerativeMetricsSummary = Field(
        description="Sample count metrics and distributions"
    )
    seconds: GenerativeMetricsSummary = Field(
        description="Duration metrics in seconds and distributions"
    )
    bytes: GenerativeMetricsSummary = Field(
        description="Byte size metrics and distributions"
    )

    @classmethod
    def compile(
        cls,
        request_types: list[Literal["successful", "incomplete", "error"]],
        request_times: list[tuple[float, float]],
        input_metrics: list[UsageMetrics],
        output_metrics: list[UsageMetrics],
    ) -> GenerativeAudioMetricsSummary:
        """
        Compile audio metrics summary from request usage data.

        :param request_types: Status types for each request
        :param request_times: Start and end times for each request
        :param input_metrics: Input usage metrics for each request
        :param output_metrics: Output usage metrics for each request
        :return: Compiled audio metrics summary
        """
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
        description=(
            "Distribution of output tokens (without first) generated per "
            "streaming iteration"
        )
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
    def update_estimate(
        cls,
        state: EstimatedBenchmarkState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Update real-time generative metrics estimates with new request data.

        :param state: Current estimated benchmark state to update
        :param response: Response received from the backend
        :param request: Original request sent to the backend
        :param request_info: Metadata about the request execution
        :param scheduler_state: Current state of the scheduler
        """
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
            (request_end_time - request_start_time)
            if request_end_time and request_start_time else None
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
        """
        Compile final generative metrics from request statistics.

        :param completed: Successfully completed request statistics
        :param errored: Failed request statistics
        :param incomplete: Incomplete/cancelled request statistics
        :return: Compiled generative metrics with full distributions
        """
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

    strategy: SchedulingStrategy = Field(
        description="Scheduling strategy used for request distribution"
    )
    constraints: dict[str, dict[str, Any]] = Field(
        description="Execution constraints applied during benchmarking"
    )
    state: SchedulerState = Field(
        description="Final state of the scheduler after execution"
    )


class BenchmarkerDict(StandardBaseDict):
    """Benchmarker configuration and component settings dictionary."""

    profile: Profile = Field(description="Benchmark profile configuration")
    requests: dict[str, Any] = Field(
        description="Request configuration and dataset information"
    )
    backend: dict[str, Any] = Field(
        description="Backend configuration and connection details"
    )
    environment: dict[str, Any] = Field(
        description="Execution environment configuration"
    )


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
        args: BenchmarkerArgs,
        state: EstimatedBenchmarkState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: RequestInfo,
        scheduler_state: SchedulerState,
    ):
        """
        Update generative benchmark estimates with new request data.

        Handles warmup/cooldown filtering, request sampling via reservoir sampling,
        and delegates metric updates to child metric classes.

        :param args: Benchmark configuration arguments
        :param state: Current estimated benchmark state to update
        :param response: Response received from the backend
        :param request: Original request sent to the backend
        :param request_info: Metadata about the request execution
        :param scheduler_state: Current state of the scheduler
        """
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
        args: BenchmarkerArgs,
        estimated_state: EstimatedBenchmarkState,
        scheduler_state: SchedulerState,
        profile: Profile,
        requests: Iterable,
        backend: BackendInterface,
        environment: Environment,
        strategy: SchedulingStrategy,
        constraints: dict[str, dict[str, Any]],
        data: list[Any],
    ) -> GenerativeBenchmark:
        """
        Compile final generative benchmark from accumulated state.

        :param args: Benchmark configuration arguments
        :param estimated_state: Accumulated benchmark state from execution
        :param scheduler_state: Final state of the scheduler
        :param profile: Benchmark profile configuration
        :param requests: Collection of requests executed
        :param backend: Backend interface used for execution
        :param environment: Execution environment configuration
        :param strategy: Scheduling strategy used
        :param constraints: Execution constraints applied
        :return: Compiled generative benchmark instance
        """
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
                profile=profile,
                requests={"data": data},
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


class BenchmarkGenerativeTextArgs(StandardBaseModel):
    """
    Configuration arguments for generative text benchmark execution.

    Defines all parameters for benchmark setup including target endpoint, data
    sources, backend configuration, processing pipeline, output formatting, and
    execution constraints. Supports loading from scenario files and merging with
    runtime overrides.
    """

    @classmethod
    def create(
        cls, scenario: Path | str | None, **kwargs: dict[str, Any]
    ) -> BenchmarkGenerativeTextArgs:
        """
        Create benchmark args from scenario file and/or keyword arguments.

        :param scenario: Path to scenario file or name of built-in scenario
        :param kwargs: Additional keyword arguments to override scenario values
        :return: Configured benchmark args instance
        :raises ValueError: If scenario is not found or file format is unsupported
        """
        constructor_kwargs = {}

        if scenario is not None:
            if isinstance(scenario, str) and scenario in (
                builtin_scenarios := get_builtin_scenarios()
            ):
                scenario_path = builtin_scenarios[scenario]
            elif Path(scenario).exists() and Path(scenario).is_file():
                scenario_path = Path(scenario)
            else:
                raise ValueError(f"Scenario '{scenario}' not found.")

            with scenario_path.open() as file:
                if scenario_path.suffix == ".json":
                    scenario_data = json.load(file)
                elif scenario_path.suffix in {".yaml", ".yml"}:
                    scenario_data = yaml.safe_load(file)
                else:
                    raise ValueError(
                        f"Unsupported scenario file format: {scenario_path.suffix}"
                    )
            if "args" in scenario_data:
                # loading from a report file
                scenario_data = scenario_data["args"]
            constructor_kwargs.update(scenario_data)

        for key, value in kwargs.items():
            if value != cls.get_default(key):
                constructor_kwargs[key] = value

        return cls.model_validate(constructor_kwargs)

    @classmethod
    def get_default(cls: type[BenchmarkGenerativeTextArgs], field: str) -> Any:
        """
        Get default value for a model field.

        :param field: Name of the field to retrieve default for
        :return: Default value for the specified field
        :raises ValueError: If field is not found in model
        """
        if field not in BenchmarkGenerativeTextArgs.model_fields:
            raise ValueError(
                f"Field '{field}' not found in BenchmarkGenerativeTextArgs"
            )

        field_info = BenchmarkGenerativeTextArgs.model_fields[field]
        factory = field_info.default_factory

        if factory is None:
            return field_info.default

        if len(inspect.signature(factory).parameters) == 0:
            return factory()  # type: ignore[call-arg] # Confirmed correct at runtime by code above
        else:
            return factory({})  # type: ignore[call-arg] # Confirmed correct at runtime by code above



    model_config = ConfigDict(
        extra="ignore",
        use_enum_values=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    # Required
    target: str = Field(description="Target endpoint URL for benchmark execution")
    data: list[Any] = Field(
        description="List of dataset sources or data files",
        default_factory=list,
        min_length=1,
    )
    # Benchmark configuration
    profile: StrategyType | ProfileType | Profile = Field(
        default="sweep", description="Benchmark profile or scheduling strategy type"
    )
    rate: float | list[float] | None = Field(
        default=None, description="Request rate(s) for rate-based scheduling"
    )
    # Backend configuration
    backend: BackendType | Backend = Field(
        default="openai_http", description="Backend type or instance for execution"
    )
    backend_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional backend configuration arguments"
    )
    model: str | None = Field(default=None, description="Model identifier for backend")
    # Data configuration
    processor: str | Path | PreTrainedTokenizerBase | None = Field(
        default=None, description="Tokenizer path, name, or instance for processing"
    )
    processor_args: dict[str, Any] | None = Field(
        default=None, description="Additional tokenizer configuration arguments"
    )
    data_args: list[dict[str, Any]] | None = Field(
        default_factory=list, description="Per-dataset configuration arguments"
    )
    data_samples: int = Field(
        default=-1, description="Number of samples to use from datasets (-1 for all)"
    )
    data_column_mapper: (
        DatasetPreprocessor | dict[str, str] | Literal["generative_column_mapper"]
    ) = Field(
        default="generative_column_mapper",
        description="Column mapping preprocessor for dataset fields",
    )
    data_request_formatter: DatasetPreprocessor | dict[str, str] | str = Field(
        default="chat_completions",
        description="Request formatting preprocessor or template name",
    )
    data_collator: Callable | Literal["generative"] | None = Field(
        default="generative", description="Data collator for batch processing"
    )
    data_sampler: Sampler[int] | Literal["shuffle"] | None = Field(
        default=None, description="Data sampler for request ordering"
    )
    data_num_workers: int | None = Field(
        default=None, description="Number of workers for data loading"
    )
    dataloader_kwargs: dict[str, Any] | None = Field(
        default=None, description="Additional dataloader configuration arguments"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    # Output configuration
    output_path: str | Path | None = Field(
        default_factory=Path.cwd, description="Directory path for output files"
    )
    output_formats: list[str] | dict[str, str | dict[str, Any]] | None = Field(
        default_factory=lambda: ["console", "json"],
        description="Output format names or configuration mappings",
    )
    # Benchmarker configuration
    benchmark_cls: type[GenerativeBenchmark] = Field(
        default=GenerativeBenchmark,
        description="Benchmark class to use for result compilation",
    )
    sample_requests: int | None = Field(
        default=10,
        description="Number of requests to sample for detailed metrics (None for all)",
    )
    warmup: float | None = Field(
        default=None,
        description="Warmup period in seconds, requests, or fraction (0-1)",
    )
    cooldown: float | None = Field(
        default=None,
        description="Cooldown period in seconds, requests, or fraction (0-1)",
    )
    prefer_response_metrics: bool = Field(
        default=True,
        description="Whether to prefer backend response metrics over request metrics",
    )
    # Constraints configuration
    max_seconds: int | float | None = Field(
        default=None, description="Maximum benchmark execution time in seconds"
    )
    max_requests: int | None = Field(
        default=None, description="Maximum number of requests to execute"
    )
    max_errors: int | None = Field(
        default=None, description="Maximum number of errors before stopping"
    )
    max_error_rate: float | None = Field(
        default=None, description="Maximum error rate (0-1) before stopping"
    )
    max_global_error_rate: float | None = Field(
        default=None, description="Maximum global error rate (0-1) before stopping"
    )

    @model_serializer
    def serialize_model(self):
        """
        Custom serialization logic for benchmark args.

        Converts complex types to serializable formats including Profile to type
        string, Backend to type string, and Path objects to strings.

        :return: Dictionary representation suitable for JSON/YAML serialization
        """
        return {
            # target - serialize as is
            "target": self.target,
            "data": [
                item if isinstance(item, str | type(None)) else str(item)
                for item in self.data
            ],  # data - for each item in the list, if not a str or None, save str(item)
            "profile": (
                self.profile.type_
                if isinstance(self.profile, Profile)
                else self.profile
            ),  # profile - if instance of Profile, then save as profile.type_
            "rate": self.rate,
            "backend": (
                self.backend.type_
                if isinstance(self.backend, Backend)
                else self.backend
            ),  # backend - if instance of Backend, then save as backend.type_
            "backend_kwargs": self.backend_kwargs,
            "model": self.model,
            "processor": (
                self.processor
                if isinstance(self.processor, str)
                else str(self.processor)
                if self.processor is not None
                else None
            ),  # processor - if not str, then save as str(processor)
            "processor_args": self.processor_args,
            "data_args": self.data_args,
            "data_samples": self.data_samples,
            "data_column_mapper": (
                self.data_column_mapper
                if isinstance(self.data_column_mapper, dict | str)
                else {}
            ),  # data_column_mapper - if not dict or str, then save as an empty dict
            "data_request_formatter": (
                self.data_request_formatter
                if isinstance(self.data_request_formatter, dict | str)
                else {}
            ),  # data_request_formatter - if not dict or str, then save as empty dict
            "data_collator": (
                self.data_collator if isinstance(self.data_collator, str) else None
            ),  # data_collator - if not str, then save as None
            "data_sampler": (
                self.data_sampler if isinstance(self.data_sampler, str) else None
            ),  # data_sampler - if not str, then save as None
            "data_num_workers": self.data_num_workers,
            "dataloader_kwargs": self.dataloader_kwargs,
            "random_seed": self.random_seed,
            "output_path": (
                str(self.output_path) if self.output_path is not None else None
            ),  # output_path - if not None, then ensure it's a str
            "output_formats": self.output_formats,
            # benchmark_cls - don't save at all (excluded)
            "sample_requests": self.sample_requests,
            "warmup": self.warmup,
            "cooldown": self.cooldown,
            "prefer_response_metrics": self.prefer_response_metrics,
            "max_seconds": self.max_seconds,
            "max_requests": self.max_requests,
            "max_errors": self.max_errors,
            "max_error_rate": self.max_error_rate,
            "max_global_error_rate": self.max_global_error_rate,
        }


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

    args: BenchmarkGenerativeTextArgs = Field(
        description="The benchmark arguments used for all benchmarks in the report."
    )
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
