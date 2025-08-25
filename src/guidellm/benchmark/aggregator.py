"""
Benchmark result aggregation and compilation interfaces.

Provides protocols and implementations for collecting, processing, and compiling
benchmark data from scheduler executions into final metrics and statistics.

Classes:
    Aggregator: Protocol for processing benchmark data updates.
    CompilableAggregator: Protocol for aggregators that can compile final results.
    SchedulerStatsAggregator: Aggregates scheduler timing and performance metrics.
    GenerativeRequestsStatsProgressAggregator: Tracks generation metrics during run.
    GenerativeRequestsAggregator: Compiles complete generative benchmark results.

Functions:
    add_aggregate_metric: Helper for accumulating timing and count metrics.

Type Variables:
    RequestT: Generic request object type.
    ResponseT: Generic response object type.
    RequestTimingsT: Generic request timing object type.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import Field, PrivateAttr

from guidellm.backend import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.benchmark.objects import (
    BenchmarkSchedulerStats,
    GenerativeMetrics,
    GenerativeRequestStats,
)
from guidellm.config import settings
from guidellm.scheduler import (
    MeasuredRequestTimingsT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.utils import (
    InfoMixin,
    PydanticClassRegistryMixin,
    StatusBreakdown,
    StatusDistributionSummary,
    all_defined,
    safe_divide,
    safe_getattr,
)

__all__ = [
    "Aggregator",
    "AggregatorState",
    "CompilableAggregator",
    "GenerativeRequestsAggregator",
    "GenerativeStatsProgressAggregator",
    "InjectExtrasAggregator",
    "SchedulerStatsAggregator",
    "SerializableAggregator",
]


class AggregatorState(dict[str, Any]):
    def add_metric(
        self,
        key: str,
        value: int | float | None,
        start_val: int | float | None = 0.0,
        count: int | None = 1,
        duration: float | None = None,
        duration_div: Literal["total", "avg"] = "total",
        prefix: str | None = None,
    ):
        """
        Add timing or count metrics to aggregation state.
        """
        if prefix:
            self.add_metric(
                key=f"{prefix}_{key}",
                value=value,
                start_val=start_val,
                count=count,
                duration=duration,
                duration_div=duration_div,
            )
            return

        if not all_defined(value, start_val, count):
            return

        delta_val = value - start_val
        self[f"{key}_total"] = self.get(f"{key}_total", 0) + delta_val
        self[f"{key}_count"] = self.get(f"{key}_count", 0) + count
        self[f"{key}_avg"] = safe_divide(
            self.get(f"{key}_total"), self.get(f"{key}_count")
        )

        if all_defined(duration):
            self[f"{key}_duration"] = duration
            self[f"{key}_rate"] = safe_divide(
                self.get(f"{key}_{duration_div}"), duration
            )

    def set_metric(
        self,
        key: str,
        value: int | float | None,
        type_: Literal["total", "count", "avg", "duration", "rate"],
        prefix: str | None = None,
    ):
        if prefix:
            self.set_metric(
                key=f"{prefix}_{key}",
                value=value,
                type_=type_,
                prefix=None,
            )
            return

        self[f"{key}_{type_}"] = value

    def get_metric(
        self,
        key: str,
        type_: Literal["total", "count", "avg", "duration", "rate"],
        default: int | float | None = None,
        prefix: str | None = None,
    ) -> int | float | None:
        if prefix:
            return self.get_metric(
                key=f"{prefix}_{key}",
                type_=type_,
                default=default,
            )

        return self.get(f"{key}_{type_}", default)


@runtime_checkable
class Aggregator(Protocol[ResponseT, RequestT, MeasuredRequestTimingsT]):
    """
    Protocol for processing benchmark data updates during execution.

    Defines the interface for aggregators that collect and process request/response
    data from scheduler executions. Implementations update aggregation state with
    each completed request for eventual compilation into final metrics.
    """

    def __call__(
        self,
        state: AggregatorState,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """


@runtime_checkable
class CompilableAggregator(Protocol[ResponseT, RequestT, MeasuredRequestTimingsT]):
    """
    Protocol for aggregators that compile final results from aggregated state.

    Extends the Aggregator protocol with the ability to transform accumulated
    state into final benchmark results and metrics after execution completes.
    """

    def __call__(
        self,
        state: AggregatorState,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """

    def compile(
        self, state: AggregatorState, scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile aggregated state into final benchmark results.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: Compiled benchmark results and metrics.
        """


class SerializableAggregator(
    PydanticClassRegistryMixin[type["SerializableAggregator"]],
    ABC,
    Generic[ResponseT, RequestT, MeasuredRequestTimingsT],
):
    schema_discriminator: ClassVar[str] = "type_"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[SerializableAggregator]:
        if cls.__name__ == "SerializableAggregator":
            return cls

        return SerializableAggregator

    @classmethod
    @abstractmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Validate and process arguments for constraint creation.

        Must be implemented by subclasses to handle their specific parameter patterns.

        :param args: Positional arguments passed to the constraint
        :param kwargs: Keyword arguments passed to the constraint
        :return: Validated dictionary of parameters for constraint creation
        :raises NotImplementedError: Must be implemented by subclasses
        """
        ...

    @classmethod
    def resolve(
        cls,
        aggregators: dict[
            str,
            Any | dict[str, Any] | Aggregator | CompilableAggregator,
        ],
    ) -> dict[str, Aggregator | CompilableAggregator]:
        """
        Resolve mixed aggregator specifications to callable aggregators.

        :param aggregators: Dictionary mapping aggregator keys to specifications
        :return: Dictionary mapping aggregator keys to callable functions
        :raises ValueError: If any key is not registered in the factory
        """
        resolved = {}

        for key, val in aggregators.items():
            if isinstance(val, (Aggregator, CompilableAggregator)):
                resolved[key] = val
            else:
                aggregator_class = cls.get_registered_object(key)
                kwargs = aggregator_class.validated_kwargs(**val)
                resolved[key] = aggregator_class(**kwargs)

        return resolved

    type_: Literal["aggregator"] = Field(default="aggregator", description="")

    @abstractmethod
    def __call__(
        self,
        state: AggregatorState,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Process a completed request and update aggregation state.

        :param agg_state: Current aggregation state to update in-place.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Optional intermediate updates for progress reporting.
        """

    @abstractmethod
    def compile(
        self, state: AggregatorState, scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile aggregated state into final benchmark results.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: Compiled benchmark results and metrics.
        """


@SerializableAggregator.register("inject_extras")
class InjectExtrasAggregator(
    SerializableAggregator[ResponseT, RequestT, MeasuredRequestTimingsT], InfoMixin
):
    """
    Aggregator for injecting extra metadata into the output.
    """

    @classmethod
    def validated_kwargs(cls, extras: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {"extras": extras}

    type_: Literal["inject_extras"] = Field(default="inject_extras")
    extras: dict[str, Any] | None = Field(default_factory=None)

    def __call__(
        self,
        state: AggregatorState,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Inject extra metadata into the aggregation state.

        :param agg_state: Current aggregation state to update.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state with injected extras.
        """
        return None

    def compile(
        self, state: AggregatorState, scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        return {"extras": self.extras} if self.extras else {}


@SerializableAggregator.register("scheduler_stats")
class SchedulerStatsAggregator(
    SerializableAggregator[ResponseT, RequestT, MeasuredRequestTimingsT], InfoMixin
):
    """
    Aggregates scheduler timing and performance metrics.

    Collects timing data for various scheduler phases including queuing,
    resolution, and processing delays to generate performance statistics.
    """

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        return {}

    type_: Literal["scheduler_stats"] = Field(default="scheduler_stats")

    def __call__(
        self,
        state: AggregatorState,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo[MeasuredRequestTimingsT],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Aggregate scheduler timing metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Response generated for the request, if successful.
        :param request: The processed request object.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for intermediate reporting.
        """
        if request_info.status not in ("completed", "errored", "cancelled"):
            # Only compile scheduler stats for processed requests
            return None

        state["updated_scheduler_stats"] = True
        state.add_metric(
            key="queued_time",
            value=request_info.scheduler_timings.dequeued,
            start_val=request_info.scheduler_timings.queued,
        )
        state.add_metric(
            key="worker_resolve_start_delay",
            value=request_info.scheduler_timings.resolve_start,
            start_val=request_info.scheduler_timings.scheduled_at,
        )
        state.add_metric(
            key="worker_resolve_time",
            value=request_info.scheduler_timings.resolve_end,
            start_val=request_info.scheduler_timings.resolve_start,
        )
        state.add_metric(
            key="worker_resolve_end_delay",
            value=request_info.scheduler_timings.resolve_end,
            start_val=safe_getattr(request_info.request_timings, "request_end"),
        )
        state.add_metric(
            key="finalized_delay",
            value=request_info.scheduler_timings.finalized,
            start_val=request_info.scheduler_timings.resolve_end,
        )
        state.add_metric(
            key="worker_targeted_start_delay",
            value=request_info.scheduler_timings.resolve_start,
            start_val=request_info.scheduler_timings.targeted_start,
        )
        state.add_metric(
            key="request_start_delay",
            value=request_info.scheduler_timings.resolve_start,
            start_val=safe_getattr(request_info.request_timings, "request_start"),
        )
        state.add_metric(
            key="request_time",
            value=safe_getattr(request_info.request_timings, "request_end"),
            start_val=safe_getattr(request_info.request_timings, "request_start"),
        )
        state.add_metric(
            key="request_targeted_start_delay",
            value=safe_getattr(request_info.request_timings, "request_start"),
            start_val=request_info.scheduler_timings.targeted_start,
        )

        return state

    def compile(
        self, state: AggregatorState, scheduler_state: SchedulerState
    ) -> dict[Literal["scheduler_stats"], BenchmarkSchedulerStats]:
        """
        Compile scheduler timing metrics into benchmark statistics.

        :param agg_state: Accumulated timing data and counts.
        :param scheduler_state: Final scheduler execution state.
        :return: Dictionary containing compiled scheduler statistics.
        """
        return {
            "run_stats": BenchmarkSchedulerStats(
                start_time=scheduler_state.start_time,
                end_time=scheduler_state.end_time,
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
                queued_time_avg=state.get_metric(
                    key="queued_time", type_="avg", default=0.0
                ),
                worker_resolve_start_delay_avg=state.get_metric(
                    key="worker_resolve_start_delay", type_="avg", default=0.0
                ),
                worker_resolve_time_avg=state.get_metric(
                    key="worker_resolve_time", type_="avg", default=0.0
                ),
                worker_resolve_end_delay_avg=state.get_metric(
                    key="worker_resolve_end_delay", type_="avg", default=0.0
                ),
                finalized_delay_avg=state.get_metric(
                    key="finalized_delay", type_="avg", default=0.0
                ),
                worker_targeted_start_delay_avg=state.get_metric(
                    key="worker_targeted_start_delay", type_="avg", default=0.0
                ),
                request_start_delay_avg=state.get_metric(
                    key="request_start_delay", type_="avg", default=0.0
                ),
                request_time_avg=state.get_metric(
                    key="request_time", type_="avg", default=0.0
                ),
                request_targeted_start_delay_avg=state.get_metric(
                    key="request_targeted_start_delay", type_="avg", default=0.0
                ),
            ),
        }


@SerializableAggregator.register("generative_stats_progress")
class GenerativeStatsProgressAggregator(
    SerializableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ]
):
    """
    Tracks generative model metrics during benchmark execution.

    Aggregates token-level metrics including time to first token, inter-token
    latency, and token counts for real-time progress monitoring.
    """

    @classmethod
    def validated_kwargs(cls, *args, **kwargs) -> dict[str, Any]:
        return {}

    type_: Literal["generative_stats_progress"] = Field(
        default="generative_stats_progress"
    )

    def __call__(
        self,
        state: AggregatorState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Aggregate generative model metrics for a completed request.

        :param agg_state: Current aggregation state to update.
        :param response: Generation response with token and timing data.
        :param request: The processed generation request.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: Updated aggregation state for progress reporting.
        """
        if request_info.status not in {"completed", "errored", "cancelled"}:
            # Only compile progress stats for processed requests
            return None

        state["updated_generative_stats"] = True
        start_time = scheduler_state.start_time
        end_time = (
            safe_getattr(request_info.request_timings, "request_end")
            or request_info.scheduler_timings.resolve_end
        )
        duration = end_time - start_time if end_time else None

        for prefix in (request_info.status, None):
            requests_count = (
                scheduler_state.processed_requests
                if prefix is None
                else scheduler_state.successful_requests
                if request_info.status == "completed"
                else scheduler_state.cancelled_requests
                if request_info.status == "cancelled"
                else scheduler_state.errored_requests
            )

            # Requests per Second
            if duration is not None:
                state.set_metric(
                    key="requests",
                    value=safe_divide(requests_count, duration),
                    type_="rate",
                    prefix=prefix,
                )

            # Request Concurrency
            state.set_metric(
                key="requests",
                value=scheduler_state.processing_requests,
                type_="avg",
                prefix=prefix,
            )

            # Request Latency
            state.add_metric(
                key="request_latency",
                value=safe_getattr(request_info.request_timings, "request_end"),
                start_val=safe_getattr(request_info.request_timings, "request_start"),
                prefix=prefix,
            )

            # Time to First Token
            state.add_metric(
                key="time_to_first_token",
                value=safe_getattr(request_info.request_timings, "first_iteration"),
                start_val=safe_getattr(request_info.request_timings, "request_start"),
                prefix=prefix,
            )

            output_tokens = safe_getattr(response, "output_tokens")
            prompt_tokens = safe_getattr(response, "prompt_tokens")

            # Inter Token Latency
            state.add_metric(
                key="inter_token_latency",
                value=safe_getattr(request_info.request_timings, "last_iteration"),
                start_val=safe_getattr(request_info.request_timings, "first_iteration"),
                count=(
                    output_tokens - 1 if output_tokens and output_tokens > 1 else None
                ),
                prefix=prefix,
            )

            # Time per Output Token
            state.add_metric(
                key="time_per_output_token",
                value=safe_getattr(request_info.request_timings, "request_start"),
                start_val=safe_getattr(request_info.request_timings, "last_iteration"),
                count=output_tokens,
                prefix=prefix,
            )

            # Prompt Tokens
            state.add_metric(
                key="prompt_tokens",
                value=prompt_tokens,
                duration=duration,
                prefix=prefix,
            )

            # Output Tokens
            state.add_metric(
                key="output_tokens",
                value=output_tokens,
                duration=duration,
                prefix=prefix,
            )

            # Total Tokens
            state.add_metric(
                key="total_tokens",
                value=(
                    prompt_tokens + output_tokens
                    if all_defined(prompt_tokens, output_tokens)
                    else prompt_tokens
                    if all_defined(prompt_tokens)
                    else output_tokens
                    if all_defined(output_tokens)
                    else None
                ),
                duration=duration,
                prefix=prefix,
            )

        return state

    def compile(
        self, state: AggregatorState, scheduler_state: SchedulerState
    ) -> dict[str, Any]:
        """
        Compile progress metrics into final results.

        GenerativeStatsProgressAggregator is primarily for progress tracking,
        so compilation returns the aggregated state as-is.

        :param agg_state: The accumulated aggregation state.
        :param scheduler_state: Final scheduler execution state.
        :return: The aggregated state as final results.
        """
        return {}


@SerializableAggregator.register("generative_requests")
class GenerativeRequestsAggregator(
    SerializableAggregator[
        GenerationResponse, GenerationRequest, GenerationRequestTimings
    ],
):
    """
    Compiles complete generative benchmark results with warmup/cooldown filtering.

    Aggregates request data during execution and compiles comprehensive metrics
    including timing distributions, token statistics, and throughput measurements.
    Supports filtering warmup and cooldown periods from final results.
    """

    @classmethod
    def validated_kwargs(
        cls,
        request_samples: int | None = 20,
        warmup: int | float | None = None,
        cooldown: int | float | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "request_samples": request_samples,
            "warmup": warmup,
            "cooldown": cooldown,
        }

    type_: Literal["generative_requests"] = Field(default="generative_requests")

    request_samples: int | None = Field(default=20, description="")
    warmup: int | float | None = Field(
        default=None,
        description="Number of warmup requests to ignore at benchmark start",
    )
    cooldown: int | float | None = Field(
        default=None,
        description="Number of cooldown requests to ignore at benchmark end",
    )
    _in_cooldown: bool = PrivateAttr(False)
    _in_warmup: bool = PrivateAttr(False)

    def __call__(
        self,
        state: AggregatorState,
        response: GenerationResponse | None,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> dict[str, Any] | None:
        """
        Collect completed requests for final compilation.

        Filters requests based on warmup/cooldown settings and categorizes by
        completion status for comprehensive benchmark analysis.

        :param agg_state: Current aggregation state to update.
        :param response: Generation response data.
        :param request: The processed generation request.
        :param request_info: Scheduling metadata and timing information.
        :param scheduler_state: Current scheduler execution state.
        :return: None, as this aggregator only collects for final compilation.
        """
        # Skip invalid requests
        if request_info.status not in {"completed", "canceled", "errored"} or (
            request_info.status == "canceled"
            and safe_getattr(request_info.scheduler_timings, "resolve_start") is None
            # Canceled requests that never started should not be kept
        ):
            return None

        status = {
            "updated_generative_requests": True,
            "requests_in_warmup": False,
            "requests_in_cooldown": False,
        }

        if self._is_in_warmup(request_info, scheduler_state):
            status["requests_in_warmup"] = True
            return status

        if self._is_in_cooldown(request_info, scheduler_state):
            status["requests_in_cooldown"] = True
            return status

        if "completed" not in state:
            state["completed"] = []
            state["errored"] = []
            state["incomplete"] = []

        # Categorize request by status
        if request_info.status == "completed":
            state["completed"].append((response, request, request_info))
        elif request_info.status == "canceled":
            state["incomplete"].append((response, request, request_info))
        else:
            state["errored"].append((response, request, request_info))

        return status

    def compile(
        self,
        state: AggregatorState,
        scheduler_state: SchedulerState,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Compile aggregated requests into comprehensive benchmark results.

        Transforms collected request data into detailed metrics including timing
        distributions, token statistics, throughput measurements, and status breakdowns.

        :param agg_state: Accumulated request data categorized by completion status.
        :param scheduler_state: Final scheduler execution state.
        :return: Complete benchmark results with metrics and request statistics.
        """
        successful: list[GenerativeRequestStats] = [
            self._create_generative_request_stats(response, request, request_info)
            for (response, request, request_info) in state.get("completed", [])
        ]
        incomplete: list[GenerativeRequestStats] = [
            self._create_generative_request_stats(response, request, request_info)
            for (response, request, request_info) in state.get("incomplete", [])
        ]
        errored: list[GenerativeRequestStats] = [
            self._create_generative_request_stats(response, request, request_info)
            for (response, request, request_info) in state.get("errored", [])
        ]

        # Use all requests for metrics calculations (not sampled)
        total: list[GenerativeRequestStats] = successful + incomplete + errored
        total_types: list[Literal["successful", "incomplete", "error"]] = [
            *["successful"] * len(successful),
            *["incomplete"] * len(incomplete),
            *["error"] * len(errored),
        ]
        start_time = min(
            [math.inf]
            + [
                req.scheduler_info.request_timings.request_start
                for req in total
                if req.scheduler_info.request_timings.request_start is not None
            ]
        )
        end_time = max(
            [-1 * math.inf]
            + [
                req.scheduler_info.request_timings.request_end
                for req in total
                if req.scheduler_info.request_timings.request_end is not None
            ]
        )

        return {
            "start_time": start_time,
            "end_time": end_time,
            "request_totals": StatusBreakdown[int, int, int, int](
                successful=len(successful),
                incomplete=len(incomplete),
                errored=len(errored),
                total=len(total),
            ),
            "requests": StatusBreakdown[
                list[GenerativeRequestStats],
                list[GenerativeRequestStats],
                list[GenerativeRequestStats],
                list[GenerativeRequestStats],
            ](
                successful=self._sample_request_stats(successful, self.request_samples),
                incomplete=self._sample_request_stats(incomplete, self.request_samples),
                errored=self._sample_request_stats(errored, self.request_samples),
            ),
            "metrics": GenerativeMetrics(
                requests_per_second=self._calculate_requests_per_second(
                    statuses=total_types, requests=total
                ),
                request_concurrency=self._calculate_request_concurrency(
                    statuses=total_types, requests=total
                ),
                request_latency=self._calculate_request_latency(
                    statuses=total_types, requests=total
                ),
                prompt_token_count=self._calculate_prompt_token_count(
                    statuses=total_types, requests=total
                ),
                output_token_count=self._calculate_output_token_count(
                    statuses=total_types, requests=total
                ),
                total_token_count=self._calculate_total_token_count(
                    statuses=total_types, requests=total
                ),
                time_to_first_token_ms=self._calculate_time_to_first_token_ms(
                    statuses=total_types, requests=total
                ),
                time_per_output_token_ms=self._calculate_time_per_output_token_ms(
                    statuses=total_types, requests=total
                ),
                inter_token_latency_ms=self._calculate_inter_token_latency_ms(
                    statuses=total_types, requests=total
                ),
                output_tokens_per_second=self._calculate_output_tokens_per_second(
                    statuses=total_types, requests=total
                ),
                tokens_per_second=self._calculate_tokens_per_second(
                    statuses=total_types, requests=total
                ),
            ),
        }

    def _is_in_warmup(
        self,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> bool:
        """Check if the current request is within the warmup period."""
        if self.warmup is None:
            return False

        if 0 < self.warmup < 1:  # Percentage-based warmup
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction > (1 - self.warmup)
            )

        if self.warmup >= 1:  # Count/time-based warmup
            if scheduler_state.processed_requests < self.warmup:
                return True

            current_time = request_info.scheduler_timings.targeted_start
            return (
                current_time is not None
                and (current_time - scheduler_state.start_time) < self.warmup
            )

        return False

    def _is_in_cooldown(
        self,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        scheduler_state: SchedulerState,
    ) -> bool:
        """Check if the current request is within the cooldown period."""
        if self.cooldown is None:
            return False

        if 0 < self.cooldown < 1:  # Percentage-based cooldown
            return (
                scheduler_state.remaining_fraction is not None
                and scheduler_state.remaining_fraction < self.cooldown
            )

        if self.cooldown >= 1:  # Count/time-based cooldown
            if scheduler_state.remaining_requests < self.cooldown:
                return True

            current_time = (
                request_info.scheduler_timings.resolve_end
                or request_info.scheduler_timings.targeted_start
            )
            return (
                current_time is not None
                and scheduler_state.remaining_duration is not None
                and scheduler_state.remaining_duration < self.cooldown
            )

        return False

    @classmethod
    def _create_generative_request_stats(
        cls,
        response: GenerationResponse,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
    ) -> GenerativeRequestStats:
        prompt_tokens = response.preferred_prompt_tokens(
            settings.preferred_prompt_tokens_source
        )
        output_tokens = response.preferred_output_tokens(
            settings.preferred_output_tokens_source
        )

        return GenerativeRequestStats(
            request_id=request.request_id,
            request_type=request.request_type,
            prompt=str(request.content),
            request_args=response.request_args,
            output=response.value,
            iterations=response.iterations,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                prompt_tokens + output_tokens
                if prompt_tokens is not None and output_tokens is not None
                else None
            ),
            scheduler_info=request_info,
        )

    @classmethod
    def _sample_request_stats(
        cls, stats: list[GenerativeRequestStats], sample_size: int | None
    ) -> list[GenerativeRequestStats]:
        if sample_size is None or sample_size <= 0 or not stats:
            return stats

        return random.sample(stats, min(sample_size, len(stats)))

    @classmethod
    def _calculate_requests_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_times = []

        for status, request in zip(statuses, requests):
            if not all_defined(
                safe_getattr(request.scheduler_info.request_timings, "request_start"),
                safe_getattr(request.scheduler_info.request_timings, "request_end"),
            ):
                continue

            filtered_statuses.append(status)
            filtered_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )

        return StatusDistributionSummary.from_request_times(
            request_types=filtered_statuses,
            requests=filtered_times,
            distribution_type="rate",
        )

    @classmethod
    def _calculate_request_concurrency(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_times = []

        for status, request in zip(statuses, requests):
            if not all_defined(
                safe_getattr(request.scheduler_info.request_timings, "request_start"),
                safe_getattr(request.scheduler_info.request_timings, "request_end"),
            ):
                continue

            filtered_statuses.append(status)
            filtered_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )

        return StatusDistributionSummary.from_request_times(
            request_types=filtered_statuses,
            requests=filtered_times,
            distribution_type="concurrency",
        )

    @classmethod
    def _calculate_request_latency(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.request_latency):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.request_latency)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_prompt_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.prompt_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.prompt_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_output_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.output_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.output_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_total_token_count(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.total_tokens):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.total_tokens)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_time_to_first_token_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.time_to_first_token_ms):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.time_to_first_token_ms)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
        )

    @classmethod
    def _calculate_time_per_output_token_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []
        filtered_weights = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.time_to_first_token_ms):
                continue

            # Add time to first token separately to better reflect in distribution
            filtered_statuses.append(status)
            filtered_values.append(request.time_to_first_token_ms)
            filtered_weights.append(1)

            if not all_defined(request.inter_token_latency_ms):
                continue

            # Add tokens after the first token to get the full distribution
            filtered_statuses.append(status)
            filtered_values.append(request.inter_token_latency_ms)
            filtered_weights.append(request.output_tokens - 1)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
            weights=filtered_weights,
        )

    @classmethod
    def _calculate_inter_token_latency_ms(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_values = []
        filtered_weights = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.inter_token_latency_ms):
                continue

            filtered_statuses.append(status)
            filtered_values.append(request.inter_token_latency_ms)
            filtered_weights.append(request.output_tokens - 1)

        return StatusDistributionSummary.from_values(
            value_types=filtered_statuses,
            values=filtered_values,
            weights=filtered_weights,
        )

    @classmethod
    def _calculate_output_tokens_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_request_times = []
        filtered_first_iter_times = []
        filtered_iter_counts = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.output_tokens_per_second):
                continue

            filtered_statuses.append(status)
            filtered_request_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )
            filtered_first_iter_times.append(
                request.scheduler_info.request_timings.first_iteration
            )
            filtered_iter_counts.append(request.output_tokens)

        return StatusDistributionSummary.from_iterable_request_times(
            request_types=filtered_statuses,
            requests=filtered_request_times,
            first_iter_times=filtered_first_iter_times,
            iter_counts=filtered_iter_counts,
        )

    @classmethod
    def _calculate_tokens_per_second(
        cls,
        statuses: list[Literal["successful", "incomplete", "error"]],
        requests: list[GenerativeRequestStats],
    ) -> StatusDistributionSummary:
        filtered_statuses = []
        filtered_request_times = []
        filtered_first_iter_times = []
        filtered_iter_counts = []
        filtered_first_iter_counts = []

        for status, request in zip(statuses, requests):
            if not all_defined(request.tokens_per_second):
                continue

            filtered_statuses.append(status)
            filtered_request_times.append(
                (
                    request.scheduler_info.request_timings.request_start,
                    request.scheduler_info.request_timings.request_end,
                )
            )
            filtered_first_iter_times.append(
                request.scheduler_info.request_timings.first_iteration
            )
            filtered_iter_counts.append(request.output_tokens - 1)
            filtered_first_iter_counts.append(request.prompt_tokens + 1)

        return StatusDistributionSummary.from_iterable_request_times(
            request_types=filtered_statuses,
            requests=filtered_request_times,
            first_iter_times=filtered_first_iter_times,
            iter_counts=filtered_iter_counts,
            first_iter_counts=filtered_first_iter_counts,
        )
