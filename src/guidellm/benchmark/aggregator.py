import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from pydantic import Field

from guidellm.backend import ResponseSummary
from guidellm.benchmark.benchmark import (
    BenchmarkArgs,
    BenchmarkRunStats,
    BenchmarkT,
    GenerativeBenchmark,
    GenerativeTextErrorStats,
    GenerativeTextResponseStats,
)
from guidellm.config import settings
from guidellm.objects import (
    RunningStats,
    StandardBaseModel,
    StatusBreakdown,
    TimeRunningStats,
)
from guidellm.request import (
    GenerationRequest,
    GenerativeRequestLoaderDescription,
    RequestLoaderDescription,
    RequestT,
    ResponseT,
)
from guidellm.scheduler import (
    GenerativeRequestsWorkerDescription,
    SchedulerRequestResult,
    WorkerDescription,
)
from guidellm.utils import check_load_processor

__all__ = [
    "AggregatorT",
    "BenchmarkAggregator",
    "GenerativeBenchmarkAggregator",
]


class SchedulerRunningStats(StandardBaseModel):
    """
    The metrics for the scheduler stored as running statistics for easy calculations
    of rates, averages, totals, etc.
    """

    created_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests created for this "
            "benchmark run. This includes all requests created, regardless of "
            "their status."
        ),
        default_factory=RunningStats,
    )
    queued_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests pending in queue "
            "for this benchmark run. This includes requests that are waiting to "
            "be scheduled."
        ),
        default_factory=RunningStats,
    )
    scheduled_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests scheduled (actively "
            "running but waiting for the desired start time) for this benchmark run."
        ),
        default_factory=RunningStats,
    )
    processing_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests actively being "
            "processed by the worker for this benchmark run."
        ),
        default_factory=RunningStats,
    )
    completed_requests: RunningStats = Field(
        description=(
            "The running statistics for the number of requests completed for this "
            "benchmark run. This includes requests within the warmup and cooldown "
            "period, if any, along with the final results."
        ),
        default_factory=RunningStats,
    )


class RequestsRunningStats(StandardBaseModel):
    """
    The metrics for requests that have succeeded, been canceled, or errored stored
    as running statistics for easy calculations of rates, averages, totals, etc.
    """

    totals: StatusBreakdown[RunningStats, RunningStats, RunningStats, RunningStats] = (
        Field(
            description=(
                "The running statistics for the total number of requests that "
                "completed within the benchmark run."
            ),
            default_factory=lambda: StatusBreakdown(
                successful=RunningStats(),
                errored=RunningStats(),
                incomplete=RunningStats(),
                total=RunningStats(),
            ),
        )
    )
    queued_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent in queue for all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was dequeued by the worker."
        ),
        default_factory=TimeRunningStats,
    )
    scheduled_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent from when a request was "
            "dequeued by the worker to when it was actually scheduled by the worker"
            "for all requests that completed within the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        ),
        default_factory=TimeRunningStats,
    )
    scheduled_time_sleep: TimeRunningStats = Field(
        description=(
            "The running statistics for the time for each request spent sleeping til "
            "the desired start time was reached for all requests that completed within "
            "the benchmark run. This is the time from when the request was scheduled "
            "to when the desired start time was reached. "
        ),
        default_factory=TimeRunningStats,
    )
    worker_start_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the time delay between when the request was "
            "scheduled and when the worker actually started processing subtracting any "
            "sleep time for all requests that completed within the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        ),
        default_factory=TimeRunningStats,
    )
    worker_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent processing all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was started to when it was completed."
        ),
        default_factory=TimeRunningStats,
    )
    worker_start_time_targeted_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for requests that completed within the benchmark "
            "run. This represents delays from the best case desired start time. "
            "For async strategies, this represents delays from the ideal system. "
            "For sync strategies, since those are doubled in queue, this should be "
            "as close to the time for a request to be processed as possible."
        ),
        default_factory=TimeRunningStats,
    )
    request_start_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the actual request being "
            "made and the time the worker started on the request for all requests "
            "that completed within the benchmark run. This time should be as close to "
            "0 as possible, any additional time is overhead from the system or "
            "the worker."
        ),
        default_factory=TimeRunningStats,
    )
    request_start_time_targeted_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay between the targeted start time and "
            "the actual start time for all requests that completed within the "
            "benchmark run. This represents delays from the best case desired start "
            "time. For async strategies, this represents delays from the ideal system. "
            "For sync strategies, since those are duplicated in queue, this should be "
            "as close to the time for a request to be processed."
        ),
        default_factory=TimeRunningStats,
    )
    request_time_delay: TimeRunningStats = Field(
        description=(
            "The running statistics for the delay in time between the total request "
            "time and the worker time. This should be as close to 0 as possible, any "
            "additional time is overhead from the system or the worker. "
        ),
        default_factory=TimeRunningStats,
    )
    request_time: TimeRunningStats = Field(
        description=(
            "The running statistics for the time spent processing all requests that "
            "completed within the benchmark run. This is the time from when the "
            "request was created to when it was completed."
        ),
        default_factory=TimeRunningStats,
    )


class BenchmarkAggregator(
    ABC, StandardBaseModel, Generic[BenchmarkT, RequestT, ResponseT]
):
    """
    A pydantic base class representing the base class for aggregating benchmark results.
    The purpose is to receive and process results from a Benchmarker as it iterates
    through a Scheduler for an individual benchmark run.
    As results are added, lightweight statistics are updated and stored for immediate
    progress and informational updates to the caller.
    Once the benchmark run is complete, the `compile` method is called to finalize
    the benchmark and return a Benchmark object with all the results and statistics
    fully calculated.
    """

    type_: Literal["benchmark_aggregator"] = "benchmark_aggregator"
    run_id: str = Field(
        description=(
            "The unique identifier for the encompasing benchmark run that this "
            "benchmark was a part of."
        )
    )
    args: BenchmarkArgs = Field(
        description=(
            "The arguments used to create the benchmark run that this benchmark was "
            "a part of."
        )
    )
    worker_description: Union[
        GenerativeRequestsWorkerDescription, WorkerDescription
    ] = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        ),
        discriminator="type_",
    )
    request_loader_description: Union[
        GenerativeRequestLoaderDescription, RequestLoaderDescription
    ] = Field(
        description=(
            "The description and specifics for the request loader used to create "
            "requests for this benchmark."
        ),
        discriminator="type_",
    )
    extras: dict[str, Any] = Field(
        description=(
            "Any additional information or metadata that was passed for this benchmark."
        )
    )
    in_warmup: bool = Field(
        description=(
            "A flag to indicate if the benchmark is currently in the warmup phase."
        ),
        default=False,
        exclude=True,
    )
    in_cooldown: bool = Field(
        description=(
            "A flag to indicate if the benchmark is currently in the cooldown phase."
        ),
        default=False,
        exclude=True,
    )
    scheduler_stats: SchedulerRunningStats = Field(
        description=(
            "The running statistics for the scheduler for this benchmark run. "
            "This includes all requests created, regardless of their status."
        ),
        default_factory=SchedulerRunningStats,
    )
    requests_stats: RequestsRunningStats = Field(
        description=(
            "The running statistics for the requests for this benchmark run. "
            "This includes all requests created, regardless of their status."
        ),
        default_factory=RequestsRunningStats,
    )
    results: StatusBreakdown[
        list[SchedulerRequestResult[RequestT, ResponseT]],
        list[SchedulerRequestResult[RequestT, ResponseT]],
        list[SchedulerRequestResult[RequestT, ResponseT]],
        None,
    ] = Field(
        description=(
            "The completed requests for this benchmark run broken down by status"
            "and excluding warmup and cooldown requests."
        ),
        default_factory=lambda: StatusBreakdown(  # type: ignore[arg-type]
            successful=[],
            errored=[],
            incomplete=[],
            total=None,
        ),
    )

    def add_result(
        self,
        result: SchedulerRequestResult[RequestT, ResponseT],
    ) -> bool:
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        :return: True if the result was added, False if it was added because it
            did not fit within the warmup or cooldown period, was not requested,
            or is not finished
        """
        # Add scheduler statistics
        self.scheduler_stats.created_requests += max(
            0, result.run_info.created_requests
        )
        self.scheduler_stats.queued_requests += max(0, result.run_info.queued_requests)
        self.scheduler_stats.scheduled_requests += max(
            0, result.run_info.scheduled_requests
        )
        self.scheduler_stats.processing_requests += max(
            0, result.run_info.processing_requests
        )
        self.scheduler_stats.completed_requests += max(
            0, result.run_info.completed_requests
        )

        if result.type_ != "request_complete" or (
            result.request_info.canceled and not result.request_info.requested
        ):
            # If the result is not completed yet, don't add to the results
            # If the result was canceled and not started, ignore it
            return False

        # Add request statistics
        self.requests_stats.totals.total += 1
        if result.request_info.canceled:
            self.requests_stats.totals.incomplete += 1
        elif result.request_info.errored:
            self.requests_stats.totals.errored += 1
        elif result.request_info.completed:
            self.requests_stats.totals.successful += 1
        else:
            raise ValueError(
                "Unexpected state: request_info must be either "
                "completed, canceled, or errored. "
                f"Got {result.request_info}"
            )

        self.requests_stats.queued_time.update(
            result.request_info.dequeued_time - result.request_info.queued_time
        )
        self.requests_stats.scheduled_time_delay.update(
            result.request_info.scheduled_time - result.request_info.dequeued_time
        )
        sleep_time = max(
            0.0,
            result.request_info.targeted_start_time
            - result.request_info.scheduled_time,
        )
        self.requests_stats.scheduled_time_sleep.update(sleep_time)
        time_to_worker_start = (
            result.request_info.worker_start - result.request_info.scheduled_time
        )
        self.requests_stats.worker_start_delay.update(time_to_worker_start - sleep_time)
        self.requests_stats.worker_time.update(
            result.request_info.worker_end - result.request_info.worker_start
        )
        self.requests_stats.worker_start_time_targeted_delay.update(
            result.request_info.worker_start - result.request_info.targeted_start_time
        )
        self.requests_stats.request_start_time_delay.update(
            result.request_info.worker_start - result.request_info.targeted_start_time
        )
        self.requests_stats.request_start_time_targeted_delay.update(
            result.request_info.worker_start - result.request_info.targeted_start_time
        )
        self.requests_stats.request_time_delay.update(
            (result.request_info.worker_end - result.request_info.worker_start)
            - (result.request_info.worker_end - result.request_info.worker_start)
        )
        self.requests_stats.request_time.update(
            result.request_info.worker_end - result.request_info.worker_start
        )

        # Add result to the list of results provided we are not in warmup or cooldown
        total_completed = self.requests_stats.totals.total.total
        global_start_time = self.requests_stats.totals.total.start_time

        in_warmup_number = (
            self.args.warmup_number and total_completed <= self.args.warmup_number
        )
        in_warmup_duration = (
            self.args.warmup_duration
            and result.request_info.worker_start
            <= (global_start_time + self.args.warmup_duration)
        )

        if in_warmup_number or in_warmup_duration:
            self.in_warmup = True
            return True

        self.in_warmup = False
        in_cooldown_number = (
            self.args.cooldown_number
            and self.args.max_number
            and total_completed > self.args.max_number - self.args.cooldown_number
        )
        in_cooldown_duration = (
            self.args.cooldown_duration
            and self.args.max_duration
            and result.request_info.worker_start
            > global_start_time + self.args.max_duration - self.args.cooldown_duration
        )

        if in_cooldown_number or in_cooldown_duration:
            self.in_cooldown = True
            return True

        self.in_cooldown = False

        if result.request_info.canceled:
            self.results.incomplete.append(result)
        elif result.request_info.errored:
            self.results.errored.append(result)
        elif result.request_info.completed:
            self.results.successful.append(result)
        else:
            raise ValueError(
                "Unexpected state: request_info must be either "
                "completed, canceled, or errored. "
                f"Got {result.request_info}"
            )

        return True

    @abstractmethod
    def compile(self) -> BenchmarkT:
        """
        Compile the benchmark results and statistics into a Benchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        ...


AggregatorT = TypeVar("AggregatorT", bound=BenchmarkAggregator)


class GenerativeRequestsRunningStats(RequestsRunningStats):
    """
    The metrics for generative requests that have succeeded, been canceled, or errored
    stored as running statistics for easy calculations of rates, averages, totals, etc.
    """

    time_to_first_token: TimeRunningStats = Field(
        description=(
            "The running statistics for the time from the start of the request to the "
            "first token being generated for all requests that completed within the "
            "benchmark run."
        ),
        default_factory=TimeRunningStats,
    )
    inter_token_latency: TimeRunningStats = Field(
        description=(
            "The running statistics for the time between each token being generated "
            "for all requests that completed within the benchmark run."
        ),
        default_factory=TimeRunningStats,
    )
    prompt_tokens: RunningStats = Field(
        description=(
            "The running statistics for the token count for the prompt for all "
            "requests that completed, if available in the response."
        ),
        default_factory=RunningStats,
    )
    output_tokens: RunningStats = Field(
        description=(
            "The running statistics for the token count for the output for all "
            "requests that completed, if available in the response."
        ),
        default_factory=RunningStats,
    )
    total_tokens: RunningStats = Field(
        description=(
            "The running statistics for the total token count for all requests that "
            "completed, if available in the response."
        ),
        default_factory=RunningStats,
    )


class GenerativeBenchmarkAggregator(
    BenchmarkAggregator[GenerativeBenchmark, GenerationRequest, ResponseSummary]
):
    type_: Literal["generative_benchmark_aggregator"] = (
        "generative_benchmark_aggregator"  # type: ignore[assignment]
    )
    processor: Optional[Union[str, Path, Any]] = Field(
        description=(
            "The tokenizer to use for calculating token counts when none are "
            "avaiable that match the preferred source."
        )
    )
    processor_args: Optional[dict[str, Any]] = Field(
        description=(
            "Additional arguments to pass to the tokenizer if it requires "
            "any specific configuration for loading or processing."
        ),
    )
    worker_description: GenerativeRequestsWorkerDescription = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        ),
        discriminator="type_",
    )
    request_loader_description: GenerativeRequestLoaderDescription = Field(
        description=(
            "The description and specifics for the request loader used to create "
            "requests for this benchmark."
        ),
        discriminator="type_",
    )
    requests_stats: GenerativeRequestsRunningStats = Field(
        description=(
            "The running statistics for the requests for this benchmark run. "
            "This includes all requests created, regardless of their status."
        ),
        default_factory=GenerativeRequestsRunningStats,
    )

    def add_result(
        self, result: SchedulerRequestResult[GenerationRequest, ResponseSummary]
    ) -> bool:
        """
        Add a result to the aggregator. This will update the internal statistics
        and add the result to the list of results if it is not within the warmup or
        cooldown period.

        :param result: The result to add to the aggregator.
        """
        if not super().add_result(result):
            return False

        if result.request is None:
            raise ValueError("Request is None, cannot add result.")

        if result.response is None:
            raise ValueError("Response is None, cannot add result.")

        self.requests_stats.request_start_time_delay.update(
            result.response.start_time - result.request_info.worker_start
        )
        self.requests_stats.request_start_time_targeted_delay.update(
            result.response.start_time - result.request_info.targeted_start_time
        )
        self.requests_stats.request_time_delay.update(
            (result.response.start_time - result.request_info.worker_start)
            + result.request_info.worker_end
            - result.response.end_time
        )
        self.requests_stats.request_time.update(
            result.response.end_time - result.response.start_time
        )
        if result.response.first_iter_time:
            self.requests_stats.time_to_first_token.update(
                result.response.first_iter_time - result.response.start_time
            )
        if result.response.last_iter_time and result.response.first_iter_time:
            self.requests_stats.inter_token_latency.update(
                result.response.last_iter_time - result.response.first_iter_time,
                count=(result.response.output_tokens or 1) - 1,
            )
        self.requests_stats.prompt_tokens += result.response.request_prompt_tokens or 0
        self.requests_stats.output_tokens += result.response.request_output_tokens or 0
        total_tokens = (result.response.request_prompt_tokens or 0) + (
            result.response.request_output_tokens or 0
        )
        self.requests_stats.total_tokens += total_tokens

        return True

    def compile(self) -> GenerativeBenchmark:
        """
        Compile the benchmark results and statistics into a GenerativeBenchmark object.
        This is required to be implemented by subclasses to finalize the benchmark
        and return the compiled object.
        """
        successful, incomplete, errored = self._compile_results()

        return GenerativeBenchmark.from_stats(
            run_id=self.run_id,
            successful=successful,
            incomplete=incomplete,
            errored=errored,
            args=self.args,
            run_stats=BenchmarkRunStats(
                start_time=self.requests_stats.totals.total.start_time,
                end_time=time.time(),
                requests_made=StatusBreakdown(
                    successful=int(self.requests_stats.totals.successful.total),
                    errored=int(self.requests_stats.totals.errored.total),
                    incomplete=int(self.requests_stats.totals.incomplete.total),
                    total=int(self.requests_stats.totals.total.total),
                ),
                queued_time_avg=self.requests_stats.queued_time.mean,
                scheduled_time_delay_avg=self.requests_stats.scheduled_time_delay.mean,
                scheduled_time_sleep_avg=self.requests_stats.scheduled_time_sleep.mean,
                worker_start_delay_avg=self.requests_stats.worker_start_delay.mean,
                worker_time_avg=self.requests_stats.worker_time.mean,
                worker_start_time_targeted_delay_avg=self.requests_stats.worker_start_time_targeted_delay.mean,
                request_start_time_delay_avg=self.requests_stats.request_start_time_delay.mean,
                request_start_time_targeted_delay_avg=self.requests_stats.request_start_time_targeted_delay.mean,
                request_time_delay_avg=self.requests_stats.request_time_delay.mean,
                request_time_avg=self.requests_stats.request_time.mean,
            ),
            worker=self.worker_description,
            requests_loader=self.request_loader_description,
            extras=self.extras,
        )

    def _compile_results(
        self,
    ) -> tuple[
        list[GenerativeTextResponseStats],
        list[GenerativeTextErrorStats],
        list[GenerativeTextErrorStats],
    ]:
        successful: list[GenerativeTextResponseStats] = [
            GenerativeTextResponseStats(
                request_id=result.request.request_id,
                request_type=result.request.request_type,
                scheduler_info=result.request_info,
                prompt=str(result.request.content),
                prompt_tokens=self._compile_tokens_count(
                    value=str(result.request.content),
                    requests_tokens=result.response.request_prompt_tokens,
                    response_tokens=result.response.response_prompt_tokens,
                    preferred_tokens_source=settings.preferred_prompt_tokens_source,
                    errored=False,
                ),
                output=result.response.value,
                output_tokens=self._compile_tokens_count(
                    value=result.response.value,
                    requests_tokens=result.response.request_output_tokens,
                    response_tokens=result.response.response_output_tokens,
                    preferred_tokens_source=settings.preferred_output_tokens_source,
                    errored=False,
                ),
                start_time=result.response.start_time,
                end_time=result.response.end_time,
                first_token_time=result.response.first_iter_time or -1.0,
                last_token_time=result.response.last_iter_time or -1.0,
            )
            for result in self.results.successful
            if result.request and result.response
        ]
        incomplete: list[GenerativeTextErrorStats] = [
            GenerativeTextErrorStats(
                error=result.response.error or "",
                request_id=result.request.request_id,
                request_type=result.request.request_type,
                scheduler_info=result.request_info,
                prompt=str(result.request.content),
                prompt_tokens=self._compile_tokens_count(
                    value=str(result.request.content),
                    requests_tokens=result.response.request_prompt_tokens,
                    response_tokens=result.response.response_prompt_tokens,
                    preferred_tokens_source=settings.preferred_prompt_tokens_source,
                    errored=True,
                ),
                output=result.response.value,
                output_tokens=self._compile_tokens_count(
                    value=result.response.value,
                    requests_tokens=result.response.request_output_tokens,
                    response_tokens=result.response.response_output_tokens,
                    preferred_tokens_source=settings.preferred_output_tokens_source,
                    errored=True,
                ),
                start_time=result.response.start_time,
                end_time=result.response.end_time,
                first_token_time=result.response.first_iter_time,
                last_token_time=result.response.last_iter_time,
            )
            for result in self.results.incomplete
            if result.request and result.response
        ]
        error: list[GenerativeTextErrorStats] = [
            GenerativeTextErrorStats(
                error=result.response.error or "",
                request_id=result.request.request_id,
                request_type=result.request.request_type,
                scheduler_info=result.request_info,
                prompt=str(result.request.content),
                prompt_tokens=self._compile_tokens_count(
                    value=str(result.request.content),
                    requests_tokens=result.response.request_prompt_tokens,
                    response_tokens=result.response.response_prompt_tokens,
                    preferred_tokens_source=settings.preferred_prompt_tokens_source,
                    errored=True,
                ),
                output=result.response.value,
                output_tokens=self._compile_tokens_count(
                    value=result.response.value,
                    requests_tokens=result.response.request_output_tokens,
                    response_tokens=result.response.response_output_tokens,
                    preferred_tokens_source=settings.preferred_output_tokens_source,
                    errored=True,
                ),
                start_time=result.response.start_time,
                end_time=result.response.end_time,
                first_token_time=result.response.first_iter_time,
                last_token_time=result.response.last_iter_time,
            )
            for result in self.results.errored
            if result.request and result.response
        ]

        return successful, incomplete, error

    def _compile_tokens_count(
        self,
        value: str,
        requests_tokens: Optional[int],
        response_tokens: Optional[int],
        preferred_tokens_source: Optional[Literal["request", "response", "local"]],
        errored: bool,
    ) -> int:
        if not errored and preferred_tokens_source == "response" and response_tokens:
            return response_tokens or 0

        if not errored and preferred_tokens_source == "request" and requests_tokens:
            return requests_tokens or 0

        if preferred_tokens_source in {"response", "request"} and (
            self.processor is None or errored or response_tokens or requests_tokens
        ):
            # we had a preferred tokens source that isn't local and we either
            # have the data to return something or we don't have the ability
            # to calculate locally
            return response_tokens or requests_tokens or 0

        self.processor = check_load_processor(
            self.processor,
            processor_args=self.processor_args,
            error_msg="Processor/Tokenizer is required for calculating token counts.",
        )
        return len(self.processor.tokenize(value))
