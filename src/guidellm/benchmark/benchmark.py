import random
import uuid
from typing import Any, Literal, Optional, TypeVar, Union

from pydantic import Field, computed_field

from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    Profile,
    SweepProfile,
    SynchronousProfile,
    ThroughputProfile,
)
from guidellm.objects import (
    StandardBaseModel,
    StatusBreakdown,
    StatusDistributionSummary,
)
from guidellm.request import (
    GenerativeRequestLoaderDescription,
    RequestLoaderDescription,
)
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    GenerativeRequestsWorkerDescription,
    SchedulerRequestInfo,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerDescription,
)

__all__ = [
    "Benchmark",
    "BenchmarkArgs",
    "BenchmarkMetrics",
    "BenchmarkRunStats",
    "BenchmarkT",
    "GenerativeBenchmark",
    "GenerativeMetrics",
    "GenerativeTextErrorStats",
    "GenerativeTextResponseStats",
    "StatusBreakdown",
]


class BenchmarkArgs(StandardBaseModel):
    """
    A serializable model representing the arguments used to specify a benchmark run
    and how data was collected for it.
    """

    profile: Union[
        AsyncProfile,
        SweepProfile,
        ConcurrentProfile,
        ThroughputProfile,
        SynchronousProfile,
        Profile,
    ] = Field(
        description=(
            "The profile used for the entire benchmark run that the strategy for "
            "this benchmark was pulled from."
        ),
        discriminator="type_",
    )
    strategy_index: int = Field(
        description=(
            "The index of the strategy in the profile that was used for this benchmark."
        )
    )
    strategy: Union[
        ConcurrentStrategy,
        SchedulingStrategy,
        ThroughputStrategy,
        SynchronousStrategy,
        AsyncPoissonStrategy,
        AsyncConstantStrategy,
        SchedulingStrategy,
    ] = Field(
        description="The scheduling strategy used to run this benchmark. ",
        discriminator="type_",
    )
    max_number: Optional[int] = Field(
        description="The maximum number of requests to run for this benchmark, if any."
    )
    max_duration: Optional[float] = Field(
        description="The maximum duration in seconds to run this benchmark, if any."
    )
    warmup_number: Optional[int] = Field(
        description=(
            "The number of requests to run for the warmup phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    warmup_duration: Optional[float] = Field(
        description=(
            "The duration in seconds to run for the warmup phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    cooldown_number: Optional[int] = Field(
        description=(
            "The number of requests to run for the cooldown phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )
    cooldown_duration: Optional[float] = Field(
        description=(
            "The duration in seconds to run for the cooldown phase of this benchmark, "
            "if any. These are requests that were not included in the final results."
        )
    )


class BenchmarkRunStats(StandardBaseModel):
    """
    A serializable model representing the run process statistics for the
    entire benchmark run across all requests including warmup and cooldown.
    """

    start_time: float = Field(
        description="The start time of the benchmark run.",
    )
    end_time: float = Field(
        description="The end time of the benchmark run.",
    )
    requests_made: StatusBreakdown[int, int, int, int] = Field(
        description=(
            "The number of requests made for the benchmark run broken down by "
            "status including successful, incomplete, errored, and the sum of all three"
        )
    )
    queued_time_avg: float = Field(
        description=(
            "The average time spent in the queue for each request in the benchmark "
            "run until it was dequeued by a worker."
        )
    )
    scheduled_time_delay_avg: float = Field(
        description=(
            "The average time delay between when a request was dequeued and when it "
            "was scheduled to be processed by a worker in the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        )
    )
    scheduled_time_sleep_avg: float = Field(
        description=(
            "The average time spent sleeping til the desired start time was reached "
            "after being scheduled by the worker in the benchmark run."
        )
    )
    worker_start_delay_avg: float = Field(
        description=(
            "The average time delay between when a request was scheduled and when "
            "the worker started processing it in the benchmark run. "
            "This should be as close to 0 as possible, any additional time is "
            "overheads from the system or the worker."
        )
    )
    worker_time_avg: float = Field(
        description=(
            "The average time taken by the worker to process each request in the "
            "benchmark run. This includes the time to generate the response and "
            "any additional processing time."
        )
    )
    worker_start_time_targeted_delay_avg: float = Field(
        description=(
            "The average time delay between when a request was targeted to start "
            "and when the worker actually started processing it in the benchmark "
            "run. For async strategies, this represents delays from the ideal "
            "system. For sync strategies, since those are doubled in queue, "
            "this should be as close to the time for a request to be processed "
            "as possible. Any additional time is overhead from the system or "
            "the worker."
        )
    )
    request_start_time_delay_avg: float = Field(
        description=(
            "The average time delay between the actual request being made "
            "and the time the worker started on the request for all requests "
            "that completed within the benchmark run. This time should be as close "
            "to 0 as possible, any additional time is overhead from the system or "
            "the worker."
        )
    )
    request_start_time_targeted_delay_avg: float = Field(
        description=(
            "The average time delay between when the targeted start time and "
            "the actual start time for each request in the benchmark run. "
            "For async strategies, this represents delays from the ideal "
            "system. For sync strategies, this should be as close to the "
            "time for a request to be processed as possible. Any additional "
            "time is overhead from the system or the worker."
        )
    )
    request_time_delay_avg: float = Field(
        description=(
            "The average time delay between the total request time and the "
            "worker time. This should be as close to 0 as possible, any additional "
            "time is overhead from the system or the worker. "
        )
    )
    request_time_avg: float = Field(
        description=(
            "The average time spent processing all requests in the benchmark run. "
            "This is the time from when the actual request was started to when "
            "it was completed."
        )
    )


class BenchmarkMetrics(StandardBaseModel):
    """
    A serializable model representing the metrics for a benchmark run.
    """

    requests_per_second: StatusDistributionSummary = Field(
        description="The distribution of requests per second for the benchmark.",
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="The distribution of requests concurrency for the benchmark.",
    )


class Benchmark(StandardBaseModel):
    """
    The base serializable model representing a benchmark run and its results.
    Specific benchmarker implementations should extend this model to include
    additional information or metadata as needed.

    Note, requests_per_second and request_concurrency are kept at this level
    and are expected to be populated by the subclass implementation to ensure
    the logic for Profiles can include more complicated logic for determining
    what rates and concurrency values to use for subsequent strategies.
    """

    type_: Literal["benchmark"] = "benchmark"
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the benchmark.",
    )
    run_id: str = Field(
        description=(
            "The unique identifier for the encompasing benchmark run that this "
            "benchmark was a part of."
        )
    )
    args: BenchmarkArgs = Field(
        description=(
            "The arguments used to specify how to run the benchmark and collect data."
        )
    )
    run_stats: BenchmarkRunStats = Field(
        description=(
            "The process statistics for the entire benchmark run across all requests."
        )
    )
    worker: Union[WorkerDescription] = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        ),
    )
    request_loader: Union[RequestLoaderDescription] = Field(
        description=(
            "The description and specifics for the request loader used to create "
            "requests for this benchmark."
        ),
    )
    extras: dict[str, Any] = Field(
        description=(
            "Any additional information or metadata that was passed for this benchmark."
        )
    )
    metrics: BenchmarkMetrics = Field(
        description=(
            "The metrics for the benchmark run represented as a distribution of "
            "various per-request statistics."
        ),
    )


BenchmarkT = TypeVar("BenchmarkT", bound=Benchmark)


class GenerativeTextResponseStats(StandardBaseModel):
    """
    A serializable model representing the request values, response values, and
    statistics for a generative text response.
    """

    type_: Literal["generative_text_response"] = "generative_text_response"
    request_id: Optional[str] = Field(
        description="The unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        description="The type of request made to the generative backend."
    )
    scheduler_info: SchedulerRequestInfo = Field(
        description=(
            "The info about the request from the scheduler about how it was run."
        ),
    )
    prompt: str = Field(
        description="The text prompt used for the generative request.",
    )
    output: str = Field(
        description="The generated text output from the generative request.",
    )
    prompt_tokens: int = Field(
        description="The number of tokens in the prompt text.",
    )
    output_tokens: int = Field(
        description="The number of tokens in the generated output text.",
    )
    start_time: float = Field(
        description="The time the request started.",
    )
    end_time: float = Field(
        description="The time the request ended.",
    )
    first_token_time: float = Field(
        description="The time the first token was received.",
    )
    last_token_time: float = Field(
        description="The time the last token was received.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> float:
        """
        :return: The duration of the request in seconds from the start to the end.
        """
        return self.end_time - self.start_time

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_ms(self) -> float:
        """
        :return: The time in milliseconds from the start of the request to the first
            token received.
        """
        return 1000 * (self.first_token_time - self.start_time)

    @computed_field  # type: ignore[misc]
    @property
    def time_per_output_token_ms(self) -> float:
        """
        :return: The average time in milliseconds per output token generated.
            This includes the time to generate the first token and all other tokens.
        """
        if self.output_tokens == 0:
            return 0.0

        return (
            1000 * (self.last_token_time - self.first_token_time) / self.output_tokens
        )

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_ms(self) -> float:
        """
        :return: The average time in milliseconds between generating tokens in the
            output text. Note, does not include the time to generate the first token.
        """
        if self.output_tokens <= 1:
            return 0.0

        return (
            1000
            * (self.last_token_time - self.first_token_time)
            / (self.output_tokens - 1)
        )

    @computed_field  # type: ignore[misc]
    @property
    def tokens_per_second(self) -> float:
        """
        :return: The average number of tokens generated per second in the prompt and
            output text.
        """
        if (latency := self.request_latency) == 0.0:
            return 0.0

        return (self.prompt_tokens + self.output_tokens) / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> float:
        """
        :return: The average number of output tokens generated per second.
        """
        if (latency := self.request_latency) == 0.0:
            return 0.0

        return self.output_tokens / latency


class GenerativeTextErrorStats(GenerativeTextResponseStats):
    """
    A serializable model representing the request values, response values, and
    statistics for a generative text response that errored.
    Extends and overrides the GenerativeTextResponseStats model to include the
    error message and optional properties given the error occurred.
    """

    type_: Literal["generative_text_error"] = "generative_text_error"  # type: ignore[assignment]
    error: str = Field(
        description=(
            "The error message for the error that occurred while making the request."
        )
    )
    output: Optional[str] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "The generated text output from the generative request, if any, "
            "before the error occurred."
        ),
    )
    first_token_time: Optional[float] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "The time the first token was received, if any, before the error occurred."
        ),
    )
    last_token_time: Optional[float] = Field(  # type: ignore[assignment]
        default=None,
        description=(
            "The time the last token was received, if any, before the error occurred."
        ),
    )

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_ms(self) -> Optional[float]:  # type: ignore[override]
        """
        :return: The time in milliseconds from the start of the request to the first
            token received. None if the first token was not received.
        """
        if self.first_token_time is None:
            return None

        return super().time_to_first_token_ms

    @computed_field  # type: ignore[misc]
    @property
    def time_per_output_token_ms(self) -> Optional[float]:  # type: ignore[override]
        """
        :return: The average time in milliseconds per output token generated.
            This includes the time to generate the first token and all other tokens.
            None if the output_tokens is None or 0.
        """
        if (
            self.output_tokens is None
            or self.output_tokens == 0
            or self.first_token_time is None
            or self.last_token_time is None
        ):
            return None

        return super().time_per_output_token_ms

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_ms(self) -> Optional[float]:  # type: ignore[override]
        """
        :return: The average time in milliseconds between generating tokens in the
            output text. Note, does not include the time to generate the first token.
            None if there were no output_tokens or the first token was not received.
        """
        if (
            self.output_tokens is None
            or self.first_token_time is None
            or self.last_token_time is None
        ):
            return None

        return super().inter_token_latency_ms

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> Optional[float]:  # type: ignore[override]
        """
        :return: The average number of tokens generated per second in the output text.
            Note, does not include the time to generate the first token. None if there
            were no output_tokens or the first token was not received.
        """
        if self.inter_token_latency_ms is None:
            return None

        return super().output_tokens_per_second


class GenerativeMetrics(BenchmarkMetrics):
    """
    A serializable model representing the metrics for a generative benchmark run.
    """

    request_latency: StatusDistributionSummary = Field(
        description="The distribution of latencies for the completed requests.",
    )
    prompt_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the prompts for completed, "
            "errored, and all requests."
        )
    )
    output_token_count: StatusDistributionSummary = Field(
        description=(
            "The distribution of token counts in the outputs for completed, "
            "errored, and all requests."
        )
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies to receiving the first token in "
            "milliseconds for completed, errored, and all requests."
        ),
    )
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies per output token in milliseconds for "
            "completed, errored, and all requests. "
            "This includes the time to generate the first token and all other tokens."
        ),
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description=(
            "The distribution of latencies between tokens in milliseconds for "
            "completed, errored, and all requests."
        ),
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of output tokens per second for completed, "
            "errored, and all requests."
        ),
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description=(
            "The distribution of tokens per second, including prompt and output tokens "
            "for completed, errored, and all requests."
        ),
    )


class GenerativeBenchmark(Benchmark):
    """
    A serializable model representing a benchmark run and its results for generative
    requests and responses. Includes the completed and errored requests, the start
    and end times for the benchmark, and the statistics for the requests and responses.
    """

    type_: Literal["generative_benchmark"] = "generative_benchmark"  # type: ignore[assignment]
    start_time: float = Field(
        description="The start time of the first request for the benchmark.",
    )
    end_time: float = Field(
        description="The end time of the last request for the benchmark.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def duration(self) -> float:
        """
        :return: The duration of the benchmark in seconds from the start of the
            first request to the end of the last request.
        """
        return self.end_time - self.start_time

    worker: GenerativeRequestsWorkerDescription = Field(
        description=(
            "The description and specifics for the worker used to resolve requests "
            "for this benchmark."
        ),
    )
    request_loader: GenerativeRequestLoaderDescription = Field(
        description=(
            "The description and specifics for the request loader used to create "
            "requests for this benchmark."
        ),
    )
    metrics: GenerativeMetrics = Field(
        description=(
            "The metrics for the benchmark run represented as a distribution of "
            "various per-request statistics."
        ),
    )
    # Output is ordered so keep the requests at the end for better readability in files
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description=(
            "The number of requests made for the benchmark broken down by status "
            "including successful, incomplete, errored, and the sum of all three"
        )
    )
    request_samples: Optional[StatusBreakdown[int, int, int, None]] = Field(
        description=(
            "The number of requests that were randomly sampled for "
            "the benchmark. None if no sampling was applied."
        ),
        default=None,
    )
    requests: StatusBreakdown[
        list[GenerativeTextResponseStats],
        list[GenerativeTextErrorStats],
        list[GenerativeTextErrorStats],
        None,
    ] = Field(
        description=(
            "The breakdown of requests for the benchmark run including successful, "
            "incomplete, and errored requests."
        ),
    )

    def set_sample_size(self, sample_size: Optional[int]) -> "GenerativeBenchmark":
        """
        Set the sample size for the benchmark. This will randomly sample the
        requests for each status type to the given sample size or the maximum
        number of requests for that status type, whichever is smaller.
        This is applied to requests.successful, requests.errored, and
        requests.incomplete.
        If None, no sampling is applied and the state is kept.

        :param sample_size: The number of requests to sample for each status type.
        :return: The benchmark with the sampled requests.
        :raises ValueError: If the sample size is invalid.
        """

        if sample_size is not None:
            if sample_size < 0 or not isinstance(sample_size, int):
                raise ValueError(
                    f"Sample size must be non-negative integer, given {sample_size}"
                )

            sample_size = min(sample_size, len(self.requests.successful))
            error_sample_size = min(sample_size, len(self.requests.errored))
            incomplete_sample_size = min(sample_size, len(self.requests.incomplete))

            self.requests.successful = random.sample(
                self.requests.successful, sample_size
            )
            self.requests.errored = random.sample(
                self.requests.errored, error_sample_size
            )
            self.requests.incomplete = random.sample(
                self.requests.incomplete, incomplete_sample_size
            )
            self.request_samples = StatusBreakdown(
                successful=len(self.requests.successful),
                incomplete=len(self.requests.incomplete),
                errored=len(self.requests.errored),
            )

        return self

    @staticmethod
    def from_stats(
        run_id: str,
        successful: list[GenerativeTextResponseStats],
        incomplete: list[GenerativeTextErrorStats],
        errored: list[GenerativeTextErrorStats],
        args: BenchmarkArgs,
        run_stats: BenchmarkRunStats,
        worker: GenerativeRequestsWorkerDescription,
        requests_loader: GenerativeRequestLoaderDescription,
        extras: Optional[dict[str, Any]],
    ) -> "GenerativeBenchmark":
        """
        Create a GenerativeBenchmark instance from the given statistics and metadata.
        Given the completed and errored requests, the benchmark will fill in the
        remaining statistics for the various metrics required for a benchmark.
        This is the preferred method for creating a GenerativeBenchmark instance
        to ensure all statistics are properly calculated and populated.

        :param run_id: The unique identifier for the benchmark run.
        :param completed: The list of completed requests.
        :param errored: The list of errored requests.
        :param args: The arguments used to specify how to run the benchmark
            and collect data.
        :param run_stats: The process statistics for the entire benchmark run across
            all requests.
        :param worker: The description and specifics for the worker used to resolve
            requests.
        :param requests_loader: The description and specifics for the request loader
            used to create requests.
        :param extras: Any additional information or metadata that was passed for
            this benchmark.
        :return: A GenerativeBenchmark instance with the given statistics and metadata
            populated and calculated
        """
        total = successful + incomplete + errored
        total_types: list[Literal["successful", "incomplete", "error"]] = [
            *["successful"] * len(successful),  # type: ignore[list-item]
            *["incomplete"] * len(incomplete),  # type: ignore[list-item]
            *["error"] * len(errored),  # type: ignore[list-item]
        ]
        start_time = min(req.start_time for req in total)
        end_time = max(req.end_time for req in total)

        total_with_prompt, total_types_with_prompt = (
            zip(*filtered)
            if (
                filtered := list(
                    filter(lambda val: bool(val[0].prompt), zip(total, total_types))
                )
            )
            else ([], [])
        )
        total_with_output_first, total_types_with_output_first = (
            zip(*filtered)
            if (
                filtered := list(
                    filter(
                        lambda val: bool(val[0].output_tokens > 0),
                        zip(total, total_types),
                    )
                )
            )
            else ([], [])
        )
        total_with_output_multi, total_types_with_output_multi = (
            zip(*filtered)
            if (
                filtered := list(
                    filter(
                        lambda val: bool(val[0].output_tokens > 1),
                        zip(total, total_types),
                    )
                )
            )
            else ([], [])
        )

        return GenerativeBenchmark(
            run_id=run_id,
            args=args,
            run_stats=run_stats,
            extras=extras or {},
            start_time=start_time,
            end_time=end_time,
            worker=worker,
            request_loader=requests_loader,
            metrics=GenerativeMetrics(
                requests_per_second=StatusDistributionSummary.from_request_times(
                    request_types=total_types,
                    requests=[(req.start_time, req.end_time) for req in total],
                    distribution_type="rate",
                ),
                request_concurrency=StatusDistributionSummary.from_request_times(
                    request_types=total_types,
                    requests=[(req.start_time, req.end_time) for req in total],
                    distribution_type="concurrency",
                ),
                request_latency=StatusDistributionSummary.from_values(
                    value_types=total_types,
                    values=[req.request_latency for req in total],
                ),
                prompt_token_count=StatusDistributionSummary.from_values(
                    value_types=list(total_types_with_prompt),
                    values=[req.prompt_tokens for req in total_with_prompt],
                ),
                output_token_count=StatusDistributionSummary.from_values(
                    value_types=list(total_types_with_output_first),
                    values=[req.output_tokens for req in total_with_output_first],
                ),
                time_to_first_token_ms=StatusDistributionSummary.from_values(
                    value_types=list(total_types_with_output_first),
                    values=[
                        req.time_to_first_token_ms or 0
                        for req in total_with_output_first
                    ],
                ),
                time_per_output_token_ms=StatusDistributionSummary.from_values(
                    value_types=list(total_types_with_output_first),
                    values=[
                        req.time_per_output_token_ms or 0
                        for req in total_with_output_first
                    ],
                    weights=[req.output_tokens for req in total_with_output_first],
                ),
                inter_token_latency_ms=StatusDistributionSummary.from_values(
                    value_types=list(total_types_with_output_multi),
                    values=[
                        req.inter_token_latency_ms or 0
                        for req in total_with_output_multi
                    ],
                    weights=[req.output_tokens - 1 for req in total_with_output_multi],
                ),
                output_tokens_per_second=StatusDistributionSummary.from_iterable_request_times(
                    request_types=list(total_types_with_output_first),
                    requests=[
                        (req.start_time, req.end_time)
                        for req in total_with_output_first
                    ],
                    first_iter_times=[
                        req.first_token_time or req.start_time
                        for req in total_with_output_first
                    ],
                    iter_counts=[req.output_tokens for req in total_with_output_first],
                ),
                tokens_per_second=StatusDistributionSummary.from_iterable_request_times(
                    request_types=list(total_types_with_output_first),
                    requests=[
                        (req.start_time, req.end_time)
                        for req in total_with_output_first
                    ],
                    first_iter_times=[
                        req.first_token_time or req.start_time
                        for req in total_with_output_first
                    ],
                    iter_counts=[req.output_tokens for req in total_with_output_first],
                    first_iter_counts=[
                        # prompt tokens + first token
                        req.prompt_tokens + 1
                        for req in total_with_output_first
                    ],
                ),
            ),
            request_totals=StatusBreakdown(
                successful=len(successful),
                incomplete=len(incomplete),
                errored=len(errored),
                total=len(total),
            ),
            requests=StatusBreakdown(
                successful=successful,
                incomplete=incomplete,
                errored=errored,
            ),
        )
