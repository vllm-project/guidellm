import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Union,
)

from pydantic import Field
from transformers import PreTrainedTokenizerBase  # type: ignore  # noqa: PGH003

from guidellm.backend import Backend, ResponseSummary
from guidellm.benchmark.aggregator import (
    AggregatorT,
    BenchmarkT,
    GenerativeBenchmarkAggregator,
)
from guidellm.benchmark.benchmark import BenchmarkArgs, GenerativeBenchmark
from guidellm.benchmark.profile import Profile
from guidellm.objects import StandardBaseModel
from guidellm.request import (
    GenerationRequest,
    GenerativeRequestLoaderDescription,
    RequestLoaderDescription,
    RequestT,
    ResponseT,
)
from guidellm.scheduler import (
    GenerativeRequestsWorker,
    RequestsWorker,
    Scheduler,
    SchedulerRequestResult,
    SchedulingStrategy,
)

__all__ = ["Benchmarker", "BenchmarkerResult", "GenerativeBenchmarker"]


class BenchmarkerResult(
    StandardBaseModel, Generic[AggregatorT, BenchmarkT, RequestT, ResponseT]
):
    type_: Literal[
        "run_start",
        "run_complete",
        "scheduler_start",
        "scheduler_update",
        "scheduler_complete",
        "benchmark_compiled",
    ]
    start_time: float
    end_number: int
    profile: Profile
    current_index: int
    current_strategy: Optional[SchedulingStrategy] = None
    current_aggregator: Optional[AggregatorT] = None
    current_benchmark: Optional[BenchmarkT] = None
    current_result: Optional[SchedulerRequestResult[RequestT, ResponseT]] = None


class BenchmarkerStrategyLimits(StandardBaseModel):
    requests_loader_size: Optional[int] = Field(
        description="Size of the request loader.",
    )
    max_number_per_strategy: Optional[int] = Field(
        description="Maximum number of requests to process per strategy.",
        ge=0,
    )
    max_duration_per_strategy: Optional[float] = Field(
        description="Maximum duration (in seconds) to process requests per strategy.",
        ge=0,
    )
    warmup_percent_per_strategy: Optional[float] = Field(
        description="Percentage of requests to use for warmup.",
        ge=0,
        le=1,
    )
    cooldown_percent_per_strategy: Optional[float] = Field(
        description="Percentage of requests to use for cooldown.",
        ge=0,
        le=1,
    )

    @property
    def max_number(self) -> Optional[int]:
        if self.max_number_per_strategy is not None:
            return self.max_number_per_strategy

        if self.requests_loader_size is not None:
            return self.requests_loader_size

        return None

    @property
    def max_duration(self) -> Optional[float]:
        return self.max_duration_per_strategy

    @property
    def warmup_number(self) -> Optional[int]:
        if self.warmup_percent_per_strategy is None or self.max_number is None:
            return None

        return int(self.warmup_percent_per_strategy * self.max_number)

    @property
    def warmup_duration(self) -> Optional[float]:
        if self.warmup_percent_per_strategy is None or self.max_duration is None:
            return None

        return self.warmup_percent_per_strategy * self.max_duration

    @property
    def cooldown_number(self) -> Optional[int]:
        if self.cooldown_percent_per_strategy is None or self.max_number is None:
            return None

        return int(self.cooldown_percent_per_strategy * self.max_number)

    @property
    def cooldown_duration(self) -> Optional[float]:
        if self.cooldown_percent_per_strategy is None or self.max_duration is None:
            return None

        return self.cooldown_percent_per_strategy * self.max_duration


class Benchmarker(Generic[AggregatorT, BenchmarkT, RequestT, ResponseT], ABC):
    def __init__(
        self,
        worker: RequestsWorker[RequestT, ResponseT],
        request_loader: Iterable[RequestT],
        requests_loader_description: RequestLoaderDescription,
        benchmark_save_extras: Optional[dict[str, Any]] = None,
    ):
        self.worker = worker
        self.scheduler: Scheduler[RequestT, ResponseT] = Scheduler(
            worker=worker, request_loader=request_loader
        )
        self.requests_loader_description = requests_loader_description
        self.benchmark_save_extras = benchmark_save_extras

    async def run(
        self,
        profile: Profile,
        max_number_per_strategy: Optional[int],
        max_duration_per_strategy: Optional[float],
        warmup_percent_per_strategy: Optional[float],
        cooldown_percent_per_strategy: Optional[float],
    ) -> AsyncGenerator[
        BenchmarkerResult[AggregatorT, BenchmarkT, RequestT, ResponseT], None
    ]:
        try:
            requests_loader_size = len(self.scheduler.request_loader)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            requests_loader_size = None

        strategy_limits = BenchmarkerStrategyLimits(
            requests_loader_size=requests_loader_size,
            max_number_per_strategy=max_number_per_strategy,
            max_duration_per_strategy=max_duration_per_strategy,
            warmup_percent_per_strategy=warmup_percent_per_strategy,
            cooldown_percent_per_strategy=cooldown_percent_per_strategy,
        )
        start_time = time.time()
        end_number = len(profile.strategy_types)
        current_index = -1
        run_id = str(uuid.uuid4())

        yield BenchmarkerResult(
            type_="run_start",
            start_time=start_time,
            end_number=end_number,
            profile=profile,
            current_index=current_index,
            current_strategy=None,
            current_aggregator=None,
            current_benchmark=None,
            current_result=None,
        )

        while scheduling_strategy := profile.next_strategy():
            current_index += 1
            aggregator = self.create_benchmark_aggregator(
                run_id=run_id,
                profile=profile,
                strategy_index=current_index,
                strategy=scheduling_strategy,
                limits=strategy_limits,
            )

            async for result in self.scheduler.run(
                scheduling_strategy=scheduling_strategy,
                max_number=max_number_per_strategy,
                max_duration=max_duration_per_strategy,
            ):
                if result.type_ == "run_start":
                    yield BenchmarkerResult(
                        type_="scheduler_start",
                        start_time=start_time,
                        end_number=end_number,
                        profile=profile,
                        current_index=current_index,
                        current_strategy=scheduling_strategy,
                        current_aggregator=aggregator,
                        current_benchmark=None,
                        current_result=None,
                    )
                elif result.type_ == "run_complete":
                    yield BenchmarkerResult(
                        type_="scheduler_complete",
                        start_time=start_time,
                        end_number=end_number,
                        profile=profile,
                        current_index=current_index,
                        current_strategy=scheduling_strategy,
                        current_aggregator=aggregator,
                        current_benchmark=None,
                        current_result=None,
                    )
                elif isinstance(result, SchedulerRequestResult):
                    aggregator.add_result(result)

                    yield BenchmarkerResult(
                        type_="scheduler_update",
                        start_time=start_time,
                        end_number=end_number,
                        profile=profile,
                        current_index=current_index,
                        current_strategy=scheduling_strategy,
                        current_aggregator=aggregator,
                        current_benchmark=None,
                        current_result=result,
                    )
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")

            benchmark: BenchmarkT = aggregator.compile()
            profile.completed_strategy(
                average_rate=benchmark.metrics.requests_per_second.successful.mean,
                average_concurrency=benchmark.metrics.request_concurrency.successful.mean,
            )

            yield BenchmarkerResult(
                type_="benchmark_compiled",
                start_time=start_time,
                end_number=end_number,
                profile=profile,
                current_index=current_index,
                current_strategy=scheduling_strategy,
                current_aggregator=None,
                current_benchmark=benchmark,
                current_result=None,
            )

        yield BenchmarkerResult(
            type_="run_complete",
            start_time=start_time,
            end_number=end_number,
            profile=profile,
            current_index=current_index,
            current_strategy=None,
            current_aggregator=None,
            current_benchmark=None,
            current_result=None,
        )

    @abstractmethod
    def create_benchmark_aggregator(
        self,
        run_id: str,
        profile: Profile,
        strategy_index: int,
        strategy: SchedulingStrategy,
        limits: BenchmarkerStrategyLimits,
    ) -> AggregatorT: ...


class GenerativeBenchmarker(
    Benchmarker[
        GenerativeBenchmarkAggregator,
        GenerativeBenchmark,
        GenerationRequest,
        ResponseSummary,
    ],
):
    def __init__(
        self,
        backend: Backend,
        request_loader: Iterable[GenerationRequest],
        request_loader_description: GenerativeRequestLoaderDescription,
        benchmark_save_extras: Optional[dict[str, Any]] = None,
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None,
        processor_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            worker=GenerativeRequestsWorker(backend),
            request_loader=request_loader,
            requests_loader_description=request_loader_description,
            benchmark_save_extras=benchmark_save_extras,
        )
        self.processor = processor
        self.processor_args = processor_args

    def create_benchmark_aggregator(
        self,
        run_id: str,
        profile: Profile,
        strategy_index: int,
        strategy: SchedulingStrategy,
        limits: BenchmarkerStrategyLimits,
    ) -> GenerativeBenchmarkAggregator:
        return GenerativeBenchmarkAggregator(
            run_id=run_id,
            args=BenchmarkArgs(
                profile=profile,
                strategy_index=strategy_index,
                strategy=strategy,
                max_number=limits.max_number,
                max_duration=limits.max_duration,
                warmup_number=limits.warmup_number,
                warmup_duration=limits.warmup_duration,
                cooldown_number=limits.cooldown_number,
                cooldown_duration=limits.cooldown_duration,
            ),
            worker_description=self.worker.description,  # type: ignore[arg-type]
            request_loader_description=self.requests_loader_description,  # type: ignore[arg-type]
            extras=self.benchmark_save_extras or {},
            processor=self.processor,
            processor_args=self.processor_args,
        )
