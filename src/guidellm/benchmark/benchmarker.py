"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across execution strategies and
environments. The Benchmarker manages the complete benchmark lifecycle from
request submission through result compilation while implementing thread-safe
singleton operations for consistent state management across concurrent workflows.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from abc import ABC
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Generic

from guidellm.benchmark.profiles import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
)
from guidellm.collectors import CollectorProcess
from guidellm.logger import logger
from guidellm.scheduler import (
    BackendInterface,
    Constraint,
    DatasetIterT,
    Environment,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulingStrategy,
    prefetch_requests,
)
from guidellm.schemas.benchmark import GoodputSLO, TransientPhaseConfig
from guidellm.schemas.benchmark.warnings import (
    WarningRuleArgs,
    default_warning_rules,
)
from guidellm.utils.mixins import InfoMixin
from guidellm.utils.singleton import ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[BenchmarkT, RequestT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    """
    Orchestrates benchmark execution across scheduling strategies.

    Coordinates benchmarking runs by managing request scheduling, metric aggregation,
    and result compilation. Implements a thread-safe singleton pattern to ensure
    consistent state management across concurrent operations while supporting multiple
    scheduling strategies and execution environments.
    """

    async def run(
        self,
        accumulator_class: type[BenchmarkAccumulatorT],
        benchmark_class: type[BenchmarkT],
        requests: DatasetIterT[RequestT],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        warmup: TransientPhaseConfig,
        cooldown: TransientPhaseConfig,
        progress: BenchmarkerProgress[BenchmarkAccumulatorT, BenchmarkT],
        sample_size: int | None = None,
        prefer_response_metrics: bool = True,
        slo: GoodputSLO | None = None,
        confidence: float | None = 0.95,
        warnings: list[WarningRuleArgs] | None = None,
        collectors: CollectorProcess | None = None,
        process_response: Callable[[ResponseT | None], Awaitable[None]] | None = None,
    ) -> AsyncIterator[BenchmarkT]:
        """
        Execute benchmark runs across scheduling strategies in the profile.

        :param accumulator_class: Class for accumulating metrics during execution
        :param benchmark_class: Class for constructing final benchmark results
        :param requests: Request datasets to process across strategies
        :param backend: Backend interface for executing requests
        :param profile: Profile defining scheduling strategies and constraints
        :param environment: Environment for execution coordination
        :param warmup: Warmup phase configuration before benchmarking
        :param cooldown: Cooldown phase configuration after benchmarking
        :param progress: Tracker for benchmark lifecycle events
        :param sample_size: Maximum number of requests per status group
            (completed, errored, incomplete) to retain full data for.
            None keeps all, 0 strips all, N > 0 uses reservoir sampling.
        :param prefer_response_metrics: Whether to prefer response metrics over
            request metrics, defaults to True
        :param slo: Per-request latency objectives defining which requests count
            toward goodput, or None to disable goodput measurement
        :param confidence: Two-sided confidence level for the intervals reported
            alongside request-level metrics, or None to omit them
        :param warnings: Post-benchmark warning checks. None uses the defaults.
        :param collectors: Started collector process, told when each benchmark
            starts and stops, whose results are added to each benchmark
        :param process_response: Optional coroutine applied to each response
            before it is accumulated, such as filling in missing token counts
        :yield: Compiled benchmark result for each strategy execution
        :raises Exception: If benchmark execution or compilation fails
        """
        with self.thread_lock:
            await progress.on_initialize(profile)

            run_id = str(uuid.uuid4())
            strategies_generator = profile.strategies_generator()
            strategy: SchedulingStrategy | None
            constraints: dict[str, Constraint] | None
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                logger.info("Starting benchmark for strategy: {}", strategy)
                await progress.on_benchmark_start(strategy)

                prefetch_stop_event = threading.Event()
                try:
                    strategy_requests = await asyncio.to_thread(
                        prefetch_requests,
                        requests,
                        strategy.prefetch_count(),
                        on_fetched=progress.on_requests_prefetched,
                        stop_event=prefetch_stop_event,
                    )
                finally:
                    prefetch_stop_event.set()

                config = BenchmarkConfig(
                    run_id=run_id,
                    run_index=len(profile.completed_strategies),
                    strategy=strategy,
                    constraints=(
                        {
                            key: InfoMixin.extract_from_obj(val)
                            for key, val in constraints.items()
                        }
                        if isinstance(constraints, dict)
                        else {"constraint": InfoMixin.extract_from_obj(constraints)}
                        if constraints
                        else {}
                    ),
                    sample_size=sample_size,
                    warmup=warmup,
                    cooldown=cooldown,
                    prefer_response_metrics=prefer_response_metrics,
                    slo=slo,
                    confidence=confidence,
                    warnings=(
                        default_warning_rules() if warnings is None else warnings
                    ),
                    profile=InfoMixin.extract_from_obj(profile),
                    requests=InfoMixin.extract_from_obj(requests),
                    backend=InfoMixin.extract_from_obj(backend),
                    environment=InfoMixin.extract_from_obj(environment),
                )
                accumulator = accumulator_class(config=config)
                scheduler_state = None
                scheduler: Scheduler[RequestT, ResponseT] = Scheduler()

                if collectors:
                    await collectors.benchmark_started()
                try:
                    async for (
                        response,
                        request,
                        request_info,
                        scheduler_state,
                    ) in scheduler.run(
                        requests=strategy_requests,
                        backend=backend,
                        strategy=strategy,
                        env=environment,
                        **constraints or {},
                    ):
                        try:
                            if process_response:
                                await process_response(response)
                            accumulator.update_estimate(
                                response,
                                request,
                                request_info,
                                scheduler_state,
                            )
                            await progress.on_benchmark_update(
                                accumulator, scheduler_state
                            )
                        except Exception as err:  # noqa: BLE001
                            logger.error(
                                "Error updating benchmark estimate/progress: {}", err
                            )
                finally:
                    if collectors:
                        await collectors.benchmark_stopped()

                await progress.on_benchmark_compile()

                benchmark = benchmark_class.compile(
                    accumulator=accumulator,
                    scheduler_state=scheduler_state,  # type: ignore[arg-type]
                )
                if collectors:
                    benchmark = benchmark.model_copy(
                        update={
                            "collectors": await collectors.results(
                                benchmark.start_time, benchmark.end_time
                            )
                        }
                    )

                await progress.on_benchmark_complete(benchmark)

                yield benchmark

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    strategy = None
                    constraints = None

            logger.info("All benchmarks finalized")
            await progress.on_finalize()
