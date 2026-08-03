"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across execution strategies and
environments. The Benchmarker manages the complete benchmark lifecycle from
request submission through result compilation while implementing thread-safe
singleton operations for consistent state management across concurrent workflows.
"""

from __future__ import annotations

import uuid
from abc import ABC
from collections.abc import AsyncIterator
from typing import Generic

from guidellm.benchmark.profiles import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
)
from guidellm.benchmark.schemas.base import TransientPhaseConfig
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
)
from guidellm.tracing import start_span
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

    async def run(  # noqa: C901
        self,
        accumulator_class: type[BenchmarkAccumulatorT],
        benchmark_class: type[BenchmarkT],
        requests: DatasetIterT[RequestT],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        warmup: TransientPhaseConfig,
        cooldown: TransientPhaseConfig,
        sample_size: int | None = None,
        prefer_response_metrics: bool = True,
        progress: (
            BenchmarkerProgress[BenchmarkAccumulatorT, BenchmarkT] | None
        ) = None,
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
        :param sample_size: Maximum number of requests per status group
            (completed, errored, incomplete) to retain full data for.
            None keeps all, 0 strips all, N > 0 uses reservoir sampling.
        :param prefer_response_metrics: Whether to prefer response metrics over
            request metrics, defaults to True
        :param progress: Optional tracker for benchmark lifecycle events
        :yield: Compiled benchmark result for each strategy execution
        :raises Exception: If benchmark execution or compilation fails
        """
        with self.thread_lock:
            if progress:
                await progress.on_initialize(profile)

            run_id = str(uuid.uuid4())
            run_span = start_span(
                "guidellm.run",
                {
                    "guidellm.run.id": run_id,
                    "guidellm.profile.type": profile.kind,
                },
            )
            strategies_generator = profile.strategies_generator()
            strategy: SchedulingStrategy | None
            constraints: dict[str, Constraint] | None
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                if progress:
                    await progress.on_benchmark_start(strategy)

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
                    profile=InfoMixin.extract_from_obj(profile),
                    requests=InfoMixin.extract_from_obj(requests),
                    backend=InfoMixin.extract_from_obj(backend),
                    environment=InfoMixin.extract_from_obj(environment),
                )
                accumulator = accumulator_class(config=config)
                benchmark_span = start_span(
                    "guidellm.benchmark",
                    {
                        "guidellm.run.id": run_id,
                        "guidellm.benchmark.id": config.id_,
                        "guidellm.run.index": config.run_index,
                        "guidellm.strategy.type": strategy.type_,
                        "guidellm.backend.kind": config.backend.get("kind"),
                        "server.address": config.backend.get("target"),
                    },
                )
                scheduler_state = None
                scheduler: Scheduler[RequestT, ResponseT] = Scheduler()
                try:
                    async for (
                        response,
                        request,
                        request_info,
                        scheduler_state,
                    ) in scheduler.run(
                        requests=requests,
                        backend=backend,
                        strategy=strategy,
                        env=environment,
                        **constraints or {},
                    ):
                        try:
                            accumulator.update_estimate(
                                response,
                                request,
                                request_info,
                                scheduler_state,
                            )
                            if progress:
                                await progress.on_benchmark_update(
                                    accumulator, scheduler_state
                                )
                        except Exception as err:  # noqa: BLE001
                            logger.error(
                                "Error updating benchmark estimate/progress: {}", err
                            )

                    benchmark = benchmark_class.compile(
                        accumulator=accumulator,
                        scheduler_state=scheduler_state,  # type: ignore[arg-type]
                    )
                except BaseException as error:
                    benchmark_span.end(error)
                    run_span.end(error)
                    raise
                else:
                    benchmark_span.end()

                try:
                    if progress:
                        await progress.on_benchmark_complete(benchmark)

                    yield benchmark
                except BaseException as error:
                    run_span.end(error)
                    raise

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    strategy = None
                    constraints = None

            if progress:
                await progress.on_finalize()
            run_span.end()
