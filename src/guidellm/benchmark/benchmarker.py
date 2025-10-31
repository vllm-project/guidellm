"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine coordinating request scheduling,
data aggregation, and result compilation across execution strategies
and environments. The Benchmarker manages the complete benchmark lifecycle
from request submission through result compilation while supporting
thread-safe singleton operations for consistent state management.
"""

from __future__ import annotations

import uuid
from abc import ABC
from collections.abc import AsyncIterator, Iterable
from typing import Generic

from guidellm.benchmark.profile import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
)
from guidellm.logger import logger
from guidellm.scheduler import (
    BackendInterface,
    Constraint,
    Environment,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulingStrategy,
)
from guidellm.utils import ThreadSafeSingletonMixin
from guidellm.utils.mixins import InfoMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[BenchmarkT, RequestT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    """
    Abstract benchmark orchestrator for request processing workflows.

    Coordinates benchmarking runs across scheduling strategies, aggregating
    metrics and compiling results. Manages the complete benchmark lifecycle
    from request submission through result compilation while implementing a
    thread-safe singleton pattern for consistent state across concurrent
    operations.
    """

    async def run(
        self,
        accumulator_class: type[BenchmarkAccumulatorT],
        benchmark_class: type[BenchmarkT],
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        progress: (
            BenchmarkerProgress[BenchmarkAccumulatorT, BenchmarkT] | None
        ) = None,
        sample_requests: int | None = 20,
        warmup: float | None = None,
        cooldown: float | None = None,
        prefer_response_metrics: bool = True,
    ) -> AsyncIterator[BenchmarkT]:
        """
        Execute benchmark runs across scheduling strategies defined in the profile.

        :param accumulator_class: Class for accumulating metrics during execution
        :param benchmark_class: Class for constructing final benchmark results
        :param requests: Request datasets to process across strategies
        :param backend: Backend interface for executing requests
        :param profile: Profile defining scheduling strategies and constraints
        :param environment: Environment for execution coordination
        :param progress: Optional tracker for benchmark lifecycle events
        :param sample_requests: Number of requests to sample for estimation
        :param warmup: Warmup duration in seconds before benchmarking
        :param cooldown: Cooldown duration in seconds after benchmarking
        :param prefer_response_metrics: Whether to prefer response metrics over
            request metrics
        :yield: Compiled benchmark result for each strategy execution
        :raises Exception: If benchmark execution or compilation fails
        """
        with self.thread_lock:
            if progress:
                await progress.on_initialize(profile)

            run_id = str(uuid.uuid4())
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
                    sample_requests=sample_requests,
                    warmup=warmup,
                    cooldown=cooldown,
                    prefer_response_metrics=prefer_response_metrics,
                    profile=profile,
                    requests=InfoMixin.extract_from_obj(requests),
                    backend=InfoMixin.extract_from_obj(backend),
                    environment=InfoMixin.extract_from_obj(environment),
                )
                accumulator = accumulator_class(config=config)
                scheduler_state = None
                scheduler: Scheduler[RequestT, ResponseT] = Scheduler()

                async for (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) in scheduler.run(
                    requests=requests,
                    backend=backend,
                    strategy=strategy,
                    startup_duration=warmup if warmup and warmup >= 1 else 0.0,
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
                            f"Error updating benchmark estimate/progress: {err}"
                        )

                benchmark = benchmark_class.compile(
                    accumulator=accumulator,
                    scheduler_state=scheduler_state,
                )

                if progress:
                    await progress.on_benchmark_complete(benchmark)

                yield benchmark

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    strategy = None
                    constraints = None

            if progress:
                await progress.on_finalize()
