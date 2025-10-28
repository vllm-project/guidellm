"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across different execution strategies
and environments. The Benchmarker acts as the primary workflow coordinator,
managing the complete benchmark lifecycle from request submission through
result compilation while supporting thread-safe singleton operations.
"""

from __future__ import annotations

import uuid
from abc import ABC
from collections.abc import AsyncIterator, Iterable
from typing import Generic

from guidellm.benchmark.profile import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkerArgs,
    BenchmarkT,
    EstimatedBenchmarkState,
)
from guidellm.logger import logger
from guidellm.scheduler import (
    BackendInterface,
    Environment,
    RequestT,
    ResponseT,
    Scheduler,
)
from guidellm.utils import ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


class Benchmarker(
    Generic[BenchmarkT, RequestT, ResponseT],
    ABC,
    ThreadSafeSingletonMixin,
):
    """
    Abstract benchmark orchestrator for request processing workflows.

    Coordinates execution of benchmarking runs across different scheduling
    strategies, aggregating metrics and compiling results. Manages the complete
    benchmark lifecycle from request submission through result compilation while
    implementing thread-safe singleton pattern to ensure consistent state across
    concurrent operations.
    """

    async def run(
        self,
        benchmark_class: type[BenchmarkT],
        requests: Iterable[RequestT | Iterable[RequestT | tuple[RequestT, float]]],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        environment: Environment,
        data: list[Any],
        progress: BenchmarkerProgress[BenchmarkT] | None = None,
        sample_requests: int | None = 20,
        warmup: float | None = None,
        cooldown: float | None = None,
        prefer_response_metrics: bool = True,
    ) -> AsyncIterator[BenchmarkT]:
        """
        Execute benchmark runs across multiple scheduling strategies.

        Orchestrates the complete benchmark workflow by iterating through scheduling
        strategies from the profile, executing requests through the scheduler,
        aggregating metrics, and compiling final benchmark results.

        :param benchmark_class: Class for constructing final benchmark objects
        :param requests: Request datasets for processing across strategies
        :param backend: Backend interface for request processing
        :param profile: Benchmark profile defining strategies and constraints
        :param environment: Execution environment for coordination
        :param progress: Optional progress tracker for benchmark lifecycle events
        :param sample_requests: Number of sample requests to use for estimation
        :param warmup: Optional warmup duration in seconds before benchmarking
        :param cooldown: Optional cooldown duration in seconds after benchmarking
        :param prefer_response_metrics: Whether to prefer response-based metrics over
            request-based metrics
        :yield: Compiled benchmark results for each strategy execution
        :raises Exception: If benchmark execution or compilation fails
        """
        with self.thread_lock:
            if progress:
                await progress.on_initialize(profile)

            run_id = str(uuid.uuid4())
            strategies_generator = profile.strategies_generator()
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                if progress:
                    await progress.on_benchmark_start(strategy)

                args = BenchmarkerArgs(
                    run_id=run_id,
                    run_index=len(profile.completed_strategies),
                    sample_requests=sample_requests,
                    warmup=warmup,
                    cooldown=cooldown,
                    prefer_response_metrics=prefer_response_metrics,
                )
                estimated_state = EstimatedBenchmarkState()
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
                        benchmark_class.update_estimate(
                            args,
                            estimated_state,
                            response,
                            request,
                            request_info,
                            scheduler_state,
                        )
                        if progress:
                            await progress.on_benchmark_update(
                                estimated_state, scheduler_state
                            )
                    except Exception as err:  # noqa: BLE001
                        logger.error(
                            f"Error updating benchmark estimate/progress: {err}"
                        )

                benchmark = benchmark_class.compile(
                    args=args,
                    estimated_state=estimated_state,
                    scheduler_state=scheduler_state,
                    profile=profile,
                    requests=requests,
                    backend=backend,
                    environment=environment,
                    strategy=strategy,
                    constraints=constraints,
                    data=data,
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
