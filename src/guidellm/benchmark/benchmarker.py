"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across different execution strategies
and environments.

Classes:
    Benchmarker: Abstract benchmark orchestrator for request processing workflows.

Type Variables:
    BenchmarkT: Generic benchmark result type.
    RequestT: Generic request object type.
    RequestTimingsT: Generic request timing object type.
    ResponseT: Generic response object type.
"""

from __future__ import annotations

import uuid
from abc import ABC
from collections.abc import AsyncIterator, Iterable
from typing import Generic

from guidellm.benchmark.profile import Profile
from guidellm.benchmark.schemas import (
    BenchmarkArgs,
    BenchmarkT,
    EstimatedBenchmarkState,
)
from guidellm.scheduler import (
    BackendInterface,
    Environment,
    NonDistributedEnvironment,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
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

    Coordinates the execution of benchmarking runs across different scheduling
    strategies, aggregating metrics and compiling results. Manages the complete
    benchmark lifecycle from request submission through result compilation.

    Implements thread-safe singleton pattern to ensure consistent state across
    concurrent benchmark operations.
    """

    async def run(
        self,
        benchmark_class: type[BenchmarkT],
        requests: Iterable[RequestT | Iterable[RequestT | tuple[RequestT, float]]],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        environment: Environment | None = None,
        sample_requests: int | None = 20,
        warmup: float | None = None,
        cooldown: float | None = None,
        prefer_response_metrics: bool = True,
    ) -> AsyncIterator[
        tuple[
            EstimatedBenchmarkState | None,
            BenchmarkT | None,
            SchedulingStrategy,
            SchedulerState | None,
        ]
    ]:
        """
        Execute benchmark runs across multiple scheduling strategies.

        Orchestrates the complete benchmark workflow: iterates through scheduling
        strategies from the profile, executes requests through the scheduler,
        aggregates metrics, and compiles final benchmark results.

        :param requests: Request datasets for processing across strategies.
        :param backend: Backend interface for request processing.
        :param profile: Benchmark profile defining strategies and constraints.
        :param environment: Execution environment for coordination.
        :param benchmark_aggregators: Metric aggregation functions by name.
        :param benchmark_class: Class for constructing final benchmark objects.
        :yield: Tuples of (metrics_update, benchmark_result, strategy, state).
        :raises Exception: If benchmark execution or compilation fails.
        """
        with self.thread_lock:
            if environment is None:
                environment = NonDistributedEnvironment()

            run_id = str(uuid.uuid4())
            strategies_generator = profile.strategies_generator()
            strategy, constraints = next(strategies_generator)

            while strategy is not None:
                yield None, None, strategy, None
                args = BenchmarkArgs(
                    run_id=run_id,
                    run_index=len(profile.completed_strategies),
                    sample_requests=sample_requests,
                    warmup=warmup,
                    cooldown=cooldown,
                    prefer_response_metrics=prefer_response_metrics,
                )
                estimated_state = EstimatedBenchmarkState()
                scheduler_state = None

                async for (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) in Scheduler[RequestT, ResponseT]().run(
                    requests=requests,
                    backend=backend,
                    strategy=strategy,
                    env=environment,
                    **constraints,
                ):
                    benchmark_class.update_estimate(
                        args,
                        estimated_state,
                        response,
                        request,
                        request_info,
                        scheduler_state,
                    )
                    yield estimated_state, None, strategy, scheduler_state

                if scheduler_state is None:
                    raise RuntimeError("Scheduler state is None after execution.")

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
                )
                yield None, benchmark, strategy, None

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    strategy = None
                    constraints = None
