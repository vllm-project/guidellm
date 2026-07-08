"""
Benchmark execution orchestration and lifecycle management.

Provides the core benchmarking engine that coordinates request scheduling,
data aggregation, and result compilation across execution strategies and
environments. The Benchmarker manages the complete benchmark lifecycle from
request submission through result compilation while implementing thread-safe
singleton operations for consistent state management across concurrent workflows.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, ABCMeta
from collections.abc import AsyncIterator
from typing import Generic

from disdantic import InfoMixin, SingletonMeta

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

__all__ = ["Benchmarker"]


class BenchmarkerMeta(ABCMeta, SingletonMeta): ...


class Benchmarker(
    Generic[BenchmarkT, RequestT, ResponseT],
    ABC,
    metaclass=BenchmarkerMeta,
):
    """
    Orchestrates benchmark execution across scheduling strategies.

    Coordinates benchmarking runs by managing request scheduling, metric aggregation,
    and result compilation. Implements a thread-safe singleton pattern to ensure
    consistent state management across concurrent operations while supporting multiple
    scheduling strategies and execution environments.
    """

    init_lock = threading.Lock()

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
        with self.init_lock:
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
