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
from typing import (
    Any,
    Generic,
)

from guidellm.benchmark.aggregator import (
    Aggregator,
    AggregatorState,
    CompilableAggregator,
)
from guidellm.benchmark.objects import BenchmarkerDict, BenchmarkT, SchedulerDict
from guidellm.benchmark.profile import Profile
from guidellm.scheduler import (
    BackendInterface,
    Constraint,
    Environment,
    NonDistributedEnvironment,
    RequestT,
    ResponseT,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
)
from guidellm.utils import InfoMixin, ThreadSafeSingletonMixin
from guidellm.utils.pydantic_utils import StandardBaseDict

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
        requests: Iterable[RequestT | Iterable[RequestT | tuple[RequestT, float]]],
        backend: BackendInterface[RequestT, ResponseT],
        profile: Profile,
        benchmark_class: type[BenchmarkT],
        benchmark_aggregators: dict[
            str,
            Aggregator[ResponseT, RequestT] | CompilableAggregator[ResponseT, RequestT],
        ],
        environment: Environment | None = None,
    ) -> AsyncIterator[
        tuple[
            AggregatorState | None,
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
                aggregators_state = {
                    key: AggregatorState() for key in benchmark_aggregators
                }

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
                    aggregators_update = AggregatorState()
                    for key, aggregator in benchmark_aggregators.items():
                        update = aggregator(
                            aggregators_state[key],
                            response,
                            request,
                            request_info,
                            scheduler_state,
                        )
                        if update:
                            aggregators_update.update(update)
                    yield aggregators_update, None, strategy, scheduler_state

                benchmark_kwargs = self._compile_benchmark_kwargs(
                    run_id=run_id,
                    run_index=len(profile.completed_strategies),
                    profile=profile,
                    requests=requests,
                    backend=backend,
                    environment=environment,
                    aggregators=benchmark_aggregators,
                    aggregators_state=aggregators_state,
                    strategy=strategy,
                    constraints=constraints,
                    scheduler_state=scheduler_state,
                )
                benchmark = benchmark_class(**benchmark_kwargs)
                yield None, benchmark, strategy, None

                try:
                    strategy, constraints = strategies_generator.send(benchmark)
                except StopIteration:
                    strategy = None
                    constraints = None

    @classmethod
    def _compile_benchmark_kwargs(
        cls,
        run_id: str,
        run_index: int,
        profile: Profile,
        requests: Iterable[RequestT | Iterable[RequestT | tuple[RequestT, float]]],
        backend: BackendInterface[RequestT, ResponseT],
        environment: Environment,
        aggregators: dict[
            str,
            Aggregator[ResponseT, RequestT] | CompilableAggregator[ResponseT, RequestT],
        ],
        aggregators_state: dict[str, dict[str, Any]],
        strategy: SchedulingStrategy,
        constraints: dict[str, Any | dict[str, Any] | Constraint],
        scheduler_state: SchedulerState | None,
    ) -> dict[str, Any]:
        """
        Compile benchmark construction parameters from execution results.

        Aggregates metadata from scheduler execution and compiles it into
        structured parameters for benchmark object construction.

        :param run_id: Unique identifier for the benchmark run.
        :param run_index: Index of this strategy in the benchmark profile.
        :param profile: Benchmark profile containing strategy configuration.
        :param requests: Request datasets used for the benchmark.
        :param backend: Backend interface used for request processing.
        :param environment: Execution environment for coordination.
        :param aggregators: Metric aggregation functions by name.
        :param aggregators_state: Current state of metric aggregators.
        :param strategy: Scheduling strategy that was executed.
        :param constraints: Runtime constraints applied during execution.
        :param scheduler_state: Final state of scheduler execution.
        :return: Dictionary of parameters for benchmark object construction.
        :raises ValueError: If aggregator output conflicts with existing keys.
        """
        benchmark_kwargs = {
            "run_id": run_id,
            "run_index": run_index,
            "scheduler": SchedulerDict(
                strategy=strategy,
                constraints={
                    key: InfoMixin.extract_from_obj(val)
                    for key, val in constraints.items()
                },
                state=scheduler_state,
            ),
            "benchmarker": BenchmarkerDict(
                profile=profile,
                requests=InfoMixin.extract_from_obj(requests),
                backend=backend.info,
                environment=environment.info,
                aggregators={
                    key: InfoMixin.extract_from_obj(aggregator)
                    for key, aggregator in aggregators.items()
                },
            ),
            "env_args": StandardBaseDict(),
            "extras": StandardBaseDict(),
        }

        def _combine(
            existing: dict[str, Any] | StandardBaseDict,
            addition: dict[str, Any] | StandardBaseDict,
        ) -> dict[str, Any] | StandardBaseDict:
            if not isinstance(existing, (dict, StandardBaseDict)):
                raise ValueError(
                    f"Existing value {existing} (type: {type(existing).__name__}) "
                    f"is not a valid type for merging."
                )
            if not isinstance(addition, (dict, StandardBaseDict)):
                raise ValueError(
                    f"Addition value {addition} (type: {type(addition).__name__}) "
                    f"is not a valid type for merging."
                )

            add_kwargs = (
                addition if isinstance(addition, dict) else addition.model_dump()
            )

            if isinstance(existing, dict):
                return {**add_kwargs, **existing}

            return existing.__class__(**{**add_kwargs, **existing.model_dump()})

        for key, aggregator in aggregators.items():
            if not isinstance(aggregator, CompilableAggregator):
                continue

            compiled = aggregator.compile(aggregators_state[key], scheduler_state)

            for field_name, field_val in compiled.items():
                if field_name in benchmark_kwargs:
                    # If the key already exists, merge the values
                    benchmark_kwargs[field_name] = _combine(
                        benchmark_kwargs[field_name], field_val
                    )
                else:
                    benchmark_kwargs[field_name] = field_val

        return benchmark_kwargs
