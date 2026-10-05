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
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Generic, Literal

from guidellm.benchmark.profiles import Profile
from guidellm.benchmark.progress import BenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
)
from guidellm.data.deserializers.trace_common import TraceDataset
from guidellm.data.loaders.torch import (
    DatasetsIterator,
    TorchDataLoader,
    _reject_indefinite_full_prefetch,
)
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
from guidellm.scheduler.constraints.request import (
    MaxDurationConstraint,
    MaxNumberConstraint,
    MinNumberConstraint,
)
from guidellm.scheduler.strategies import TraceReplayStrategy
from guidellm.schemas.benchmark import GoodputSLO, TransientPhaseConfig
from guidellm.utils.mixins import InfoMixin
from guidellm.utils.singleton import ThreadSafeSingletonMixin

__all__ = ["Benchmarker"]


def _report_prefetch(
    progress: BenchmarkerProgress | None,
) -> Callable[[int, int | None], Awaitable[None]]:
    """
    Build the scheduler callback that updates the live load status.

    :param progress: Benchmark progress tracker, or ``None`` when display is off
    :return: Callback passed to the scheduler while the replay clock is held
    """

    async def on_prefetch(built: int, target: int | None) -> None:
        if progress is not None:
            await progress.on_prefetch(built, target)

    return on_prefetch


def _max_duration_seconds(constraints: dict[str, Constraint] | None) -> float | None:
    """
    Read the run's duration limit, in seconds.

    :param constraints: Constraints for the strategy about to run
    :return: The max-duration limit, or ``None`` when this run has none
    """
    if not constraints:
        return None
    for constraint in constraints.values():
        if not isinstance(constraint, MaxDurationConstraint):
            continue
        seconds = constraint.args.seconds
        if isinstance(seconds, int | float):
            return float(seconds)
        if not seconds:
            continue
        index = min(max(0, constraint.current_index), len(seconds) - 1)
        return float(seconds[index])
    return None


def _constraint_request_count(
    constraint: MaxNumberConstraint | MinNumberConstraint,
) -> int | None:
    """
    Read one request-count constraint at its current index.

    :param constraint: A max-requests or min-requests constraint
    :return: The count for this strategy, or ``None`` when the list is empty
    """
    count = constraint.args.count
    if isinstance(count, int | float):
        return int(count)
    if not count:
        return None
    index = min(max(0, constraint.current_index), len(count) - 1)
    return int(count[index])


def _scheduled_request_count(constraints: dict[str, Constraint] | None) -> int | None:
    """
    Read how many requests this run will send.

    A max-requests cap wins when both counts are set, because creation stops
    there. A min-requests count is used when it is the only request bound.

    :param constraints: Constraints for the strategy about to run
    :return: The request count to preload, or ``None`` when neither is set
    """
    if not constraints:
        return None
    minimum: int | None = None
    for constraint in constraints.values():
        if isinstance(constraint, MaxNumberConstraint):
            count = _constraint_request_count(constraint)
            if count is not None:
                return count
        elif isinstance(constraint, MinNumberConstraint):
            minimum = _constraint_request_count(constraint)
    return minimum


def _replay_prefetch_horizon(
    strategy: SchedulingStrategy,
    constraints: dict[str, Constraint] | None,
) -> float | None:
    """
    Convert a max-duration limit into scheduled-offset seconds.

    :param strategy: Strategy about to run; only replay scales the window
    :param constraints: Constraints for that strategy
    :return: Offset horizon, or ``None`` when the run has no duration limit
    """
    horizon = _max_duration_seconds(constraints)
    if (
        horizon is None
        or not isinstance(strategy, TraceReplayStrategy)
        or strategy.time_scale <= 0
    ):
        return horizon
    return horizon / strategy.time_scale


def _trace_prefetch_counts(
    requests: object,
    horizon: float | None,
) -> tuple[int, int | None]:
    """
    Count conversations that must be built before the replay clock starts.

    A duration limit counts every emitted conversation due inside that window.
    Without one, only conversations scheduled at time zero are counted.
    Exact zero misses a burst whose timestamps are a fraction of a second apart.

    :param requests: Request source passed to the benchmarker
    :param horizon: Scheduled-offset limit in seconds, or ``None`` for time zero
    :return: Opening count and schedule length, or ``(0, None)`` when the
        source is not one trace dataset
    """
    if not isinstance(requests, TorchDataLoader):
        return 0, None
    source = requests.dataset
    if not isinstance(source, DatasetsIterator):
        return 0, None
    traces = [item for item in source.datasets if isinstance(item, TraceDataset)]
    if len(traces) != 1:
        return 0, None
    offsets = list(traces[0].replay_start_offsets)
    samples = requests.info["samples"]
    if isinstance(samples, int) and samples > 0:
        offsets = offsets[:samples]
    limit = 0.0 if horizon is None else horizon
    opening = sum(1 for offset in offsets if offset <= limit)
    return opening, len(offsets)


def _resolve_profile_prefetch(
    min_prefetch: Literal["start", "scheduled"] | int | None,
    requests: object,
    strategy: SchedulingStrategy,
    constraints: dict[str, Constraint] | None,
) -> tuple[int | None, int, int | None]:
    """
    Turn a profile prefetch choice into the integer the scheduler waits on.

    ``start`` stays ``None`` so the scheduler still takes the greatest of
    workers, concurrency, and the time-zero trace count. ``scheduled`` uses a
    max-requests or min-requests count when one is set, and otherwise the
    duration window. A number is passed through and replaces that formula.

    :param min_prefetch: Profile choice
    :param requests: Request source passed to the benchmarker
    :param strategy: Strategy about to run
    :param constraints: Constraints for that strategy
    :return: Scheduler prefetch, trace count, and schedule length
    :raises ValueError: If explicit ``scheduled`` has neither a request cap
        nor a duration window on one trace dataset
    """
    if isinstance(min_prefetch, int):
        _zero_starts, schedule_length = _trace_prefetch_counts(requests, None)
        return min_prefetch, 0, schedule_length

    request_count = _scheduled_request_count(constraints)
    horizon = _replay_prefetch_horizon(strategy, constraints)
    # Explicit start ignores a request cap. Omit uses scheduled when either
    # cap can name what this run will send.
    mode = min_prefetch
    if mode is None and (request_count is not None or horizon is not None):
        mode = "scheduled"
    elif mode is None:
        mode = "start"
    if mode == "scheduled" and request_count is not None:
        _ignored, schedule_length = _trace_prefetch_counts(requests, None)
        return request_count, 0, schedule_length
    if mode == "scheduled" and horizon is None:
        raise ValueError(
            "min_prefetch=scheduled needs a max_requests, min_requests, or "
            "max_duration constraint. Use min_prefetch=start or a conversation "
            "count instead."
        )
    use_horizon = horizon if mode == "scheduled" else None
    zero_starts, schedule_length = _trace_prefetch_counts(requests, use_horizon)
    if schedule_length is None:
        if min_prefetch == "scheduled":
            raise ValueError(
                "min_prefetch=scheduled applies to a single trace dataset "
                "when neither max_requests nor min_requests is set."
            )
        # ``start`` and omit on a non-trace source wait for workers and the
        # strategy concurrency cap. There is no schedule to count.
        return None, 0, None
    return None, zero_starts, schedule_length


def _reject_full_prefetch(requests: object, min_prefetch: int | None) -> None:
    """
    Raise when a full-dataset prefetch waits on a source that never ends.

    :param requests: Request source passed to the benchmarker
    :param min_prefetch: Profile prefetch setting
    :raises ValueError: If prefetch waits for the end of an indefinite source
    """
    if not isinstance(min_prefetch, int) or min_prefetch >= 0:
        return
    if not isinstance(requests, TorchDataLoader):
        return
    source = requests.dataset
    if not isinstance(source, DatasetsIterator):
        return
    samples = requests.info["samples"]
    sample_count = samples if isinstance(samples, int) else -1
    _reject_indefinite_full_prefetch(min_prefetch, source.datasets, sample_count)


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
        sample_size: int | None = None,
        prefer_response_metrics: bool = True,
        progress: (
            BenchmarkerProgress[BenchmarkAccumulatorT, BenchmarkT] | None
        ) = None,
        slo: GoodputSLO | None = None,
        confidence: float | None = 0.95,
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
        :param slo: Per-request latency objectives defining which requests count
            toward goodput, or None to disable goodput measurement
        :param confidence: Two-sided confidence level for the intervals reported
            alongside request-level metrics, or None to omit them
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
                logger.info("Starting benchmark for strategy: {}", strategy)
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
                    slo=slo,
                    confidence=confidence,
                    profile=InfoMixin.extract_from_obj(profile),
                    requests=InfoMixin.extract_from_obj(requests),
                    backend=InfoMixin.extract_from_obj(backend),
                    environment=InfoMixin.extract_from_obj(environment),
                )
                accumulator = accumulator_class(config=config)
                scheduler_state = None
                scheduler: Scheduler[RequestT, ResponseT] = Scheduler()

                min_prefetch, zero_starts, schedule_length = _resolve_profile_prefetch(
                    profile.args.min_prefetch,
                    requests,
                    strategy,
                    constraints,
                )
                _reject_full_prefetch(requests, min_prefetch)
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
                    min_prefetch=min_prefetch,
                    zero_start_count=zero_starts,
                    schedule_length=schedule_length,
                    on_prefetch=_report_prefetch(progress),
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

            logger.info("All benchmarks finalized")
            if progress:
                await progress.on_finalize()
