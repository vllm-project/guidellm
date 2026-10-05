"""Post-run checks that a trace replay kept the trace's arrival schedule.

Prompt generation for a large conversation can finish after that conversation
was supposed to start, or after queuing has already stopped. These checks
compare the timing-only schedule captured at dataset load with the
conversations the scheduler actually enqueued.
"""

from __future__ import annotations

from guidellm.benchmark.schemas.benchmark import GenerativeBenchmark
from guidellm.data.deserializers.trace_common import TraceDataset
from guidellm.data.loaders.torch import DatasetsIterator, TorchDataLoader
from guidellm.logger import logger
from guidellm.scheduler.schemas.state import SchedulerState
from guidellm.scheduler.strategies import SchedulingStrategy, TraceReplayStrategy

__all__ = [
    "TRACE_LATE_CONVERSATION_TOLERANCE_SEC",
    "apply_trace_replay_warnings",
    "trace_replay_fidelity_warnings",
    "validate_trace_replay",
]

TRACE_LATE_CONVERSATION_TOLERANCE_SEC = 0.1
"""Enqueue lateness, in seconds, ignored as scheduler jitter."""


def apply_trace_replay_warnings(benchmark: GenerativeBenchmark, loader: object) -> None:
    """Store and log trace replay fidelity warnings for one benchmark.

    Sets ``unscheduled_conversations`` when the fidelity check runs. Leaves
    it unset when the run is not a single trace replay.

    :param benchmark: Compiled benchmark to annotate
    :param loader: Request loader used for the run
    """
    unscheduled, warnings = validate_trace_replay(benchmark, loader)
    if unscheduled is not None:
        benchmark.unscheduled_conversations = unscheduled
    benchmark.warnings.extend(warnings)
    for message in warnings:
        logger.warning("{}", message)


def validate_trace_replay(
    benchmark: GenerativeBenchmark, loader: object
) -> tuple[int | None, list[str]]:
    """Warn when a replay missed due conversations or queued them late.

    No-op unless the strategy is trace replay and the loader holds exactly
    one trace dataset. The run is not failed.

    :param benchmark: Compiled benchmark, including scheduler state
    :param loader: Request loader used for the run
    :return: Unscheduled count and warning strings. The count is ``None``
        when the check does not run, and ``0`` when every due conversation
        was enqueued.
    """
    dataset, samples = _trace_schedule(loader)
    if dataset is None:
        return None, []
    return trace_replay_fidelity_warnings(
        strategy=benchmark.config.strategy,
        state=benchmark.scheduler_state,
        start_offsets=dataset.replay_start_offsets,
        samples=samples,
    )


def trace_replay_fidelity_warnings(
    *,
    strategy: SchedulingStrategy,
    state: SchedulerState,
    start_offsets: list[float],
    samples: int,
) -> tuple[int | None, list[str]]:
    """Compare a replay schedule with the conversations that were enqueued.

    A conversation is due when its scheduled start is at or before queuing
    stopped. Emission is in schedule order, so a shortfall is
    ``due_count - enqueued_count``. ``samples`` greater than zero keeps only
    the first that many offsets.

    :param strategy: Strategy that ran this benchmark
    :param state: Final scheduler state
    :param start_offsets: Scheduled root starts, in emission order
    :param samples: Loader sample cap, or a non-positive value for no cap
    :return: Unscheduled count and warning strings. The count is ``None``
        when the strategy is not trace replay, and ``0`` when every due
        conversation was enqueued.
    """
    if not isinstance(strategy, TraceReplayStrategy):
        return None, []

    offsets = start_offsets
    if samples > 0:
        offsets = offsets[:samples]
    due = _due_offsets(offsets, state, strategy.time_scale)
    arrivals = state.trace_conversation_arrivals
    warnings: list[str] = []

    left_out = max(0, len(due) - len(arrivals))
    if left_out:
        sent = len(due) - left_out
        warnings.append(
            f"Trace replay sent {sent} of {len(due)} conversations due by the "
            "time queuing stopped. Prompt generation fell behind; cap "
            "max_context_len for large traces."
        )

    late = _late_delays(state, strategy.time_scale)
    if late:
        warnings.append(
            f"Trace replay queued {len(late)} conversations after their "
            f"scheduled start (max {max(late):.1f}s). The arrival pattern no "
            "longer matches the trace."
        )
    return left_out, warnings


def _trace_schedule(loader: object) -> tuple[TraceDataset | None, int]:
    """Return the single trace dataset on ``loader``, plus its sample cap.

    :param loader: Request loader used for the run
    :return: Dataset and sample cap, or ``(None, -1)`` when replay fidelity
        cannot be checked
    """
    if not isinstance(loader, TorchDataLoader):
        return None, -1
    source = loader.dataset
    if not isinstance(source, DatasetsIterator):
        return None, -1
    traces = [item for item in source.datasets if isinstance(item, TraceDataset)]
    if len(traces) != 1:
        return None, -1
    return traces[0], int(loader.info["samples"])


def _due_offsets(
    offsets: list[float],
    state: SchedulerState,
    time_scale: float,
) -> list[float]:
    """Keep offsets whose scheduled start falls before queuing stopped.

    :param offsets: Scheduled root starts, in emission order
    :param state: Final scheduler state
    :param time_scale: Replay profile scale applied on top of dataset offsets
    :return: Offsets that should have been enqueued
    """
    if state.end_queuing_time is None:
        return list(offsets)
    window = state.end_queuing_time - state.start_time
    limit = window / time_scale
    return [offset for offset in offsets if offset <= limit]


def _late_delays(state: SchedulerState, time_scale: float) -> list[float]:
    """Return enqueue lateness for conversations that missed their start.

    :param state: Final scheduler state
    :param time_scale: Replay profile scale applied on top of dataset offsets
    :return: Positive delays, in seconds, above the jitter tolerance
    """
    delays: list[float] = []
    for arrival in state.trace_conversation_arrivals:
        expected = state.start_time + time_scale * arrival.scheduled_offset
        delay = arrival.queued_at - expected
        if delay > TRACE_LATE_CONVERSATION_TOLERANCE_SEC:
            delays.append(delay)
    return delays
