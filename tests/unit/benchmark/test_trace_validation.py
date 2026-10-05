"""Tests for post-run trace replay fidelity warnings."""

from __future__ import annotations

import pytest

from guidellm.benchmark.trace_validation import (
    TRACE_LATE_CONVERSATION_TOLERANCE_SEC,
    trace_replay_fidelity_warnings,
    validate_trace_replay,
)
from guidellm.scheduler.schemas.state import (
    SchedulerState,
    TraceConversationArrival,
)
from guidellm.scheduler.strategies import SynchronousStrategy, TraceReplayStrategy


def _state(
    *,
    start: float = 100.0,
    end_queuing: float | None = 110.0,
    arrivals: list[tuple[float, float]] | None = None,
) -> SchedulerState:
    return SchedulerState(
        start_time=start,
        end_queuing_time=end_queuing,
        trace_conversation_arrivals=[
            TraceConversationArrival(scheduled_offset=offset, queued_at=queued_at)
            for offset, queued_at in (arrivals or [])
        ],
    )


@pytest.mark.sanity
def test_on_time_replay_is_silent() -> None:
    """
    A replay that enqueues every due conversation on time does not warn.

    ## WRITTEN BY AI ##
    """
    unscheduled, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(),
        state=_state(arrivals=[(0.0, 100.0), (2.0, 102.0)]),
        start_offsets=[0.0, 2.0, 50.0],
        samples=-1,
    )

    assert unscheduled == 0
    assert warnings == []


@pytest.mark.sanity
def test_missing_due_conversations_warn() -> None:
    """
    Conversations scheduled before queuing stopped must be enqueued.

    ## WRITTEN BY AI ##
    """
    unscheduled, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(),
        state=_state(arrivals=[(0.0, 100.0)]),
        start_offsets=[0.0, 1.0, 2.0],
        samples=-1,
    )

    assert unscheduled == 2
    assert warnings == [
        "Trace replay sent 1 of 3 conversations due by the time queuing "
        "stopped. Prompt generation fell behind; cap max_context_len for "
        "large traces."
    ]


@pytest.mark.sanity
def test_late_enqueue_warns_with_max_delay() -> None:
    """
    Queuing a conversation after its scheduled start taints the replay.

    ## WRITTEN BY AI ##
    """
    _, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(time_scale=2.0),
        state=_state(
            start=100.0,
            end_queuing=200.0,
            arrivals=[(1.0, 100.0 + 2.0 + 18.2)],
        ),
        start_offsets=[1.0],
        samples=-1,
    )

    assert len(warnings) == 1
    assert "queued 1 conversations" in warnings[0]
    assert "max 18.2s" in warnings[0]


@pytest.mark.sanity
def test_jitter_within_tolerance_is_silent() -> None:
    """
    Enqueue lateness inside the jitter tolerance is not a warning.

    ## WRITTEN BY AI ##
    """
    queued_at = 101.0 + TRACE_LATE_CONVERSATION_TOLERANCE_SEC
    _, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(),
        state=_state(arrivals=[(1.0, queued_at)]),
        start_offsets=[1.0],
        samples=-1,
    )

    assert warnings == []


@pytest.mark.sanity
def test_conversations_after_the_window_are_not_due() -> None:
    """
    A conversation scheduled after queuing stopped is not counted as missed.

    ## WRITTEN BY AI ##
    """
    _, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(),
        state=_state(end_queuing=105.0, arrivals=[(0.0, 100.0)]),
        start_offsets=[0.0, 20.0],
        samples=-1,
    )

    assert warnings == []


@pytest.mark.sanity
def test_samples_cap_limits_the_expected_prefix() -> None:
    """
    Only the first ``samples`` conversations are expected from the loader.

    ## WRITTEN BY AI ##
    """
    _, warnings = trace_replay_fidelity_warnings(
        strategy=TraceReplayStrategy(),
        state=_state(arrivals=[(0.0, 100.0)]),
        start_offsets=[0.0, 1.0, 2.0],
        samples=1,
    )

    assert warnings == []


@pytest.mark.sanity
def test_non_replay_strategy_is_skipped() -> None:
    """
    Profiles that do not replay a trace arrival schedule are not checked.

    ## WRITTEN BY AI ##
    """
    unscheduled, warnings = trace_replay_fidelity_warnings(
        strategy=SynchronousStrategy(),
        state=_state(arrivals=[]),
        start_offsets=[0.0, 1.0],
        samples=-1,
    )

    assert unscheduled is None
    assert warnings == []


@pytest.mark.smoke
def test_validate_trace_replay_ignores_unrelated_loaders() -> None:
    """
    Loaders that are not a single trace dataset produce no warnings.

    ## WRITTEN BY AI ##
    """
    # The loader is rejected before the benchmark object is read.
    assert validate_trace_replay(benchmark=object(), loader=object()) == (None, [])  # type: ignore[arg-type]
