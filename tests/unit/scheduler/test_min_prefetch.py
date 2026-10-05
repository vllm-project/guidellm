from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from guidellm.scheduler import MaxDurationConstraint, SynchronousStrategy
from guidellm.scheduler.schemas.state import TraceConversationArrival
from guidellm.scheduler.worker_group import (
    WorkerGroupState,
    WorkerProcessGroup,
)
from guidellm.schemas.scheduler import MaxDurationConstraintArgs


def _state(
    stop: threading.Event | None = None,
) -> WorkerGroupState:
    return WorkerGroupState(
        start_time=time.time() - 10,
        processes=[],
        strategy=SynchronousStrategy(),
        constraints={
            "max_duration": MaxDurationConstraint(
                args=MaxDurationConstraintArgs(seconds=0.01)
            ).create_constraint(),
        },
        stop_send_requests_event=stop or threading.Event(),
        send_requests_stopped_event=threading.Event(),
        requests_generated_event=multiprocessing.Event(),
        constraint_reached_event=multiprocessing.Event(),
        shutdown_event=multiprocessing.Event(),
        error_event=multiprocessing.Event(),
        messaging=None,  # type: ignore[arg-type]  # state methods under test do not send
    )


@pytest.mark.sanity
def test_min_prefetch_zero_does_not_wait():
    """
    A zero count is satisfied before any conversation is built.

    ## WRITTEN BY AI ##
    """
    state = _state()
    assert state.prefetch_ready(0)


@pytest.mark.sanity
def test_min_prefetch_all_waits_until_generator_finishes():
    """
    -1 stays unsatisfied until the generator is done, then starts.

    ## WRITTEN BY AI ##
    """
    stop = threading.Event()
    state = _state(stop)
    state.update_state(generation_delay=0.0)
    state.update_state(generation_delay=0.0)
    assert not state.prefetch_ready(-1)
    stop.set()
    assert state.prefetch_ready(-1)


@pytest.mark.sanity
def test_min_prefetch_past_end_starts_when_generator_finishes():
    """
    A count larger than the dataset is satisfied once the generator ends.

    ## WRITTEN BY AI ##
    """
    stop = threading.Event()
    state = _state(stop)
    state.update_state(generation_delay=0.0)
    assert not state.prefetch_ready(5)
    stop.set()
    assert state.prefetch_ready(5)


@pytest.mark.sanity
def test_min_prefetch_counts_conversations_without_trace_arrivals():
    """
    A positive count is met by built conversations that are not trace arrivals.

    ## WRITTEN BY AI ##
    """
    state = _state()
    assert not state.prefetch_ready(1)
    state.update_state(generation_delay=0.0)
    assert state.prefetch_ready(1)
    assert state.conversations_built() == 1


@pytest.mark.regression
def test_arrival_is_queued_before_the_delayed_start():
    """
    A conversation built during prefetch is recorded before the clock starts.

    ## WRITTEN BY AI ##
    """
    state = _state()
    state.hold_clock()
    queued_at = time.time()
    update = state.update_state(
        generation_delay=0.0,
        trace_arrival=TraceConversationArrival(
            scheduled_offset=0.0,
            queued_at=queued_at,
        ),
    )
    assert update.stop_queueing is False
    assert state.prefetch_ready(1)
    start_time = time.time() + 1.0
    state.set_start_time(start_time)
    assert queued_at < state._state.start_time


@pytest.mark.regression
def test_generator_end_unblocks_a_larger_prefetch():
    """
    Finishing the generator unblocks a prefetch count the dataset cannot reach.

    ## WRITTEN BY AI ##
    """
    stop = threading.Event()
    state = _state(stop)
    state.update_state(
        generation_delay=0.0,
        trace_arrival=TraceConversationArrival(
            scheduled_offset=0.0,
            queued_at=time.time(),
        ),
    )
    assert not state.prefetch_ready(4)
    stop.set()
    assert state.prefetch_ready(4)
    assert state.prefetch_ready(-1)


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_wait_returns_when_the_generator_finishes():
    """
    The clock wait returns once the generator finishes, including for -1.

    ## WRITTEN BY AI ##
    """
    stop = threading.Event()
    state = _state(stop)
    error_event = multiprocessing.Event()

    async def finish() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    task = asyncio.create_task(finish())
    group = WorkerProcessGroup.__new__(WorkerProcessGroup)
    group.state = state
    group.error_event = error_event
    group._worker_error_details = None
    await group._wait_for_min_prefetch(-1)
    await task


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_min_prefetch_zero_does_not_call_the_callback():
    """
    Starting immediately does not report dataset-load progress.

    ## WRITTEN BY AI ##
    """
    called = False

    async def on_prefetch(built: int, target: int | None) -> None:
        nonlocal called
        called = True
        _ = (built, target)

    group = WorkerProcessGroup.__new__(WorkerProcessGroup)
    group.processes = [object()]  # type: ignore[list-item]
    group.requests_generated_event = threading.Event()  # type: ignore[assignment]
    group.constraint_reached_event = threading.Event()  # type: ignore[assignment]
    group.shutdown_event = threading.Event()  # type: ignore[assignment]
    group.error_event = threading.Event()  # type: ignore[assignment]
    group.messaging = SimpleNamespace(start=AsyncMock())  # type: ignore[assignment]
    group.strategy = SimpleNamespace(  # type: ignore[assignment]
        init_processes_start=lambda start_time: None
    )
    group.constraints = {}
    group.requests = iter(())
    group._worker_error_details = None
    # The events above stand in for multiprocessing primitives; start() only
    # stores them when min_prefetch is 0.
    await group.start(
        time.time() - 1,
        min_prefetch=0,
        on_prefetch=on_prefetch,
    )
    assert called is False


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_wait_reports_arrivals_and_an_open_target(monkeypatch):
    """
        The wait reports each newly built conversation, with no target when prefetch is -1.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(
        "guidellm.scheduler.worker_group.asyncio.sleep",
        AsyncMock(),
    )
    stop = threading.Event()
    state = _state(stop)
    reported: list[tuple[int, int | None]] = []

    async def on_prefetch(built: int, target: int | None) -> None:
        reported.append((built, target))

    added = False

    def ready(_min_prefetch: int) -> bool:
        nonlocal added
        if not added:
            state.update_state(generation_delay=0.0)
            added = True
            return False
        return True

    state.prefetch_ready = ready  # type: ignore[method-assign]
    group = WorkerProcessGroup.__new__(WorkerProcessGroup)
    group.state = state
    group.error_event = multiprocessing.Event()
    group._worker_error_details = None
    await group._wait_for_min_prefetch(-1, on_prefetch)
    assert reported == [(0, None), (1, None)]


@pytest.mark.regression
@pytest.mark.asyncio
async def test_wait_logs_again_after_ten_seconds_with_the_same_count(monkeypatch):
    """
    Prefetch logs at the start and again after 10 seconds if nothing new is built.

    ## WRITTEN BY AI ##
    """
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "guidellm.scheduler.worker_group.time.monotonic",
        lambda: clock["t"],
    )
    monkeypatch.setattr(
        "guidellm.scheduler.worker_group.asyncio.sleep",
        AsyncMock(),
    )
    messages: list[str] = []

    def info(_template: str, text: str) -> None:
        messages.append(text)

    monkeypatch.setattr("guidellm.scheduler.worker_group.logger.info", info)
    state = _state()
    checks = {"n": 0}

    def ready(_min_prefetch: int) -> bool:
        checks["n"] += 1
        if checks["n"] == 2:
            clock["t"] = 10.0
        return checks["n"] >= 3

    state.prefetch_ready = ready  # type: ignore[method-assign]
    group = WorkerProcessGroup.__new__(WorkerProcessGroup)
    group.state = state
    group.error_event = multiprocessing.Event()
    group._worker_error_details = None
    await group._wait_for_min_prefetch(4)
    assert messages == [
        "Loading dataset: 0/4 conversations",
        "Loading dataset: 0/4 conversations",
    ]
