from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from guidellm.benchmark.benchmarker import (
    _replay_prefetch_horizon,
    _resolve_profile_prefetch,
    _trace_prefetch_counts,
)
from guidellm.data.deserializers.trace_common import TraceDataset
from guidellm.data.loaders.torch import DatasetsIterator, TorchDataLoader
from guidellm.scheduler import (
    MaxDurationConstraint,
    MaxNumberConstraint,
    MinNumberConstraint,
    SynchronousStrategy,
    TraceReplayStrategy,
)
from guidellm.scheduler.schemas.state import TraceConversationArrival
from guidellm.scheduler.worker_group import (
    WorkerGroupState,
    WorkerProcessGroup,
    prefetch_progress_target,
    resolve_min_prefetch,
)
from guidellm.schemas.scheduler import (
    MaxDurationConstraintArgs,
    MaxRequestsConstraintArgs,
    MinRequestsConstraintArgs,
)


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
def test_automatic_prefetch_uses_the_greatest_demand():
    """
    An omitted count waits for workers, concurrency, or time-zero conversations.

    ## WRITTEN BY AI ##
    """
    assert (
        resolve_min_prefetch(
            None,
            num_processes=10,
            requests_limit=None,
            zero_start_count=400,
        )
        == 400
    )
    assert (
        resolve_min_prefetch(
            None,
            num_processes=4,
            requests_limit=8,
            zero_start_count=0,
        )
        == 8
    )


@pytest.mark.sanity
def test_automatic_prefetch_ignores_the_global_concurrency_ceiling():
    """
    An uncapped strategy does not prefetch settings.max_concurrency.

    ## WRITTEN BY AI ##
    """
    assert (
        resolve_min_prefetch(
            None,
            num_processes=10,
            requests_limit=None,
            zero_start_count=0,
        )
        == 10
    )


@pytest.mark.sanity
def test_explicit_prefetch_replaces_the_formula():
    """
    A set count is used as given, including zero and a full-dataset wait.

    ## WRITTEN BY AI ##
    """
    assert (
        resolve_min_prefetch(
            3,
            num_processes=10,
            requests_limit=None,
            zero_start_count=400,
        )
        == 3
    )
    assert (
        resolve_min_prefetch(
            0,
            num_processes=10,
            requests_limit=8,
            zero_start_count=400,
        )
        == 0
    )
    assert (
        resolve_min_prefetch(
            -1,
            num_processes=10,
            requests_limit=None,
            zero_start_count=400,
        )
        == -1
    )


@pytest.mark.sanity
def test_progress_target_is_the_required_count():
    """
    The load display uses the resolved need, or the trace length for a full wait.

    ## WRITTEN BY AI ##
    """
    assert prefetch_progress_target(400, 900) == 400
    assert prefetch_progress_target(-1, 900) == 900
    assert prefetch_progress_target(-1, None) is None
    assert prefetch_progress_target(0, 900) is None


class _Trace(TraceDataset):
    def __init__(self, offsets: list[float]) -> None:
        self.replay_start_offsets = offsets


class _Loader(TorchDataLoader):
    def __init__(self, offsets: list[float], samples: int) -> None:
        self.dataset = DatasetsIterator.__new__(DatasetsIterator)
        self.dataset.datasets = [_Trace(offsets)]  # type: ignore[attr-defined]
        self._info = {"samples": samples}

    @property
    def info(self) -> dict[str, int]:
        return self._info


@pytest.mark.sanity
def test_replay_horizon_follows_the_duration_constraint():
    """
    The opening window is the duration limit, scaled by the replay time scale.

    ## WRITTEN BY AI ##
    """
    constraint = MaxDurationConstraint(args=MaxDurationConstraintArgs(seconds=60.0))
    constraints = {"max_duration": constraint.create_constraint()}
    assert (
        _replay_prefetch_horizon(TraceReplayStrategy(time_scale=2.0), constraints)
        == 30.0
    )
    assert _replay_prefetch_horizon(SynchronousStrategy(), constraints) == 60.0
    assert _replay_prefetch_horizon(TraceReplayStrategy(), None) is None


@pytest.mark.sanity
def test_prefetch_modes_select_the_trace_window():
    """
    start counts time zero, scheduled counts the duration, and a number passes through.

    ## WRITTEN BY AI ##
    """
    loader = _Loader([0.0, 0.0, 5.0, 0.0], samples=3)
    strategy = TraceReplayStrategy()
    constraints = {
        "max_duration": MaxDurationConstraint(
            args=MaxDurationConstraintArgs(seconds=60.0)
        ).create_constraint()
    }
    request_cap = {
        "max_requests": MaxNumberConstraint(
            args=MaxRequestsConstraintArgs(count=1000)
        ).create_constraint()
    }
    assert _resolve_profile_prefetch("start", loader, strategy, constraints) == (
        None,
        2,
        3,
    )
    assert _resolve_profile_prefetch("start", loader, strategy, request_cap) == (
        None,
        2,
        3,
    )
    assert _resolve_profile_prefetch("scheduled", loader, strategy, constraints) == (
        None,
        3,
        3,
    )
    assert _resolve_profile_prefetch("scheduled", loader, strategy, request_cap) == (
        1000,
        0,
        3,
    )
    assert _resolve_profile_prefetch("scheduled", object(), strategy, request_cap) == (
        1000,
        0,
        None,
    )
    minimum = {
        "min_requests": MinNumberConstraint(
            args=MinRequestsConstraintArgs(count=250)
        ).create_constraint()
    }
    assert _resolve_profile_prefetch("scheduled", object(), strategy, minimum) == (
        250,
        0,
        None,
    )
    both = {**request_cap, **minimum}
    assert _resolve_profile_prefetch("scheduled", object(), strategy, both) == (
        1000,
        0,
        None,
    )
    assert _resolve_profile_prefetch(400, loader, strategy, constraints) == (400, 0, 3)
    with pytest.raises(ValueError, match="max_requests, min_requests, or"):
        _resolve_profile_prefetch("scheduled", loader, strategy, None)
    # Non-trace sources have no schedule. start matches omit, so the scheduler
    # waits for workers and the strategy concurrency cap.
    assert _resolve_profile_prefetch("start", object(), strategy, None) == (
        None,
        0,
        None,
    )
    assert _resolve_profile_prefetch(None, object(), strategy, None) == (
        None,
        0,
        None,
    )
    with pytest.raises(ValueError, match="trace dataset"):
        _resolve_profile_prefetch("scheduled", object(), strategy, constraints)


@pytest.mark.sanity
def test_trace_prefetch_counts_time_zero_within_the_sample_cap():
    """
    Only the sample prefix counts. A duration window includes later starts.

    ## WRITTEN BY AI ##
    """
    loader = _Loader([0.0, 0.0, 5.0, 0.0], samples=3)
    assert _trace_prefetch_counts(loader, None) == (2, 3)
    assert _trace_prefetch_counts(loader, 60.0) == (3, 3)
    assert _trace_prefetch_counts(object(), 60.0) == (0, None)


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
    The wait reports each newly built conversation against the required total.

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
    await group._wait_for_min_prefetch(-1, on_prefetch, display_target=4)
    assert reported == [(0, 4), (1, 4)]


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
