from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import defaultdict
from functools import wraps
from typing import Any

import pytest
from pydantic import BaseModel, Field

from guidellm.scheduler import (
    BackendInterface,
    ConstraintInitializer,
    Environment,
    MaxNumberConstraint,
    NonDistributedEnvironment,
    Scheduler,
    SchedulerState,
    SchedulingStrategy,
    SynchronousStrategy,
    worker_group,
)
from guidellm.scheduler.loop_lag import LoopLagHistograms
from guidellm.scheduler.schemas import ConversationGraph, ConversationNode
from guidellm.schemas import RequestInfo, RequestSettings
from guidellm.schemas.scheduler import MaxRequestsConstraintArgs
from guidellm.settings import settings


def async_timeout(delay: float):
    """Decorator to add timeout to async test functions."""

    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class MockRequest(BaseModel):
    payload: str
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))


class MockConversationGraph(ConversationGraph[MockRequest]):
    """Bound graph type so IPC deserialization restores MockRequest nodes."""


def _single_request_graph(payload: str) -> MockConversationGraph:
    """Build a one-node conversation graph for scheduler tests.

    ## WRITTEN BY AI ##
    """
    return MockConversationGraph(
        graph_id=str(uuid.uuid4()),
        nodes={
            "turn_0": ConversationNode(
                node_id="turn_0",
                agent_id="default",
                request=MockRequest(payload=payload),
                settings=RequestSettings(),
            )
        },
        edges=[],
    )


class MockBackend(BackendInterface):
    """Mock backend for integration testing with predictable responses."""

    def __init__(
        self,
        processes_limit_value: int | None = None,
        requests_limit_value: int | None = None,
        error_rate: float = 0.2,
        response_delay: float = 0.0,
    ):
        self._processes_limit = processes_limit_value
        self._requests_limit = requests_limit_value
        self._error_rate = error_rate
        self._response_delay = response_delay

    @property
    def processes_limit(self) -> int | None:
        return self._processes_limit

    @property
    def requests_limit(self) -> int | None:
        return self._requests_limit

    def info(self) -> dict[str, Any]:
        return {"type": "mock_integration", "delay": self._response_delay}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request: MockRequest, request_info, request_history):
        """Return predictable response based on input request."""
        await asyncio.sleep(self._response_delay)

        if (
            self._error_rate
            and self._error_rate > 0
            and random.random() < self._error_rate
        ):
            raise RuntimeError(f"mock_error_for_{request.payload}")

        yield f"response_for_{request.payload}", request_info


@pytest.mark.smoke
@pytest.mark.asyncio
@async_timeout(60.0)
@pytest.mark.parametrize(
    ("strategy", "env", "constraint_inits"),
    [
        (
            SynchronousStrategy(),
            NonDistributedEnvironment(),
            {
                "max_requests": MaxNumberConstraint(
                    args=MaxRequestsConstraintArgs(count=100)
                )
            },
        ),
    ],
)
async def test_scheduler_run_integration(
    strategy: SchedulingStrategy,
    env: Environment,
    constraint_inits: dict[str, ConstraintInitializer],
):
    """Integration test for full scheduler workflow.

    Timeout is 60s: 100 multiprocess requests plus worker spawn routinely
    exceeds 10s on loaded CI runners.
    """
    # Clear singleton state
    if hasattr(Scheduler, "singleton_instance"):
        Scheduler.singleton_instance = None

    scheduler = Scheduler()
    constraints = {
        key: init.create_constraint() for key, init in constraint_inits.items()
    }
    received_updates = defaultdict(list)
    received_responses = []
    last_state = None
    num_requests = 100

    async for resp, req, info, state in scheduler.run(
        requests=[_single_request_graph(f"req_{ind}") for ind in range(num_requests)],
        backend=MockBackend(),
        strategy=strategy,
        env=env,
        **constraints,
    ):
        assert req is not None
        assert isinstance(req, MockRequest)
        assert isinstance(info, RequestInfo)
        assert info.status != "cancelled"
        assert isinstance(state, SchedulerState)
        if info.status == "completed":
            assert resp == f"response_for_{req.payload}"
            received_responses.append(resp)
        elif info.status == "errored":
            assert resp is None
            assert info.error is not None
            assert info.error == f"RuntimeError('mock_error_for_{req.payload}')"
            # The assert protects us but this is ugly and should be improved
            error_str = str(eval(info.error))  # noqa: S307
            received_responses.append(error_str)

        if len(received_updates[req.payload]) < 3:
            received_updates[req.payload].append(info.status)
        last_state = state

    assert len(received_updates) == num_requests
    assert len(received_responses) == constraints["max_requests"].args.count
    assert last_state.created_requests == constraints["max_requests"].args.count
    assert last_state.queued_requests == 0
    assert last_state.processing_requests == 0
    assert last_state.processed_requests == constraints["max_requests"].args.count
    assert last_state.cancelled_requests == 0
    assert (
        last_state.successful_requests + last_state.errored_requests
    ) == constraints["max_requests"].args.count

    def _request_indices():
        while True:
            yield from range(num_requests)

    for index, req, statuses, resp in zip(
        _request_indices(),
        received_updates.keys(),
        received_updates.values(),
        received_responses,
        strict=False,
    ):
        assert req == f"req_{index}"
        assert resp in (f"response_for_{req}", f"mock_error_for_{req}")
        assert statuses in (
            ["queued", "in_progress", "completed"],
            ["queued", "in_progress", "errored"],
            ["queued", "pending", "in_progress"],
        )


def _block_event_loop(seconds: float) -> None:
    """Hold the event loop the way a CPU-starved worker does."""
    time.sleep(seconds)


class _LoopBlockingBackend(MockBackend):
    """Hold the worker's event loop for each request, as a CPU-starved worker does."""

    def __init__(self, block: float):
        super().__init__(error_rate=0.0)
        self._block = block

    async def resolve(self, request: MockRequest, request_info, request_history):
        await asyncio.sleep(0.02)
        _block_event_loop(self._block)
        yield f"response_for_{request.payload}", request_info


class _EngineTimedBackend(_LoopBlockingBackend):
    """Backend whose token times do not come from the worker's event loop."""

    stamps_tokens_on_event_loop = False


@pytest.mark.regression
@pytest.mark.asyncio
@async_timeout(60.0)
@pytest.mark.parametrize(
    ("backend", "low", "high"),
    [
        (_LoopBlockingBackend(block=0.0), 0.0, 0.1),
        (_LoopBlockingBackend(block=0.2), 0.1, 1.0),
    ],
)
async def test_scheduler_reports_worker_loop_lag(
    monkeypatch: pytest.MonkeyPatch, backend: MockBackend, low: float, high: float
):
    """Report how long workers held their event loops in the final state.

    ## WRITTEN BY AI ##
    """
    monkeypatch.setattr(settings, "loop_lag_interval", 0.01)
    created: list[LoopLagHistograms] = []

    def recording_histograms(*args: Any) -> LoopLagHistograms:
        histograms = LoopLagHistograms(*args)
        created.append(histograms)
        return histograms

    monkeypatch.setattr(worker_group, "LoopLagHistograms", recording_histograms)
    if hasattr(Scheduler, "singleton_instance"):
        Scheduler.singleton_instance = None
    last_state = None

    async for _, _, _, state in Scheduler().run(
        requests=[_single_request_graph(f"req_{ind}") for ind in range(5)],
        backend=backend,
        strategy=SynchronousStrategy(),
        env=NonDistributedEnvironment(),
        max_requests=MaxNumberConstraint(args=MaxRequestsConstraintArgs(count=5)),
    ):
        last_state = state

    assert [histograms.interval for histograms in created] == [0.01]
    assert last_state is not None
    assert sum(count for _, count in last_state.worker_loop_lag_samples) > 0
    assert low <= max(lag for lag, _ in last_state.worker_loop_lag_samples) < high


@pytest.mark.sanity
@pytest.mark.asyncio
@async_timeout(60.0)
async def test_scheduler_skips_loop_lag_for_engine_timed_backends():
    """Leave loop lag empty for a backend that does not stamp tokens on the loop.

    ## WRITTEN BY AI ##
    """
    if hasattr(Scheduler, "singleton_instance"):
        Scheduler.singleton_instance = None
    last_state = None

    async for _, _, _, state in Scheduler().run(
        requests=[_single_request_graph(f"req_{ind}") for ind in range(3)],
        backend=_EngineTimedBackend(block=0.0),
        strategy=SynchronousStrategy(),
        env=NonDistributedEnvironment(),
        max_requests=MaxNumberConstraint(args=MaxRequestsConstraintArgs(count=3)),
    ):
        last_state = state

    assert last_state is not None
    assert last_state.worker_loop_lag_samples == []
