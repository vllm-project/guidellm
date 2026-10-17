"""
Integration tests for WorkerProcessGroup.

Tests the complete lifecycle of the worker group with real multiprocessing
worker processes and a mock backend. Validates end-to-end functionality
across different scheduling strategies and constraints.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import random
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any
from unittest.mock import patch

import pytest

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    BackendInterface,
    ConcurrentStrategy,
    MaxDurationConstraint,
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
    MaxNumberConstraint,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
)
from guidellm.scheduler.constraints import ConstraintInitializer
from guidellm.scheduler.schemas import ConversationGraph, ConversationNode
from guidellm.scheduler.strategies import SchedulingStrategy
from guidellm.scheduler.worker_group import WorkerGroupState
from guidellm.schemas import RequestInfo, RequestSettings, RequestTimings
from guidellm.schemas.scheduler import (
    MaxDurationConstraintArgs,
    MaxErrorRateConstraintArgs,
    MaxErrorsConstraintArgs,
    MaxGlobalErrorRateConstraintArgs,
    MaxRequestsConstraintArgs,
)
from guidellm.utils.encoding import MessageEncoding
from guidellm.utils.messaging import (
    InterProcessMessagingManagerQueue,
    InterProcessMessagingPipe,
    InterProcessMessagingQueue,
)


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


def _single_request_graph(request: str) -> ConversationGraph[str]:
    """Build a one-node conversation graph for worker group tests.

    ## WRITTEN BY AI ##
    """
    return ConversationGraph(
        graph_id=f"graph_{request}",
        nodes={
            "turn_0": ConversationNode(
                node_id="turn_0",
                agent_id="default",
                request=request,
                settings=RequestSettings(),
            )
        },
        edges=[],
    )


class MockRequestTimings(RequestTimings):
    """Mock timing implementation for integration testing."""


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

    async def resolve(self, request, request_info, request_history):
        """Return predictable response based on input request."""
        # Simulate processing time
        await asyncio.sleep(self._response_delay)

        if (
            self._error_rate
            and self._error_rate > 0
            and random.random() < self._error_rate
        ):
            raise RuntimeError("Mock error for testing")

        yield f"response_for_{request}", request_info


class TestWorkerGroup:
    @pytest.mark.regression
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("messaging_kind", ["queue", "manager_queue", "pipe"])
    @pytest.mark.parametrize("status", ["completed", "errored", "cancelled"])
    async def test_request_updates_delivers_terminal_update_after_shutdown(
        self, messaging_kind, status
    ):
        """A terminal callback can set shutdown before its update is buffered.

        Hold the real receive callback across two consumer timeouts, then let
        each IPC transport finish delivering the update. The iterator must
        yield the terminal status and its final scheduler counts.

        ## WRITTEN BY AI ##
        """
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            kwargs = {"mp_context": context, "poll_interval": 0.01}
            if messaging_kind == "manager_queue":
                messaging = InterProcessMessagingManagerQueue(manager=manager, **kwargs)
            elif messaging_kind == "pipe":
                messaging = InterProcessMessagingPipe(num_workers=2, **kwargs)
            else:
                messaging = InterProcessMessagingQueue(**kwargs)
            group = WorkerProcessGroup(
                requests=[], backend=MockBackend(), strategy=SynchronousStrategy()
            )
            group.messaging = messaging
            group.error_event = context.Event()
            group.shutdown_event = context.Event()
            group.requests_generated_event = context.Event()
            group.constraint_reached_event = context.Event()
            state = WorkerGroupState(
                start_time=time.time(),
                processes=[],
                strategy=group.strategy,
                constraints={},
                stop_send_requests_event=threading.Event(),
                send_requests_stopped_event=threading.Event(),
                requests_generated_event=group.requests_generated_event,
                constraint_reached_event=group.constraint_reached_event,
                shutdown_event=group.shutdown_event,
                error_event=group.error_event,
                messaging=messaging,
            )
            group.state = state
            for initial_status in ("queued", "pending", "in_progress"):
                state.update_state(
                    RequestInfo(request_id="last", status=initial_status)
                )
            state.stop_send_requests_event.set()
            group.requests_generated_event.set()
            group.constraint_reached_event.set()

            callback_entered = threading.Event()
            release_callback = threading.Event()

            def delayed_callback(update):
                result = state.received_callback(update)
                assert group.shutdown_event.is_set()
                callback_entered.set()
                assert release_callback.wait(timeout=20)
                return result

            await messaging.start(
                receive_callback=delayed_callback,
                receive_stop_criteria=[group.shutdown_event],
                pydantic_models=[RequestInfo],
            )
            original_get = messaging.get
            timeouts = 0

            async def observed_get(*, timeout=None):
                nonlocal timeouts
                try:
                    return await original_get(timeout=0.01)
                except asyncio.TimeoutError:
                    timeouts += 1
                    if timeouts == 2:
                        release_callback.set()
                    raise

            try:
                encoding = MessageEncoding(
                    serialization=messaging.serialization,
                    encoding=messaging.encoding,
                    pydantic_models=[RequestInfo],
                )
                update = encoding.encode(
                    (
                        "response",
                        "request",
                        RequestInfo(request_id="last", status=status),
                    )
                )
                # Feed the actual receive transport without starting model workers.
                if isinstance(messaging, InterProcessMessagingPipe):
                    messaging.pipes[0][0].send(update)
                else:
                    messaging.done_queue.put(update, timeout=5)
                assert await asyncio.to_thread(callback_entered.wait, 20)
                with patch.object(messaging, "get", side_effect=observed_get):
                    results = [update async for update in group.request_updates()]
                assert len(results) == 1
                assert results[0][0:2] == ("response", "request")
                assert results[0][2].status == status
                assert results[0][3].created_requests == 1
                assert results[0][3].processed_requests == 1
                assert results[0][3].successful_requests == (status == "completed")
                assert results[0][3].errored_requests == (status == "errored")
                assert results[0][3].cancelled_requests == (status == "cancelled")
                assert timeouts >= 2
                assert messaging.receive_stopped_event.is_set()
                assert messaging.buffer_receive_queue.empty()
            finally:
                release_callback.set()
                await messaging.stop()

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5)
    @pytest.mark.parametrize(
        "strategy",
        [
            SynchronousStrategy(),
            ConcurrentStrategy(streams=10),
            ThroughputStrategy(max_concurrency=20),
            AsyncConstantStrategy(rate=1000.0),
            AsyncPoissonStrategy(rate=1000.0),
        ],
    )
    @pytest.mark.parametrize(
        "constraints_inits",
        [
            {
                "max_requests": MaxNumberConstraint(
                    args=MaxRequestsConstraintArgs(count=100)
                )
            },
            {
                "max_duration": MaxDurationConstraint(
                    args=MaxDurationConstraintArgs(seconds=0.5)
                )
            },
            {"max_errors": MaxErrorsConstraint(args=MaxErrorsConstraintArgs(count=20))},
            {
                "max_error_rate": MaxErrorRateConstraint(
                    args=MaxErrorRateConstraintArgs(rate=0.1)
                )
            },
            {
                "max_global_error_rate": MaxGlobalErrorRateConstraint(
                    args=MaxGlobalErrorRateConstraintArgs(rate=0.1)
                )
            },
        ],
    )
    async def test_lifecycle(
        self,
        strategy: SchedulingStrategy,
        constraints_inits: dict[str, ConstraintInitializer],
    ):
        """Test comprehensive lifecycle with different strategies and constraints."""
        # Setup
        backend = MockBackend(response_delay=0.01, processes_limit_value=1)
        requests = [_single_request_graph(f"request_{ind}") for ind in range(1000)]
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=strategy,
            startup_duration=0.1,
            constraints={
                key: init.create_constraint() for key, init in constraints_inits.items()
            },
        )

        try:
            # Create processes
            await group.create_processes()
            assert group.processes is not None
            assert len(group.processes) > 0
            assert group.mp_context is not None

            # Start processing
            start_time = time.time() + 0.1
            await group.start(start_time)
            actual_start = time.time()
            assert actual_start == pytest.approx(start_time)

            # Validate scheduler state
            assert group.scheduler_state is not None
            assert group.scheduler_state.start_time == start_time
            assert group.scheduler_state.num_processes == len(group.processes)

            # Collect all request updates
            received_updates = defaultdict(list)
            received_responses = []

            async for (
                response,
                request,
                request_info,
                _state,
            ) in group.request_updates():
                received_updates[request].append(request_info.status)
                if response is not None:
                    received_responses.append(response)
        finally:
            # Clean shutdown
            exceptions = await group.shutdown()
            assert len(exceptions) == 0, f"Shutdown errors: {exceptions}"
