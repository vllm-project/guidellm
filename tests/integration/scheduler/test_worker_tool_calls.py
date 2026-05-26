"""
Integration tests for WorkerProcess handling of missing tool calls.

Tests all three tool_call_missing_behavior modes using a mock backend that
simulates the streaming control flow of OpenAIHTTPBackend.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from multiprocessing import Barrier, Event
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from guidellm.scheduler import SynchronousStrategy, WorkerProcess
from guidellm.schemas import GenerationRequest, RequestInfo
from guidellm.utils.messaging import InterProcessMessagingQueue


def async_timeout(delay: float):
    """Decorator to add timeout to async test functions."""

    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class _MockToolBackend:
    """Mock backend that raises exceptions for missing tool calls.

    Mimics the streaming path of OpenAIHTTPBackend.resolve():

    * ``_check_tool_call_expectations`` runs BEFORE the final yield.
    * If it raises ``CancelledError`` (ignore_stop), the ``except`` block
      yields the response once, then re-raises -- so the consumer sees the
      response followed by ``CancelledError`` on the next iteration.
    * If it raises ``ValueError`` (error_stop), the exception is NOT caught
      by the ``except asyncio.CancelledError`` block, so it propagates
      without ever yielding the final response.
    """

    def __init__(
        self,
        has_tool_calls: bool = True,
        tool_call_missing_behavior: str = "error_stop",
    ):
        self.has_tool_calls = has_tool_calls
        self.tool_call_missing_behavior = tool_call_missing_behavior
        self.process_startup_called = False
        self.validate_called = False
        self.process_shutdown_called = False

    @property
    def processes_limit(self):
        return None

    @property
    def requests_limit(self):
        return None

    @property
    def info(self):
        return {"type": "mock_tool"}

    async def process_startup(self):
        self.process_startup_called = True

    async def validate(self):
        self.validate_called = True

    async def process_shutdown(self):
        self.process_shutdown_called = True

    async def resolve(self, request, request_info, history=None):
        response = MagicMock()
        response.tool_calls = (
            [{"id": "call_1", "type": "function", "function": {"name": "fn"}}]
            if self.has_tool_calls
            else None
        )

        if request.expects_tool_call and not self.has_tool_calls:
            if self.tool_call_missing_behavior == "ignore_stop":
                yield response, request_info
                raise asyncio.CancelledError(
                    "Expected tool call but model produced none"
                )
            elif self.tool_call_missing_behavior == "error_stop":
                raise ValueError("Expected tool call but model produced none")

        yield response, request_info


def _make_conversation(
    num_turns: int, tool_call_turns: int
) -> list[tuple[Any, RequestInfo]]:
    """Build a pre-planned conversation list for testing.

    ## WRITTEN BY AI ##
    """
    conv = []
    for i in range(num_turns):
        req = GenerationRequest(
            columns={"text_column": [f"turn_{i}"]},
            expects_tool_call=(i < tool_call_turns),
        )
        info = RequestInfo(
            request_id=req.request_id,
            conversation_id="conv_1",
            turn_index=i,
            status="queued",
        )
        conv.append((req, info))
    return conv


class TestWorkerMissingToolCallBehavior:
    """Test worker handling of missing tool calls for all 3 behaviors.

    ## WRITTEN BY AI ##
    """

    @pytest_asyncio.fixture
    async def make_worker(self):
        """Factory fixture that creates a worker with the given backend.

        ## WRITTEN BY AI ##
        """
        workers = []

        async def _factory(backend):
            messaging = InterProcessMessagingQueue(
                serialization="dict",
                encoding=None,
                max_buffer_receive_size=10,
                poll_interval=0.01,
            )
            await messaging.start(pydantic_models=[])

            worker = WorkerProcess(
                worker_index=0,
                messaging=messaging.create_worker_copy(0),
                backend=backend,
                strategy=SynchronousStrategy(),
                async_limit=1,
                fut_scheduling_time_limit=10.0,
                startup_barrier=Barrier(1),
                requests_generated_event=Event(),
                constraint_reached_event=Event(),
                shutdown_event=Event(),
                error_event=Event(),
            )
            workers.append((worker, messaging))
            return worker, messaging

        yield _factory

        for _, msg in workers:
            await msg.stop()

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_ignore_continue_keeps_remaining_turns(self, make_worker):
        """ignore_continue: conversation continues to next turn normally.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_continue",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        assert len(remaining_conv) == 2
        assert remaining_conv[0][0].expects_tool_call is True
        assert remaining_conv[1][0].expects_tool_call is False

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_ignore_stop_cancels_all_turns(self, make_worker):
        """ignore_stop: current turn and all remaining turns are cancelled.

        The backend raises CancelledError which the worker catches and
        marks the current request as cancelled, then cancels remaining turns.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        with pytest.raises(asyncio.CancelledError):
            await worker._process_next_request(target_start=time.time())

        cancelled_updates = [u for u in updates if u[0] == "cancelled"]
        assert len(cancelled_updates) == 3

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_error_stop_errors_and_cancels_remaining(self, make_worker):
        """error_stop: current turn errored, remaining turns cancelled.

        The backend raises ValueError (before yielding the response) which
        the worker catches via its generic exception handler, setting
        request_info.error and sending an "errored" status update. The
        remaining conversation turns are cancelled in the finally block.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        assert len(remaining_conv) == 0

        errored_updates = [u for u in updates if u[0] == "errored"]
        assert len(errored_updates) == 1

        cancelled_updates = [u for u in updates if u[0] == "cancelled"]
        assert len(cancelled_updates) == 2

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_tool_call_present_continues_normally(self, make_worker):
        """When tool call IS present, conversation continues normally.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=True,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        assert len(remaining_conv) == 2

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_non_tool_turn_ignores_behavior(self, make_worker):
        """Non-tool turns don't trigger missing-tool-call logic at all.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=2, tool_call_turns=0)
        await messaging.put(conv)

        await worker._processing_startup()

        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        assert len(remaining_conv) == 1
        errored = [u for u in updates if u[0] == "errored"]
        cancelled = [u for u in updates if u[0] == "cancelled"]
        assert len(errored) == 0
        assert len(cancelled) == 0

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_ignore_continue_per_turn_independence(self, make_worker):
        """ignore_continue: each tool turn succeeds or fails independently.

        Process two turns sequentially. First turn misses tool call but
        conversation continues. Second tool turn gets its own chance.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_continue",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        history, remaining_conv, _ = await worker._process_next_request(
            target_start=time.time()
        )

        assert len(remaining_conv) == 2
        assert remaining_conv[0][0].expects_tool_call is True
