from __future__ import annotations

import asyncio
import contextlib
import inspect
import random
import time
from dataclasses import dataclass
from functools import wraps
from multiprocessing import Barrier, Event, Process
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from typing import Any, Generic, Literal

import pytest
import pytest_asyncio

from guidellm.scheduler import (
    BackendInterface,
    ConstantRateRequestTimings,
    LastCompletionRequestTimings,
    MeasuredRequestTimings,
    NoDelayRequestTimings,
    PoissonRateRequestTimings,
    ScheduledRequestInfo,
    ScheduledRequestTimings,
    SchedulerMessagingPydanticRegistry,
    WorkerProcess,
)
from guidellm.utils import InterProcessMessagingQueue

STANDARD_NUM_REQUESTS: int = 200


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


@dataclass
class TimingsBounds:
    exact: float | None = None
    lower: float | None = None
    upper: float | None = None
    prev_request: Literal["greater", "greater_equal", "less", "less_equal"] | None = (
        None
    )
    tolerance: float = 10e-4
    actual_tolerance: float = 10e-4


class MockRequestTimings(MeasuredRequestTimings):
    """Mock timing implementation for testing."""


SchedulerMessagingPydanticRegistry.register("ScheduledRequestInfo")(
    ScheduledRequestInfo
)


class MockBackend(BackendInterface):
    """Mock backend for testing worker functionality."""

    def __init__(
        self,
        lifecycle_delay: float = 0.1,
        resolve_delay: float = 0.0,
        should_fail: bool = False,
        request_error_rate: float = 0.0,
    ):
        self.lifecycle_delay = lifecycle_delay
        self.resolve_delay = resolve_delay
        self.should_fail = should_fail
        self.request_error_rate = request_error_rate
        self.process_startup_called = False
        self.validate_called = False
        self.process_shutdown_called = False
        self.resolve_called = False

    @property
    def processes_limit(self) -> int | None:
        return None

    @property
    def requests_limit(self) -> int | None:
        return None

    @property
    def info(self) -> dict[str, Any]:
        return {
            "type": "mock",
            "lifecycle_delay": self.lifecycle_delay,
            "resolve_delay": self.resolve_delay,
        }

    async def process_startup(self):
        await asyncio.sleep(self.lifecycle_delay)
        self.process_startup_called = True

    async def validate(self):
        await asyncio.sleep(self.lifecycle_delay)
        self.validate_called = True
        if self.should_fail:
            raise RuntimeError("Mock validation failed")

    async def process_shutdown(self):
        await asyncio.sleep(self.lifecycle_delay)
        self.process_shutdown_called = True

    async def resolve(self, request, request_info, request_history):
        self.resolve_called = True
        await asyncio.sleep(
            self.resolve_delay if not str(request).startswith("cancel") else 1000.0
        )
        if self.should_fail:
            raise RuntimeError("Mock resolve failed")
        if self.request_error_rate > 0.0 and random.random() < self.request_error_rate:
            raise RuntimeError("Mock resolve failed")
        yield f"response_for_{request}", request_info


class TestWorkerProcess:
    """Test suite for WorkerProcess class."""

    @pytest_asyncio.fixture(
        params=[
            {
                "messaging": {
                    "serialization": "dict",
                    "encoding": None,
                    "max_buffer_receive_size": 2,
                },
                "worker": {
                    "async_limit": 1,
                },
            },
            {
                "messaging": {
                    "serialization": "dict",
                    "encoding": None,
                    "max_buffer_receive_size": 100,
                },
                "worker": {
                    "async_limit": 1000,
                },
            },
        ],
    )
    async def valid_instances(self, request):
        """Fixture providing test data for WorkerProcess."""
        constructor_args = request.param
        main_messaging = InterProcessMessagingQueue(
            **constructor_args["messaging"], poll_interval=0.01
        )

        try:
            instance = WorkerProcess(
                messaging=main_messaging.create_worker_copy(0),
                **constructor_args["worker"],
                startup_barrier=Barrier(2),
                shutdown_event=Event(),
                error_event=Event(),
                requests_completed_event=Event(),
                backend=MockBackend(),
                request_timings=LastCompletionRequestTimings(),
            )
            await main_messaging.start(
                pydantic_models=list(
                    SchedulerMessagingPydanticRegistry.registry.values()
                )
            )
            yield instance, main_messaging, constructor_args
        finally:
            await main_messaging.stop()

    @pytest.mark.smoke
    def test_class_signatures(
        self,
        valid_instances: tuple[WorkerProcess, InterProcessMessagingQueue, dict],
    ):
        """Test inheritance and type relationships."""
        worker_process, main_messaging, constructor_args = valid_instances

        # Class
        assert isinstance(worker_process, Generic)
        assert issubclass(WorkerProcess, Generic)

        # Generics
        orig_bases = getattr(WorkerProcess, "__orig_bases__", ())
        assert len(orig_bases) > 0
        generic_base = next(
            (
                base
                for base in orig_bases
                if hasattr(base, "__origin__") and base.__origin__ is Generic
            ),
            None,
        )
        assert generic_base is not None
        type_args = getattr(generic_base, "__args__", ())
        assert len(type_args) == 2  # RequestT, ResponseT

        # Function signatures
        run_sig = inspect.signature(WorkerProcess.run)
        assert len(run_sig.parameters) == 1
        assert "self" in run_sig.parameters

        run_async_sig = inspect.signature(WorkerProcess.run_async)
        assert len(run_async_sig.parameters) == 1
        assert "self" in run_async_sig.parameters

        stop_processing_sig = inspect.signature(
            WorkerProcess._run_async_stop_processing
        )
        assert len(stop_processing_sig.parameters) == 1
        assert "self" in stop_processing_sig.parameters

        requests_processing_sig = inspect.signature(
            WorkerProcess._run_async_requests_processing
        )
        assert len(requests_processing_sig.parameters) == 1
        assert "self" in requests_processing_sig.parameters

    @pytest.mark.smoke
    def test_initialization(
        self,
        valid_instances: tuple[WorkerProcess, InterProcessMessagingQueue, dict],
    ):
        """Test basic initialization of WorkerProcess."""
        instance, main_messaging, constructor_args = valid_instances

        # messaging
        assert instance.messaging is not None
        assert isinstance(instance.messaging, InterProcessMessagingQueue)
        assert instance.messaging is not main_messaging
        assert instance.messaging.worker_index is not None
        assert instance.messaging.worker_index == 0
        assert (
            instance.messaging.serialization
            == constructor_args["messaging"]["serialization"]
        )
        assert instance.messaging.encoding == constructor_args["messaging"]["encoding"]
        assert (
            instance.messaging.max_buffer_receive_size
            == constructor_args["messaging"]["max_buffer_receive_size"]
        )

        # worker
        assert instance.async_limit == constructor_args["worker"]["async_limit"]
        assert instance.startup_barrier is not None
        assert isinstance(instance.startup_barrier, ProcessingBarrier)
        assert instance.shutdown_event is not None
        assert isinstance(instance.shutdown_event, ProcessingEvent)
        assert instance.error_event is not None
        assert isinstance(instance.error_event, ProcessingEvent)
        assert instance.requests_completed_event is not None
        assert isinstance(instance.requests_completed_event, ProcessingEvent)
        assert instance.backend is not None
        assert isinstance(instance.backend, MockBackend)
        assert instance.request_timings is not None
        assert isinstance(instance.request_timings, LastCompletionRequestTimings)
        assert not instance.startup_completed

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test that invalid initialization raises appropriate errors."""

        # Test with missing required parameters
        with pytest.raises(TypeError):
            WorkerProcess()

        # Create a complete set of valid parameters
        backend = MockBackend()
        request_timings = LastCompletionRequestTimings()
        barrier = Barrier(2)
        shutdown_event = Event()
        error_event = Event()
        completed_event = Event()
        messaging = InterProcessMessagingQueue()

        # Test missing each required parameter one by one
        required_params = [
            "messaging",
            "async_limit",
            "startup_barrier",
            "shutdown_event",
            "error_event",
            "requests_completed_event",
            "backend",
            "request_timings",
        ]

        for param_to_remove in required_params:
            kwargs = {
                "messaging": messaging,
                "async_limit": 5,
                "startup_barrier": barrier,
                "shutdown_event": shutdown_event,
                "error_event": error_event,
                "requests_completed_event": completed_event,
                "backend": backend,
                "request_timings": request_timings,
            }

            del kwargs[param_to_remove]

            with pytest.raises(TypeError):
                WorkerProcess(**kwargs)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(15)
    @pytest.mark.parametrize(
        ("num_requests", "num_canceled", "error_rate"),
        [
            (20, 0, 0),
            (STANDARD_NUM_REQUESTS, 20, 0.5),
        ],
    )
    @pytest.mark.parametrize(
        "stop_method", ["task_cancel", "shutdown_event", "error_event"]
    )
    async def test_run_async_request_processing(  # noqa: C901, PLR0912
        self,
        valid_instances: tuple[WorkerProcess, InterProcessMessagingQueue, dict],
        stop_method: Literal["task_cancel", "shutdown_event", "error_event"],
        num_requests: int,
        num_canceled: int,
        error_rate: float,
    ):
        """Test the asynchronous request processing of WorkerProcess."""
        instance, main_messaging, constructor_args = valid_instances

        if num_canceled > constructor_args["worker"]["async_limit"]:
            pytest.skip("Canceled requests exceed async limit")

        instance.backend.request_error_rate = error_rate
        instance_task = asyncio.create_task(instance.run_async())

        try:
            await asyncio.to_thread(instance.startup_barrier.wait)
            start_time = time.time()

            # Send regular requests
            requests_tracker = {}
            for i in range(num_requests):
                request = f"request_{i}"
                requests_tracker[request] = {
                    "sent": True,
                    "received_in_progress": False,
                    "received_resolved": False,
                }
                await main_messaging.put(
                    (
                        request,
                        ScheduledRequestInfo(scheduler_start_time=start_time),
                    ),
                    timeout=2.0,
                )

            # Process regular requests
            error_count = 0
            for _ in range(num_requests * 2):
                response, request, request_info = await main_messaging.get(timeout=2.0)
                if request_info.status == "in_progress":
                    requests_tracker[request]["received_in_progress"] = True
                elif request_info.status == "completed":
                    assert response == f"response_for_{request}"
                    requests_tracker[request]["received_resolved"] = True
                elif request_info.status == "errored":
                    assert response is None
                    requests_tracker[request]["received_resolved"] = True
                    error_count += 1
                else:
                    raise ValueError(f"Unexpected status: {request_info.status}")

            assert float(error_count) / num_requests == pytest.approx(
                error_rate, rel=0.2
            )

            # Send cancel requests and wait for in_progress
            cancel_requests = []
            for ind in range(num_canceled):
                cancel_request = f"cancel_request_{ind}"
                cancel_requests.append(cancel_request)
                requests_tracker[cancel_request] = {
                    "sent": True,
                    "received_in_progress": False,
                    "received_resolved": False,
                }
                await main_messaging.put(
                    (
                        cancel_request,
                        ScheduledRequestInfo(scheduler_start_time=start_time),
                    ),
                    timeout=2.0,
                )

            # Signal that all requests have been sent
            instance.requests_completed_event.set()

            for _ in range(num_canceled):
                response, request, request_info = await main_messaging.get(timeout=2.0)
                if request_info.status == "in_progress":
                    requests_tracker[request]["received_in_progress"] = True
                else:
                    raise ValueError(f"Unexpected status: {request_info.status}")

            # Trigger shutdown/cancel
            if stop_method == "task_cancel":
                instance_task.cancel()
            elif stop_method == "shutdown_event":
                instance.shutdown_event.set()
            elif stop_method == "error_event":
                instance.error_event.set()
            await asyncio.sleep(0.5)

            # Collect any cancelled
            for _ in range(num_canceled):
                response, request, request_info = await main_messaging.get(timeout=1.0)
                if request_info.status == "cancelled":
                    requests_tracker[request]["received_resolved"] = True
                else:
                    raise ValueError(f"Unexpected status: {request_info.status}")

            # Verify all requests were processed
            for request, status in requests_tracker.items():
                assert status["received_in_progress"], (
                    f"Request {request} never went in_progress"
                )
                assert status["received_resolved"], f"Request {request} never completed"

        finally:
            if not instance_task.done() and not instance_task.cancelled():
                instance_task.cancel()

            final_error = None
            try:
                await asyncio.wait_for(instance_task, timeout=2.0)
            except asyncio.TimeoutError:
                # If it times out, force cancel
                instance_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(instance_task, timeout=1.0)
            except (asyncio.CancelledError, RuntimeError) as err:
                # Expected exceptions depending on stop method
                final_error = err

            if stop_method == "task_cancel":
                assert isinstance(final_error, asyncio.CancelledError)
            elif stop_method == "error_event":
                assert isinstance(final_error, RuntimeError)
            else:
                assert final_error is None

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(15)
    @pytest.mark.parametrize(
        ("request_timings", "timing_bounds"),
        [
            (
                LastCompletionRequestTimings(offset=0.1),
                [
                    TimingsBounds(lower=0.1, prev_request="greater_equal")
                    for _ in range(STANDARD_NUM_REQUESTS)
                ],
            ),
            (
                NoDelayRequestTimings(offset=0.05),
                [
                    TimingsBounds(lower=0.05, upper=0.05, actual_tolerance=1.0)
                    for _ in range(STANDARD_NUM_REQUESTS)
                ],
            ),
            (
                ConstantRateRequestTimings(rate=100, offset=0.2),
                [
                    TimingsBounds(
                        exact=0.2 + ind * 0.01,
                        lower=0.2,
                        prev_request="greater",
                        actual_tolerance=10e-2,
                    )
                    for ind in range(STANDARD_NUM_REQUESTS)
                ],
            ),
            (
                PoissonRateRequestTimings(rate=200, offset=0.01),
                [
                    TimingsBounds(lower=0.01, prev_request="greater")
                    for ind in range(STANDARD_NUM_REQUESTS)
                ],
            ),
        ],
        ids=[
            "LastCompletion",
            "NoDelay",
            "ConstantRate",
            "PoissonRate",
        ],
    )
    async def test_run_with_timings(  # noqa: C901, PLR0912
        self,
        valid_instances: tuple[WorkerProcess, InterProcessMessagingQueue, dict],
        request_timings: ScheduledRequestTimings,
        timing_bounds: list[TimingsBounds],
    ):
        instance, main_messaging, constructor_args = valid_instances
        instance.request_timings = request_timings
        num_requests = STANDARD_NUM_REQUESTS
        assert len(timing_bounds) == num_requests

        # Start process
        process = Process(target=instance.run)
        process.start()

        try:
            await asyncio.to_thread(instance.startup_barrier.wait)
            start_time = time.time() + 0.1

            # Send regular requests
            requests_tracker = {}
            for ind in range(num_requests):
                request = f"request_{ind}"
                requests_tracker[request] = {
                    "sent": True,
                    "target_start_time": -1,
                    "actual_start_time": -1,
                    "received_in_progress": False,
                    "received_resolved": False,
                }
                await main_messaging.put(
                    (
                        request,
                        ScheduledRequestInfo(scheduler_start_time=start_time),
                    ),
                    timeout=2.0,
                )

            # Process regular requests
            for _ in range(num_requests * 2):
                response, request, request_info = await main_messaging.get(timeout=2.0)
                if request_info.status == "in_progress":
                    requests_tracker[request]["received_in_progress"] = True
                    requests_tracker[request]["target_start_time"] = (
                        request_info.scheduler_timings.targeted_start
                    )
                    requests_tracker[request]["actual_start_time"] = (
                        request_info.scheduler_timings.resolve_start
                    )
                elif request_info.status == "completed":
                    assert response == f"response_for_{request}"
                    requests_tracker[request]["received_resolved"] = True
                else:
                    raise ValueError(f"Unexpected status: {request_info.status}")

            # Validate request values are correct
            for ind in range(num_requests):
                request = f"request_{ind}"
                assert requests_tracker[request]["received_in_progress"]
                assert requests_tracker[request]["received_resolved"]

                bounds = timing_bounds[ind]
                target_offset = (
                    requests_tracker[request]["target_start_time"] - start_time
                )
                actual_offset = (
                    requests_tracker[request]["actual_start_time"] - start_time
                )
                prev_offset = (
                    requests_tracker[f"request_{ind - 1}"]["target_start_time"]
                    - start_time
                    if ind > 0
                    else None
                )

                if bounds.exact is not None:
                    assert target_offset == pytest.approx(
                        bounds.exact, rel=bounds.tolerance
                    )
                    assert target_offset == pytest.approx(
                        actual_offset, rel=bounds.actual_tolerance or bounds.tolerance
                    )
                if bounds.lower is not None:
                    assert target_offset >= bounds.lower - bounds.tolerance
                    assert actual_offset >= bounds.lower - (
                        bounds.actual_tolerance or bounds.tolerance
                    )
                if bounds.upper is not None:
                    assert target_offset <= bounds.upper + bounds.tolerance
                    assert actual_offset <= bounds.upper + (
                        bounds.actual_tolerance or bounds.tolerance
                    )
                if bounds.prev_request is not None and prev_offset is not None:
                    if bounds.prev_request == "greater":
                        assert target_offset > prev_offset - bounds.tolerance
                    elif bounds.prev_request == "greater_equal":
                        assert target_offset >= prev_offset - bounds.tolerance
                    elif bounds.prev_request == "less":
                        assert target_offset < prev_offset + bounds.tolerance
                    elif bounds.prev_request == "less_equal":
                        assert target_offset <= prev_offset + bounds.tolerance

            # Trigger shutdown
            instance.requests_completed_event.set()
            instance.shutdown_event.set()
            await asyncio.to_thread(process.join, timeout=2.0)
        finally:
            instance.shutdown_event.set()
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, timeout=2.0)
            assert process.exitcode <= 0, (
                f"Process exited with error code: {process.exitcode}"
            )
