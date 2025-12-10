from __future__ import annotations

import asyncio
import inspect
import time
from multiprocessing.context import BaseContext
from multiprocessing.managers import BaseManager
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier, Event
from typing import Any, Generic, Literal
from unittest.mock import patch

import pytest
from pydantic import Field

from guidellm.scheduler import (
    AsyncConstantStrategy,
    BackendInterface,
    ConcurrentStrategy,
    MaxDurationConstraint,
    MaxNumberConstraint,
    SchedulerState,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
)
from guidellm.scheduler.worker_group import WorkerGroupState
from guidellm.schemas import RequestInfo, RequestTimings
from guidellm.utils import InterProcessMessaging
from tests.unit.testing_utils import async_timeout


class MockRequestTimings(RequestTimings):
    """Mock timing implementation for testing."""

    timings_type: Literal["mock"] = Field(default="mock")


class MockTime:
    """Deterministic time mock for testing."""

    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.increment = 0.1

    def time(self) -> float:
        """Return current mock time and increment for next call."""
        current = self.current_time
        self.current_time += self.increment
        return current


mock_time = MockTime()


class MockBackend(BackendInterface):
    """Mock backend for testing worker group functionality."""

    def __init__(
        self,
        processes_limit_value: int | None = None,
        requests_limit_value: int | None = None,
    ):
        self._processes_limit = processes_limit_value
        self._requests_limit = requests_limit_value

    @property
    def processes_limit(self) -> int | None:
        return self._processes_limit

    @property
    def requests_limit(self) -> int | None:
        return self._requests_limit

    @property
    def info(self) -> dict[str, Any]:
        return {"type": "mock"}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request, request_info, request_history):
        request_info.timings = MockRequestTimings(
            request_start=time.time(), request_end=time.time()
        )
        yield f"response_for_{request}", request_info


class TestWorkerProcessGroup:
    """Test suite for WorkerProcessGroup class."""

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    @pytest.fixture(
        params=[
            {
                "requests": ["request1", "request2", "request3"],
                "strategy": SynchronousStrategy(),
                "startup_duration": 0.1,
                "max_num": MaxNumberConstraint(max_num=10),
            },
            {
                "requests": ["req_a", "req_b"],
                "strategy": ConcurrentStrategy(streams=2),
                "startup_duration": 0.1,
                "max_num": MaxNumberConstraint(max_num=5),
            },
            {
                "requests": ["req_x", "req_y", "req_z"],
                "strategy": ThroughputStrategy(max_concurrency=5),
                "startup_duration": 0.1,
            },
            {
                "requests": ["req_8", "req_9", "req_10"],
                "strategy": AsyncConstantStrategy(rate=20),
                "startup_duration": 0.1,
                "max_duration": MaxDurationConstraint(max_duration=1),
            },
        ],
        ids=["sync_max", "concurrent_max", "throughput_no_cycle", "constant_duration"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for WorkerProcessGroup."""
        constructor_args = request.param.copy()
        base_params = {
            k: v
            for k, v in request.param.items()
            if k in ["requests", "strategy", "startup_duration"]
        }
        constraint_params = {
            k: v
            for k, v in request.param.items()
            if k not in ["requests", "strategy", "startup_duration"]
        }
        instance = WorkerProcessGroup(
            **base_params, backend=MockBackend(), **constraint_params
        )
        yield instance, constructor_args

        # Shutting down. Attempting shut down.
        try:
            if hasattr(instance, "processes") and instance.processes is not None:
                asyncio.run(instance.shutdown())
        # It's not...it's-it's not...it's not shutting down...it's not...
        except Exception:  # noqa: BLE001
            if hasattr(instance, "processes") and instance.processes is not None:
                # Gahhh...!
                for proc in instance.processes:
                    proc.kill()
                    proc.join(timeout=1.0)

    @pytest.mark.smoke
    def test_class_signatures(self, valid_instances):
        """Test inheritance and type relationships."""
        instance, _ = valid_instances

        # Class
        assert isinstance(instance, Generic)
        assert issubclass(WorkerProcessGroup, Generic)

        # Generics
        orig_bases = getattr(WorkerProcessGroup, "__orig_bases__", ())
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
        assert len(type_args) == 2

        # Function signatures
        create_processes_sig = inspect.signature(WorkerProcessGroup.create_processes)
        assert len(create_processes_sig.parameters) == 1
        assert "self" in create_processes_sig.parameters

        start_sig = inspect.signature(WorkerProcessGroup.start)
        assert len(start_sig.parameters) == 2
        assert "self" in start_sig.parameters
        assert "start_time" in start_sig.parameters

        request_updates_sig = inspect.signature(WorkerProcessGroup.request_updates)
        assert len(request_updates_sig.parameters) == 1
        assert "self" in request_updates_sig.parameters

        shutdown_sig = inspect.signature(WorkerProcessGroup.shutdown)
        assert len(shutdown_sig.parameters) == 1
        assert "self" in shutdown_sig.parameters

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test basic initialization of WorkerProcessGroup."""
        instance, constructor_args = valid_instances

        # Core attributes
        assert isinstance(instance.backend, MockBackend)
        # requests is now an iterator, not a list
        assert hasattr(instance.requests, "__iter__")
        assert hasattr(instance.requests, "__next__")
        assert isinstance(instance.strategy, type(constructor_args["strategy"]))
        assert isinstance(instance.constraints, dict)

        # Multiprocessing attributes (should be None initially)
        assert instance.mp_context is None
        assert instance.mp_manager is None
        assert instance.processes is None

        # Synchronization primitives (should be None initially)
        assert instance.startup_barrier is None
        assert instance.shutdown_event is None
        assert instance.error_event is None
        assert instance.requests_generated_event is None
        assert instance.constraint_reached_event is None

        # Scheduler state and messaging (should be None initially)
        assert instance.state is None
        assert instance.messaging is None

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("requests", "expected_error"),
        [
            (None, TypeError),
        ],
        ids=["no_requests"],
    )
    def test_invalid_initialization_values(self, requests, expected_error):
        """Test WorkerProcessGroup with invalid initialization values."""
        with pytest.raises(expected_error):
            WorkerProcessGroup(
                requests=requests,
                backend=MockBackend(),
                strategy=SynchronousStrategy(),
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test WorkerProcessGroup initialization without required fields."""
        with pytest.raises(TypeError):
            WorkerProcessGroup()

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @async_timeout(10)
    @pytest.mark.asyncio
    @patch.object(time, "time", mock_time.time)
    async def test_lifecycle(self, valid_instances: tuple[WorkerProcessGroup, dict]):  # noqa: C901, PLR0912
        """Test the lifecycle methods of WorkerProcessGroup."""
        instance, constructor_args = valid_instances
        assert instance.requests or instance.cycle_requests
        assert instance.backend
        assert instance.strategy
        assert instance.constraints is not None

        # Validate create_processes works and sets correct state
        await instance.create_processes()
        assert instance.mp_context is not None
        assert isinstance(instance.mp_context, BaseContext)
        assert instance.mp_manager is not None
        assert isinstance(instance.mp_manager, BaseManager)
        assert instance.processes is not None
        assert isinstance(instance.processes, list)
        assert len(instance.processes) > 0
        assert all(isinstance(proc, BaseProcess) for proc in instance.processes)
        assert all(proc.is_alive() for proc in instance.processes)
        assert instance.startup_barrier is not None
        assert isinstance(instance.startup_barrier, Barrier)
        assert instance.requests_generated_event is not None
        assert isinstance(instance.requests_generated_event, Event)
        assert instance.constraint_reached_event is not None
        assert isinstance(instance.constraint_reached_event, Event)
        assert instance.shutdown_event is not None
        assert isinstance(instance.shutdown_event, Event)
        assert instance.error_event is not None
        assert isinstance(instance.error_event, Event)
        assert instance.messaging is not None
        assert isinstance(instance.messaging, InterProcessMessaging)
        assert instance.messaging.worker_index is None

        # Validate start works and sets correct state
        start_time = time.time() + 0.1
        await instance.start(start_time=start_time)
        assert instance.state is not None
        assert isinstance(instance.state, WorkerGroupState)
        assert not instance.requests_generated_event.is_set()
        assert not instance.constraint_reached_event.is_set()
        assert not instance.shutdown_event.is_set()
        assert not instance.error_event.is_set()

        # Test iter updates
        requests_tracker = {}

        async for (
            response,
            request,
            request_info,
            scheduler_state,
        ) in instance.request_updates():
            # Validate returned request
            assert request is not None

            # Validate returned request info and response
            assert request_info is not None
            assert isinstance(request_info, RequestInfo)
            assert request_info.request_id is not None
            assert request_info.status is not None
            if request_info.request_id not in requests_tracker:
                requests_tracker[request_info.request_id] = {
                    "received_pending": 0,
                    "received_in_progress": 0,
                    "received_resolved": 0,
                    "received_cancelled": 0,
                }
            assert request_info.scheduler_node_id > -1
            assert request_info.scheduler_process_id > -1
            assert request_info.scheduler_start_time == start_time
            assert request_info.scheduler_timings is not None
            if request_info.status == "pending":
                requests_tracker[request_info.request_id]["received_pending"] += 1
                assert request_info.scheduler_timings.dequeued is not None
                assert request_info.scheduler_timings.targeted_start is not None
                assert request_info.scheduler_timings.targeted_start >= start_time
            elif request_info.status == "in_progress":
                requests_tracker[request_info.request_id]["received_in_progress"] += 1
                assert request_info.scheduler_timings.scheduled_at is not None
                assert (
                    request_info.scheduler_timings.scheduled_at
                    >= request_info.scheduler_timings.dequeued
                )
                assert request_info.scheduler_timings.resolve_start is not None
                assert (
                    request_info.scheduler_timings.resolve_start
                    >= request_info.scheduler_timings.scheduled_at
                )
            elif request_info.status == "completed":
                requests_tracker[request_info.request_id]["received_resolved"] += 1
                assert response is not None
                assert request_info.scheduler_timings.resolve_end is not None
                assert (
                    request_info.scheduler_timings.resolve_end
                    > request_info.scheduler_timings.resolve_start
                )
                assert request_info.request_timings is not None
                assert isinstance(request_info.request_timings, MockRequestTimings)
                assert request_info.request_timings.request_start is not None
                assert (
                    request_info.request_timings.request_start
                    >= request_info.scheduler_timings.targeted_start
                )
                assert request_info.request_timings.request_end is not None
                assert (
                    request_info.request_timings.request_end
                    >= request_info.request_timings.request_start
                )
            elif request_info.status in ("errored", "cancelled"):
                assert response is None
                requests_tracker[request_info.request_id]["received_resolved"] += 1
                assert request_info.scheduler_timings.resolve_end is not None
                assert (
                    request_info.scheduler_timings.resolve_end
                    > request_info.scheduler_start_time
                )
                if request_info.status == "cancelled":
                    requests_tracker[request_info.request_id]["received_cancelled"] += 1

            # Validate state structure
            assert scheduler_state is not None
            assert isinstance(scheduler_state, SchedulerState)
            assert scheduler_state.node_id > -1
            assert scheduler_state.start_time == start_time
            assert scheduler_state.end_time is not None
            if constructor_args.get("constraints"):
                assert scheduler_state.remaining_fraction is not None
                assert scheduler_state.remaining_fraction >= 0.0
                assert scheduler_state.remaining_fraction <= 1.0
            if constructor_args.get("constraints", {}).get("max_num") is not None:
                assert scheduler_state.remaining_requests is not None
                assert scheduler_state.remaining_requests >= 0
                assert (
                    scheduler_state.remaining_requests
                    <= constructor_args["constraints"]["max_num"].max_num
                )
            if constructor_args.get("constraints", {}).get("max_duration") is not None:
                assert scheduler_state.remaining_duration is not None
                assert scheduler_state.remaining_duration >= 0.0
                assert (
                    scheduler_state.remaining_duration
                    <= constructor_args["constraints"]["max_duration"].max_duration
                )
            assert scheduler_state.created_requests >= 0
            assert scheduler_state.queued_requests >= 0
            assert scheduler_state.pending_requests >= 0
            assert scheduler_state.processing_requests >= 0
            assert scheduler_state.processed_requests >= 0
            assert scheduler_state.successful_requests >= 0
            assert scheduler_state.errored_requests >= 0
            assert scheduler_state.cancelled_requests >= 0

        # Validate correctness of all updates
        for _, counts in requests_tracker.items():
            assert counts["received_cancelled"] in (0, 1)
            if counts["received_cancelled"] == 0:
                assert counts["received_pending"] == 1
                assert counts["received_in_progress"] >= 1
                assert counts["received_resolved"] == 1
        assert scheduler_state is not None  # last yielded state
        assert scheduler_state.end_time > scheduler_state.start_time
        assert scheduler_state.end_queuing_time is not None
        assert scheduler_state.end_queuing_constraints is not None
        assert scheduler_state.end_processing_time is not None
        assert scheduler_state.end_processing_time >= scheduler_state.start_time
        assert scheduler_state.end_processing_constraints is not None
        assert scheduler_state.scheduler_constraints is not None
        assert scheduler_state.created_requests == len(requests_tracker)
        assert scheduler_state.queued_requests == 0
        assert scheduler_state.pending_requests == 0
        assert scheduler_state.processing_requests == 0
        assert scheduler_state.processed_requests == len(requests_tracker)
        assert scheduler_state.successful_requests >= 0
        assert scheduler_state.errored_requests >= 0
        assert scheduler_state.cancelled_requests >= 0
        assert (
            scheduler_state.successful_requests
            + scheduler_state.errored_requests
            + scheduler_state.cancelled_requests
            == len(requests_tracker)
        )
        if constructor_args.get("constraints"):
            assert list(scheduler_state.scheduler_constraints.keys()) == list(
                constructor_args["constraints"].keys()
            )
            assert scheduler_state.remaining_fraction == 0.0
            if "max_num" in constructor_args["constraints"]:
                assert "max_num" in scheduler_state.end_queuing_constraints
                assert "max_num" in scheduler_state.end_processing_constraints
                max_num = constructor_args["constraints"]["max_num"].max_num
                assert scheduler_state.created_requests == max_num
                assert scheduler_state.successful_requests == max_num
                assert scheduler_state.errored_requests == 0
                assert scheduler_state.cancelled_requests == 0
            if "max_duration" in constructor_args["constraints"]:
                assert "max_duration" in scheduler_state.end_queuing_constraints
                assert "max_duration" in scheduler_state.end_processing_constraints
                assert scheduler_state.remaining_duration == 0.0
        else:
            assert "requests_exhausted" in scheduler_state.scheduler_constraints
            assert "requests_exhausted" in scheduler_state.end_queuing_constraints
            assert "requests_exhausted" in scheduler_state.end_processing_constraints
            assert scheduler_state.remaining_fraction is None
            assert scheduler_state.remaining_requests is None
            assert scheduler_state.remaining_duration is None

        # Test shutdown
        exceptions = await instance.shutdown()

        # Check valid shutdown behavior
        assert isinstance(exceptions, list)
        assert len(exceptions) == 0
        assert instance.messaging is None
        assert instance.state is None
        assert instance.processes is None
        assert instance.startup_barrier is None
        assert instance.requests_generated_event is None
        assert instance.constraint_reached_event is None
        assert instance.shutdown_event is None
        assert instance.error_event is None
        assert instance.mp_manager is None
        assert instance.mp_context is None
