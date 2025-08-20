from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Generic

import pytest

from guidellm.scheduler import (
    AsyncConstantStrategy,
    BackendInterface,
    ConcurrentStrategy,
    MaxDurationConstraint,
    MaxNumberConstraint,
    MeasuredRequestTimings,
    ScheduledRequestInfo,
    SchedulerMessagingPydanticRegistry,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
)


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class MockRequestTimings(MeasuredRequestTimings):
    """Mock timing implementation for testing."""


SchedulerMessagingPydanticRegistry.register("MockRequestTimings")(ScheduledRequestInfo)


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

    def info(self) -> dict[str, Any]:
        return {"type": "mock"}

    async def process_startup(self):
        pass

    async def validate(self):
        pass

    async def process_shutdown(self):
        pass

    async def resolve(self, request, request_info, request_history):
        yield f"response_for_{request}", request_info


class TestWorkerProcessGroup:
    """Test suite for WorkerProcessGroup class."""

    @pytest.fixture(
        params=[
            {
                "requests": None,
                "cycle_requests": ["request1", "request2", "request3"],
                "strategy": SynchronousStrategy(),
                "constraints": {"max_num": MaxNumberConstraint(max_num=10)},
            },
            {
                "requests": None,
                "cycle_requests": ["req_a", "req_b"],
                "strategy": ConcurrentStrategy(streams=2),
                "constraints": {"max_num": MaxNumberConstraint(max_num=5)},
            },
            {
                "requests": ["req_x", "req_y", "req_z"],
                "cycle_requests": None,
                "strategy": ThroughputStrategy(max_concurrency=5),
                "constraints": {},
            },
            {
                "requests": None,
                "cycle_requests": ["req_8", "req_9", "req_10"],
                "strategy": AsyncConstantStrategy(rate=20),
                "constraints": {"max_duration": MaxDurationConstraint(max_duration=1)},
            },
        ],
        ids=["sync_max", "concurrent_max", "throughput_no_cycle", "constant_duration"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for WorkerProcessGroup."""
        constructor_args = request.param.copy()
        instance = WorkerProcessGroup(**request.param, backend=MockBackend())
        return instance, constructor_args

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
        assert len(type_args) == 3

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
        assert instance.requests is constructor_args["requests"]
        assert instance.cycle_requests is constructor_args["cycle_requests"]
        assert isinstance(instance.strategy, type(constructor_args["strategy"]))
        assert isinstance(instance.constraints, dict)
        assert instance.constraints == constructor_args["constraints"]

        # Multiprocessing attributes (should be None initially)
        assert instance.mp_context is None
        assert instance.mp_manager is None
        assert instance.processes is None

        # Synchronization primitives (should be None initially)
        assert instance.startup_barrier is None
        assert instance.shutdown_event is None
        assert instance.error_event is None

        # Scheduler state and messaging (should be None initially)
        assert instance._state is None
        assert instance.messaging is None

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("requests", "cycle_requests", "expected_error"),
        [
            (None, None, ValueError),
            ([], iter([]), ValueError),  # cycle_requests as Iterator
            (None, iter(["req1"]), ValueError),  # cycle_requests as Iterator
        ],
        ids=["no_requests", "cycle_as_iterator_empty", "cycle_as_iterator_data"],
    )
    def test_invalid_initialization_values(
        self, requests, cycle_requests, expected_error
    ):
        """Test WorkerProcessGroup with invalid initialization values."""
        with pytest.raises(expected_error):
            WorkerProcessGroup(
                requests=requests,
                cycle_requests=cycle_requests,
                backend=MockBackend(),
                strategy=SynchronousStrategy(),
                constraints={},
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test WorkerProcessGroup initialization without required fields."""
        with pytest.raises(TypeError):
            WorkerProcessGroup()

    @pytest.mark.smoke
    @async_timeout(10)
    @pytest.mark.asyncio
    async def test_lifecycle(self, valid_instances: tuple[WorkerProcessGroup, dict]):
        """Test the lifecycle methods of WorkerProcessGroup."""
        instance, constructor_args = valid_instances

        # Test create processes
        await instance.create_processes()

        # Check valid process creation
        assert instance.mp_context is not None
        assert instance.mp_manager is not None
        assert instance.processes is not None
        assert len(instance.processes) > 0
        assert all(proc.is_alive() for proc in instance.processes)
        assert instance.startup_barrier is not None
        assert instance.shutdown_event is not None
        assert instance.error_event is not None
        assert instance.requests_completed_event is not None
        assert instance.messaging is not None

        # Test start
        start_time = time.time() + 0.1
        await instance.start(start_time=start_time)

        # Check valid start behavior
        assert instance.messaging is not None
        assert instance._state is not None
        assert instance._state._start_time == start_time
        assert instance._state._state.num_processes == len(instance.processes)
        assert not instance.error_event.is_set()

        # Test iter updates
        updates_list = []
        responses_count = 0

        async for (
            response,
            request,
            request_info,
            scheduler_state,
        ) in instance.request_updates():
            updates_list.append((response, request, request_info, scheduler_state))
            if response is not None:
                responses_count += 1

            # Validate request info structure
            assert hasattr(request_info, "request_id")
            assert hasattr(request_info, "status")
            valid_statuses = [
                "queued",
                "in_progress",
                "completed",
                "errored",
                "cancelled",
            ]
            assert request_info.status in valid_statuses

            # Validate state structure
            assert hasattr(scheduler_state, "created_requests")
            assert hasattr(scheduler_state, "processed_requests")
            assert hasattr(scheduler_state, "successful_requests")
            assert scheduler_state.created_requests >= 0
            assert scheduler_state.processed_requests >= 0
            assert scheduler_state.successful_requests >= 0

        # Validate correctness of all updates
        if constructor_args.get("requests") is not None:
            assert len(updates_list) == 2 * len(constructor_args["requests"]), (
                "Should have received updates for all requests"
            )
        if constructor_args.get("constraints", {}).get("max_num") is not None:
            assert (
                len(updates_list)
                == 2 * constructor_args["constraints"]["max_num"].max_num
            ), "Should not have received more updates than max_num constraint"

        assert len(updates_list) > 0, "Should have received at least one update"

        # Constraints should be satisfied
        for constraint_name, _ in constructor_args["constraints"].items():
            constraint_check = (
                "max" in constraint_name.lower()
                or "duration" in constraint_name.lower()
            )
            if constraint_check:
                assert scheduler_state.end_processing_time is not None, (
                    f"Should have stopped processing due to {constraint_name}"
                )

        # Test shutdown
        exceptions = await instance.shutdown()

        # Check valid shutdown behavior
        assert isinstance(exceptions, list), "Shutdown should return list of exceptions"
        assert instance.messaging is None, "Messaging should be cleared after shutdown"
        assert instance._state is None, "State should be cleared after shutdown"
        assert instance.processes is None, "Processes should be cleared after shutdown"
        assert instance.mp_manager is None, (
            "MP manager should be cleared after shutdown"
        )
        assert instance.mp_context is None, (
            "MP context should be cleared after shutdown"
        )
