from __future__ import annotations

import asyncio
import inspect
import math
import os
import queue
import threading
import time
from collections import defaultdict
from functools import wraps
from multiprocessing import get_context
from queue import Empty
from typing import Any, Generic

import culsans
import pytest

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    BackendInterface,
    ConcurrentStrategy,
    MaxNumberConstraint,
    MeasuredRequestTimings,
    ScheduledRequestInfo,
    SchedulerState,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
    worker_group,
)
from guidellm.utils import MessageEncoding


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class MockWorker:
    """Picklable mock worker used to validate create_processes logic."""

    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(
        self,
        local_rank,
        local_world_size,
        async_limit,
        startup_barrier,
        shutdown_event,
        error_event,
        requests_queue,
        updates_queue,
        backend,
        request_timings,
        poll_intervals,
    ):
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.async_limit = async_limit
        self.startup_barrier = startup_barrier
        self.shutdown_event = shutdown_event
        self.error_event = error_event
        self.requests_queue = requests_queue
        self.updates_queue = updates_queue
        self.backend = backend
        self.request_timings = request_timings
        self.poll_intervals = poll_intervals

    def run(self):
        try:
            # Access parameters to ensure they're usable and wait for barrier
            shutdown_is_set = self.shutdown_event.is_set()
            error_is_set = self.error_event.is_set()
            backend_info = self.backend.info()

            self.startup_barrier.wait()

            # Publish diagnostics back to parent for assertions
            payload = (
                "diag",
                self.local_rank,
                {
                    "child_pid": os.getpid(),
                    "local_rank": self.local_rank,
                    "local_world_size": self.local_world_size,
                    "async_limit": self.async_limit,
                    "backend_info": backend_info,
                    "shutdown_is_set": shutdown_is_set,
                    "error_is_set": error_is_set,
                    "passed_barrier": True,
                    "request_timings_type": type(self.request_timings).__name__,
                },
            )
            self.updates_queue.put(payload)
        except Exception as err:  # noqa: BLE001
            try:
                self.error_event.set()
                self.updates_queue.put(("error", self.local_rank, repr(err)))
            finally:
                raise


class MockWorkerProcessor(MockWorker):
    def run(self):
        self.startup_barrier.wait()

        while not self.shutdown_event.is_set() and not self.error_event.is_set():
            try:
                request_msg = self.requests_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            request, request_info = MessageEncoding.decode_message(request_msg)
            request_info.status = "in_progress"
            self.updates_queue.put(
                MessageEncoding.encode_message((None, request, request_info))
            )
            time.sleep(0.01)
            request_info.status = "completed"
            response = f"response_for_{request}"
            self.updates_queue.put(
                MessageEncoding.encode_message((response, request, request_info))
            )


class MockRequestTimings(MeasuredRequestTimings):
    """Mock timing implementation for testing."""


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
        yield f"response_for_{request}"


class TestWorkerProcessGroup:
    """Test suite for WorkerProcessGroup class."""

    @pytest.fixture(
        params=[
            {
                "requests": ["request1", "request2", "request3"],
                "strategy": SynchronousStrategy(),
                "constraints": {"max_requests": MaxNumberConstraint(max_num=10)},
            },
            {
                "requests": ["req_a", "req_b"],
                "strategy": ConcurrentStrategy(streams=2),
                "constraints": {},
            },
            {
                "requests": iter(["req_x", "req_y", "req_z"]),
                "strategy": ThroughputStrategy(max_concurrency=5),
                "constraints": {"max_num": MaxNumberConstraint(max_num=5)},
                "infinite_requests": False,
            },
        ],
        ids=["basic_sync", "concurrent", "throughput_iterator"],
    )
    def valid_instances(self, request):
        """Fixture providing test data for WorkerProcessGroup."""
        constructor_args = request.param.copy()
        backend = MockBackend()
        constructor_args["backend"] = backend

        instance = WorkerProcessGroup(**constructor_args)
        return instance, constructor_args

    @pytest.fixture
    def worker_process_group(self):
        """Create a basic WorkerProcessGroup instance for testing."""
        backend = MockBackend()
        requests = ["request1", "request2", "request3"]
        strategy = SynchronousStrategy()
        constraints = {"max_requests": MaxNumberConstraint(max_num=10)}

        return WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=strategy,
            constraints=constraints,
        )

    @pytest.mark.smoke
    def test_class_signatures(self, worker_process_group: WorkerProcessGroup):
        """Test inheritance and type relationships."""
        # Class
        assert isinstance(worker_process_group, Generic)
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
        assert isinstance(instance.strategy, type(constructor_args["strategy"]))
        assert isinstance(instance.constraints, dict)
        assert instance.constraints == constructor_args["constraints"]

        # Optional attributes
        expected_infinite = constructor_args.get("infinite_requests", None)
        assert instance.infinite_requests == expected_infinite

        # Multiprocessing attributes (should be None initially)
        assert instance.mp_context is None
        assert instance.processes is None

        # Synchronization primitives (should be None initially)
        assert instance.startup_barrier is None
        assert instance.shutdown_event is None
        assert instance.error_event is None

        # Queues (should be None initially)
        assert instance.requests_queue is None
        assert instance.updates_queue is None
        assert instance.pending_updates_queue is None
        assert instance.pending_requests_complete is None
        assert instance.pending_updates_complete is None

        # Scheduler state and tasks (should be None initially)
        assert instance.state_update_lock is None
        assert instance.scheduler_state is None
        assert instance.populate_requests_task is None
        assert instance.populate_updates_task is None

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test WorkerProcessGroup with invalid field values."""
        backend = MockBackend()
        requests = ["req1"]
        strategy = SynchronousStrategy()
        constraints = {}

        # Test with None requests (will likely fail during create_processes)
        group1 = WorkerProcessGroup(
            requests=None,
            backend=backend,
            strategy=strategy,
            constraints=constraints,
        )
        assert group1.requests is None

        # Test with None backend (will likely fail during create_processes)
        group2 = WorkerProcessGroup(
            requests=requests,
            backend=None,
            strategy=strategy,
            constraints=constraints,
        )
        assert group2.backend is None

        # Test with None strategy (will likely fail during create_processes)
        group3 = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=None,
            constraints=constraints,
        )
        assert group3.strategy is None

        # Test with None constraints (will likely fail during create_processes)
        group4 = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=strategy,
            constraints=None,
        )
        assert group4.constraints is None

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("strategy", "expected_num_procs", "expected_max_conc"),
        [
            (SynchronousStrategy(), 1, 1),
            (ConcurrentStrategy(streams=3), 3, 3),
            (ThroughputStrategy(max_concurrency=6), 3, 6),
            (AsyncConstantStrategy(rate=100.0), 3, 12),
            (AsyncPoissonStrategy(rate=100.0), 3, 12),
        ],
    )
    async def test_create_processes(
        self,
        monkeypatch,
        strategy,
        expected_num_procs,
        expected_max_conc,
    ):
        # Patch required mock settings
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 3, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 12, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

        # Setup group to test
        backend = MockBackend()
        requests = [f"r{i}" for i in range(10)]
        constraints = {"max_requests": MaxNumberConstraint(max_num=100)}
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=strategy,
            constraints=constraints,
        )

        # Run within a reasonable time limit
        try:
            await asyncio.wait_for(group.create_processes(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("create_processes() timed out after 5 seconds")

        # Check expected attributes are created
        assert group.mp_context is not None
        assert hasattr(group.mp_context, "Barrier")
        assert hasattr(group.mp_context, "Event")
        assert hasattr(group.mp_context, "Queue")
        assert group.processes is not None
        assert len(group.processes) == expected_num_procs

        # Validate processes ran correctly
        diags: dict[int, dict] = {}
        for _ in range(expected_num_procs):
            kind, rank, payload = group.updates_queue.get(timeout=3)
            if kind == "error":
                pytest.fail(f"Worker {rank} reported error: {payload}")
            assert kind == "diag"
            diags[rank] = payload

        # Verify returned processes state
        main_pid = os.getpid()
        assert len(diags) == expected_num_procs
        for rank, payload in diags.items():
            assert payload["local_rank"] == rank
            assert payload["local_world_size"] == expected_num_procs
            assert payload["passed_barrier"] is True
            assert payload["shutdown_is_set"] is False
            assert payload["error_is_set"] is False
            assert isinstance(payload["backend_info"], dict)
            assert payload["child_pid"] != main_pid
        per_proc = math.ceil(expected_max_conc / expected_num_procs)
        expected_last = expected_max_conc - per_proc * (expected_num_procs - 1)
        for rank, payload in diags.items():
            exp_limit = per_proc if rank < expected_num_procs - 1 else expected_last
            assert payload["async_limit"] == exp_limit

        exceptions = await group.shutdown()
        assert len(exceptions) == 0, f"Shutdown encountered exceptions: {exceptions}"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_start(self, monkeypatch):
        # Patch required mock settings
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

        # Setup group and mimic create_processes
        backend = MockBackend()
        requests = [f"r{i}" for i in range(5)]  # to few requests, test new iter logic
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=SynchronousStrategy(),
            constraints={"max_num": MaxNumberConstraint(max_num=10)},
        )
        group.mp_context = get_context("fork")
        group.startup_barrier = group.mp_context.Barrier(2)
        group.shutdown_event = group.mp_context.Event()
        group.error_event = group.mp_context.Event()
        group.requests_queue = group.mp_context.Queue()
        group.updates_queue = group.mp_context.Queue()
        group.pending_updates_queue = culsans.Queue()
        group.pending_updates_complete = threading.Event()
        group.processes = [None]

        # Validate function runs and returns at start_time
        start_time = time.time() + 0.2
        await asyncio.wait_for(group.start(start_time), timeout=3.0)
        end_time = time.time()
        assert end_time == pytest.approx(start_time, abs=0.01)

        # Validate instance state
        assert group.state_update_lock is not None
        assert hasattr(group.state_update_lock, "acquire")
        assert group.scheduler_state is not None
        assert group.scheduler_state.num_processes == 1
        assert group.scheduler_state.start_time == start_time
        assert isinstance(group.populate_requests_task, asyncio.Task)
        assert isinstance(group.populate_updates_task, asyncio.Task)

        # Pull the queued requests
        await asyncio.sleep(0.1)
        sent_requests = []
        while True:
            await asyncio.sleep(0)
            try:
                req = group.requests_queue.get(timeout=1.0)
                sent_requests.append(req)
            except Empty:
                break
        assert len(sent_requests) == 10

        # Enqueue lifecycle updates
        for req in requests + requests:
            group.updates_queue.put(
                MessageEncoding.encode_message(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="in_progress",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )
            group.updates_queue.put(
                MessageEncoding.encode_message(
                    (
                        None,
                        req,
                        ScheduledRequestInfo[MockRequestTimings](
                            request_id=str(req),
                            status="completed",
                            scheduler_node_id=0,
                            scheduler_process_id=0,
                            scheduler_start_time=start_time,
                        ),
                    )
                )
            )
            await asyncio.sleep(0)

        # Drain 3 updates per request (queued, started, completed)
        await asyncio.sleep(0.1)
        updates = []
        for _ in range(3 * 10):
            try:
                update = await asyncio.wait_for(
                    group.pending_updates_queue.async_get(), timeout=1.0
                )
                updates.append(update)
            except asyncio.TimeoutError:
                break
        assert len(updates) == 3 * 10

        # Ensure tasks finish
        if not group.populate_requests_task.done():
            await asyncio.wait_for(group.populate_requests_task, timeout=1.0)
        if not group.populate_updates_task.done():
            await asyncio.wait_for(group.populate_updates_task, timeout=1.0)

        # Clean up resources
        group.processes = None
        exceptions = await group.shutdown()
        assert len(exceptions) == 0, f"Shutdown encountered exceptions: {exceptions}"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(3.0)
    async def test_error_handling_basic(self, monkeypatch):
        """Test basic error handling patterns."""
        self._setup_test_environment(monkeypatch)

        backend = MockBackend()
        requests = ["req1"]
        # Create group directly without using helper (which calls start automatically)
        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )

        # Test that error_event can be accessed when not initialized
        # First save the existing error_event
        original_error_event = group.error_event

        # Temporarily set to None to test this state
        group.error_event = None
        assert group.error_event is None

        # Restore it for the start test
        group.error_event = original_error_event

        # Test basic group state validation
        with pytest.raises(
            RuntimeError, match="create_processes.*must be called before start"
        ):
            await group.start(time.time())

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_shutdown_event_stops_tasks(self, monkeypatch):
        """Test that setting shutdown event stops background tasks."""
        self._setup_test_environment(monkeypatch)

        # Setup group
        backend = MockBackend()
        requests = [f"req_{i}" for i in range(5)]
        group = self._create_test_group(backend, requests)

        # Start and verify tasks
        start_time = time.time() + 0.1
        await group.start(start_time)

        # Simulate some processing
        self._process_test_requests(group, start_time, count=2)
        await asyncio.sleep(0.05)

        # Set shutdown event and verify tasks stop
        group.shutdown_event.set()
        await asyncio.sleep(0.1)  # Allow propagation

        assert group.pending_requests_complete.is_set()
        assert group.populate_requests_task.done()

        # Clean up
        await group.shutdown()

    def _setup_test_environment(self, monkeypatch):
        """Helper to setup test environment with mocked settings."""
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(worker_group, "WorkerProcess", MockWorker, raising=True)

    def _create_test_group(self, backend, requests):
        """Helper to create a test group with mocked multiprocessing components."""
        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )
        group.mp_context = get_context("fork")
        group.startup_barrier = group.mp_context.Barrier(2)
        group.shutdown_event = group.mp_context.Event()
        group.error_event = group.mp_context.Event()
        group.requests_queue = group.mp_context.Queue(maxsize=1)
        group.updates_queue = group.mp_context.Queue()
        group.pending_updates_queue = culsans.Queue()
        group.pending_updates_complete = threading.Event()
        # Create mock process objects instead of None
        mock_process = type(
            "MockProcess",
            (),
            {"join": lambda self, timeout=None: None, "exitcode": 0, "pid": 12345},
        )()
        group.processes = [mock_process]
        return group

    def _process_test_requests(self, group, start_time, count=1):
        """Helper to process test requests and generate updates."""
        for _ in range(count):
            try:
                req, req_info = MessageEncoding.decode_message(
                    group.requests_queue.get(timeout=0.1)
                )
                # Simulate in_progress update
                group.updates_queue.put(
                    MessageEncoding.encode_message(
                        (
                            None,
                            req,
                            ScheduledRequestInfo[MockRequestTimings](
                                request_id=str(req),
                                status="in_progress",
                                scheduler_node_id=0,
                                scheduler_process_id=0,
                                scheduler_start_time=start_time,
                            ),
                        )
                    )
                )
                # Simulate completed update
                group.updates_queue.put(
                    MessageEncoding.encode_message(
                        (
                            None,
                            req,
                            ScheduledRequestInfo[MockRequestTimings](
                                request_id=str(req),
                                status="completed",
                                scheduler_node_id=0,
                                scheduler_process_id=0,
                                scheduler_start_time=start_time,
                            ),
                        )
                    )
                )
            except Empty:
                break

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_request_updates(self, monkeypatch):
        """Test the request_updates async iterator functionality."""
        # Configure settings for controlled testing
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)
        monkeypatch.setattr(
            worker_group.settings, "scheduler_poll_interval", 0.01, raising=False
        )
        monkeypatch.setattr(
            worker_group, "WorkerProcess", MockWorkerProcessor, raising=True
        )

        # Setup group
        backend = MockBackend()
        requests = [f"req_{index}" for index in range(20)]
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=SynchronousStrategy(),
            constraints={"max_num": MaxNumberConstraint(max_num=10)},
        )

        # Mimic create_processes to set required state
        await group.create_processes()
        await group.start(time.time() + 0.05)

        # Collect all updates from request_updates iterator
        received_updates = defaultdict(list)
        received_responses = []
        count = 0
        async for resp, req, req_info, state in group.request_updates():
            assert isinstance(req_info, ScheduledRequestInfo)
            assert isinstance(state, SchedulerState)
            received_updates[req].append(req_info.status)
            if resp is not None:
                received_responses.append(resp)
            count += 1

        # Check we have all expected updates (10 requests)
        assert len(received_updates) == 10
        for index, (req, statuses, resp) in enumerate(
            zip(received_updates.keys(), received_updates.values(), received_responses)
        ):
            assert req == f"req_{index}"
            assert resp == f"response_for_req_{index}"
            assert statuses == ["queued", "in_progress", "completed"]

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_shutdown_basic(self):
        """Test basic shutdown functionality."""
        backend = MockBackend()
        requests = ["req1", "req2"]
        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )

        # Test shutdown with empty state - should return no exceptions
        exceptions = await group.shutdown()
        assert len(exceptions) == 0
        assert group.processes is None
        assert group.mp_context is None
        assert group.shutdown_event is None

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_start_without_create_processes(self):
        """Test that start() raises error when create_processes() not called."""
        backend = MockBackend()
        requests = ["req1", "req2"]
        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )

        with pytest.raises(
            RuntimeError,
            match="create_processes\\(\\) must be called before start\\(\\)",
        ):
            await group.start(time.time())

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_create_processes_invalid_limits(self, monkeypatch):
        """Test create_processes with invalid process and concurrency limits."""
        # Test zero processes limit
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 0, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 1, raising=False)

        backend = MockBackend()
        requests = ["req1"]
        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )

        with pytest.raises(RuntimeError, match="num_processes resolved to 0"):
            await group.create_processes()

        # Test zero concurrency limit
        monkeypatch.setattr(
            worker_group.settings, "max_worker_processes", 1, raising=False
        )
        monkeypatch.setattr(worker_group.settings, "max_concurrency", 0, raising=False)

        group2 = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=SynchronousStrategy(),
            constraints={},
        )

        with pytest.raises(RuntimeError, match="max_concurrency resolved to 0"):
            await group2.create_processes()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_request_updates_error_handling(self, monkeypatch):
        """Test request_updates handles error events correctly."""
        # Use the helper method that creates mocked multiprocessing components
        self._setup_test_environment(monkeypatch)

        backend = MockBackend()
        requests = ["req1"]
        group = self._create_test_group(backend, requests)

        # Start the group with mocked components
        start_time = time.time() + 0.1
        await group.start(start_time)

        # Set error event to simulate error
        group.error_event.set()

        # Test that request_updates raises RuntimeError when error event is set
        with pytest.raises(
            RuntimeError, match="error_event is set in WorkerProcessGroup"
        ):
            async for _ in group.request_updates():
                pass

        # Clean up
        await group.shutdown()

    @pytest.mark.smoke
    def test_valid_instances_fixture(self):
        """Test the valid_instances fixture provides correct data."""
        backend = MockBackend()
        requests = ["request1", "request2", "request3"]
        strategy = SynchronousStrategy()
        constraints = {"max_requests": MaxNumberConstraint(max_num=10)}

        instance = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=strategy,
            constraints=constraints,
        )

        assert isinstance(instance, WorkerProcessGroup)
        assert instance.requests is requests
        assert instance.backend is backend
        assert instance.strategy is strategy
        assert instance.constraints is constraints

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "infinite_requests",
        [
            None,
            True,
            False,
        ],
    )
    def test_initialization_infinite_requests(self, infinite_requests):
        """Test initialization with different infinite_requests values."""
        backend = MockBackend()
        requests = ["req1", "req2"]
        strategy = SynchronousStrategy()
        constraints = {}

        group = WorkerProcessGroup(
            requests=requests,
            backend=backend,
            strategy=strategy,
            constraints=constraints,
            infinite_requests=infinite_requests,
        )

        assert group.infinite_requests == infinite_requests

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "missing_param",
        [
            "requests",
            "backend",
            "strategy",
            "constraints",
        ],
    )
    def test_invalid_initialization_missing_params(self, missing_param):
        """Test invalid initialization with missing required parameters."""
        # Create complete valid parameters
        params = {
            "requests": ["req1"],
            "backend": MockBackend(),
            "strategy": SynchronousStrategy(),
            "constraints": {},
        }

        # Remove the specified parameter
        del params[missing_param]

        with pytest.raises(TypeError):
            WorkerProcessGroup(**params)
