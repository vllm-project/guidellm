from __future__ import annotations

import asyncio
import inspect
import random
import uuid
from functools import wraps
from typing import Any, Generic

import pytest
from pydantic import BaseModel, Field

from guidellm.scheduler import (
    BackendInterface,
    MaxNumberConstraint,
    NonDistributedEnvironment,
    ScheduledRequestInfo,
    Scheduler,
    SchedulerState,
    SynchronousStrategy,
)
from guidellm.utils.singleton import ThreadSafeSingletonMixin


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

        yield f"response_for_{request.payload}"


class TestScheduler:
    """Test suite for Scheduler class."""

    @pytest.fixture
    def valid_instances(self):
        """Fixture providing test data for Scheduler."""
        # Clear singleton state between tests
        if hasattr(Scheduler, "singleton_instance"):
            Scheduler.singleton_instance = None

        instance = Scheduler()
        constructor_args = {}
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Scheduler inheritance and type relationships."""
        # Clear singleton before testing
        if hasattr(Scheduler, "singleton_instance"):
            Scheduler.singleton_instance = None

        assert issubclass(Scheduler, ThreadSafeSingletonMixin)
        assert issubclass(Scheduler, Generic)
        assert hasattr(Scheduler, "run")
        assert callable(Scheduler.run)

        # Check method signature
        run_sig = inspect.signature(Scheduler.run)
        expected_params = [
            "self",
            "requests",
            "backend",
            "strategy",
            "env",
            "constraints",
        ]
        param_names = list(run_sig.parameters.keys())
        assert param_names == expected_params

        # Check that run is async generator (returns AsyncIterator)
        assert hasattr(Scheduler.run, "__code__")
        code = Scheduler.run.__code__
        # Check for async generator flags or return annotation
        assert (
            inspect.iscoroutinefunction(Scheduler.run)
            or "AsyncIterator" in str(run_sig.return_annotation)
            or code.co_flags & 0x100  # CO_GENERATOR flag
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test Scheduler initialization as singleton."""
        instance1, _ = valid_instances
        instance2 = Scheduler()

        assert isinstance(instance1, Scheduler)
        assert instance1 is instance2
        assert id(instance1) == id(instance2)
        assert hasattr(instance1, "thread_lock")

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @pytest.mark.parametrize(
        ("num_requests", "constraint_args"),
        [
            (5, {"max_number": MaxNumberConstraint(max_num=10)}),
            (20, {"max_number": MaxNumberConstraint(max_num=25)}),
            (1, {"max_number": MaxNumberConstraint(max_num=5)}),
        ],
    )
    async def test_run_basic_functionality(
        self, valid_instances, num_requests, constraint_args
    ):
        """Test Scheduler.run basic functionality with various parameters."""
        instance, _ = valid_instances
        requests = [MockRequest(payload=f"req_{i}") for i in range(num_requests)]
        backend = MockBackend(error_rate=0.0, response_delay=0.001)
        strategy = SynchronousStrategy()
        env = NonDistributedEnvironment()

        results = []
        async for response, _request, info, _state in instance.run(
            requests=requests,
            backend=backend,
            strategy=strategy,
            env=env,
            **constraint_args,
        ):
            results.append((response, _request, info, _state))

        assert len(results) > 0
        assert all(isinstance(r[1], MockRequest) for r in results)
        assert all(isinstance(r[2], ScheduledRequestInfo) for r in results)
        assert all(isinstance(r[3], SchedulerState) for r in results)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_run_with_errors(self, valid_instances):
        """Test Scheduler.run error handling."""
        instance, _ = valid_instances
        requests = [MockRequest(payload=f"req_{i}") for i in range(5)]
        backend = MockBackend(error_rate=1.0)  # Force all requests to error
        strategy = SynchronousStrategy()
        env = NonDistributedEnvironment()

        error_count = 0
        async for response, _request, info, _state in instance.run(
            requests=requests,
            backend=backend,
            strategy=strategy,
            env=env,
            max_number=MaxNumberConstraint(max_num=10),
        ):
            if info.status == "errored":
                error_count += 1
                assert response is None
                assert info.error is not None

        assert error_count > 0

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_run_invalid_parameters(self, valid_instances):
        """Test Scheduler.run with invalid parameters."""
        instance, _ = valid_instances

        with pytest.raises((TypeError, ValueError, AttributeError)):
            async for _ in instance.run(
                requests=None,  # Invalid requests
                backend=None,  # Invalid backend
                strategy=SynchronousStrategy(),
                env=NonDistributedEnvironment(),
            ):
                pass

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_run_constraint_variations(self, valid_instances):
        """Test Scheduler.run with different constraint types."""
        instance, _ = valid_instances
        requests = [MockRequest(payload=f"req_{i}") for i in range(3)]
        backend = MockBackend(error_rate=0.0, response_delay=0.001)
        strategy = SynchronousStrategy()
        env = NonDistributedEnvironment()

        # Test with multiple constraints
        results = []
        async for response, request, info, state in instance.run(
            requests=requests,
            backend=backend,
            strategy=strategy,
            env=env,
            max_number=MaxNumberConstraint(max_num=5),
            max_duration=5.0,  # Should be converted to constraint
        ):
            results.append((response, request, info, state))

        assert len(results) > 0
