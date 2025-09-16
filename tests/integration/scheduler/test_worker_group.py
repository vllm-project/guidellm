"""
Integration tests for WorkerProcessGroup.

Tests the complete lifecycle of the worker group with real multiprocessing
worker processes and a mock backend. Validates end-to-end functionality
across different scheduling strategies and constraints.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from functools import wraps
from typing import Any

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
    MeasuredRequestTimings,
    SynchronousStrategy,
    ThroughputStrategy,
    WorkerProcessGroup,
)
from guidellm.scheduler.constraints import ConstraintInitializer
from guidellm.scheduler.strategy import SchedulingStrategy


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


class MockRequestTimings(MeasuredRequestTimings):
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
            {"max_num": MaxNumberConstraint(max_num=100)},
            {"max_duration": MaxDurationConstraint(max_duration=0.5)},
            {"max_errors": MaxErrorsConstraint(max_errors=20)},
            {"max_error_rate": MaxErrorRateConstraint(max_error_rate=0.1)},
            {"max_global_error_rate": MaxGlobalErrorRateConstraint(max_error_rate=0.1)},
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
        requests = [f"request_{ind}" for ind in range(1000)]
        group = WorkerProcessGroup(
            backend=backend,
            requests=requests,
            strategy=strategy,
            constraints={
                key: init.create_constraint() for key, init in constraints_inits.items()
            },
            infinite_requests=False,
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
