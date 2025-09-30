import inspect
import time
from abc import ABC
from typing import Generic
from unittest.mock import patch

import pytest

from guidellm.scheduler import (
    Environment,
    MaxNumberConstraint,
    NonDistributedEnvironment,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
    SynchronousStrategy,
)
from guidellm.utils import InfoMixin


class TestEnvironment:
    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Environment inheritance and type relationships."""
        # Inheritance and abstract class properties
        assert issubclass(Environment, ABC)
        assert issubclass(Environment, Generic)
        assert issubclass(Environment, InfoMixin)
        assert inspect.isabstract(Environment)
        assert hasattr(Environment, "info")

        # Abstract methods validation
        expected_abstract_methods = {
            "sync_run_params",
            "sync_run_start",
            "update_run_iteration",
            "sync_run_error",
            "sync_run_end",
        }
        assert Environment.__abstractmethods__ == expected_abstract_methods

        # Method signatures and async properties
        method_signatures = {
            "sync_run_params": ["self", "requests", "strategy", "constraints"],
            "sync_run_start": ["self"],
            "update_run_iteration": [
                "self",
                "response",
                "request",
                "request_info",
                "state",
            ],
            "sync_run_error": ["self", "err"],
            "sync_run_end": ["self"],
        }

        for method_name, expected_params in method_signatures.items():
            method = getattr(Environment, method_name)
            sig = inspect.signature(method)

            # Check parameter names and count
            param_names = list(sig.parameters.keys())
            assert param_names == expected_params

            # Check async nature
            assert inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(
                method
            )

        # Generic type parameters
        orig_bases = getattr(Environment, "__orig_bases__", ())
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
        assert RequestT in type_args
        assert ResponseT in type_args

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise TypeError."""

        class InvalidImplementation(Environment):
            pass

        with pytest.raises(TypeError):
            InvalidImplementation()

    @pytest.mark.sanity
    def test_partial_invalid_implementation(self):
        """Test that partial implementations raise TypeError."""

        class PartialImplementation(Environment):
            async def sync_run_params(self, requests, strategy, constraints):
                return requests, strategy, constraints

            async def sync_run_start(self):
                return 0.0

            # Missing other required methods

        with pytest.raises(TypeError):
            PartialImplementation()

    @pytest.mark.smoke
    def test_implementation_construction(self):
        """Test that concrete implementations can be constructed."""

        class TestEnvironment(Environment):
            async def sync_run_params(self, requests, strategy, constraints):
                return requests, strategy, constraints

            async def sync_run_start(self):
                return 0.0

            async def update_run_iteration(self, response, request, request_info):
                pass

            async def sync_run_error(self, err):
                pass

            async def sync_run_end(self):
                yield

        env = TestEnvironment()
        assert isinstance(env, Environment)


class TestNonDistributedEnvironment:
    @pytest.fixture
    def valid_instances(self):
        """Fixture providing test data for NonDistributedEnvironment."""
        instance = NonDistributedEnvironment()
        return instance, {}

    @pytest.mark.smoke
    def test_class_signatures(self, valid_instances):
        """Test NonDistributedEnvironment inheritance and type relationships."""
        instance, constructor_args = valid_instances
        assert issubclass(NonDistributedEnvironment, Environment)
        assert issubclass(NonDistributedEnvironment, InfoMixin)
        assert not inspect.isabstract(NonDistributedEnvironment)

        # Should inherit from Environment
        assert isinstance(instance, Environment)
        assert issubclass(NonDistributedEnvironment, Environment)

        # Should implement all required methods
        required_methods = [
            "sync_run_params",
            "sync_run_start",
            "update_run_iteration",
            "sync_run_error",
            "sync_run_end",
        ]

        for method_name in required_methods:
            assert hasattr(instance, method_name)
            assert callable(getattr(instance, method_name))

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test NonDistributedEnvironment initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, NonDistributedEnvironment)
        assert isinstance(instance, Environment)
        assert instance.run_errors == []

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test that initialization doesn't accept invalid arguments."""
        with pytest.raises(TypeError):
            NonDistributedEnvironment("invalid_arg")

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("requests", "strategy", "constraints"),
        [
            (
                ["request1", "request2"],
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=10)},
            ),
            (
                [],
                SynchronousStrategy(),
                {},
            ),
            (
                ["single_request"],
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=1)},
            ),
            (
                range(5),
                SynchronousStrategy(),
                {"max_requests": MaxNumberConstraint(max_num=5)},
            ),
        ],
        ids=[
            "multiple_requests",
            "empty_requests",
            "single_request",
            "range_requests",
        ],
    )
    async def test_sync_run_params(
        self, valid_instances, requests, strategy, constraints
    ):
        """Test sync_run_params returns parameters unchanged."""
        instance, constructor_args = valid_instances

        (
            returned_requests,
            returned_strategy,
            returned_constraints,
        ) = await instance.sync_run_params(requests, strategy, constraints)

        assert returned_requests is requests
        assert returned_strategy is strategy
        assert returned_constraints is constraints

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mock_time", "delay", "expected"),
        [
            (1000.0, 0.0, 1000.0),
            (500.0, 1.5, 501.5),
            (100.0, 10.0, 110.0),
            (0.0, 2.5, 2.5),
        ],
        ids=["no_delay", "small_delay", "large_delay", "zero_time"],
    )
    async def test_sync_run_start(self, valid_instances, mock_time, delay, expected):
        """Test sync_run_start uses configuration value correctly."""
        instance, constructor_args = valid_instances

        with (
            patch("time.time", return_value=mock_time),
            patch("guidellm.scheduler.environment.settings") as mock_settings,
        ):
            mock_settings.scheduler_start_delay_non_distributed = delay
            start_time = await instance.sync_run_start()
            assert start_time == expected

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response", "req"),
        [
            ("mock_response", "mock_request"),
            (None, "mock_request"),
            ("mock_response", None),
            (None, None),
        ],
        ids=["both_present", "no_response", "no_request", "both_none"],
    )
    async def test_update_run_iteration(self, valid_instances, response, req):
        """Test update_run_iteration no-op behavior."""
        instance, constructor_args = valid_instances

        mock_request_info = ScheduledRequestInfo(
            request_id="test-123",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=time.time(),
        )
        mock_state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=time.time(),
        )

        # Should not raise any errors and is a no-op
        await instance.update_run_iteration(
            response, req, mock_request_info, mock_state
        )

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_sync_run_error(self, valid_instances):
        """Test sync_run_error stores errors correctly."""
        instance, constructor_args = valid_instances

        error1 = RuntimeError("First error")
        error2 = ValueError("Second error")

        await instance.sync_run_error(error1)
        assert error1 in instance.run_errors
        assert len(instance.run_errors) == 1

        await instance.sync_run_error(error2)
        assert len(instance.run_errors) == 2

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_sync_run_end(self, valid_instances):
        """Test sync_run_end behavior with no errors and multiple errors."""
        instance, constructor_args = valid_instances

        # No errors - empty iterator
        results = []
        async for result in instance.sync_run_end():
            results.append(result)
        assert results == []

        # Single error - raises original error
        error = RuntimeError("Test error")
        await instance.sync_run_error(error)
        with pytest.raises(RuntimeError):
            async for _ in instance.sync_run_end():
                pass

        # Multiple errors - raises RuntimeError with combined message
        await instance.sync_run_error(ValueError("Second error"))
        with pytest.raises(RuntimeError) as exc_info:
            async for _ in instance.sync_run_end():
                pass
        assert "Errors occurred during execution" in str(exc_info.value)
