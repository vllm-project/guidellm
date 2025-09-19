"""
Unit tests for the Backend base class and registry functionality.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import wraps
from typing import Any
from unittest.mock import Mock, patch

import pytest

from guidellm.backend.backend import Backend, BackendType
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
)
from guidellm.scheduler import BackendInterface, ScheduledRequestInfo
from guidellm.utils import RegistryMixin


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


def test_backend_type():
    """Test that BackendType is defined correctly as a Literal type."""
    assert BackendType is not None
    # BackendType should be a literal type containing "openai_http"
    assert "openai_http" in str(BackendType)


class TestBackend:
    """Test cases for Backend base class."""

    @pytest.fixture(
        params=[
            {"type_": "openai_http"},
            {"type_": "openai_http"},  # Test multiple instances with same type
        ]
    )
    def valid_instances(self, request):
        """Fixture providing valid Backend instances."""
        constructor_args = request.param

        class TestBackend(Backend):
            def info(self) -> dict[str, Any]:
                return {"type": self.type_}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(
                self, request, request_info, history=None
            ) -> AsyncIterator[tuple[Any, Any]]:
                yield request, request_info

            async def default_model(self) -> str | None:
                return "test-model"

        instance = TestBackend(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Backend inheritance and type relationships."""
        assert issubclass(Backend, RegistryMixin)
        assert issubclass(Backend, BackendInterface)
        assert hasattr(Backend, "create")
        assert hasattr(Backend, "register")
        assert hasattr(Backend, "get_registered_object")

        # Check properties exist
        assert hasattr(Backend, "processes_limit")
        assert hasattr(Backend, "requests_limit")

        # Check abstract method exists
        assert hasattr(Backend, "default_model")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test Backend initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, Backend)
        assert instance.type_ == constructor_args["type_"]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("type_", None),
            ("type_", 123),
            ("type_", ""),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test Backend with invalid field values."""

        class TestBackend(Backend):
            def info(self) -> dict[str, Any]:
                return {}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self) -> str | None:
                return "test-model"

        data = {field: value}
        # Backend itself doesn't validate types, but we test that it accepts the value
        backend = TestBackend(**data)
        assert getattr(backend, field) == value

    @pytest.mark.smoke
    def test_default_properties(self, valid_instances):
        """Test Backend default property implementations."""
        instance, _ = valid_instances
        assert instance.processes_limit is None
        assert instance.requests_limit is None

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_default_model_abstract(self):
        """Test that default_model is abstract and must be implemented."""
        # Backend itself is abstract and cannot be instantiated
        with pytest.raises(TypeError):
            Backend("openai_http")  # type: ignore

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_interface_compatibility(self, valid_instances):
        """Test that Backend is compatible with BackendInterface."""
        instance, _ = valid_instances

        # Test that Backend uses the correct generic types
        request = GenerationRequest(content="test")
        request_info = ScheduledRequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )

        # Test resolve method
        async for response, info in instance.resolve(request, request_info):
            assert response == request
            assert info == request_info
            break  # Only test first iteration

    @pytest.mark.smoke
    def test_create_method_valid(self):
        """Test Backend.create class method with valid backend."""
        # Mock a registered backend
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        with patch.object(
            Backend, "get_registered_object", return_value=mock_backend_class
        ):
            result = Backend.create("openai_http", test_arg="value")

            Backend.get_registered_object.assert_called_once_with("openai_http")
            mock_backend_class.assert_called_once_with(test_arg="value")
            assert result == mock_backend_instance

    @pytest.mark.sanity
    def test_create_method_invalid(self):
        """Test Backend.create class method with invalid backend type."""
        with pytest.raises(
            ValueError, match="Backend type 'invalid_type' is not registered"
        ):
            Backend.create("invalid_type")

    @pytest.mark.regression
    def test_docstring_example_pattern(self):
        """Test that Backend docstring examples work as documented."""

        # Test the pattern shown in docstring
        class MyBackend(Backend):
            def __init__(self, api_key: str):
                super().__init__("mock_backend")  # type: ignore [arg-type]
                self.api_key = api_key

            def info(self) -> dict[str, Any]:
                return {"api_key": "***"}

            async def process_startup(self):
                self.client = Mock()  # Simulate API client

            async def process_shutdown(self):
                self.client = None  # type: ignore[assignment]

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self) -> str | None:
                return "my-model"

        # Register the backend
        Backend.register("my_backend")(MyBackend)

        # Create instance
        backend = Backend.create("my_backend", api_key="secret")
        assert isinstance(backend, MyBackend)
        assert backend.api_key == "secret"
        assert backend.type_ == "mock_backend"


class TestBackendRegistry:
    """Test cases for Backend registry functionality."""

    @pytest.mark.smoke
    def test_openai_backend_registered(self):
        """Test that OpenAI HTTP backend is registered."""
        from guidellm.backend.openai import OpenAIHTTPBackend

        # OpenAI backend should be registered
        backend = Backend.create("openai_http", target="http://test")
        assert isinstance(backend, OpenAIHTTPBackend)
        assert backend.type_ == "openai_http"

    @pytest.mark.sanity
    def test_backend_create_invalid_type(self):
        """Test Backend.create with invalid type raises appropriate error."""
        with pytest.raises(
            ValueError, match="Backend type 'invalid_type' is not registered"
        ):
            Backend.create("invalid_type")

    @pytest.mark.smoke
    def test_backend_registry_functionality(self):
        """Test that backend registry functions work."""
        from guidellm.backend.openai import OpenAIHTTPBackend

        # Test that we can get registered backends
        openai_class = Backend.get_registered_object("openai_http")
        assert openai_class == OpenAIHTTPBackend

        # Test creating with kwargs
        backend = Backend.create(
            "openai_http", target="http://localhost:8000", model="gpt-4"
        )
        assert backend.target == "http://localhost:8000"
        assert backend.model == "gpt-4"

    @pytest.mark.smoke
    def test_backend_is_registered(self):
        """Test Backend.is_registered method."""
        # Test with a known registered backend
        assert Backend.is_registered("openai_http")

        # Test with unknown backend
        assert not Backend.is_registered("unknown_backend")

    @pytest.mark.regression
    def test_backend_registration_decorator(self):
        """Test that backend registration decorator works."""

        # Create a test backend class
        @Backend.register("test_backend")
        class TestBackend(Backend):
            def __init__(self, test_param="default"):
                super().__init__("test_backend")  # type: ignore
                self._test_param = test_param

            def info(self):
                return {"test_param": self._test_param}

            async def process_startup(self):
                pass

            async def process_shutdown(self):
                pass

            async def validate(self):
                pass

            async def resolve(self, request, request_info, history=None):
                yield request, request_info

            async def default_model(self):
                return "test-model"

        # Test that it's registered and can be created
        backend = Backend.create("test_backend", test_param="custom")
        assert isinstance(backend, TestBackend)
        assert backend.info() == {"test_param": "custom"}

    @pytest.mark.smoke
    def test_backend_registered_objects(self):
        """Test Backend.registered_objects method returns registered backends."""
        # Should include at least the openai_http backend
        registered = Backend.registered_objects()
        assert isinstance(registered, tuple)
        assert len(registered) > 0

        # Check that openai backend is in the registered objects
        from guidellm.backend.openai import OpenAIHTTPBackend

        assert OpenAIHTTPBackend in registered
