"""
Unit tests for the Backend base class and registry functionality.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from guidellm.backends import Backend, BackendArgs
from guidellm.schemas import GenerationRequest, RequestInfo
from guidellm.utils.registry import RegistryMixin
from tests.unit.testing_utils import async_timeout


class _TestBackendArgs(BackendArgs):
    """Minimal backend args model for test backends."""

    type_: str = "test_backend"
    target: str | None = None
    model: str | None = None


class TestBackend:
    """Test cases for Backend base class."""

    @pytest.fixture(
        params=[
            {"type_": "openai_http"},
        ],
        ids=["openai_http_type"],
    )
    def valid_instances(self, request):
        """Fixture providing valid Backend instances."""
        constructor_args = request.param

        class TestBackendImpl(Backend):
            @property
            def info(self) -> dict[str, Any]:
                return {"type": self.type_, "test": "backend"}

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

            async def default_model(self) -> str:
                return "test-model"

        instance = TestBackendImpl(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test Backend inheritance and type relationships."""
        # Test inheritance
        assert issubclass(Backend, RegistryMixin)

        # Check that Backend implements BackendInterface methods
        assert hasattr(Backend, "resolve")
        assert hasattr(Backend, "process_startup")
        assert hasattr(Backend, "process_shutdown")
        assert hasattr(Backend, "validate")
        assert hasattr(Backend, "info")

        # Check registry methods exist
        assert hasattr(Backend, "create")
        assert hasattr(Backend, "register")
        assert hasattr(Backend, "get_registered_object")
        assert hasattr(Backend, "is_registered")
        assert hasattr(Backend, "registered_objects")

        # Check properties exist
        assert hasattr(Backend, "processes_limit")
        assert hasattr(Backend, "requests_limit")

        # Check abstract methods exist
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

        class TestBackendImpl(Backend):
            @property
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

            async def default_model(self) -> str:
                return "test-model"

        data = {field: value}
        # Backend itself doesn't validate types, but we test that it accepts the value
        backend = TestBackendImpl(**data)
        assert getattr(backend, field) == value

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test Backend initialization without required field."""

        class TestBackendImpl(Backend):
            @property
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

            async def default_model(self) -> str:
                return "test-model"

        with pytest.raises(TypeError):
            TestBackendImpl()  # type: ignore

    @pytest.mark.smoke
    def test_default_properties(self, valid_instances):
        """Test Backend default property implementations."""
        instance, _ = valid_instances
        assert instance.processes_limit is None
        assert instance.requests_limit is None

    @pytest.mark.smoke
    def test_info_property(self, valid_instances):
        """Test Backend info property."""
        instance, constructor_args = valid_instances
        info = instance.info
        assert isinstance(info, dict)
        assert info["type"] == constructor_args["type_"]
        assert "test" in info

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_default_model(self, valid_instances):
        """Test that default_model is abstract and must be implemented."""
        instance, _ = valid_instances
        # Test that it returns a string
        model = await instance.default_model()
        assert isinstance(model, str)
        assert model == "test-model"

        # Test that Backend itself is abstract and cannot be instantiated
        with pytest.raises(TypeError):
            Backend("openai_http")  # type: ignore

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_process_startup(self, valid_instances):
        """Test Backend.process_startup lifecycle method."""
        instance, _ = valid_instances
        # Should not raise any exceptions
        await instance.process_startup()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_process_shutdown(self, valid_instances):
        """Test Backend.process_shutdown lifecycle method."""
        instance, _ = valid_instances
        # Should not raise any exceptions
        await instance.process_shutdown()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_validate(self, valid_instances):
        """Test Backend.validate lifecycle method."""
        instance, _ = valid_instances
        # Should not raise any exceptions
        await instance.validate()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_resolve(self, valid_instances):
        """Test Backend.resolve method."""
        instance, _ = valid_instances
        request = GenerationRequest(
            columns={"text_column": ["test prompt"]},
        )
        request_info = RequestInfo(request_id="test-id")

        # Test resolve method
        results = []
        async for response, info in instance.resolve(request, request_info):
            results.append((response, info))

        assert len(results) == 1
        assert results[0][0] == request
        assert results[0][1] == request_info

    @pytest.mark.smoke
    def test_create(self):
        """Test Backend.create class method with valid backend."""
        mock_backend_class = Mock()
        mock_backend_instance = Mock()
        mock_backend_class.return_value = mock_backend_instance

        mock_args = Mock(spec=BackendArgs)
        mock_args.type_ = "openai_http"

        with patch.object(
            Backend, "get_registered_object", return_value=mock_backend_class
        ):
            result = Backend.create(mock_args)

            Backend.get_registered_object.assert_called_once_with("openai_http")
            mock_backend_class.assert_called_once_with(mock_args)
            assert result == mock_backend_instance

    @pytest.mark.sanity
    def test_create_invalid(self):
        """Test Backend.create class method with invalid backend type."""
        mock_args = Mock(spec=BackendArgs)
        mock_args.type_ = "invalid_type"

        with pytest.raises(
            ValueError, match="Backend type 'invalid_type' is not registered"
        ):
            Backend.create(mock_args)

    @pytest.mark.regression
    def test_docstring_example_pattern(self):
        """Test that Backend docstring examples work as documented."""

        @BackendArgs.register("my_backend")
        class MyBackendArgs(BackendArgs):
            type_: str = "my_backend"  # type: ignore[assignment]
            api_key: str = ""

        # Test the pattern shown in docstring
        class MyBackend(Backend):
            def __init__(self, arguments: MyBackendArgs):
                super().__init__("my_backend")
                self.api_key = arguments.api_key

            @property
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

            async def default_model(self) -> str:
                return "my-model"

        # Register the backend
        Backend.register("my_backend")(MyBackend)

        # Create instance using BackendArgs
        args = MyBackendArgs(api_key="secret")
        backend = Backend.create(args)
        assert isinstance(backend, MyBackend)
        assert backend.api_key == "secret"
        assert backend.type_ == "my_backend"

    @pytest.mark.smoke
    def test_openai_backend_registered(self):
        """Test that OpenAI HTTP backend is registered."""
        from guidellm.backends.openai import OpenAIHTTPBackend
        from guidellm.backends.openai.http import OpenAIHTTPBackendArgs

        # OpenAI backend should be registered
        args = OpenAIHTTPBackendArgs(target="http://test")
        backend = Backend.create(args)
        assert isinstance(backend, OpenAIHTTPBackend)
        assert backend.type_ == "openai_http"

    @pytest.mark.smoke
    def test_vllm_python_backend_registered(self):
        """
        Test that vllm_python backend is registered and createable.
        ## WRITTEN BY AI ##
        """
        from guidellm.backends.vllm_python.vllm import (
            VLLMPythonBackend,
            VLLMPythonBackendArgs,
        )

        assert Backend.is_registered("vllm_python")
        with patch("guidellm.backends.vllm_python.vllm._check_vllm_available"):
            args = VLLMPythonBackendArgs(model="test-model")
            backend = Backend.create(args)
        assert isinstance(backend, VLLMPythonBackend)
        assert backend._args.model == "test-model"
        assert backend.type_ == "vllm_python"

    @pytest.mark.smoke
    def test_backend_registry_functionality(self):
        """Test that backend registry functions work."""
        from guidellm.backends.openai import OpenAIHTTPBackend
        from guidellm.backends.openai.http import OpenAIHTTPBackendArgs

        # Test that we can get registered backends
        openai_class = Backend.get_registered_object("openai_http")
        assert openai_class == OpenAIHTTPBackend

        # Test creating with BackendArgs
        args = OpenAIHTTPBackendArgs(target="http://localhost:8000", model="gpt-4")
        backend = Backend.create(args)
        assert backend._args.target == "http://localhost:8000"
        assert backend._args.model == "gpt-4"

    @pytest.mark.smoke
    def test_is_registered(self):
        """Test Backend.is_registered method."""
        # Test with a known registered backend
        assert Backend.is_registered("openai_http")

        # Test with unknown backend
        assert not Backend.is_registered("unknown_backend")

    @pytest.mark.regression
    def test_registration_decorator(self):
        """Test that backend registration decorator works."""

        @BackendArgs.register("test_decorator_backend")
        class TestDecoratorArgs(BackendArgs):
            type_: str = "test_decorator_backend"  # type: ignore[assignment]
            test_param: str = "default"

        # Create a test backend class
        @Backend.register("test_decorator_backend")
        class TestDecoratorBackend(Backend):
            def __init__(self, arguments: TestDecoratorArgs):
                super().__init__("test_decorator_backend")  # type: ignore
                self._test_param = arguments.test_param

            @property
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
        args = TestDecoratorArgs(test_param="custom")
        backend = Backend.create(args)
        assert isinstance(backend, TestDecoratorBackend)
        assert backend.info == {"test_param": "custom"}

    @pytest.mark.smoke
    def test_registered_objects(self):
        """Test Backend.registered_objects method returns registered backends."""
        # Should include at least the openai_http backend
        registered = Backend.registered_objects()
        assert isinstance(registered, tuple)
        assert len(registered) > 0

        # Check that openai backend is in the registered objects
        from guidellm.backends.openai import OpenAIHTTPBackend

        assert OpenAIHTTPBackend in registered
