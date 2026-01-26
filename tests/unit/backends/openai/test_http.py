"""
Unit tests for OpenAIHTTPBackend implementation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock, IteratorStream

from guidellm.backends.backend import Backend
from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.backends.openai.request_handlers import (
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)
from tests.unit.testing_utils import async_timeout


class TestOpenAIHTTPBackend:
    """Test cases for OpenAIHTTPBackend."""

    @pytest.fixture(
        params=[
            {"target": "http://localhost:8000"},
            {
                "target": "https://api.openai.com",
                "model": "gpt-4",
                "timeout": 30.0,
            },
            {
                "target": "http://test-server:8080",
                "model": "test-model",
                "timeout": 120.0,
                "http2": False,
                "follow_redirects": False,
                "verify": True,
            },
        ]
    )
    def valid_instances(self, request):
        """Fixture providing valid OpenAIHTTPBackend instances."""
        constructor_args = request.param
        instance = OpenAIHTTPBackend(**constructor_args)
        return instance, constructor_args

    @pytest.fixture
    def mock_request_handler(self):
        """
        Fixture providing a mocked GenerationResponseHandlerFactory.

        Returns a tuple of (mock_handler, patch_context) where:
        - mock_handler: The mocked GenerationResponseHandler instance
        - patch_context: The patch object for GenerationResponseHandlerFactory.create

        This fixture patches GenerationResponseHandlerFactory.create to return
        the mock_handler, allowing tests to configure the mock as needed.
        """
        mock_handler = Mock(spec=OpenAIRequestHandler)
        # Set default return value for format() method
        mock_handler.format.return_value = GenerationRequestArguments(
            body={"model": "test-model"}
        )
        patch_context = patch.object(
            OpenAIRequestHandlerFactory, "create", return_value=mock_handler
        )
        return mock_handler, patch_context

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test OpenAIHTTPBackend inheritance and type relationships."""
        assert issubclass(OpenAIHTTPBackend, Backend)
        # Check that required methods exist
        assert hasattr(OpenAIHTTPBackend, "process_startup")
        assert hasattr(OpenAIHTTPBackend, "process_shutdown")
        assert hasattr(OpenAIHTTPBackend, "validate")
        assert hasattr(OpenAIHTTPBackend, "resolve")
        assert hasattr(OpenAIHTTPBackend, "default_model")
        assert hasattr(OpenAIHTTPBackend, "available_models")
        # Check that inherited properties exist
        assert hasattr(OpenAIHTTPBackend, "processes_limit")
        assert hasattr(OpenAIHTTPBackend, "requests_limit")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test OpenAIHTTPBackend initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, OpenAIHTTPBackend)
        expected_target = constructor_args["target"].rstrip("/").removesuffix("/v1")
        assert instance.target == expected_target
        if "model" in constructor_args:
            assert instance.model == constructor_args["model"]
        if "timeout" in constructor_args:
            assert instance.timeout == constructor_args["timeout"]
        else:
            assert instance.timeout == 60.0

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target", ""),
            ("timeout", -1.0),
            ("http2", "invalid"),
            ("verify", "invalid"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test OpenAIHTTPBackend with invalid field values."""
        base_args = {"target": "http://localhost:8000"}
        base_args[field] = value
        # OpenAI backend doesn't validate types at init, accepts whatever is passed
        backend = OpenAIHTTPBackend(**base_args)
        assert getattr(backend, field) == value

    @pytest.mark.sanity
    def test_invalid_validate_backend_parameter(self):
        """Test OpenAIHTTPBackend with invalid validate_backend parameter."""
        # Invalid dict without url
        with pytest.raises(ValueError, match="validate_backend must be"):
            OpenAIHTTPBackend(
                target="http://localhost:8000",
                validate_backend={"method": "GET"},
            )

        # Invalid type (number)
        with pytest.raises(ValueError, match="validate_backend must be"):
            OpenAIHTTPBackend(
                target="http://localhost:8000",
                validate_backend=123,  # type: ignore[arg-type]
            )

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that OpenAIHTTPBackend is registered with Backend factory."""
        assert Backend.is_registered("openai_http")
        backend = Backend.create("openai_http", target="http://test")
        assert isinstance(backend, OpenAIHTTPBackend)
        assert backend.type_ == "openai_http"

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test minimal OpenAIHTTPBackend initialization."""
        backend = OpenAIHTTPBackend(target="http://localhost:8000")

        assert backend.target == "http://localhost:8000"
        assert backend.model == ""
        assert backend.timeout == 60.0
        assert backend.http2 is True
        assert backend.follow_redirects is True
        assert backend.verify is False
        assert backend._in_process is False
        assert backend._async_client is None
        assert backend.processes_limit is None
        assert backend.requests_limit is None

    @pytest.mark.smoke
    def test_initialization_full(self):
        """Test full OpenAIHTTPBackend initialization."""
        api_routes = {"health": "custom/health", "models": "custom/models"}
        request_handlers = {"test": "handler"}

        backend = OpenAIHTTPBackend(
            target="https://localhost:8000/v1",
            model="test-model",
            api_routes=api_routes,
            request_handlers=request_handlers,
            timeout=120.0,
            http2=False,
            follow_redirects=False,
            verify=True,
            validate_backend=False,
        )

        assert backend.target == "https://localhost:8000"
        assert backend.model == "test-model"
        assert backend.timeout == 120.0
        assert backend.http2 is False
        assert backend.follow_redirects is False
        assert backend.verify is True
        assert backend.api_routes["health"] == "custom/health"
        assert backend.api_routes["models"] == "custom/models"
        assert backend.request_handlers == request_handlers
        assert backend.processes_limit is None
        assert backend.requests_limit is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("validate_backend", "expected_validate_backend"),
        [
            (True, {"method": "GET", "url": "http://test/health"}),
            (False, None),
            ("/health", {"method": "GET", "url": "http://test/health"}),
            (
                "http://custom/endpoint",
                {"method": "GET", "url": "http://custom/endpoint"},
            ),
            (
                {"url": "http://custom/url", "method": "POST"},
                {"url": "http://custom/url", "method": "POST"},
            ),
            (
                {"url": "http://custom/url"},
                {"url": "http://custom/url", "method": "GET"},
            ),
        ],
        ids=[
            "bool_true",
            "bool_false",
            "str_api_route",
            "str_custom_url",
            "dict_with_method",
            "dict_without_method",
        ],
    )
    def test_validate_backend_parameter(
        self, validate_backend, expected_validate_backend
    ):
        """Test validate_backend parameter with various input types."""
        backend = OpenAIHTTPBackend(
            target="http://test",
            validate_backend=validate_backend,
        )
        assert backend.validate_backend == expected_validate_backend

    @pytest.mark.sanity
    def test_target_normalization(self):
        """Test target URL normalization."""
        # Remove trailing slashes and /v1
        backend1 = OpenAIHTTPBackend(target="http://localhost:8000/")
        assert backend1.target == "http://localhost:8000"

        backend2 = OpenAIHTTPBackend(target="http://localhost:8000/v1")
        assert backend2.target == "http://localhost:8000"

        backend3 = OpenAIHTTPBackend(target="http://localhost:8000/v1/")
        assert backend3.target == "http://localhost:8000"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_info(self):
        """Test info method."""
        backend = OpenAIHTTPBackend(
            target="http://test", model="test-model", timeout=30.0
        )

        info = backend.info

        assert info["target"] == "http://test"
        assert info["model"] == "test-model"
        assert info["timeout"] == 30.0
        assert info["openai_paths"]["/health"] == "health"
        assert info["openai_paths"]["/models"] == "v1/models"
        assert info["openai_paths"]["/completions"] == "v1/completions"
        assert info["openai_paths"]["/chat/completions"] == "v1/chat/completions"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_startup(self):
        """Test process startup."""
        backend = OpenAIHTTPBackend(target="http://test")

        assert not backend._in_process
        assert backend._async_client is None

        await backend.process_startup()

        assert backend._in_process
        assert backend._async_client is not None
        assert isinstance(backend._async_client, httpx.AsyncClient)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_startup_already_started(self):
        """Test process startup when already started."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        with pytest.raises(RuntimeError, match="Backend already started up"):
            await backend.process_startup()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_shutdown(self):
        """Test process shutdown."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        assert backend._in_process
        assert backend._async_client is not None

        await backend.process_shutdown()

        assert not backend._in_process
        assert backend._async_client is None

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_shutdown_not_started(self):
        """Test process shutdown when not started."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_available_models(self, httpx_mock: HTTPXMock):
        """Test available_models method."""
        httpx_mock.add_response(
            url="http://test/v1/models",
            json={"data": [{"id": "test-model1"}, {"id": "test-model2"}]},
        )

        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        models = await backend.available_models()
        assert models == ["test-model1", "test-model2"]

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_default_model(self):
        """Test default_model method."""
        # Test when model is already set
        backend1 = OpenAIHTTPBackend(target="http://test", model="test-model")
        result1 = await backend1.default_model()
        assert result1 == "test-model"

        # Test when not in process
        backend2 = OpenAIHTTPBackend(target="http://test")
        result2 = await backend2.default_model()
        assert result2 == ""

        # Test when in process but no model set
        backend3 = OpenAIHTTPBackend(target="http://test")
        await backend3.process_startup()

        with patch.object(backend3, "available_models", return_value=["test-model2"]):
            result3 = await backend3.default_model()
            assert result3 == "test-model2"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_with_model(self, httpx_mock: HTTPXMock):
        """Test validate method when model is set."""
        httpx_mock.add_response(
            url="http://test/health",
            method="GET",
            headers={},
        )

        backend = OpenAIHTTPBackend(target="http://test", model="test-model")
        await backend.process_startup()

        await backend.validate()  # Should not raise

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_without_model(self):
        """Test validate method when no model is set."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch.object(backend._async_client, "request", return_value=mock_response):
            await backend.validate()  # Should not raise

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_not_in_process(self):
        """Test validate method when backend is not started."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_disabled(self):
        """Test validate method when validation is disabled."""
        backend = OpenAIHTTPBackend(target="http://test", validate_backend=False)
        await backend.process_startup()

        # Should not raise and should not make any requests
        await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_failure(self):
        """Test validate method when validation fails."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        def mock_fail(*args, **kwargs):
            raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

        with (
            patch.object(backend._async_client, "request", side_effect=mock_fail),
            pytest.raises(RuntimeError, match="Backend validation request failed"),
        ):
            await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_not_implemented_history(self, httpx_mock: HTTPXMock):
        """Test resolve method raises error for conversation history."""
        backend = OpenAIHTTPBackend(
            target="http://test", request_format="text_completions"
        )
        await backend.process_startup()

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )
        history = [
            (request, GenerationResponse(request_id="test", request_args="test args"))
        ]

        with pytest.raises(NotImplementedError, match="Multi-turn requests"):
            async for _ in backend.resolve(request, request_info, history):
                pass

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_invalid_request_format(self, httpx_mock: HTTPXMock):
        """Test resolve method raises error for invalid request type."""
        with pytest.raises(ValueError, match="Invalid request_format 'invalid_type'."):
            OpenAIHTTPBackend(
                target="http://test",
                request_format="invalid_type",  # type: ignore[arg-type]
            )

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_not_in_process(self, httpx_mock: HTTPXMock):
        """Test resolve method raises error when backend is not started."""
        backend = OpenAIHTTPBackend(
            target="http://test", request_format="text_completions"
        )

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )

        with pytest.raises(RuntimeError, match="Backend not started up"):
            async for _ in backend.resolve(request, request_info):
                pass

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_text_completions(
        self,
        httpx_mock: HTTPXMock,
        mock_request_handler,
    ):
        """Test resolve method for text completions."""
        httpx_mock.add_response(
            url="http://test/v1/completions",
            json={"choices": [{"text": "Hello world"}]},
        )

        backend = OpenAIHTTPBackend(
            target="http://test",
            model="test-model",
            request_format="text_completions",
        )
        await backend.process_startup()

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )

        # Configure mock handler
        mock_handler, handler_patch = mock_request_handler
        mock_handler.format.return_value = GenerationRequestArguments(
            body={"prompt": "test prompt", "temperature": 0.7, "max_tokens": 100}
        )
        mock_handler.compile_non_streaming.return_value = GenerationResponse(
            request_id="test-id", request_args="test args"
        )

        with handler_patch:
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) == 1
        final_response = responses[0][0]
        assert final_response.request_id == "test-id"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_chat_completions(
        self,
        httpx_mock: HTTPXMock,
        mock_request_handler,
    ):
        """Test resolve method for chat completions."""
        httpx_mock.add_response(
            url="http://test/v1/chat/completions",
            json={"choices": [{"message": {"content": "Response"}}]},
        )

        backend = OpenAIHTTPBackend(
            target="http://test",
            model="test-model",
            request_format="chat_completions",
        )
        await backend.process_startup()

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )

        # Configure mock handler
        mock_handler, handler_patch = mock_request_handler
        mock_handler.format.return_value = GenerationRequestArguments(
            body={
                "messages": [{"role": "user", "content": "test message"}],
                "temperature": 0.5,
            }
        )
        mock_handler.compile_non_streaming.return_value = GenerationResponse(
            request_id="test-id", request_args="test args"
        )

        with handler_patch:
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) == 1
        final_response = responses[0][0]
        assert final_response.request_id == "test-id"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_with_files(
        self,
        httpx_mock: HTTPXMock,
        mock_request_handler,
    ):
        """Test resolve method with file uploads."""
        httpx_mock.add_response(
            url="http://test/v1/audio/transcriptions",
            json={"choices": [{"message": {"content": "Response"}}]},
        )

        backend = OpenAIHTTPBackend(
            target="http://test",
            model="test-model",
            request_format="audio_transcriptions",
        )
        await backend.process_startup()

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )

        # Configure mock handler
        mock_handler, handler_patch = mock_request_handler
        mock_handler.format.return_value = GenerationRequestArguments(
            body={"model": "whisper-1"},
            files={"file": ["audio.mp3", b"audio_data", "audio/mpeg"]},
        )
        mock_handler.compile_non_streaming.return_value = GenerationResponse(
            request_id="test-id", request_args="test args"
        )

        with handler_patch:
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) == 1

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_stream(
        self,
        httpx_mock: HTTPXMock,
        mock_request_handler,
    ):
        """Test resolve method handles asyncio.CancelledError during streaming."""
        httpx_mock.add_response(
            url="http://test/v1/chat/completions",
            stream=IteratorStream(
                [
                    b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
                    b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
                ]
            ),
        )

        backend = OpenAIHTTPBackend(
            target="http://test",
            model="test-model",
            stream=True,
            validate_backend=False,
            request_format="/chat/completions",
        )
        await backend.process_startup()

        request = GenerationRequest()
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
        )

        # Configure mock handler
        mock_handler, handler_patch = mock_request_handler
        mock_handler.format.return_value = GenerationRequestArguments(
            body={
                "messages": [{"role": "user", "content": "test message"}],
                "temperature": 0.5,
                "stream": True,
            },
            stream=True,
        )
        mock_handler.add_streaming_line.return_value = 1
        mock_handler.compile_streaming.return_value = GenerationResponse(
            request_id="test-id", request_args="test args"
        )

        with handler_patch:
            async for _response, _info in backend.resolve(request, request_info):
                pass
