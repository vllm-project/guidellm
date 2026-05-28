"""
Unit tests for OpenAIHTTPBackend implementation.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import ValidationError
from pytest_httpx import HTTPXMock, IteratorStream

from guidellm.backends.backend import Backend
from guidellm.backends.openai.http import OpenAIHTTPBackend, OpenAIHTTPBackendArgs
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


def _make_backend(**kwargs) -> OpenAIHTTPBackend:
    """Create an OpenAIHTTPBackend from keyword arguments via BackendArgs."""
    args = OpenAIHTTPBackendArgs(**kwargs)
    return OpenAIHTTPBackend(args)


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
        instance = _make_backend(**constructor_args)
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
        assert instance._args.target == expected_target
        if "model" in constructor_args:
            assert instance._args.model == constructor_args["model"]
        if "timeout" in constructor_args:
            assert instance._args.timeout == constructor_args["timeout"]
        else:
            assert instance._args.timeout is None

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("http2", "not-a-bool"),
            ("verify", "not-a-bool"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test OpenAIHTTPBackend rejects invalid field types via BackendArgs."""
        base_args = {"target": "http://localhost:8000"}
        base_args[field] = value
        with pytest.raises(ValidationError):
            _make_backend(**base_args)

    @pytest.mark.sanity
    def test_invalid_validate_backend_parameter(self):
        """Test OpenAIHTTPBackend with invalid validate_backend parameter types."""
        # Dict is not a valid bool — raises ValidationError
        with pytest.raises(ValidationError):
            _make_backend(
                target="http://localhost:8000",
                validate_backend={"method": "GET"},  # type: ignore[arg-type]
            )

        # Integer is not a valid bool coercion for non-0/1 values — depends on Pydantic
        # The field is typed as bool, so Pydantic may accept 0/1 as False/True
        # Test with a non-bool object that can't coerce
        with pytest.raises((ValidationError, TypeError)):
            _make_backend(
                target="http://localhost:8000",
                validate_backend="not-a-bool",  # type: ignore[arg-type]
            )

    @pytest.mark.sanity
    def test_server_history_requires_responses_api(self):
        """
        Test server_history=True raises ValidationError for non-responses formats.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            _make_backend(
                target="http://localhost:8000",
                request_format="/v1/chat/completions",
                server_history=True,
            )

    @pytest.mark.sanity
    def test_server_history_with_responses_api(self):
        """
        Test server_history=True is accepted with /v1/responses.

        ## WRITTEN BY AI ##
        """
        backend = _make_backend(
            target="http://localhost:8000",
            request_format="/v1/responses",
            server_history=True,
        )
        assert backend._args.server_history is True

    @pytest.mark.smoke
    def test_factory_registration(self):
        """Test that OpenAIHTTPBackend is registered with Backend factory."""
        assert Backend.is_registered("openai_http")
        args = OpenAIHTTPBackendArgs(target="http://test")
        backend = Backend.create(args)
        assert isinstance(backend, OpenAIHTTPBackend)
        assert backend.kind == "openai_http"

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test minimal OpenAIHTTPBackend initialization."""
        backend = _make_backend(target="http://localhost:8000")

        assert backend._args.target == "http://localhost:8000"
        assert backend._args.model == ""
        assert backend._args.timeout is None
        assert backend._args.timeout_connect == 5.0
        assert backend._args.http2 is True
        assert backend._args.follow_redirects is True
        assert backend._args.verify is False
        assert backend._in_process is False
        assert backend._async_client is None
        assert backend.processes_limit is None
        assert backend.requests_limit is None

    @pytest.mark.smoke
    def test_initialization_full(self):
        """Test full OpenAIHTTPBackend initialization."""
        api_routes = {"health": "custom/health", "models": "custom/models"}

        backend = _make_backend(
            target="https://localhost:8000/v1",
            model="test-model",
            api_routes=api_routes,
            timeout=120.0,
            http2=False,
            follow_redirects=False,
            verify=True,
            validate_backend=False,
        )

        assert backend._args.target == "https://localhost:8000"
        assert backend._args.model == "test-model"
        assert backend._args.timeout == 120.0
        assert backend._args.http2 is False
        assert backend._args.follow_redirects is False
        assert backend._args.verify is True
        assert backend._args.api_routes["health"] == "custom/health"
        assert backend._args.api_routes["models"] == "custom/models"
        assert backend.processes_limit is None
        assert backend.requests_limit is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("validate_backend", "expected_validate_backend"),
        [
            (True, True),
            (False, False),
        ],
        ids=[
            "bool_true",
            "bool_false",
        ],
    )
    def test_validate_backend_parameter(
        self, validate_backend, expected_validate_backend
    ):
        """Test validate_backend parameter stores boolean value."""
        backend = _make_backend(
            target="http://test",
            validate_backend=validate_backend,
        )
        assert backend._args.validate_backend == expected_validate_backend

    @pytest.mark.sanity
    def test_target_normalization(self):
        """Test target URL normalization."""
        # Remove trailing slashes and /v1
        backend1 = _make_backend(target="http://localhost:8000/")
        assert backend1._args.target == "http://localhost:8000"

        backend2 = _make_backend(target="http://localhost:8000/v1")
        assert backend2._args.target == "http://localhost:8000"

        backend3 = _make_backend(target="http://localhost:8000/v1/")
        assert backend3._args.target == "http://localhost:8000"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_info(self):
        """Test info method."""
        backend = _make_backend(target="http://test", model="test-model", timeout=30.0)

        info = backend.info

        assert info["target"] == "http://test"
        assert info["model"] == "test-model"
        assert info["timeout"] == 30.0
        assert info["api_routes"]["/health"] == "health"
        assert info["api_routes"]["/v1/models"] == "v1/models"
        assert info["api_routes"]["/v1/completions"] == "v1/completions"
        assert info["api_routes"]["/v1/chat/completions"] == "v1/chat/completions"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_startup(self):
        """Test process startup."""
        backend = _make_backend(target="http://test")

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
        backend = _make_backend(target="http://test")
        await backend.process_startup()

        with pytest.raises(RuntimeError, match="Backend already started up"):
            await backend.process_startup()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_shutdown(self):
        """Test process shutdown."""
        backend = _make_backend(target="http://test")
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
        backend = _make_backend(target="http://test")

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

        backend = _make_backend(target="http://test")
        await backend.process_startup()

        models = await backend.available_models()
        assert models == ["test-model1", "test-model2"]

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_default_model(self):
        """Test default_model method."""
        # Test when model is already set
        backend1 = _make_backend(target="http://test", model="test-model")
        result1 = await backend1.default_model()
        assert result1 == "test-model"

        # Test when not in process
        backend2 = _make_backend(target="http://test")
        result2 = await backend2.default_model()
        assert result2 == ""

        # Test when in process but no model set
        backend3 = _make_backend(target="http://test")
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

        backend = _make_backend(target="http://test", model="test-model")
        await backend.process_startup()

        await backend.validate()  # Should not raise

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_without_model(self):
        """Test validate method when no model is set."""
        backend = _make_backend(target="http://test")
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
        backend = _make_backend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_disabled(self):
        """Test validate method when validation is disabled."""
        backend = _make_backend(target="http://test", validate_backend=False)
        await backend.process_startup()

        # Should not raise and should not make any requests
        await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_failure(self):
        """Test validate method when validation fails."""
        backend = _make_backend(target="http://test")
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
    async def test_resolve_with_history(self, httpx_mock: HTTPXMock):
        """Test resolve method handles conversation history."""
        backend = _make_backend(target="http://test", request_format="/v1/completions")

        # Mock the models endpoint
        httpx_mock.add_response(
            url="http://test/v1/models",
            json={"data": [{"id": "test-model"}]},
        )

        await backend.process_startup()

        request = GenerationRequest(columns={"text_column": ["world"]})
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            timings=RequestTimings(),
        )
        history = [
            (
                GenerationRequest(columns={"text_column": ["hello"]}),
                GenerationResponse(
                    request_id="test", request_args="test args", text="hi"
                ),
            )
        ]

        # Mock the completions endpoint
        httpx_mock.add_response(
            url="http://test/v1/completions",
            json={
                "choices": [{"text": "response"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

        results = []
        async for response, info in backend.resolve(request, request_info, history):
            results.append((response, info))

        assert len(results) > 0
        # Verify the prompt includes history
        last_request = httpx_mock.get_requests()[-1]
        body = json.loads(last_request.content)
        assert "hi world" in body["prompt"]

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_invalid_request_format(self, httpx_mock: HTTPXMock):
        """Test resolve method raises error for invalid request type."""
        with pytest.raises(ValidationError):
            _make_backend(
                target="http://test",
                request_format="invalid_type",  # type: ignore[arg-type]
            )

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_not_in_process(self, httpx_mock: HTTPXMock):
        """Test resolve method raises error when backend is not started."""
        backend = _make_backend(target="http://test", request_format="/v1/completions")

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

        backend = _make_backend(
            target="http://test",
            model="test-model",
            request_format="/v1/completions",
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

        backend = _make_backend(
            target="http://test",
            model="test-model",
            request_format="/v1/chat/completions",
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

        backend = _make_backend(
            target="http://test",
            model="test-model",
            request_format="/v1/audio/transcriptions",
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

        backend = _make_backend(
            target="http://test",
            model="test-model",
            stream=True,
            validate_backend=False,
            request_format="/v1/chat/completions",
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

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_filters_none_from_request_body(
        self,
        httpx_mock: HTTPXMock,
        mock_request_handler,
    ):
        """
        Test that None values are filtered from request body.

        This is a simple integration test confirming the backend works with
        None filtering. Deep testing of None filtering is covered by test_dict.py.

        ### WRITTEN BY AI ###
        """
        # Track the actual request body sent
        sent_body = None

        def capture_request(request: httpx.Request):
            nonlocal sent_body
            sent_body = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "Response"}}]},
            )

        httpx_mock.add_callback(capture_request, url="http://test/v1/chat/completions")

        backend = _make_backend(
            target="http://test",
            model="test-model",
            request_format="/v1/chat/completions",
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

        # Configure mock handler to return body with None values
        mock_handler, handler_patch = mock_request_handler
        mock_handler.format.return_value = GenerationRequestArguments(
            body={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.7,
                "max_tokens": None,  # Should be filtered out
                "top_p": None,  # Should be filtered out
                "stream": None,  # Should be filtered out
            }
        )
        mock_handler.compile_non_streaming.return_value = GenerationResponse(
            request_id="test-id", request_args="test args"
        )

        with handler_patch:
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        # Verify the backend processed the request successfully
        assert len(responses) == 1
        assert responses[0][0].request_id == "test-id"

        # Verify that None values were filtered out from the sent body
        assert sent_body is not None
        assert "model" in sent_body
        assert "messages" in sent_body
        assert "temperature" in sent_body
        assert "max_tokens" not in sent_body  # None value filtered
        assert "top_p" not in sent_body  # None value filtered
        assert "stream" not in sent_body  # None value filtered


class TestOpenAIBackendToolCallMissingBehavior:
    """Validate tool_call_missing_behavior field on the backend.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_default_is_error_stop(self):
        """Default tool_call_missing_behavior is error_stop.

        ## WRITTEN BY AI ##
        """
        args = OpenAIHTTPBackendArgs(target="http://localhost:8000")
        backend = OpenAIHTTPBackend(args)
        assert backend._args.tool_call_missing_behavior == "error_stop"

    @pytest.mark.sanity
    def test_valid_behaviors_accepted(self):
        """All valid tool_call_missing_behavior values are accepted.

        ## WRITTEN BY AI ##
        """
        for behavior in ("ignore_continue", "ignore_stop", "error_stop"):
            args = OpenAIHTTPBackendArgs(
                target="http://localhost:8000",
                tool_call_missing_behavior=behavior,
            )
            backend = OpenAIHTTPBackend(args)
            assert backend._args.tool_call_missing_behavior == behavior

    @pytest.mark.sanity
    def test_invalid_behavior_rejected(self):
        """Invalid tool_call_missing_behavior is rejected by the Literal type.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            OpenAIHTTPBackendArgs(
                target="http://localhost:8000",
                tool_call_missing_behavior="invalid_mode",
            )


class TestCheckToolCallExpectations:
    """Verify _check_tool_call_expectations raises the right exceptions.

    ## WRITTEN BY AI ##
    """

    def _make_backend(self, behavior: str) -> OpenAIHTTPBackend:
        """
        ## WRITTEN BY AI ##
        """
        args = OpenAIHTTPBackendArgs(
            target="http://localhost:8000",
            tool_call_missing_behavior=behavior,
        )
        return OpenAIHTTPBackend(args)

    def _make_request(self, expects_tool_call: bool) -> GenerationRequest:
        """
        ## WRITTEN BY AI ##
        """
        return GenerationRequest(
            columns={"text_column": ["test"]},
            expects_tool_call=expects_tool_call,
        )

    def _make_response(self, has_tool_calls: bool):
        """
        ## WRITTEN BY AI ##
        """
        from unittest.mock import MagicMock

        from guidellm.schemas.tool_call import (
            ToolCall,
            ToolCallFunction,
        )

        resp = MagicMock()
        resp.tool_calls = (
            [
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="fn"),
                )
            ]
            if has_tool_calls
            else None
        )
        return resp

    @pytest.mark.smoke
    def test_no_op_when_tool_call_present(self):
        """No exception when the model produced a tool call.

        ## WRITTEN BY AI ##
        """
        backend = self._make_backend("error_stop")
        req = self._make_request(expects_tool_call=True)
        resp = self._make_response(has_tool_calls=True)

        backend._check_tool_call_expectations(req, resp)

    @pytest.mark.smoke
    def test_no_op_when_not_expecting_tool_call(self):
        """No exception when the turn doesn't expect a tool call.

        ## WRITTEN BY AI ##
        """
        backend = self._make_backend("error_stop")
        req = self._make_request(expects_tool_call=False)
        resp = self._make_response(has_tool_calls=False)

        backend._check_tool_call_expectations(req, resp)

    @pytest.mark.smoke
    def test_ignore_continue_raises_nothing(self):
        """ignore_continue: no exception even when tool call is missing.

        ## WRITTEN BY AI ##
        """
        backend = self._make_backend("ignore_continue")
        req = self._make_request(expects_tool_call=True)
        resp = self._make_response(has_tool_calls=False)

        backend._check_tool_call_expectations(req, resp)

    @pytest.mark.smoke
    def test_ignore_stop_raises_cancelled_error(self):
        """ignore_stop: raises CancelledError when tool call is missing.

        ## WRITTEN BY AI ##
        """
        import asyncio

        backend = self._make_backend("ignore_stop")
        req = self._make_request(expects_tool_call=True)
        resp = self._make_response(has_tool_calls=False)

        with pytest.raises(asyncio.CancelledError, match="tool call"):
            backend._check_tool_call_expectations(req, resp)

    @pytest.mark.smoke
    def test_error_stop_raises_value_error(self):
        """error_stop: raises ValueError when tool call is missing.

        ## WRITTEN BY AI ##
        """
        backend = self._make_backend("error_stop")
        req = self._make_request(expects_tool_call=True)
        resp = self._make_response(has_tool_calls=False)

        with pytest.raises(ValueError, match="tool call"):
            backend._check_tool_call_expectations(req, resp)

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_stream_reasoning_tokens_ttft(
        self,
        httpx_mock: HTTPXMock,
    ):
        """
        Test that TTFT is measured correctly when reasoning tokens arrive first.

        Validates that the first_token_iteration timing is set on the first
        reasoning token, not waiting for the first content token.
        This is an integration test for issue #737.

        ### WRITTEN BY AI ###
        """
        # Create a realistic reasoning model response stream
        stream_chunks = [
            # First chunk: reasoning token (should trigger TTFT)
            b'data: {"id": "chatcmpl-123", "choices": '
            b'[{"index": 0, "delta": {"reasoning": "Let me think"}}]}\n\n',
            # More reasoning tokens
            b'data: {"choices": [{"delta": {"reasoning": " about this..."}}]}\n\n',
            # Finally, content tokens
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}}], '
            b'"usage": {"prompt_tokens": 5, "completion_tokens": 10}}\n\n',
            b"data: [DONE]\n\n",
        ]

        httpx_mock.add_response(
            url="http://test/v1/chat/completions",
            stream=IteratorStream(stream_chunks),
        )

        backend = _make_backend(
            target="http://test",
            model="reasoning-model",
            stream=True,
            validate_backend=False,
            request_format="/v1/chat/completions",
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

        responses = []
        async for response, info in backend.resolve(request, request_info):
            responses.append((response, info))

        # Verify we got a response
        assert len(responses) > 0
        final_response, final_info = responses[-1]

        # Verify TTFT was measured (first_token_iteration should be set)
        assert final_info.timings.first_token_iteration is not None, (
            "TTFT (first_token_iteration) should be set when reasoning tokens arrive"
        )

        # Reasoning tokens should NOT appear in text; only content is captured
        assert final_response.text is not None
        assert "Let me think" not in final_response.text
        assert "about this..." not in final_response.text
        assert "Hello world" in final_response.text

        # Verify token counts
        assert final_info.timings.token_iterations > 0
        assert final_response.output_metrics.text_tokens == 10
