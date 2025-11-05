"""
Unit tests for OpenAIHTTPBackend implementation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import httpx
import pytest

from guidellm.backends.backend import Backend
from guidellm.backends.openai import OpenAIHTTPBackend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)
from guidellm.schemas.request import GenerationRequestArguments, UsageMetrics
from tests.unit.testing_utils import async_timeout


def test_usage_metrics():
    """Test that UsageMetrics is defined correctly."""
    metrics = UsageMetrics()
    assert hasattr(metrics, "text_tokens")
    assert hasattr(metrics, "text_characters")
    assert hasattr(metrics, "total_tokens")

    metrics_with_values = UsageMetrics(text_tokens=10, text_characters=50)
    assert metrics_with_values.text_tokens == 10
    assert metrics_with_values.text_characters == 50


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
        assert backend.model is None
        assert backend.timeout == 60.0
        assert backend.http2 is True
        assert backend.follow_redirects is True
        assert backend.verify is False
        assert backend._in_process is False
        assert backend._async_client is None

    @pytest.mark.smoke
    def test_initialization_full(self):
        """Test full OpenAIHTTPBackend initialization."""
        api_routes = {"health": "custom/health", "models": "custom/models"}
        response_handlers = {"test": "handler"}

        backend = OpenAIHTTPBackend(
            target="https://localhost:8000/v1",
            model="test-model",
            api_routes=api_routes,
            response_handlers=response_handlers,
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
        assert backend.response_handlers == response_handlers

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
        assert info["openai_paths"]["health"] == "health"
        assert info["openai_paths"]["models"] == "v1/models"
        assert info["openai_paths"]["text_completions"] == "v1/completions"
        assert info["openai_paths"]["chat_completions"] == "v1/chat/completions"

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
    async def test_available_models(self):
        """Test available_models method."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "test-model1"}, {"id": "test-model2"}]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(backend._async_client, "get", return_value=mock_response):
            models = await backend.available_models()

            assert models == ["test-model1", "test-model2"]
            backend._async_client.get.assert_called_once()

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
        assert result2 is None

        # Test when in process but no model set
        backend3 = OpenAIHTTPBackend(target="http://test")
        await backend3.process_startup()

        with patch.object(backend3, "available_models", return_value=["test-model2"]):
            result3 = await backend3.default_model()
            assert result3 == "test-model2"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_with_model(self):
        """Test validate method when model is set."""
        backend = OpenAIHTTPBackend(target="http://test", model="test-model")
        await backend.process_startup()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch.object(backend._async_client, "request", return_value=mock_response):
            await backend.validate()  # Should not raise

            backend._async_client.request.assert_called_once_with(
                method="GET", url="http://test/health"
            )

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
    async def test_resolve_not_implemented_history(self):
        """Test resolve method raises error for conversation history."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        request = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(body={"prompt": "test"}),
        )
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=RequestTimings(),
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
    async def test_resolve_text_completions(self):
        """Test resolve method for text completions."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        request = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(
                body={"prompt": "test prompt", "temperature": 0.7, "max_tokens": 100}
            ),
        )
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=RequestTimings(),
        )

        # Mock response handler
        from guidellm.backends.response_handlers import GenerationResponseHandler

        mock_handler = Mock(spec=GenerationResponseHandler)
        mock_response = GenerationResponse(
            request_id="test-id", request_args="test args"
        )
        mock_handler.compile_non_streaming.return_value = mock_response

        with (
            patch.object(
                backend, "_resolve_response_handler", return_value=mock_handler
            ),
            patch.object(backend._async_client, "request") as mock_request,
        ):
            mock_http_response = Mock()
            mock_http_response.json.return_value = {
                "choices": [{"text": "Hello world"}]
            }
            mock_http_response.raise_for_status = Mock()
            mock_request.return_value = mock_http_response

            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) == 1
        final_response = responses[0][0]
        assert final_response.request_id == "test-id"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_chat_completions(self):
        """Test resolve method for chat completions."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        request = GenerationRequest(
            request_type="chat_completions",
            arguments=GenerationRequestArguments(
                body={
                    "messages": [{"role": "user", "content": "test message"}],
                    "temperature": 0.5,
                }
            ),
        )
        request_info = RequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=RequestTimings(),
        )

        # Mock response handler
        from guidellm.backends.response_handlers import GenerationResponseHandler

        mock_handler = Mock(spec=GenerationResponseHandler)
        mock_response = GenerationResponse(
            request_id="test-id", request_args="test args"
        )
        mock_handler.compile_non_streaming.return_value = mock_response

        with (
            patch.object(
                backend, "_resolve_response_handler", return_value=mock_handler
            ),
            patch.object(backend._async_client, "request") as mock_request,
        ):
            mock_http_response = Mock()
            mock_http_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_http_response.raise_for_status = Mock()
            mock_request.return_value = mock_http_response

            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) == 1
        final_response = responses[0][0]
        assert final_response.request_id == "test-id"
