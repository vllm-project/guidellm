"""
Unit tests for OpenAIHTTPBackend implementation.
"""

from __future__ import annotations

import asyncio
import base64
from functools import wraps
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from PIL import Image

from guidellm.backend.backend import Backend
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.backend.openai import OpenAIHTTPBackend, UsageStats
from guidellm.scheduler import ScheduledRequestInfo


def async_timeout(delay):
    def decorator(func):
        @wraps(func)
        async def new_func(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=delay)

        return new_func

    return decorator


def test_usage_stats():
    """Test that UsageStats is defined correctly as a dataclass."""
    stats = UsageStats()
    assert stats.prompt_tokens is None
    assert stats.output_tokens is None

    stats_with_values = UsageStats(prompt_tokens=10, output_tokens=5)
    assert stats_with_values.prompt_tokens == 10
    assert stats_with_values.output_tokens == 5


class TestOpenAIHTTPBackend:
    """Test cases for OpenAIHTTPBackend."""

    @pytest.fixture(
        params=[
            {"target": "http://localhost:8000"},
            {
                "target": "https://api.openai.com",
                "model": "gpt-4",
                "api_key": "test-key",
                "timeout": 30.0,
                "stream_response": False,
            },
            {
                "target": "http://test-server:8080",
                "model": "test-model",
                "api_key": "Bearer test-token",
                "organization": "test-org",
                "project": "test-proj",
                "timeout": 120.0,
                "http2": False,
                "follow_redirects": False,
                "max_output_tokens": 500,
                "extra_query": {"param": "value"},
                "extra_body": {"setting": "test"},
                "remove_from_body": ["unwanted"],
                "headers": {"Custom": "header"},
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
        assert hasattr(OpenAIHTTPBackend, "HEALTH_PATH")
        assert OpenAIHTTPBackend.HEALTH_PATH == "/health"
        assert hasattr(OpenAIHTTPBackend, "MODELS_PATH")
        assert OpenAIHTTPBackend.MODELS_PATH == "/v1/models"
        assert hasattr(OpenAIHTTPBackend, "TEXT_COMPLETIONS_PATH")
        assert OpenAIHTTPBackend.TEXT_COMPLETIONS_PATH == "/v1/completions"
        assert hasattr(OpenAIHTTPBackend, "CHAT_COMPLETIONS_PATH")
        assert OpenAIHTTPBackend.CHAT_COMPLETIONS_PATH == "/v1/chat/completions"
        assert hasattr(OpenAIHTTPBackend, "MODELS_KEY")
        assert OpenAIHTTPBackend.MODELS_KEY == "models"
        assert hasattr(OpenAIHTTPBackend, "TEXT_COMPLETIONS_KEY")
        assert OpenAIHTTPBackend.TEXT_COMPLETIONS_KEY == "text_completions"
        assert hasattr(OpenAIHTTPBackend, "CHAT_COMPLETIONS_KEY")
        assert OpenAIHTTPBackend.CHAT_COMPLETIONS_KEY == "chat_completions"

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
        assert backend.stream_response is True
        assert backend._in_process is False
        assert backend._async_client is None

    @pytest.mark.smoke
    def test_initialization_full(self):
        """Test full OpenAIHTTPBackend initialization."""
        extra_query = {"param": "value"}
        extra_body = {"setting": "test"}
        remove_from_body = ["unwanted"]
        headers = {"Custom-Header": "value"}

        backend = OpenAIHTTPBackend(
            target="https://localhost:8000/v1",
            model="test-model",
            api_key="test-key",
            organization="test-org",
            project="test-project",
            timeout=120.0,
            http2=False,
            follow_redirects=False,
            max_output_tokens=1000,
            stream_response=False,
            extra_query=extra_query,
            extra_body=extra_body,
            remove_from_body=remove_from_body,
            headers=headers,
            verify=True,
        )

        assert backend.target == "https://localhost:8000"
        assert backend.model == "test-model"
        assert backend.timeout == 120.0
        assert backend.http2 is False
        assert backend.follow_redirects is False
        assert backend.verify is True
        assert backend.max_output_tokens == 1000
        assert backend.stream_response is False
        assert backend.extra_query == extra_query
        assert backend.extra_body == extra_body
        assert backend.remove_from_body == remove_from_body

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

    @pytest.mark.sanity
    def test_header_building(self):
        """Test header building logic."""
        # Test with API key
        backend1 = OpenAIHTTPBackend(target="http://test", api_key="test-key")
        assert "Authorization" in backend1.headers
        assert backend1.headers["Authorization"] == "Bearer test-key"

        # Test with Bearer prefix already
        backend2 = OpenAIHTTPBackend(target="http://test", api_key="Bearer test-key")
        assert backend2.headers["Authorization"] == "Bearer test-key"

        # Test with organization and project
        backend3 = OpenAIHTTPBackend(
            target="http://test", organization="test-org", project="test-project"
        )
        assert backend3.headers["OpenAI-Organization"] == "test-org"
        assert backend3.headers["OpenAI-Project"] == "test-project"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @async_timeout(5.0)
    async def test_info(self):
        """Test info method."""
        backend = OpenAIHTTPBackend(
            target="http://test", model="test-model", timeout=30.0
        )

        info = backend.info()

        assert info["target"] == "http://test"
        assert info["model"] == "test-model"
        assert info["timeout"] == 30.0
        assert info["health_path"] == "/health"
        assert info["models_path"] == "/v1/models"
        assert info["text_completions_path"] == "/v1/completions"
        assert info["chat_completions_path"] == "/v1/chat/completions"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @async_timeout(5.0)
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
    @async_timeout(5.0)
    async def test_process_startup_already_started(self):
        """Test process startup when already started."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        with pytest.raises(RuntimeError, match="Backend already started up"):
            await backend.process_startup()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @async_timeout(5.0)
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
    @async_timeout(5.0)
    async def test_process_shutdown_not_started(self):
        """Test process shutdown when not started."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @async_timeout(5.0)
    async def test_check_in_process(self):
        """Test _check_in_process method."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            backend._check_in_process()

        await backend.process_startup()
        backend._check_in_process()  # Should not raise

        await backend.process_shutdown()
        with pytest.raises(RuntimeError, match="Backend not started up"):
            backend._check_in_process()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    @async_timeout(5.0)
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
    @async_timeout(5.0)
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
    @async_timeout(10.0)
    async def test_validate_with_model(self):
        """Test validate method when model is set."""
        backend = OpenAIHTTPBackend(target="http://test", model="test-model")
        await backend.process_startup()

        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        with patch.object(backend._async_client, "get", return_value=mock_response):
            await backend.validate()  # Should not raise

            backend._async_client.get.assert_called_once_with(
                "http://test/health", headers={"Content-Type": "application/json"}
            )

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_without_model(self):
        """Test validate method when no model is set."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        with patch.object(backend, "available_models", return_value=["test-model"]):
            await backend.validate()
            assert backend.model == "test-model"

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_fallback_to_text_completions(self):
        """Test validate method fallback to text completions."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        # Mock health and models endpoints to fail
        def mock_get(*args, **kwargs):
            raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

        # Mock text_completions to succeed
        async def mock_text_completions(*args, **kwargs):
            yield "test", UsageStats()

        with (
            patch.object(backend._async_client, "get", side_effect=mock_get),
            patch.object(
                backend, "text_completions", side_effect=mock_text_completions
            ),
        ):
            await backend.validate()  # Should not raise

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_failure(self):
        """Test validate method when all validation methods fail."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        def mock_fail(*args, **kwargs):
            raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

        def mock_http_error(*args, **kwargs):
            raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

        with (
            patch.object(backend._async_client, "get", side_effect=mock_http_error),
            patch.object(backend, "text_completions", side_effect=mock_http_error),
            pytest.raises(RuntimeError, match="Backend validation failed"),
        ):
            await backend.validate()

    @pytest.mark.sanity
    def test_get_headers(self):
        """Test _get_headers method."""
        backend = OpenAIHTTPBackend(
            target="http://test", api_key="test-key", headers={"Custom": "value"}
        )

        headers = backend._get_headers()

        expected = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key",
            "Custom": "value",
        }
        assert headers == expected

    @pytest.mark.sanity
    def test_get_params(self):
        """Test _get_params method."""
        extra_query = {
            "general": "value",
            "text_completions": {"specific": "text"},
            "chat_completions": {"specific": "chat"},
        }

        backend = OpenAIHTTPBackend(target="http://test", extra_query=extra_query)

        # Test endpoint-specific params
        text_params = backend._get_params("text_completions")
        assert text_params == {"specific": "text"}

        # Test fallback to general params
        other_params = backend._get_params("other")
        assert other_params == extra_query

    @pytest.mark.regression
    def test_get_chat_messages_string(self):
        """Test _get_chat_messages with string content."""
        backend = OpenAIHTTPBackend(target="http://test")

        messages = backend._get_chat_messages("Hello world")

        expected = [{"role": "user", "content": "Hello world"}]
        assert messages == expected

    @pytest.mark.regression
    def test_get_chat_messages_list(self):
        """Test _get_chat_messages with list content."""
        backend = OpenAIHTTPBackend(target="http://test")

        content = [
            "Hello",
            {"type": "text", "text": "world"},
            {"role": "assistant", "content": "existing message"},
        ]

        messages = backend._get_chat_messages(content)

        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "world"},
                    {"role": "assistant", "content": "existing message"},
                ],
            }
        ]
        assert messages == expected

    @pytest.mark.regression
    def test_get_chat_messages_invalid(self):
        """Test _get_chat_messages with invalid content."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(ValueError, match="Unsupported content type"):
            backend._get_chat_messages(123)

        with pytest.raises(ValueError, match="Unsupported content item type"):
            backend._get_chat_messages([123])

    @pytest.mark.regression
    def test_get_chat_message_media_item_image(self):
        """Test _get_chat_message_media_item with PIL Image."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Create a mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_image.tobytes.return_value = b"fake_image_data"

        result = backend._get_chat_message_media_item(mock_image)

        expected_data = base64.b64encode(b"fake_image_data").decode("utf-8")
        expected = {
            "type": "image",
            "image": {"url": f"data:image/jpeg;base64,{expected_data}"},
        }
        assert result == expected

    @pytest.mark.regression
    def test_get_chat_message_media_item_path(self):
        """Test _get_chat_message_media_item with file paths."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Test unsupported file type
        unsupported_path = Path("test.txt")
        with pytest.raises(ValueError, match="Unsupported file type: .txt"):
            backend._get_chat_message_media_item(unsupported_path)

    @pytest.mark.regression
    def test_get_body(self):
        """Test _get_body method."""
        extra_body = {"general": "value", "text_completions": {"temperature": 0.5}}

        backend = OpenAIHTTPBackend(
            target="http://test",
            model="test-model",
            max_output_tokens=1000,
            extra_body=extra_body,
        )

        request_kwargs = {"temperature": 0.7}

        body = backend._get_body(
            endpoint_type="text_completions",
            request_kwargs=request_kwargs,
            max_output_tokens=500,
            prompt="test",
        )

        # Check that max_tokens settings are applied
        assert body["temperature"] == 0.7  # request_kwargs override extra_body
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 500
        assert body["max_completion_tokens"] == 500
        assert body["ignore_eos"] is True
        assert body["prompt"] == "test"
        # stop: None is filtered out by the None filter
        assert "stop" not in body

    @pytest.mark.regression
    def test_get_completions_text_content(self):
        """Test _get_completions_text_content method."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Test with text field
        data1 = {"choices": [{"text": "generated text"}]}
        result1 = backend._get_completions_text_content(data1)
        assert result1 == "generated text"

        # Test with delta content field
        data2 = {"choices": [{"delta": {"content": "delta text"}}]}
        result2 = backend._get_completions_text_content(data2)
        assert result2 == "delta text"

        # Test with no choices
        data3: dict[str, list] = {"choices": []}
        result3 = backend._get_completions_text_content(data3)
        assert result3 is None

        # Test with no choices key
        data4: dict[str, str] = {}
        result4 = backend._get_completions_text_content(data4)
        assert result4 is None

    @pytest.mark.regression
    def test_get_completions_usage_stats(self):
        """Test _get_completions_usage_stats method."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Test with usage data
        data1 = {"usage": {"prompt_tokens": 50, "completion_tokens": 100}}
        result1 = backend._get_completions_usage_stats(data1)
        assert isinstance(result1, UsageStats)
        assert result1.prompt_tokens == 50
        assert result1.output_tokens == 100

        # Test with no usage data
        data2: dict[str, str] = {}
        result2 = backend._get_completions_usage_stats(data2)
        assert result2 is None

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_not_implemented_history(self):
        """Test resolve method raises error for conversation history."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        request = GenerationRequest(content="test")
        request_info = ScheduledRequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )
        history = [(request, GenerationResponse(request_id="test", request_args={}))]

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
            content="test prompt",
            request_type="text_completions",
            params={"temperature": 0.7},
            constraints={"output_tokens": 100},
        )
        request_info = ScheduledRequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )

        # Mock text_completions method
        async def mock_text_completions(*args, **kwargs):
            yield None, None  # Start signal
            yield "Hello", None  # First token
            yield " world", UsageStats(prompt_tokens=10, output_tokens=2)  # Final

        with patch.object(
            backend, "text_completions", side_effect=mock_text_completions
        ):
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        assert len(responses) >= 2
        final_response = responses[-1][0]
        assert final_response.value == "Hello world"
        assert final_response.request_id == request.request_id
        assert final_response.iterations == 2

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_chat_completions(self):
        """Test resolve method for chat completions."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        request = GenerationRequest(
            content="test message",
            request_type="chat_completions",
            params={"temperature": 0.5},
        )
        request_info = ScheduledRequestInfo(
            request_id="test-id",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )

        # Mock chat_completions method
        async def mock_chat_completions(*args, **kwargs):
            yield None, None  # Start signal
            yield "Response", UsageStats(prompt_tokens=5, output_tokens=1)

        with patch.object(
            backend, "chat_completions", side_effect=mock_chat_completions
        ):
            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

        final_response = responses[-1][0]
        assert final_response.value == "Response"
        assert final_response.request_id == request.request_id


class TestOpenAICompletions:
    """Test cases for completion methods."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_text_completions_not_in_process(self):
        """Test text_completions when backend not started."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            async for _ in backend.text_completions("test", "req-id"):
                pass

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_text_completions_basic(self):
        """Test basic text_completions functionality."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "choices": [{"text": "Generated text"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            with patch.object(
                backend._async_client, "post", return_value=mock_response
            ):
                results = []
                async for result in backend.text_completions(
                    prompt="test prompt", request_id="req-123", stream_response=False
                ):
                    results.append(result)

            assert len(results) == 2
            assert results[0] == (None, None)  # Initial yield
            assert results[1][0] == "Generated text"
            assert isinstance(results[1][1], UsageStats)
            assert results[1][1].prompt_tokens == 10
            assert results[1][1].output_tokens == 5
        finally:
            await backend.process_shutdown()

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_chat_completions_not_in_process(self):
        """Test chat_completions when backend not started."""
        backend = OpenAIHTTPBackend(target="http://test")

        with pytest.raises(RuntimeError, match="Backend not started up"):
            async for _ in backend.chat_completions("test"):
                pass

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_chat_completions_basic(self):
        """Test basic chat_completions functionality."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "choices": [{"delta": {"content": "Chat response"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            }

            with patch.object(
                backend._async_client, "post", return_value=mock_response
            ):
                results = []
                async for result in backend.chat_completions(
                    content="Hello", request_id="req-456", stream_response=False
                ):
                    results.append(result)

            assert len(results) == 2
            assert results[0] == (None, None)
            assert results[1][0] == "Chat response"
            assert isinstance(results[1][1], UsageStats)
            assert results[1][1].prompt_tokens == 8
            assert results[1][1].output_tokens == 3
        finally:
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_text_completions_with_parameters(self):
        """Test text_completions with additional parameters."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "choices": [{"text": "response"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }

            with patch.object(
                backend._async_client, "post", return_value=mock_response
            ) as mock_post:
                async for _ in backend.text_completions(
                    prompt="test",
                    request_id="req-123",
                    output_token_count=50,
                    temperature=0.7,
                    stream_response=False,
                ):
                    pass

            # Check that the request body contains expected parameters
            call_args = mock_post.call_args
            body = call_args[1]["json"]
            assert body["max_tokens"] == 50
            assert body["temperature"] == 0.7
            assert body["model"] == "gpt-4"
        finally:
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_chat_completions_content_formatting(self):
        """Test chat_completions content formatting."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "choices": [{"delta": {"content": "response"}}]
            }

            with patch.object(
                backend._async_client, "post", return_value=mock_response
            ) as mock_post:
                async for _ in backend.chat_completions(
                    content="Hello world", stream_response=False
                ):
                    pass

            call_args = mock_post.call_args
            body = call_args[1]["json"]
            expected_messages = [{"role": "user", "content": "Hello world"}]
            assert body["messages"] == expected_messages
        finally:
            await backend.process_shutdown()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_validate_no_models_available(self):
        """Test validate method when no models are available."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        try:
            # Mock endpoints to fail, then available_models to return empty list
            def mock_get_fail(*args, **kwargs):
                raise httpx.HTTPStatusError("Error", request=Mock(), response=Mock())

            with (
                patch.object(backend._async_client, "get", side_effect=mock_get_fail),
                patch.object(backend, "available_models", return_value=[]),
                patch.object(backend, "text_completions", side_effect=mock_get_fail),
                pytest.raises(
                    RuntimeError,
                    match="No model available and could not set a default model",
                ),
            ):
                await backend.validate()
        finally:
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_text_completions_streaming(self):
        """Test text_completions with streaming enabled."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            # Mock streaming response
            mock_stream = Mock()
            mock_stream.raise_for_status = Mock()

            async def mock_aiter_lines():
                lines = [
                    'data: {"choices":[{"text":"Hello"}], "usage":{"prompt_tokens":5,"completion_tokens":1}}',  # noqa: E501
                    'data: {"choices":[{"text":" world"}], "usage":{"prompt_tokens":5,"completion_tokens":2}}',  # noqa: E501
                    'data: {"choices":[{"text":"!"}], "usage":{"prompt_tokens":5,"completion_tokens":3}}',  # noqa: E501
                    "data: [DONE]",
                ]
                for line in lines:
                    yield line

            mock_stream.aiter_lines = mock_aiter_lines

            mock_client_stream = AsyncMock()
            mock_client_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_client_stream.__aexit__ = AsyncMock(return_value=None)

            with patch.object(
                backend._async_client, "stream", return_value=mock_client_stream
            ):
                results = []
                async for result in backend.text_completions(
                    prompt="test prompt", request_id="req-123", stream_response=True
                ):
                    results.append(result)

            # Should get initial None, then tokens, then final with usage
            assert len(results) >= 3
            assert results[0] == (None, None)  # Initial yield
            assert all(
                isinstance(result[0], str) for result in results[1:]
            )  # Has text content
            assert all(
                isinstance(result[1], UsageStats) for result in results[1:]
            )  # Has usage stats
            assert all(
                result[1].output_tokens == i for i, result in enumerate(results[1:], 1)
            )
        finally:
            await backend.process_shutdown()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_chat_completions_streaming(self):
        """Test chat_completions with streaming enabled."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            # Mock streaming response
            mock_stream = Mock()
            mock_stream.raise_for_status = Mock()

            async def mock_aiter_lines():
                lines = [
                    'data: {"choices":[{"delta":{"content":"Hi"}}]}',
                    'data: {"choices":[{"delta":{"content":" there"}}]}',
                    'data: {"choices":[{"delta":{"content":"!"}}]}',
                    'data: {"usage":{"prompt_tokens":3,"completion_tokens":3}}',
                    "data: [DONE]",
                ]
                for line in lines:
                    yield line

            mock_stream.aiter_lines = mock_aiter_lines

            mock_client_stream = AsyncMock()
            mock_client_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_client_stream.__aexit__ = AsyncMock(return_value=None)

            with patch.object(
                backend._async_client, "stream", return_value=mock_client_stream
            ):
                results = []
                async for result in backend.chat_completions(
                    content="Hello", request_id="req-456", stream_response=True
                ):
                    results.append(result)

            # Should get initial None, then deltas, then final with usage
            assert len(results) >= 3
            assert results[0] == (None, None)  # Initial yield
            assert any(result[0] for result in results if result[0])  # Has content
            assert any(result[1] for result in results if result[1])  # Has usage stats
        finally:
            await backend.process_shutdown()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_streaming_response_edge_cases(self):
        """Test streaming response edge cases for line processing."""
        backend = OpenAIHTTPBackend(target="http://test", model="gpt-4")
        await backend.process_startup()

        try:
            # Mock streaming response with edge cases
            mock_stream = Mock()
            mock_stream.raise_for_status = Mock()

            async def mock_aiter_lines():
                lines = [
                    "",  # Empty line
                    "   ",  # Whitespace only
                    "not data line",  # Line without data prefix
                    'data: {"choices":[{"text":"Hello"}]}',  # Valid data
                    "data: [DONE]",  # End marker
                ]
                for line in lines:
                    yield line

            mock_stream.aiter_lines = mock_aiter_lines

            mock_client_stream = AsyncMock()
            mock_client_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_client_stream.__aexit__ = AsyncMock(return_value=None)

            with patch.object(
                backend._async_client, "stream", return_value=mock_client_stream
            ):
                results = []
                async for result in backend.text_completions(
                    prompt="test", request_id="req-123", stream_response=True
                ):
                    results.append(result)

            # Should get initial None and the valid response
            assert len(results) == 2
            assert results[0] == (None, None)
            assert results[1][0] == "Hello"
        finally:
            await backend.process_shutdown()

    @pytest.mark.sanity
    def test_get_chat_message_media_item_jpeg_file(self):
        """Test _get_chat_message_media_item with JPEG file path."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Create a mock Path object for JPEG file
        mock_jpeg_path = Mock(spec=Path)
        mock_jpeg_path.suffix.lower.return_value = ".jpg"

        # Mock Image.open to return a mock image
        mock_image = Mock(spec=Image.Image)
        mock_image.tobytes.return_value = b"fake_jpeg_data"

        with patch("guidellm.backend.openai.Image.open", return_value=mock_image):
            result = backend._get_chat_message_media_item(mock_jpeg_path)

        expected_data = base64.b64encode(b"fake_jpeg_data").decode("utf-8")
        expected = {
            "type": "image",
            "image": {"url": f"data:image/jpeg;base64,{expected_data}"},
        }
        assert result == expected

    @pytest.mark.sanity
    def test_get_chat_message_media_item_wav_file(self):
        """Test _get_chat_message_media_item with WAV file path."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Create a mock Path object for WAV file
        mock_wav_path = Mock(spec=Path)
        mock_wav_path.suffix.lower.return_value = ".wav"
        mock_wav_path.read_bytes.return_value = b"fake_wav_data"

        result = backend._get_chat_message_media_item(mock_wav_path)

        expected_data = base64.b64encode(b"fake_wav_data").decode("utf-8")
        expected = {
            "type": "input_audio",
            "input_audio": {"data": expected_data, "format": "wav"},
        }
        assert result == expected

    @pytest.mark.sanity
    def test_get_chat_messages_with_pil_image(self):
        """Test _get_chat_messages with PIL Image in content list."""
        backend = OpenAIHTTPBackend(target="http://test")

        # Create a mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_image.tobytes.return_value = b"fake_image_bytes"

        content = ["Hello", mock_image, "world"]

        result = backend._get_chat_messages(content)

        # Should have one user message with mixed content
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 3

        # Check text items
        assert result[0]["content"][0] == {"type": "text", "text": "Hello"}
        assert result[0]["content"][2] == {"type": "text", "text": "world"}

        # Check image item
        image_item = result[0]["content"][1]
        assert image_item["type"] == "image"
        assert "data:image/jpeg;base64," in image_item["image"]["url"]

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_resolve_timing_edge_cases(self):
        """Test resolve method timing edge cases."""
        backend = OpenAIHTTPBackend(target="http://test")
        await backend.process_startup()

        try:
            request = GenerationRequest(
                content="test prompt",
                request_type="text_completions",
                constraints={"output_tokens": 50},
            )
            request_info = ScheduledRequestInfo(
                request_id="test-id",
                status="pending",
                scheduler_node_id=1,
                scheduler_process_id=1,
                scheduler_start_time=123.0,
                request_timings=GenerationRequestTimings(),
            )

            # Mock text_completions to test timing edge cases
            async def mock_text_completions(*args, **kwargs):
                yield None, None  # Initial yield - tests line 343
                yield "token1", None  # First token
                yield "token2", UsageStats(prompt_tokens=10, output_tokens=2)  # Final

            with patch.object(
                backend, "text_completions", side_effect=mock_text_completions
            ):
                responses = []
                async for response, info in backend.resolve(request, request_info):
                    responses.append((response, info))

            # Check that timing was properly set
            final_response, final_info = responses[-1]
            assert final_info.request_timings.request_start is not None
            assert final_info.request_timings.first_iteration is not None
            assert final_info.request_timings.last_iteration is not None
            assert final_info.request_timings.request_end is not None
            assert final_response.delta is None  # Tests line 362

        finally:
            await backend.process_shutdown()
