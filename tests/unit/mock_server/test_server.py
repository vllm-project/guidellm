from __future__ import annotations

import asyncio
import json
import multiprocessing

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.server import MockServer


# Start server in a separate process
def _start_server_process(config: MockServerConfig):
    server = MockServer(config)
    server.run()


@pytest_asyncio.fixture(scope="class")
async def mock_server_instance():
    """Instance-level fixture that provides a running server for HTTP testing."""

    config = MockServerConfig(
        host="127.0.0.1",
        port=8012,
        model="test-model",
        ttft_ms=10.0,
        itl_ms=1.0,
        request_latency=0.1,
    )
    base_url = f"http://{config.host}:{config.port}"
    server_process = multiprocessing.Process(
        target=_start_server_process, args=(config,)
    )
    server_process.start()

    # Wait for server to start up and be ready
    async def wait_for_startup():
        poll_frequency = 1.0
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(f"{base_url}/health", timeout=1.0)
                    if response.status_code == 200:
                        break
                except (httpx.RequestError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(poll_frequency)
                poll_frequency = min(poll_frequency * 1.5, 2.0)

    timeout = 30.0
    try:
        await asyncio.wait_for(wait_for_startup(), timeout)
    except TimeoutError:
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
            server_process.join(timeout=5)
        pytest.fail(f"Server failed to start within {timeout} seconds")

    yield base_url, config

    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()
        server_process.join(timeout=5)


class TestMockServerConfig:
    """Test suite for MockServerConfig class."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test MockServerConfig initialization with default values."""
        config = MockServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1
        assert config.model == "llama-3.1-8b-instruct"
        assert config.processor is None
        assert config.request_latency == 3.0
        assert config.request_latency_std == 0.0
        assert config.ttft_ms == 150.0
        assert config.ttft_ms_std == 0.0
        assert config.itl_ms == 10.0
        assert config.itl_ms_std == 0.0
        assert config.output_tokens == 128
        assert config.output_tokens_std == 0.0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("kwargs", "expected_values"),
        [
            (
                {"host": "127.0.0.1", "port": 9000, "model": "custom-model"},
                {"host": "127.0.0.1", "port": 9000, "model": "custom-model"},
            ),
            (
                {"request_latency": 1.5, "ttft_ms": 100.0, "output_tokens": 256},
                {"request_latency": 1.5, "ttft_ms": 100.0, "output_tokens": 256},
            ),
        ],
    )
    def test_custom_initialization(self, kwargs, expected_values):
        """Test MockServerConfig initialization with custom values."""
        config = MockServerConfig(**kwargs)
        for key, expected_value in expected_values.items():
            assert getattr(config, key) == expected_value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("port", "not_int"),
            ("request_latency", "not_float"),
            ("output_tokens", "not_int"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test MockServerConfig with invalid field values."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            MockServerConfig(**kwargs)


class TestMockServer:
    """Test suite for MockServer class."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test MockServer class signatures and attributes."""
        assert hasattr(MockServer, "__init__")
        assert hasattr(MockServer, "run")
        assert hasattr(MockServer, "_setup_middleware")
        assert hasattr(MockServer, "_setup_routes")
        assert hasattr(MockServer, "_setup_error_handlers")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test MockServer initialization without required config."""
        with pytest.raises(TypeError):
            MockServer()


class TestMockServerEndpoints:
    """Test suite for MockServer HTTP endpoints with real server instances."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_server_instance):
        """Test the health check endpoint."""
        server_url, _ = mock_server_instance

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/health", timeout=5.0)
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert isinstance(data["timestamp"], int | float)

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_models_endpoint(self, mock_server_instance):
        """Test the models listing endpoint."""
        server_url, _ = mock_server_instance

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/v1/models", timeout=5.0)
            assert response.status_code == 200

            data = response.json()
            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert isinstance(data["data"], list)
            assert len(data["data"]) > 0

            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model
            assert model["object"] == "model"
            assert model["owned_by"] == "guidellm-mock"
            assert model["id"] == "test-model"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 10,
                },
                ["choices", "usage", "model", "object"],
            ),
            (
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 5,
                    "temperature": 0.7,
                },
                ["choices", "usage", "model", "object"],
            ),
            (
                {
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello from list format!"}
                            ],
                        }
                    ],
                    "max_tokens": 10,
                },
                ["choices", "usage", "model", "object"],
            ),
        ],
    )
    async def test_chat_completions_endpoint(
        self, mock_server_instance, payload, expected_fields
    ):
        """Test the chat completions endpoint."""
        server_url, _ = mock_server_instance

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/v1/chat/completions", json=payload, timeout=10.0
            )
            assert response.status_code == 200

            data = response.json()
            for field in expected_fields:
                assert field in data

            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "message" in choice
            assert "content" in choice["message"]
            assert "role" in choice["message"]
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)
            assert len(choice["message"]["content"]) > 0

            # Verify usage information
            assert "prompt_tokens" in data["usage"]
            assert "completion_tokens" in data["usage"]
            assert "total_tokens" in data["usage"]
            assert data["usage"]["total_tokens"] == (
                data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
            )

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_streaming_chat_completions(self, mock_server_instance):
        """Test streaming chat completions endpoint."""
        server_url, _ = mock_server_instance

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi!"}],
            "max_tokens": 5,
            "stream": True,
        }

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=10.0,
            ) as response,
        ):
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        continue

            assert len(chunks) > 0
            # Verify chunk structure
            for chunk in chunks:
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0
                assert "delta" in chunk["choices"][0]

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_streaming_chat_completions_list_content(self, mock_server_instance):
        """
        Test streaming chat completions with list-of-content-parts message format.

        ## WRITTEN BY AI ##
        """
        server_url, _ = mock_server_instance

        payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi!"}],
                }
            ],
            "max_tokens": 5,
            "stream": True,
        }

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=10.0,
            ) as response,
        ):
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        continue

            assert len(chunks) > 0
            for chunk in chunks:
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0
                assert "delta" in chunk["choices"][0]

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {
                    "model": "test-model",
                    "prompt": "Hello",
                    "max_tokens": 10,
                },
                ["choices", "usage", "model", "object"],
            ),
            (
                {
                    "model": "test-model",
                    "prompt": "Test prompt",
                    "max_tokens": 5,
                    "temperature": 0.8,
                },
                ["choices", "usage", "model", "object"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_completions_endpoint(
        self, mock_server_instance, payload, expected_fields
    ):
        """Test the legacy completions endpoint."""
        server_url, _ = mock_server_instance

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/v1/completions", json=payload, timeout=10.0
            )
            assert response.status_code == 200

            data = response.json()
            for field in expected_fields:
                assert field in data

            assert len(data["choices"]) > 0
            choice = data["choices"][0]
            assert "text" in choice
            assert isinstance(choice["text"], str)
            assert len(choice["text"]) > 0

            # Verify usage information
            assert "prompt_tokens" in data["usage"]
            assert "completion_tokens" in data["usage"]
            assert "total_tokens" in data["usage"]

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_streaming_completions(self, mock_server_instance):
        """Test streaming completions endpoint."""
        server_url, _ = mock_server_instance
        payload = {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": True,
        }

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{server_url}/v1/completions",
                json=payload,
                timeout=10.0,
            ) as response,
        ):
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        continue

            assert len(chunks) > 0
            # Verify chunk structure
            for chunk in chunks:
                assert "choices" in chunk
                assert len(chunk["choices"]) > 0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {"text": "Hello world!"},
                ["tokens", "count"],
            ),
            (
                {"text": "This is a test sentence."},
                ["tokens", "count"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_tokenize_endpoint(
        self, mock_server_instance, payload, expected_fields
    ):
        """Test the tokenize endpoint."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/tokenize", json=payload, timeout=5.0
            )
            assert response.status_code == 200

            data = response.json()
            for field in expected_fields:
                assert field in data

            assert isinstance(data["tokens"], list)
            assert isinstance(data["count"], int)
            assert data["count"] == len(data["tokens"])
            assert len(data["tokens"]) > 0

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {"tokens": [123, 456, 789]},
                ["text"],
            ),
            (
                {"tokens": [100, 200]},
                ["text"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_detokenize_endpoint(
        self, mock_server_instance, payload, expected_fields
    ):
        """Test the detokenize endpoint."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/detokenize", json=payload, timeout=5.0
            )
            assert response.status_code == 200

            data = response.json()
            for field in expected_fields:
                assert field in data

            assert isinstance(data["text"], str)
            assert len(data["text"]) > 0

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_options_endpoint(self, mock_server_instance):
        """Test the OPTIONS endpoint for CORS support."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            response = await client.options(
                f"{server_url}/v1/chat/completions", timeout=5.0
            )
            assert response.status_code == 204
            assert response.text == ""

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_cors_headers(self, mock_server_instance):
        """Test CORS headers are properly set."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/health", timeout=5.0)
            assert response.status_code == 200

            # Check for CORS headers
            assert response.headers.get("Access-Control-Allow-Origin") == "*"
            methods_header = response.headers.get("Access-Control-Allow-Methods", "")
            assert "GET, POST, OPTIONS" in methods_header
            headers_header = response.headers.get("Access-Control-Allow-Headers", "")
            assert "Content-Type, Authorization" in headers_header
            assert response.headers.get("Server") == "guidellm-mock-server"

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "expected_fields"),
        [
            (
                {
                    "model": "test-model",
                    "input": "Hello!",
                    "max_output_tokens": 10,
                },
                ["output", "usage", "model", "object", "status"],
            ),
            (
                {
                    "model": "test-model",
                    "input": [{"role": "user", "content": "Hello from list format!"}],
                    "max_output_tokens": 10,
                },
                ["output", "usage", "model", "object", "status"],
            ),
            (
                {
                    "model": "test-model",
                    "input": "Test with instructions",
                    "instructions": "You are a helpful assistant.",
                    "max_output_tokens": 5,
                },
                ["output", "usage", "model", "object", "status"],
            ),
        ],
    )
    async def test_responses_endpoint(
        self, mock_server_instance, payload, expected_fields
    ):
        """
        Test the Responses API endpoint with various input formats.

        ## WRITTEN BY AI ##
        """
        server_url, _ = mock_server_instance

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{server_url}/v1/responses", json=payload, timeout=10.0
            )
            assert resp.status_code == 200

            data = resp.json()
            for field in expected_fields:
                assert field in data

            assert data["object"] == "response"
            assert data["status"] == "completed"
            assert len(data["output"]) > 0

            msg = data["output"][0]
            assert msg["type"] == "message"
            assert msg["role"] == "assistant"
            assert msg["status"] == "completed"
            assert len(msg["content"]) > 0
            assert msg["content"][0]["type"] == "output_text"
            assert isinstance(msg["content"][0]["text"], str)
            assert len(msg["content"][0]["text"]) > 0

            assert "input_tokens" in data["usage"]
            assert "output_tokens" in data["usage"]
            assert "total_tokens" in data["usage"]
            assert data["usage"]["total_tokens"] == (
                data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
            )

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_streaming_responses(self, mock_server_instance):
        """
        Test streaming Responses API endpoint with SSE event format.

        ## WRITTEN BY AI ##
        """
        server_url, _ = mock_server_instance

        payload = {
            "model": "test-model",
            "input": "Hi!",
            "max_output_tokens": 5,
            "stream": True,
        }

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{server_url}/v1/responses",
                json=payload,
                timeout=10.0,
            ) as resp,
        ):
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events: list[dict] = []
            event_types: list[str] = []
            current_event_type: str | None = None
            done_received = False

            async for line in resp.aiter_lines():
                if not line:
                    current_event_type = None
                    continue
                if line.startswith("event: "):
                    current_event_type = line[7:]
                elif line == "data: [DONE]":
                    done_received = True
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append(data)
                    if current_event_type:
                        event_types.append(current_event_type)

            assert done_received, "Stream should end with data: [DONE]"
            assert "response.created" in event_types
            assert "response.in_progress" in event_types
            assert "response.output_item.added" in event_types
            assert "response.content_part.added" in event_types
            assert "response.output_text.delta" in event_types
            assert "response.output_text.done" in event_types
            assert "response.output_item.done" in event_types
            assert "response.completed" in event_types

            deltas = [
                e for e in events if e.get("type") == "response.output_text.delta"
            ]
            assert len(deltas) > 0
            for d in deltas:
                assert "delta" in d

            completed = [e for e in events if e.get("type") == "response.completed"]
            assert len(completed) == 1
            assert "response" in completed[0]
            assert completed[0]["response"]["usage"] is not None
            assert "input_tokens" in completed[0]["response"]["usage"]
            assert "output_tokens" in completed[0]["response"]["usage"]

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("endpoint", "method", "payload"),
        [
            ("/v1/chat/completions", "POST", {"invalid": "payload"}),
            ("/v1/completions", "POST", {"invalid": "payload"}),
            ("/v1/responses", "POST", {"invalid": "payload"}),
            ("/tokenize", "POST", {"invalid": "payload"}),
            ("/detokenize", "POST", {"invalid": "payload"}),
        ],
    )
    async def test_invalid_request_handling(
        self, mock_server_instance, endpoint, method, payload
    ):
        """Test handling of invalid requests."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            if method == "POST":
                response = await client.post(
                    f"{server_url}{endpoint}", json=payload, timeout=5.0
                )
            else:
                response = await client.get(f"{server_url}{endpoint}", timeout=5.0)

            # Should return an error response, not crash
            assert response.status_code in [400, 422, 500]

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, mock_server_instance):
        """Test handling of requests to nonexistent endpoints."""
        server_url, _ = mock_server_instance
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/nonexistent", timeout=5.0)
            assert response.status_code == 404
