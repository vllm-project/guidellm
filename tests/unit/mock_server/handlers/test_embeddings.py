from __future__ import annotations

import base64
import struct

import pytest

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.handlers.embeddings import EmbeddingsHandler
from guidellm.mock_server.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
)


class TestEmbeddingsHandler:
    """Tests for embeddings mock server handler."""

    @pytest.fixture
    def handler(self):
        """Create embeddings handler with default config."""
        config = MockServerConfig()
        return EmbeddingsHandler(config)

    @pytest.fixture
    def handler_with_ttft(self):
        """Create embeddings handler with TTFT delay."""
        config = MockServerConfig(ttft_ms=100.0)
        return EmbeddingsHandler(config)

    @pytest.mark.smoke
    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler is not None
        assert handler.config is not None

    @pytest.mark.sanity
    async def test_handle_basic_request(self, handler):
        """Test handling a basic embeddings request."""
        request = EmbeddingsRequest(
            input="Test sentence for embedding.",
            model="test-embedding-model",
        )

        response = await handler.handle(request)

        assert isinstance(response, EmbeddingsResponse)
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.model == "test-embedding-model"

    @pytest.mark.sanity
    async def test_handle_single_string_input(self, handler):
        """Test handling request with single string input."""
        request = EmbeddingsRequest(
            input="Single string input.",
            model="test-model",
        )

        response = await handler.handle(request)

        assert len(response.data) == 1
        assert response.data[0].index == 0
        assert response.data[0].object == "embedding"

    @pytest.mark.sanity
    async def test_handle_list_input(self, handler):
        """Test handling request with list of strings."""
        inputs = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        request = EmbeddingsRequest(
            input=inputs,
            model="test-model",
        )

        response = await handler.handle(request)

        assert len(response.data) == 3
        for i, emb_obj in enumerate(response.data):
            assert emb_obj.index == i
            assert emb_obj.object == "embedding"

    @pytest.mark.sanity
    async def test_float_encoding(self, handler):
        """Test float encoding format (default)."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            encoding_format="float",
        )

        response = await handler.handle(request)

        # Embedding should be a list of floats
        embedding = response.data[0].embedding
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.sanity
    async def test_base64_encoding(self, handler):
        """Test base64 encoding format."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            encoding_format="base64",
        )

        response = await handler.handle(request)

        # Embedding should be a base64-encoded string
        embedding = response.data[0].embedding
        assert isinstance(embedding, str)

        # Verify it's valid base64
        try:
            decoded_bytes = base64.b64decode(embedding)
            assert len(decoded_bytes) > 0
        except Exception:  # noqa: BLE001
            pytest.fail("Invalid base64 encoding")

    @pytest.mark.regression
    async def test_base64_encoding_decodes_to_floats(self, handler):
        """Test that base64 encoding can be decoded back to floats."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            encoding_format="base64",
        )

        response = await handler.handle(request)

        # Decode base64 to float array
        embedding_b64 = response.data[0].embedding
        decoded_bytes = base64.b64decode(embedding_b64)

        # Unpack as floats
        num_floats = len(decoded_bytes) // 4  # 4 bytes per float
        floats = struct.unpack(f"{num_floats}f", decoded_bytes)

        # Should be a valid array of floats
        assert len(floats) > 0
        assert all(isinstance(x, float) for x in floats)

    @pytest.mark.sanity
    async def test_usage_metrics(self, handler):
        """Test that usage metrics are populated."""
        request = EmbeddingsRequest(
            input="Test sentence with some tokens.",
            model="test-model",
        )

        response = await handler.handle(request)

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0
        # Embeddings don't have completion tokens
        assert response.usage.completion_tokens == 0

    @pytest.mark.regression
    async def test_usage_metrics_batch(self, handler):
        """Test usage metrics with batch input."""
        inputs = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        request = EmbeddingsRequest(
            input=inputs,
            model="test-model",
        )

        response = await handler.handle(request)

        # Total tokens should sum across all inputs
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens

    @pytest.mark.sanity
    async def test_dimensions_parameter(self, handler):
        """Test dimensions parameter (Matryoshka embeddings)."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            dimensions=128,
            encoding_format="float",
        )

        response = await handler.handle(request)

        # Embedding should have specified dimensions
        embedding = response.data[0].embedding
        assert len(embedding) == 128

    @pytest.mark.regression
    async def test_dimensions_default(self, handler):
        """Test default dimensions when not specified."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            encoding_format="float",
        )

        response = await handler.handle(request)

        # Default dimensions should be used (typically 384 or similar)
        embedding = response.data[0].embedding
        assert len(embedding) > 0
        # Common default dimension sizes
        assert len(embedding) in [384, 512, 768, 1024, 1536]

    @pytest.mark.sanity
    async def test_truncate_prompt_tokens(self, handler):
        """Test truncate_prompt_tokens parameter."""
        request = EmbeddingsRequest(
            input="A very long sentence with many tokens that should be truncated.",
            model="test-model",
            truncate_prompt_tokens=10,
        )

        response = await handler.handle(request)

        # Usage should reflect truncation
        assert response.usage.prompt_tokens <= 10

    @pytest.mark.regression
    async def test_embedding_normalized(self, handler):
        """Test that embeddings are normalized (unit length)."""
        import math

        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            encoding_format="float",
        )

        response = await handler.handle(request)

        embedding = response.data[0].embedding

        # Calculate norm (should be 1.0 for normalized vector)
        norm = math.sqrt(sum(x * x for x in embedding))
        assert norm == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.regression
    async def test_multiple_embeddings_different(self, handler):
        """Test that different inputs produce different embeddings."""
        request = EmbeddingsRequest(
            input=["First sentence.", "Second sentence."],
            model="test-model",
            encoding_format="float",
        )

        response = await handler.handle(request)

        emb1 = response.data[0].embedding
        emb2 = response.data[1].embedding

        # Embeddings should be different (random generation)
        assert emb1 != emb2

    @pytest.mark.sanity
    async def test_ttft_delay(self, handler_with_ttft):
        """Test that TTFT delay is applied."""
        import time

        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
        )

        start = time.time()
        await handler_with_ttft.handle(request)
        elapsed = time.time() - start

        # Should have some delay (at least 50ms for 100ms TTFT config)
        assert elapsed >= 0.05  # Reduced threshold for test reliability

    @pytest.mark.regression
    async def test_empty_input(self, handler):
        """Test handling empty input string."""
        request = EmbeddingsRequest(
            input="",
            model="test-model",
        )

        response = await handler.handle(request)

        # Should still produce an embedding (possibly all zeros or minimal)
        assert len(response.data) == 1
        assert response.usage.prompt_tokens >= 0

    @pytest.mark.regression
    async def test_response_model_matches_request(self, handler):
        """Test that response model matches request model."""
        model_name = "custom-embedding-model-v2"
        request = EmbeddingsRequest(
            input="Test sentence.",
            model=model_name,
        )

        response = await handler.handle(request)

        assert response.model == model_name

    @pytest.mark.sanity
    async def test_embedding_object_fields(self, handler):
        """Test that embedding objects have correct fields."""
        request = EmbeddingsRequest(
            input=["First.", "Second."],
            model="test-model",
        )

        response = await handler.handle(request)

        for emb_obj in response.data:
            assert hasattr(emb_obj, "object")
            assert hasattr(emb_obj, "embedding")
            assert hasattr(emb_obj, "index")
            assert emb_obj.object == "embedding"

    @pytest.mark.regression
    async def test_large_batch_input(self, handler):
        """Test handling large batch of inputs."""
        inputs = [f"Sentence number {i}." for i in range(100)]

        request = EmbeddingsRequest(
            input=inputs,
            model="test-model",
        )

        response = await handler.handle(request)

        assert len(response.data) == 100
        for i, emb_obj in enumerate(response.data):
            assert emb_obj.index == i

    @pytest.mark.regression
    async def test_user_parameter(self, handler):
        """Test user parameter (should be accepted but not affect output)."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
            user="test-user-123",
        )

        response = await handler.handle(request)

        # Should complete successfully
        assert isinstance(response, EmbeddingsResponse)
        assert len(response.data) == 1

    @pytest.mark.sanity
    async def test_response_object_field(self, handler):
        """Test that response object field is 'list'."""
        request = EmbeddingsRequest(
            input="Test sentence.",
            model="test-model",
        )

        response = await handler.handle(request)

        assert response.object == "list"
