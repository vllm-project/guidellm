"""
Mock server handler for OpenAI-compatible /v1/embeddings endpoint.

Generates synthetic normalized embedding vectors with configurable dimensions and
encoding formats. Simulates realistic embedding API behavior including timing delays,
token counting, and batch processing while providing deterministic outputs for testing.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import random
import struct
from typing import TYPE_CHECKING

from pydantic import ValidationError
from sanic import response
from sanic.request import Request
from sanic.response import HTTPResponse

from guidellm.mock_server.models import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorDetail,
    ErrorResponse,
    Usage,
)
from guidellm.mock_server.utils import MockTokenizer

if TYPE_CHECKING:
    from guidellm.mock_server.config import MockServerConfig

__all__ = ["EmbeddingsHandler"]


class EmbeddingsHandler:
    """
    Handler for /v1/embeddings endpoint in mock server.

    Processes embeddings requests and generates synthetic normalized embedding
    vectors with realistic timing simulation. Supports both float and base64
    encoding formats, batch processing, and optional dimension reduction.

    Example:
    ::
        handler = EmbeddingsHandler(config)
        response = await handler.handle(request)
    """

    def __init__(self, config: MockServerConfig):
        """
        Initialize embeddings handler with server configuration.

        :param config: Mock server configuration with timing and model parameters
        """
        self.config = config
        self.tokenizer = MockTokenizer()

    async def handle(self, request: Request) -> HTTPResponse:
        """
        Process embeddings request and return response.

        :param request: HTTP request containing embeddings parameters
        :return: HTTP response with embeddings data or error
        """
        try:
            # Parse request body
            req = EmbeddingsRequest(**request.json)
        except ValidationError as exc:
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid request: {str(exc)}",
                        type="invalid_request_error",
                        code="invalid_request",
                    )
                ).model_dump(),
                status=400,
            )
        except (json.JSONDecodeError, TypeError):
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message="Invalid JSON in request body",
                        type="invalid_request_error",
                        code="invalid_json",
                    )
                ).model_dump(),
                status=400,
            )

        # Handle input as list
        inputs = [req.input] if isinstance(req.input, str) else req.input

        # Determine embedding dimensions
        dimensions = req.dimensions if req.dimensions is not None else 384  # Default dim

        # Validate encoding format
        encoding_format = req.encoding_format or "float"
        if encoding_format not in {"float", "base64"}:
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid encoding_format: {encoding_format}. Must be 'float' or 'base64'",
                        type="invalid_request_error",
                        code="invalid_encoding_format",
                    )
                ).model_dump(),
                status=400,
            )

        # Count total tokens (for timing and usage)
        total_tokens = 0
        for text in inputs:
            tokens = len(self.tokenizer.tokenize(text))

            # Apply truncation if requested
            if req.truncate_prompt_tokens is not None:
                tokens = min(tokens, req.truncate_prompt_tokens)

            total_tokens += tokens

        # Simulate time-to-first-token delay based on input tokens
        # TTFT is proportional to input processing time
        if self.config.ttft_ms > 0:
            delay_ms = max(
                0,
                random.gauss(
                    self.config.ttft_ms,
                    self.config.ttft_ms_std if self.config.ttft_ms_std > 0 else 0,
                ),
            )
            await asyncio.sleep(delay_ms / 1000.0)

        # Generate embeddings for each input
        embeddings_data = []
        for index, text in enumerate(inputs):
            # Generate synthetic normalized embedding
            embedding_vector = self._generate_embedding(dimensions)

            # Encode based on requested format
            if encoding_format == "base64":
                embedding_encoded = self._encode_to_base64(embedding_vector)
            else:
                embedding_encoded = embedding_vector

            embeddings_data.append(
                EmbeddingObject(
                    embedding=embedding_encoded,
                    index=index,
                )
            )

        # Build usage stats (embeddings have no completion_tokens)
        usage = Usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,  # Embeddings don't generate tokens
        )

        # Build response
        embeddings_response = EmbeddingsResponse(
            data=embeddings_data,
            model=req.model,
            usage=usage,
        )

        return HTTPResponse(
            body=embeddings_response.model_dump_json(),
            status=200,
            headers={"Content-Type": "application/json"},
        )

    def _generate_embedding(self, dimensions: int) -> list[float]:
        """
        Generate synthetic normalized embedding vector.

        Creates a random vector and normalizes it to unit length (L2 norm = 1),
        which is standard for embedding models.

        :param dimensions: Number of dimensions for the embedding
        :return: Normalized embedding vector as list of floats

        Example:
        ::
            emb = handler._generate_embedding(384)
            norm = math.sqrt(sum(x*x for x in emb))  # Should be ≈1.0
        """
        # Generate random vector from Gaussian distribution
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _encode_to_base64(self, embedding: list[float]) -> str:
        """
        Encode embedding vector as base64-encoded binary string.

        Converts float list to packed binary format (little-endian floats)
        and encodes as base64 string for efficient transmission.

        :param embedding: Embedding vector as list of floats
        :return: Base64-encoded binary representation

        Example:
        ::
            embedding = [0.1, 0.2, 0.3]
            encoded = handler._encode_to_base64(embedding)
            # Returns base64 string like "MzMzPz8/Pz8/Pz8="
        """
        # Pack floats as little-endian binary
        # Format: 'f' = single-precision float (4 bytes each)
        bytes_data = struct.pack(f"{len(embedding)}f", *embedding)

        # Encode as base64
        encoded = base64.b64encode(bytes_data).decode("utf-8")

        return encoded

    @staticmethod
    def decode_from_base64(encoded: str, dimensions: int) -> list[float]:
        """
        Decode base64-encoded embedding back to float list.

        Utility method for testing and validation. Reverses the encoding
        performed by _encode_to_base64.

        :param encoded: Base64-encoded binary string
        :param dimensions: Number of dimensions to decode
        :return: Decoded embedding vector as list of floats

        Example:
        ::
            encoded = "MzMzPz8/Pz8/Pz8="
            decoded = EmbeddingsHandler.decode_from_base64(encoded, 3)
            # Returns approximately [0.1, 0.2, 0.3]
        """
        # Decode base64 to bytes
        bytes_data = base64.b64decode(encoded)

        # Unpack floats
        embedding = list(struct.unpack(f"{dimensions}f", bytes_data))

        return embedding
