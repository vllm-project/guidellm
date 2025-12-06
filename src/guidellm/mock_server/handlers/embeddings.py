"""
Embeddings API handler for the mock server.

This module provides the EmbeddingsHandler class that implements the /v1/embeddings
endpoint for the guidellm mock server. It generates synthetic embedding vectors
to simulate realistic embedding model behavior for benchmarking and testing purposes.
The handler supports both single and batch text inputs with configurable dimensions
and realistic timing delays.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid

from pydantic import ValidationError
from sanic import response
from sanic.request import Request
from sanic.response import HTTPResponse
from transformers import PreTrainedTokenizer

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.models import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorDetail,
    ErrorResponse,
    Usage,
)
from guidellm.mock_server.utils import MockTokenizer, sample_number

__all__ = ["EmbeddingsHandler"]


class EmbeddingsHandler:
    """
    Handler for the OpenAI /v1/embeddings endpoint in the mock server.

    This handler simulates the OpenAI embeddings API by processing incoming
    requests and generating synthetic embedding vectors. It applies realistic
    timing delays based on input token count to mimic actual embedding model
    behavior for benchmarking purposes.

    Example:
    ::
        config = MockServerConfig(ttft_ms=50)
        handler = EmbeddingsHandler(config)
        response = await handler.handle(sanic_request)
    """

    def __init__(self, config: MockServerConfig) -> None:
        """
        Initialize the embeddings handler with configuration settings.

        :param config: Mock server configuration containing timing parameters
            and tokenizer settings
        """
        self.config = config
        self.tokenizer = (
            MockTokenizer()
            if config.processor is None
            else PreTrainedTokenizer.from_pretrained(config.processor)
        )

    async def handle(self, request: Request) -> HTTPResponse:
        """
        Process an embeddings request and return the appropriate response.

        Validates the incoming request, generates synthetic embedding vectors,
        and returns the response with usage statistics.

        :param request: Sanic request object containing the embeddings request data
        :return: HTTP response with embedding data or error information
        :raises ValidationError: When request validation fails
        :raises json.JSONDecodeError: When request JSON is malformed
        """
        try:
            req_data = EmbeddingsRequest(**request.json)
        except ValidationError as e:
            return response.json(
                ErrorResponse(
                    error=ErrorDetail(
                        message=f"Invalid request: {str(e)}",
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

        return await self._handle_embeddings(req_data)

    async def _handle_embeddings(self, req: EmbeddingsRequest) -> HTTPResponse:
        """
        Generate embeddings for the input text(s).

        Creates synthetic embedding vectors with realistic timing delays based on
        the number of input tokens. Supports both single text and batch processing.

        :param req: Validated embeddings request containing input text(s)
        :return: JSON HTTP response with embedding vectors and usage data
        """
        inputs = [req.input] if isinstance(req.input, str) else req.input

        total_tokens = sum(len(self.tokenizer(text)) for text in inputs)

        # Simulate processing delay based on token count
        # Use TTFT config as base delay per token
        processing_delay = (
            sample_number(self.config.ttft_ms, self.config.ttft_ms_std) / 1000.0
        )
        await asyncio.sleep(processing_delay)

        # Determine embedding dimensions
        # Default to 1536 (OpenAI ada-002 dimension) or use requested dimensions
        dimensions = req.dimensions if req.dimensions else 1536

        # Generate synthetic embeddings for each input
        embeddings_data = []
        for index, _text in enumerate(inputs):
            # Generate random normalized embedding vector
            embedding = self._generate_embedding(dimensions)
            embeddings_data.append(
                EmbeddingObject(
                    embedding=embedding,
                    index=index,
                )
            )

        embeddings_response = EmbeddingsResponse(
            id=f"embd-{uuid.uuid4().hex[:29]}",
            model=req.model,
            data=embeddings_data,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
            ),
        )

        return response.json(embeddings_response.model_dump())

    def _generate_embedding(self, dimensions: int) -> list[float]:
        """
        Generate a random normalized embedding vector.

        Creates a synthetic embedding vector with the specified number of
        dimensions, normalized to unit length to mimic real embedding outputs.

        :param dimensions: Number of dimensions in the embedding vector
        :return: Normalized embedding vector as a list of floats
        """
        # Generate random vector
        embedding = [random.gauss(0, 1) for _ in range(dimensions)]

        # Normalize to unit length
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
