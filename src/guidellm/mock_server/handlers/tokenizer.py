"""
HTTP request handler for vLLM tokenization API endpoints in the mock server.

This module provides the TokenizerHandler class that implements vLLM-compatible
tokenization and detokenization endpoints for testing and development purposes.
It handles text-to-token conversion, token-to-text reconstruction, request
validation, and error responses with proper HTTP status codes and JSON formatting.
"""

from __future__ import annotations

from pydantic import ValidationError
from sanic import response
from sanic.request import Request
from sanic.response import HTTPResponse
from transformers import AutoTokenizer

from guidellm.mock_server.config import MockServerConfig
from guidellm.mock_server.models import (
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorDetail,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from guidellm.mock_server.utils import MockTokenizer

__all__ = ["TokenizerHandler"]


class TokenizerHandler:
    """
    HTTP request handler for vLLM tokenization and detokenization endpoints.

    Provides mock implementations of vLLM's tokenization API endpoints including
    /tokenize for converting text to tokens and /detokenize for reconstructing
    text from token sequences. Handles request validation, error responses, and
    JSON serialization with proper HTTP status codes.

    Example:
    ::
        handler = TokenizerHandler(config)
        response = await handler.tokenize(request)
        response = await handler.detokenize(request)
    """

    def __init__(self, config: MockServerConfig) -> None:
        """
        Initialize the tokenizer handler with configuration.

        :param config: Server configuration object containing tokenizer settings
        """
        self.config = config
        self.tokenizer = (
            MockTokenizer()
            if config.processor is None
            else AutoTokenizer.from_pretrained(config.processor)
        )

    async def tokenize(self, request: Request) -> HTTPResponse:
        """
        Convert input text to token IDs via the /tokenize endpoint.

        Validates the request payload, extracts text content, and returns a JSON
        response containing the token sequence and count. Handles validation errors
        and malformed JSON with appropriate HTTP error responses.

        :param request: Sanic HTTP request containing JSON payload with text field
        :return: JSON response with tokens list and count, or error response
        """
        try:
            req_data = TokenizeRequest(**request.json)
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
        except (ValueError, TypeError, KeyError):
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

        tokens = self.tokenizer.tokenize(req_data.text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return response.json(
            TokenizeResponse(tokens=token_ids, count=len(token_ids)).model_dump()
        )

    async def detokenize(self, request: Request) -> HTTPResponse:
        """
        Convert token IDs back to text via the /detokenize endpoint.

        Validates the request payload, extracts token sequences, and returns a JSON
        response containing the reconstructed text. Handles validation errors and
        malformed JSON with appropriate HTTP error responses.

        :param request: Sanic HTTP request containing JSON payload with tokens field
        :return: JSON response with reconstructed text, or error response
        """
        try:
            req_data = DetokenizeRequest(**request.json)
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
        except (ValueError, TypeError, KeyError):
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

        text = self.tokenizer.decode(req_data.tokens, skip_special_tokens=False)

        return response.json(DetokenizeResponse(text=text).model_dump())
