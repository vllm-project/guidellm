"""
Configuration settings for the mock server component.

Provides centralized configuration management for mock server behavior including
network binding, model identification, response timing characteristics, and token
generation parameters. Supports environment variable configuration for deployment
flexibility with automatic validation through Pydantic settings.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

__all__ = ["MockServerConfig"]


class MockServerConfig(BaseSettings):
    """
    Configuration settings for mock server behavior and deployment.

    Centralizes all configurable parameters for mock server operation including
    network settings, model identification, response timing characteristics, and
    token generation behavior. Environment variables with GUIDELLM_MOCK_SERVER_
    prefix override default values for deployment flexibility.

    Example:
    ::
        config = MockServerConfig(host="0.0.0.0", port=8080, model="custom-model")
        # Use with environment variables:
        # GUIDELLM_MOCK_SERVER_HOST=127.0.0.1 GUIDELLM_MOCK_SERVER_PORT=9000
    """

    host: str = Field(
        default="127.0.0.1", description="Host address to bind the server to"
    )
    port: int = Field(default=8000, description="Port number to bind the server to")
    workers: int = Field(default=1, description="Number of worker processes to spawn")
    model: str = Field(
        default="llama-3.1-8b-instruct",
        description="Model name to present in API responses",
    )
    processor: str | None = Field(
        default=None,
        description=(
            "Processor type to use for token stats, tokenize, and detokenize. "
            "If None, a mock one is created."
        ),
    )
    request_latency: float = Field(
        default=3.0,
        description="Base request latency in seconds for non-streaming responses",
    )
    request_latency_std: float = Field(
        default=0.0,
        description="Standard deviation for request latency variation",
    )
    ttft_ms: float = Field(
        default=150.0,
        description="Time to first token in milliseconds for streaming responses",
    )
    ttft_ms_std: float = Field(
        default=0.0,
        description="Standard deviation for time to first token variation",
    )
    itl_ms: float = Field(
        default=10.0,
        description="Inter-token latency in milliseconds for streaming responses",
    )
    itl_ms_std: float = Field(
        default=0.0,
        description="Standard deviation for inter-token latency variation",
    )
    output_tokens: int = Field(
        default=128, description="Number of output tokens to generate in responses"
    )
    output_tokens_std: float = Field(
        default=0.0,
        description="Standard deviation for output token count variation",
    )

    class Config:
        env_prefix = "GUIDELLM_MOCK_SERVER_"
        case_sensitive = False
