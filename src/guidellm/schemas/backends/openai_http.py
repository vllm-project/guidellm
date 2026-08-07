"""
OpenAI HTTP backend Args schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator

from guidellm.schemas import GenerationRequestArguments
from guidellm.schemas.backends.backend import BackendArgs

__all__ = ["OpenAIHTTPBackendArgs"]

DEFAULT_API_PATHS = {
    "/health": "health",
    "/v1/models": "v1/models",
    "/v1/completions": "v1/completions",
    "/v1/chat/completions": "v1/chat/completions",
    "/v1/embeddings": "v1/embeddings",
    "/v1/responses": "v1/responses",
    "/v1/audio/transcriptions": "v1/audio/transcriptions",
    "/v1/audio/translations": "v1/audio/translations",
    "/pooling": "pooling",
}


@BackendArgs.register("openai_http")
class OpenAIHTTPBackendArgs(BackendArgs):
    """Pydantic model for OpenAI HTTP backend creation arguments."""

    kind: Literal["openai_http"] = Field(
        default="openai_http",
        description="Type identifier for the backend configuration.",
    )
    target: str = Field(
        description="Base URL of an OpenAI-compatible inference server",
        examples=["http://localhost:8000"],
    )
    model: str = Field(
        default_factory=str,
        description="Huggingface model identifier or local path to a model",
        examples=["gpt-4o", "Qwen/Qwen3-0.6B"],
    )
    request_format: Literal[
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/responses",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/pooling",
    ] = Field(
        default="/v1/chat/completions",
        description=(
            "Request format for desired API endpoint of the OpenAI-compatible server."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="HTTP Bearer token API key for authentication to server",
        examples=["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"],
    )
    api_routes: dict[str, str] = Field(
        default_factory=dict,
        validate_default=True,
        description=(
            "Custom API endpoint routes mapping. Keys should be request types "
            "like '/v1/completions' and values should be the corresponding "
            "endpoint paths relative to the target URL."
        ),
        examples=[
            {
                "/v1/chat/completions": "/v1/chat/completions",
                "/v1/embeddings": "/v1/embeddings",
                "/v1/responses": "/v1/responses",
                "/v1/audio/translations": "/v1/audio/translations",
            }
        ],
    )
    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds when reading a server response.",
        examples=[10.0],
    )
    timeout_connect: float | None = Field(
        default=5.0,
        description="Request timeout in seconds for establishing server connection.",
        examples=[10.0],
    )
    http2: bool = Field(
        default=True,
        description="Enable HTTP/2 protocol.",
    )
    follow_redirects: bool = Field(
        default=True,
        description="Follow HTTP redirect response headers automatically.",
    )
    verify: bool = Field(
        default=False,
        description="Verify the server's TLS certificate.",
    )
    validate_backend: bool = Field(
        default=True,
        description="Send a health check request to validate backend configuration.",
    )
    stream: bool = Field(
        default=True,
        description="Use streaming responses for generation requests when supported.",
    )
    extras: GenerationRequestArguments | None = Field(
        default=None,
        description="Additional parameters to include in generation requests.",
    )
    max_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
        description="Maximum number of tokens to request in any response.",
        examples=[1024],
    )
    server_history: bool = Field(
        default=False,
        description=(
            "Use server-side conversation history (previous_response_id) for "
            "multi-turn requests. Only supported with /v1/responses."
        ),
    )
    tool_call_missing_behavior: Literal[
        "ignore_continue", "ignore_stop", "error_stop"
    ] = Field(
        default="error_stop",
        description=(
            "Specify behavior when a tool call is expected but the model does not "
            "produce one. Options: ignore_continue (continue to next turn), "
            "ignore_stop (cancel remaining turns), error_stop (error and "
            "cancel remaining turns)."
        ),
    )
    multiturn_reasoning: bool | str = Field(
        default=False,
        description=(
            "Include reasoning/chain-of-thought text in multi-turn "
            "conversation history. False disables (default). True wraps "
            "reasoning in <think>...</think> tags (equivalent to the string "
            "'<think>{reasoning}</think>'). A string value is used as a "
            "format template and must contain the '{reasoning}' placeholder text."
        ),
    )

    @field_validator("multiturn_reasoning", mode="after")
    @classmethod
    def validate_multiturn_reasoning(cls, value: bool | str) -> bool | str:
        """Reject non-empty strings that don't contain the {reasoning} placeholder."""
        if isinstance(value, str) and "{reasoning}" not in value:
            raise ValueError(
                "multiturn_reasoning string must contain '{reasoning}' "
                f"placeholder, got: {value!r}"
            )
        return value

    @field_validator("target", mode="after")
    @classmethod
    def strip_target(cls, value: str) -> str:
        """Strip trailing slashes and API paths from the target URL."""
        return value.rstrip("/").removesuffix("/v1")

    @field_validator("api_routes", mode="after")
    @classmethod
    def merge_api_routes(cls, value: dict[str, str]) -> dict[str, str]:
        """Merge user-provided API routes with default routes."""
        return DEFAULT_API_PATHS | value

    @model_validator(mode="after")
    def validate_server_history(self):
        """Validate that server_history is only True with supported endpoints."""
        if self.server_history and self.request_format != "/v1/responses":
            raise ValueError(
                "server_history=True is only supported with the /v1/responses "
                "request format. Current request_format: "
                f"'{self.request_format}'"
            )
        return self
