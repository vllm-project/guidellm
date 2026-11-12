"""
OpenAI WebSocket backend Args schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from guidellm.schemas.backends.backend import BackendArgs, resolve_api_key_file

__all__ = ["OpenAIWebSocketBackendArgs"]

FALLBACK_TIMEOUT = 5.0


@BackendArgs.register("openai_websocket")
class OpenAIWebSocketBackendArgs(BackendArgs):
    """Typed configuration for :class:`OpenAIWebSocketBackend`."""

    kind: Literal["openai_websocket"] = Field(
        default="openai_websocket",
        description="Type identifier for the backend configuration.",
    )
    target: str = Field(
        description=(
            "HTTP(S) base URL of the server (WebSocket URL is derived from it)."
        ),
    )
    model: str = Field(
        default_factory=str,
        description="Model identifier for generation requests.",
    )
    request_format: str = Field(
        default="/v1/realtime",
        description=(
            "Realtime WebSocket path (only /v1/realtime is supported today). "
            "Use the same top-level CLI flags as ``openai_http``: "
            "--request-format / --request-type."
        ),
    )
    chunk_samples: int = Field(
        default=3200,
        ge=1,
        description="PCM16 frames per input_audio_buffer.append chunk (16 kHz).",
    )
    api_key: SecretStr | None = Field(
        default=None, description="Bearer token if required."
    )
    api_key_file: Path | None = Field(
        default=None,
        description=(
            "Path to a file containing the bearer token; surrounding whitespace is "
            "stripped. Mutually exclusive with api_key."
        ),
    )
    verify: bool = Field(default=False, description="Verify TLS certificates.")
    timeout: float | None = Field(
        default=None,
        description="Per-message read timeout for WebSocket receives (seconds).",
    )
    timeout_connect: float = Field(
        default=FALLBACK_TIMEOUT,
        description="Timeout for establishing the WebSocket connection.",
    )
    validate_backend: bool | str | dict[str, Any] = Field(
        default=True,
        description=(
            "HTTP health check before benchmarks (same semantics as openai_http)."
        ),
    )
    extras: dict[str, Any] | None = Field(
        default=None,
        description="Extra fields merged into session.update (backend model wins).",
    )

    @field_validator("target", mode="after")
    @classmethod
    def strip_target(cls, value: str) -> str:
        """Strip trailing slashes and ``/v1`` suffix from the target URL."""
        return value.rstrip("/").removesuffix("/v1")

    @field_validator("request_format")
    @classmethod
    def validate_request_format(cls, v: str) -> str:
        """Validate ``request_format`` against allowed WebSocket paths."""
        stripped = v.strip()
        if stripped != "/v1/realtime":
            raise ValueError(f"request_format must be '/v1/realtime', got {stripped!r}")
        return stripped

    @model_validator(mode="after")
    def load_api_key_file(self):
        """
        Load api_key from api_key_file when the latter is set.

        api_key_file is cleared once consumed so dumped args re-validate cleanly
        without needing the file again.
        """
        self.api_key = resolve_api_key_file(self.api_key, self.api_key_file)
        self.api_key_file = None
        return self
