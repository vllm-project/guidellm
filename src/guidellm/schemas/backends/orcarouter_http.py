"""
OrcaRouter HTTP backend Args schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr

from guidellm.schemas.backends.backend import BackendArgs
from guidellm.schemas.backends.openai_http import OpenAIHTTPBackendArgs

__all__ = ["OrcaRouterHTTPBackendArgs"]


@BackendArgs.register("orcarouter_http")
class OrcaRouterHTTPBackendArgs(OpenAIHTTPBackendArgs):
    """
    Pydantic model for OrcaRouter HTTP backend creation arguments.

    OrcaRouter is an OpenAI-compatible AI gateway, so this schema extends the
    ``openai_http`` backend arguments and specializes the default target and
    model identifiers for the OrcaRouter API.
    """

    kind: Literal["orcarouter_http"] = Field(  # type: ignore[assignment]
        default="orcarouter_http",
        description="Type identifier for the backend configuration.",
    )
    target: str = Field(
        default="https://api.orcarouter.ai",
        description="Base URL of the OrcaRouter API",
        examples=["https://api.orcarouter.ai"],
    )
    model: str = Field(
        default_factory=str,
        description="OrcaRouter model identifier, e.g. an 'orcarouter/*' or "
        "provider-prefixed model",
        examples=["orcarouter/auto", "anthropic/claude-sonnet-5"],
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="HTTP Bearer token API key for authentication to OrcaRouter",
        examples=["sk-orca-..."],
    )
