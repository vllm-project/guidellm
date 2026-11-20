"""Configuration schema for the example ``echo_plugin`` backend."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.backends import BackendArgs

__all__ = ["EchoPluginBackendArgs"]


@BackendArgs.register("echo_plugin")
class EchoPluginBackendArgs(BackendArgs):
    """Configuration schema for the example ``echo_plugin`` backend."""

    kind: Literal["echo_plugin"] = Field(  # type: ignore[assignment]
        default="echo_plugin",
        description="Type identifier for the example plugin backend.",
    )
    target: str = Field(
        description="Base URL of an OpenAI-compatible HTTP server.",
    )
    model: str = Field(
        default="",
        description="Optional model name; defaults to first server model.",
    )
    timeout: float = Field(
        default=10.0,
        description="HTTP client timeout in seconds.",
    )
