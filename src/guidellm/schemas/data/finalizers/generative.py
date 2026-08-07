from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.data.entrypoints import DataFinalizerArgs


@DataFinalizerArgs.register("generative")
class GenerativeRequestFinalizerArgs(DataFinalizerArgs):
    """Model for generative request finalizer arguments."""

    kind: Literal["generative"] = Field(
        default="generative",
        description="Type identifier for the generative request finalizer.",
    )
    tool_call_mode: Literal["client", "server"] = Field(
        default="client",
        description="How to handle turns with tool definitions. "
        "'client' (default) creates client_tool_call + injection turns. "
        "'server' creates server_tool_call turns (no injection, "
        "tools are server-managed).",
    )
