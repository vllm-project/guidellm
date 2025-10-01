"""
Backend object models for request and response handling.

Provides standardized models for generation requests, responses, and timing
information to ensure consistent data handling across different backend
implementations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.data import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationRequestTimings,
)
from guidellm.scheduler import (
    SchedulerMessagingPydanticRegistry,
)
from guidellm.utils import StandardBaseModel

__all__ = [
    "GenerationRequest",
    "GenerationRequestArguments",
    "GenerationRequestTimings",
    "GenerationResponse",
    "GenerationTokenStats",
]


@SchedulerMessagingPydanticRegistry.register()
class GenerationTokenStats(StandardBaseModel):
    """Token statistics for generation requests and responses."""

    request: int | None = Field(
        default=None, description="Number of tokens in the original request."
    )
    response: int | None = Field(
        default=None, description="Number of tokens in the generated response."
    )

    def value(
        self, preference: Literal["request", "response"] | None = None
    ) -> int | None:
        if preference == "request":
            return self.request
        if preference == "response":
            return self.response
        return self.response if self.response is not None else self.request


@SchedulerMessagingPydanticRegistry.register()
class GenerationResponse(StandardBaseModel):
    """Response model for backend generation operations."""

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    request_args: GenerationRequestArguments = Field(
        description="Arguments passed to the backend for this request."
    )
    text: str | None = Field(
        default=None,
        description="The generated response text.",
    )
    iterations: int = Field(
        default=0, description="Number of generation iterations completed."
    )

    prompt_stats: GenerationTokenStats = Field(
        default_factory=GenerationTokenStats,
        description="Token statistics from the prompt.",
    )
    output_stats: GenerationTokenStats = Field(
        default_factory=GenerationTokenStats,
        description="Token statistics from the generated output.",
    )

    def total_tokens(
        self, preference: Literal["request", "response"] | None = None
    ) -> int | None:
        prompt_tokens = self.prompt_stats.value(preference=preference)
        output_tokens = self.output_stats.value(preference=preference)

        if prompt_tokens is None and output_tokens is None:
            return None
        return (prompt_tokens or 0) + (output_tokens or 0)
