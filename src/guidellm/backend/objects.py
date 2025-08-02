"""
Backend object models for request and response handling.

Provides standardized models for generation requests, responses, and timing
information to ensure consistent data handling across different backend
implementations.
"""

import uuid
from typing import Any, Literal, Optional

from pydantic import Field

from guidellm.objects.pydantic import StandardBaseModel
from guidellm.scheduler import RequestTimings

__all__ = [
    "GenerationRequest",
    "GenerationRequestTimings",
    "GenerationResponse",
]


class GenerationRequest(StandardBaseModel):
    """Request model for backend generation operations."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the request.",
    )
    request_type: Literal["text_completions", "chat_completions"] = Field(
        default="text_completions",
        description=(
            "Type of request. 'text_completions' uses backend.text_completions(), "
            "'chat_completions' uses backend.chat_completions()."
        ),
    )
    content: Any = Field(
        description=(
            "Request content. For text_completions: string or list of strings. "
            "For chat_completions: string, list of messages, or raw content "
            "(set raw_content=True in params)."
        )
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters passed to backend methods. "
            "Common: max_tokens, temperature, stream."
        ),
    )
    stats: dict[Literal["prompt_tokens"], int] = Field(
        default_factory=dict,
        description="Request statistics including prompt token count.",
    )
    constraints: dict[Literal["output_tokens"], int] = Field(
        default_factory=dict,
        description="Request constraints such as maximum output tokens.",
    )


class GenerationResponse(StandardBaseModel):
    """Response model for backend generation operations."""

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    request_args: dict[str, Any] = Field(
        description="Arguments passed to the backend for this request."
    )
    value: Optional[str] = Field(
        default=None,
        description="Complete generated text content. None for streaming responses.",
    )
    delta: Optional[str] = Field(
        default=None, description="Incremental text content for streaming responses."
    )
    iterations: int = Field(
        default=0, description="Number of generation iterations completed."
    )
    request_prompt_tokens: Optional[int] = Field(
        default=None, description="Token count from the original request prompt."
    )
    request_output_tokens: Optional[int] = Field(
        default=None,
        description="Expected output token count from the original request.",
    )
    response_prompt_tokens: Optional[int] = Field(
        default=None, description="Actual prompt token count reported by the backend."
    )
    response_output_tokens: Optional[int] = Field(
        default=None, description="Actual output token count reported by the backend."
    )


class GenerationRequestTimings(RequestTimings):
    """Timing model for tracking generation request lifecycle events."""

    first_iteration: Optional[float] = Field(
        default=None,
        description="Unix timestamp when the first generation iteration began.",
    )
    last_iteration: Optional[float] = Field(
        default=None,
        description="Unix timestamp when the last generation iteration completed.",
    )
