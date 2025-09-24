"""
Backend object models for request and response handling.

Provides standardized models for generation requests, responses, and timing
information to ensure consistent data handling across different backend
implementations.
"""

from typing import Any, Literal, Optional

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
]


@SchedulerMessagingPydanticRegistry.register()
class GenerationResponse(StandardBaseModel):
    """Response model for backend generation operations."""

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    request_args: dict[str, Any] = Field(
        description="Arguments passed to the backend for this request."
    )
    values: list[str] = Field(
        default_factory=list,
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

    @property
    def prompt_tokens(self) -> Optional[int]:
        """
        :return: The number of prompt tokens used in the request
            (response_prompt_tokens if available, otherwise request_prompt_tokens).
        """
        return self.response_prompt_tokens or self.request_prompt_tokens

    @property
    def output_tokens(self) -> Optional[int]:
        """
        :return: The number of output tokens generated in the response
            (response_output_tokens if available, otherwise request_output_tokens).
        """
        return self.response_output_tokens or self.request_output_tokens

    @property
    def total_tokens(self) -> Optional[int]:
        """
        :return: The total number of tokens used in the request and response.
            Sum of prompt_tokens and output_tokens.
        """
        if self.prompt_tokens is None or self.output_tokens is None:
            return None
        return self.prompt_tokens + self.output_tokens

    def preferred_prompt_tokens(
        self, preferred_source: Literal["request", "response"]
    ) -> Optional[int]:
        if preferred_source == "request":
            return self.request_prompt_tokens or self.response_prompt_tokens
        else:
            return self.response_prompt_tokens or self.request_prompt_tokens

    def preferred_output_tokens(
        self, preferred_source: Literal["request", "response"]
    ) -> Optional[int]:
        if preferred_source == "request":
            return self.request_output_tokens or self.response_output_tokens
        else:
            return self.response_output_tokens or self.request_output_tokens
