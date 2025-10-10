"""
Backend object models for request and response handling.

Provides standardized models for generation requests, responses, and timing
information to ensure consistent data handling across different backend
implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from guidellm.schemas.info import RequestInfo
from guidellm.schemas.request import GenerationRequest, UsageMetrics
from guidellm.utils import StandardBaseModel

if TYPE_CHECKING:
    from guidellm.schemas.stats import GenerativeRequestStats

__all__ = ["GenerationResponse"]


class GenerationResponse(StandardBaseModel):
    """Response model for backend generation operations."""

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    request_args: str | None = Field(
        description="Arguments passed to the backend for this request."
    )
    text: str | None = Field(
        default=None,
        description="The generated response text.",
    )
    input_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token statistics from the input.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token statistics from the generated output.",
    )

    def compile_stats(
        self,
        request: GenerationRequest,
        info: RequestInfo,
        prefer_response: bool = True,
    ) -> GenerativeRequestStats:
        """Compile and return request statistics.

        :param request: The original generation request.
        :param info: Metadata and timing information for the request.
        :return: A GenerativeRequestStats object containing detailed statistics.
        """
        if request.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between request and response.")

        if info.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between info and response.")

        if info.status != "completed":
            # clear out request output metrics if the request failed since those are not valid
            request.output_metrics = UsageMetrics()

        base_input = request.input_metrics if prefer_response else self.input_metrics
        override_input = (
            self.input_metrics if prefer_response else request.input_metrics
        )
        base_output = request.output_metrics if prefer_response else self.output_metrics
        override_output = (
            self.output_metrics if prefer_response else request.output_metrics
        )

        input_metrics_dict = base_input.model_dump()
        for key, value in override_input.model_dump().items():
            if value is not None:
                input_metrics_dict[key] = value
        output_metrics_dict = base_output.model_dump()
        for key, value in override_output.model_dump().items():
            if value is not None:
                output_metrics_dict[key] = value

        return GenerativeRequestStats(
            request_id=self.request_id,
            request_type=request.request_type,
            request_args=str(
                request.arguments.model_dump() if request.arguments else {}
            ),
            output=self.text,
            info=info,
            input_metrics=UsageMetrics(**input_metrics_dict),
            output_metrics=UsageMetrics(**output_metrics_dict),
        )
