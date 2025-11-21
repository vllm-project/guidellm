"""
Backend response models for request and response handling.

Provides standardized response models for generation operations that capture
output text, usage metrics, and compilation of request statistics. Ensures
consistent data handling and statistics aggregation across different backend
implementations.
"""

from __future__ import annotations

from pydantic import Field

from guidellm.schemas.base import StandardBaseModel
from guidellm.schemas.info import RequestInfo
from guidellm.schemas.request import GenerationRequest, UsageMetrics
from guidellm.schemas.request_stats import GenerativeRequestStats

__all__ = ["GenerationResponse"]


class GenerationResponse(StandardBaseModel):
    """
    Response model for backend generation operations.

    Captures the output and metrics from a generation request, providing structured
    data for text output, token usage statistics, and compilation of detailed
    request statistics for analysis and monitoring purposes.

    Example:
    ::
        response = GenerationResponse(
            request_id="req-123",
            text="Generated response text",
            input_metrics=UsageMetrics(token_count=50),
            output_metrics=UsageMetrics(token_count=25)
        )
        stats = response.compile_stats(request, info)
    """

    request_id: str = Field(
        description="Unique identifier matching the original GenerationRequest."
    )
    response_id: str | None = Field(
        default=None,
        description="Unique identifier matching the original vLLM Response ID.",
    )
    request_args: str | None = Field(
        description="Arguments passed to the backend for request processing."
    )
    text: str | None = Field(
        default=None,
        description="The generated response text.",
    )
    input_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token usage statistics from the input prompt.",
    )
    output_metrics: UsageMetrics = Field(
        default_factory=UsageMetrics,
        description="Token usage statistics from the generated output.",
    )

    def compile_stats(
        self,
        request: GenerationRequest,
        info: RequestInfo,
        prefer_response: bool = True,
    ) -> GenerativeRequestStats:
        """
        Compile and return comprehensive request statistics.

        Merges metrics from the request and response objects to create a complete
        statistical record, with preference given to response-level metrics when
        available to ensure accuracy of actual execution data.

        :param request: The original generation request containing input data
        :param info: Metadata and timing information for the request execution
        :param prefer_response: Whether to prefer response metrics over request
            metrics when both are available
        :return: A GenerativeRequestStats object containing detailed statistics
        :raises ValueError: When request IDs don't match between objects
        """
        if request.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between request and response.")

        if info.request_id != self.request_id:
            raise ValueError("Mismatched request IDs between info and response.")

        if info.status != "completed":
            # clear out request output metrics if the request failed since
            # those are not valid
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
            response_id=self.response_id,
            request_type=request.request_type,
            request_args=str(
                request.arguments.model_dump() if request.arguments else {}
            ),
            output=self.text,
            info=info,
            input_metrics=UsageMetrics(**input_metrics_dict),
            output_metrics=UsageMetrics(**output_metrics_dict),
        )
