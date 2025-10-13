"""
Request statistics and metrics for generative AI benchmark analysis.

Provides data structures for capturing and analyzing performance metrics from
generative AI workloads. Contains request-level statistics including token counts,
latency measurements, and throughput calculations for text generation benchmarks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from guidellm.schemas.info import RequestInfo
from guidellm.schemas.request import GenerativeRequestType, UsageMetrics
from guidellm.utils import StandardBaseDict

__all__ = ["GenerativeRequestStats"]


class GenerativeRequestStats(StandardBaseDict):
    """
    Request statistics for generative AI text generation workloads.

    Captures comprehensive performance metrics for individual generative requests,
    including token counts, timing measurements, and derived performance statistics.
    Provides computed properties for latency analysis, throughput calculations,
    and token generation metrics essential for benchmark evaluation.

    Example:
    ::
        stats = GenerativeRequestStats(
            request_id="req_123",
            request_type="text_completion",
            info=request_info,
            input_metrics=input_usage,
            output_metrics=output_usage
        )
        throughput = stats.output_tokens_per_second
    """

    type_: Literal["generative_request_stats"] = "generative_request_stats"
    request_id: str = Field(description="Unique identifier for the request")
    request_type: GenerativeRequestType | str = Field(
        description="Type of generative request: text or chat completion"
    )
    request_args: str | None = Field(
        default=None, description="Arguments passed to the backend for this request"
    )
    output: str | None = Field(
        description="Generated text output, if request completed successfully"
    )
    info: RequestInfo = Field(
        description="Metadata and timing information for the request"
    )
    input_metrics: UsageMetrics = Field(
        description="Usage statistics for the input prompt"
    )
    output_metrics: UsageMetrics = Field(
        description="Usage statistics for the generated output"
    )

    # Request stats
    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> float | None:
        """
        End-to-end request processing latency in seconds.

        :return: Duration from request start to completion, or None if unavailable.
        """
        if not self.info.timings.request_end or not self.info.timings.request_start:
            return None

        return self.info.timings.request_end - self.info.timings.request_start

    # General token stats
    @computed_field  # type: ignore[misc]
    @property
    def prompt_tokens(self) -> int | None:
        """
        Number of tokens in the input prompt.

        :return: Input prompt token count, or None if unavailable.
        """
        return self.input_metrics.text_tokens

    @computed_field  # type: ignore[misc]
    @property
    def input_tokens(self) -> int | None:
        """
        Number of tokens in the input prompt.

        :return: Input prompt token count, or None if unavailable.
        """
        return self.input_metrics.total_tokens

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens(self) -> int | None:
        """
        Number of tokens in the generated output.

        :return: Generated output token count, or None if unavailable.
        """
        return self.output_metrics.total_tokens

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int | None:
        """
        Total token count including prompt and output tokens.

        :return: Sum of prompt and output tokens, or None if either is unavailable.
        """
        input_tokens = self.input_metrics.total_tokens
        output_tokens = self.output_metrics.total_tokens

        if input_tokens is None and output_tokens is None:
            return None

        return (input_tokens or 0) + (output_tokens or 0)

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_ms(self) -> float | None:
        """
        Time to first token generation in milliseconds.

        :return: Latency from request start to first token, or None if unavailable.
        """
        if (
            not self.info.timings.first_iteration
            or not self.info.timings.request_start
            or self.info.timings.first_iteration == self.info.timings.last_iteration
        ):
            return None

        return 1000 * (
            self.info.timings.first_iteration - self.info.timings.request_start
        )

    @computed_field  # type: ignore[misc]
    @property
    def time_per_output_token_ms(self) -> float | None:
        """
        Average time per output token in milliseconds.

        Includes time for first token and all subsequent tokens.

        :return: Average milliseconds per output token, or None if unavailable.
        """
        if (
            not self.info.timings.request_start
            or not self.info.timings.last_iteration
            or not self.output_metrics.total_tokens
        ):
            return None

        return (
            1000
            * (self.info.timings.last_iteration - self.info.timings.request_start)
            / self.output_metrics.total_tokens
        )

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_ms(self) -> float | None:
        """
        Average inter-token latency in milliseconds.

        Measures time between token generations, excluding first token.

        :return: Average milliseconds between tokens, or None if unavailable.
        """
        if (
            not self.info.timings.first_iteration
            or not self.info.timings.last_iteration
            or not self.output_metrics.total_tokens
            or self.output_metrics.total_tokens <= 1
        ):
            return None

        return (
            1000
            * (self.info.timings.last_iteration - self.info.timings.first_iteration)
            / (self.output_metrics.total_tokens - 1)
        )

    @computed_field  # type: ignore[misc]
    @property
    def tokens_per_second(self) -> float | None:
        """
        Overall token throughput including prompt and output tokens.

        :return: Total tokens per second, or None if unavailable.
        """
        if not (latency := self.request_latency) or self.total_tokens is None:
            return None

        return self.total_tokens / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> float | None:
        """
        Output token generation throughput.

        :return: Output tokens per second, or None if unavailable.
        """
        if not (latency := self.request_latency) or self.output_tokens is None:
            return None

        return self.output_tokens / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_iteration(self) -> float | None:
        """
        Average output tokens generated per iteration.

        :return: Output tokens per iteration, or None if unavailable.
        """
        if self.output_tokens is None or not self.info.timings.iterations:
            return None

        return self.output_tokens / self.info.timings.iterations
