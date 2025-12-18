"""
Request statistics and metrics for generative AI benchmark analysis.

Provides data structures for capturing and analyzing performance metrics from
generative AI workloads. The module contains request-level statistics including
token counts, latency measurements, and throughput calculations essential for
evaluating text generation benchmark performance. Computed properties enable
analysis of time-to-first-token, inter-token latency, and token generation rates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, computed_field

from guidellm.schemas.base import StandardBaseDict
from guidellm.schemas.info import RequestInfo
from guidellm.schemas.request import GenerativeRequestType, UsageMetrics

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
        description="Type of generative request (text_completion or chat_completion)"
    )
    response_id: str | None = Field(
        default=None, description="Unique identifier matching vLLM Response ID"
    )
    request_args: str | None = Field(
        default=None, description="Backend arguments used for this request"
    )
    output: str | None = Field(
        default=None, description="Generated text output from the request"
    )
    info: RequestInfo = Field(description="Request metadata and timing information")
    input_metrics: UsageMetrics = Field(
        description="Token usage statistics for the input prompt"
    )
    output_metrics: UsageMetrics = Field(
        description="Token usage statistics for the generated output"
    )

    # Request stats
    @computed_field  # type: ignore[misc]
    @property
    def request_start_time(self) -> float | None:
        """
        :return: Timestamp when the request started, or None if unavailable
        """
        return (
            self.info.timings.request_start
            if self.info.timings.request_start is not None
            else self.info.timings.resolve_start
        )

    @computed_field  # type: ignore[misc]
    @property
    def request_end_time(self) -> float:
        """
        :return: Timestamp when the request ended, or None if unavailable
        """
        if self.info.timings.resolve_end is None:
            raise ValueError("resolve_end timings should be set but is None.")

        return (
            self.info.timings.request_end
            if self.info.timings.request_end is not None
            else self.info.timings.resolve_end
        )

    @computed_field  # type: ignore[misc]
    @property
    def request_latency(self) -> float | None:
        """
        End-to-end request processing latency in seconds.

        :return: Duration from request start to completion, or None if unavailable
        """
        start = self.info.timings.request_start
        end = self.info.timings.request_end
        if start is None or end is None:
            return None

        return end - start

    # General token stats
    @computed_field  # type: ignore[misc]
    @property
    def prompt_tokens(self) -> int | None:
        """
        :return: Number of tokens in the input prompt, or None if unavailable
        """
        return self.input_metrics.total_tokens

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens(self) -> int | None:
        """
        :return: Number of tokens in the generated output, or None if unavailable
        """
        # Fallback if we did not get usage metrics from the server
        # NOTE: This assumes each iteration is one token
        if self.output_metrics.total_tokens is None:
            return self.info.timings.token_iterations or None

        return self.output_metrics.total_tokens

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int | None:
        """
        :return: Sum of prompt and output tokens, or None if both unavailable
        """
        input_tokens = self.prompt_tokens
        output_tokens = self.output_tokens

        if input_tokens is None and output_tokens is None:
            return None

        return (input_tokens or 0) + (output_tokens or 0)

    @computed_field  # type: ignore[misc]
    @property
    def time_to_first_token_ms(self) -> float | None:
        """
        :return: Time to first token generation in milliseconds, or None if unavailable
        """
        first_token = self.first_token_iteration
        start = self.info.timings.request_start
        if first_token is None or start is None:
            return None

        return 1000 * (first_token - start)

    @computed_field  # type: ignore[misc]
    @property
    def time_per_output_token_ms(self) -> float | None:
        """
        Average time per output token in milliseconds including first token.

        :return: Average milliseconds per output token, or None if unavailable
        """
        if (
            (start := self.info.timings.request_start) is None
            or (
                (last_token := self.last_token_iteration or self.request_end_time)
                is None
            )
            or (output_tokens := self.output_tokens) is None
            or output_tokens == 0
        ):
            return None

        return 1000 * (last_token - start) / output_tokens

    @computed_field  # type: ignore[misc]
    @property
    def inter_token_latency_ms(self) -> float | None:
        """
        Average inter-token latency in milliseconds excluding first token.

        :return: Average milliseconds between token generations, or None if unavailable
        """
        first_token = self.first_token_iteration
        last_token = self.last_token_iteration
        output_tokens = self.output_tokens
        if (
            first_token is None
            or last_token is None
            or output_tokens is None
            or output_tokens <= 1
        ):
            return None

        return 1000 * (last_token - first_token) / (output_tokens - 1)

    @computed_field  # type: ignore[misc]
    @property
    def tokens_per_second(self) -> float | None:
        """
        :return: Total tokens per second throughput, or None if unavailable
        """
        if not (latency := self.request_latency) or self.total_tokens is None:
            return None

        return self.total_tokens / latency

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_second(self) -> float | None:
        """
        :return: Output token generation throughput, or None if unavailable
        """
        if not (latency := self.request_latency) or self.output_tokens is None:
            return None

        return self.output_tokens / latency

    @computed_field  # type: ignore[misc]
    @property
    def iter_tokens_per_iteration(self) -> float | None:
        """
        :return: Average tokens per iteration excluding first token, or None if
            unavailable
        """
        if (
            self.output_tokens is None
            or self.output_tokens <= 1
            or self.token_iterations <= 1
        ):
            return None

        return (self.output_tokens - 1.0) / (
            self.token_iterations - 1.0
        )  # subtract 1 for first token from the prompt, assume first iter is 1 token

    @computed_field  # type: ignore[misc]
    @property
    def output_tokens_per_iteration(self) -> float | None:
        """
        :return: Average output tokens per iteration, or None if unavailable
        """
        if self.output_tokens is None or self.token_iterations < 1:
            return None

        return self.output_tokens / self.token_iterations

    @property
    def first_token_iteration(self) -> float | None:
        """
        :return: Timestamp of first token generation, or None if unavailable
        """
        return self.info.timings.first_token_iteration

    @property
    def last_token_iteration(self) -> float | None:
        """
        :return: Timestamp of last token generation, or None if unavailable
        """
        return self.info.timings.last_token_iteration

    @property
    def token_iterations(self) -> int:
        """
        :return: Total number of token generation iterations
        """
        return self.info.timings.token_iterations

    @property
    def prompt_tokens_timing(self) -> tuple[float, float]:
        """
        :return: Tuple of (timestamp, token_count) for prompt processing
        :raises ValueError: If resolve_end timings are not set
        """
        return (
            (
                self.first_token_iteration
                if self.first_token_iteration is not None
                else self.request_end_time
            ),
            self.prompt_tokens or 0.0,
        )

    @property
    def output_tokens_timings(self) -> list[tuple[float, float]]:
        """
        :return: List of (timestamp, token_count) tuples for output token generations
        :raises ValueError: If resolve_end timings are not set
        """
        if (
            self.first_token_iteration is None
            or self.last_token_iteration is None
            or self.token_iterations <= 1
        ):
            # No iteration data, return single timing at end with all tokens
            return [
                (
                    (
                        self.last_token_iteration
                        if self.last_token_iteration is not None
                        else self.request_end_time
                    ),
                    self.output_tokens or 0.0,
                )
            ]

        # Return first token timing as 1 token plus per-iteration timings
        return [
            (self.first_token_iteration, 1.0 * bool(self.output_tokens))
        ] + self.iter_tokens_timings

    @property
    def iter_tokens_timings(self) -> list[tuple[float, float]]:
        """
        :return: List of (timestamp, token_count) tuples for iterations excluding
            first token
        """
        if (
            self.first_token_iteration is None
            or self.last_token_iteration is None
            or (tok_per_iter := self.iter_tokens_per_iteration) is None
            or self.token_iterations <= 1
        ):
            return []

        # evenly space the iterations since we don't have per-iteration timings
        # / we don't know the individual token counts per iteration
        iter_times = np.linspace(
            self.first_token_iteration,
            self.last_token_iteration,
            num=self.token_iterations,
        )[1:]  # skip first iteration

        return [(iter_time, tok_per_iter) for iter_time in iter_times]

    @property
    def total_tokens_timings(self) -> list[tuple[float, float]]:
        """
        :return: List of (timestamp, token_count) tuples for all token generations
        """
        prompt_timings = self.prompt_tokens_timing
        output_timings = self.output_tokens_timings

        return ([prompt_timings] if prompt_timings else []) + output_timings
