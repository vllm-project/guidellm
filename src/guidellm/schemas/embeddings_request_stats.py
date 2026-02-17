"""
Request statistics for embeddings benchmark analysis.

Provides data structures for capturing and analyzing performance metrics from
embeddings workloads. The module contains request-level statistics including
input token counts, latency measurements, and optional quality validation metrics
such as cosine similarity for evaluating embeddings benchmark performance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from guidellm.schemas.base import StandardBaseDict
from guidellm.schemas.info import RequestInfo
from guidellm.schemas.request import UsageMetrics

__all__ = ["EmbeddingsRequestStats"]


class EmbeddingsRequestStats(StandardBaseDict):
    """
    Request statistics for embeddings workloads.

    Captures comprehensive performance metrics for individual embeddings requests,
    including input token counts, timing measurements, and optional quality validation
    metrics. Unlike generative requests, embeddings do not produce output tokens
    or have streaming behavior.

    Example:
    ::
        stats = EmbeddingsRequestStats(
            request_id="req_123",
            info=request_info,
            input_metrics=input_usage
        )
        latency = stats.request_latency
    """

    type_: Literal["embeddings_request_stats"] = "embeddings_request_stats"
    request_id: str = Field(description="Unique identifier for the request")
    response_id: str | None = Field(
        default=None, description="Unique identifier matching API Response ID"
    )
    request_args: str | None = Field(
        default=None, description="Backend arguments used for this request"
    )
    info: RequestInfo = Field(description="Request metadata and timing information")
    input_metrics: UsageMetrics = Field(
        description="Token usage statistics for the input text"
    )

    # Quality validation metrics (optional)
    cosine_similarity: float | None = Field(
        default=None,
        description="Cosine similarity score against baseline model (0.0-1.0)",
    )
    encoding_format: str | None = Field(
        default="float",
        description="Encoding format used for embeddings (float or base64)",
    )

    # Request timing stats
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

    # Input token stats (no output tokens for embeddings)
    @computed_field  # type: ignore[misc]
    @property
    def prompt_tokens(self) -> int | None:
        """
        :return: Number of tokens in the input text, or None if unavailable
        """
        return self.input_metrics.total_tokens

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int | None:
        """
        :return: Same as prompt_tokens (embeddings have no output tokens)
        """
        return self.prompt_tokens

    @computed_field  # type: ignore[misc]
    @property
    def input_tokens_timing(self) -> tuple[float, float]:
        """
        Timing tuple for input token processing.

        :return: Tuple of (timestamp, token_count) for input processing
        """
        return (
            self.request_end_time,
            self.prompt_tokens or 0.0,
        )
