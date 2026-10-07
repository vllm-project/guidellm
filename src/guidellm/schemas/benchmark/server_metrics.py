"""
Server metrics collection arguments and per-benchmark summaries.

Server metrics are scraped from the system under test while each benchmark runs
and summarized over the benchmark's measurement window, so server-side signals
such as queue depth, cache usage, and server-measured latencies can be read
alongside the client-side results.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator

from guidellm.schemas import (
    DistributionSummary,
    PydanticClassRegistryMixin,
    StandardBaseModel,
    standard_model_config,
)

__all__ = [
    "DEFAULT_VLLM_METRICS",
    "PrometheusServerMetricsArgs",
    "ServerCounterSeries",
    "ServerGaugeSeries",
    "ServerHistogramSeries",
    "ServerMetricsArgs",
    "ServerMetricsSummary",
]

# Older vLLM releases expose gpu_cache_usage_perc and time_per_output_token_seconds
# under the names that newer releases replaced, so both are listed.
DEFAULT_VLLM_METRICS: list[str] = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:num_preemptions",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
]


class ServerMetricsArgs(PydanticClassRegistryMixin["ServerMetricsArgs"], ABC):
    """Base class for server metrics source arguments.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[ServerMetricsArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base ServerMetricsArgs class for schema validation
        """
        if cls.__name__ == "ServerMetricsArgs":
            return cls

        return ServerMetricsArgs

    kind: str = Field(
        description="The kind of server metrics source to scrape.",
    )


@ServerMetricsArgs.register("prometheus")
class PrometheusServerMetricsArgs(ServerMetricsArgs):
    """Scrape a Prometheus text-format endpoint, such as vLLM's ``/metrics``."""

    kind: Literal["prometheus"] = Field(
        default="prometheus",
        description="The kind of server metrics source to scrape.",
    )
    url: str = Field(
        description="URL of the Prometheus metrics endpoint to scrape.",
        examples=["http://localhost:8000/metrics"],
    )
    interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds between scrapes while a benchmark runs.",
    )
    timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="Seconds to wait for a single scrape before counting it failed.",
    )
    metrics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_VLLM_METRICS),
        description=(
            "Metric family names to collect. Families the endpoint does not "
            "expose are skipped. Defaults to vLLM's queue, cache, token, outcome "
            "and latency metrics, including names used by older vLLM releases."
        ),
    )

    @field_validator("url")
    @classmethod
    def _validate_url(cls, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"Server metrics url must be an http or https URL, got {url!r}"
            )
        return url


class ServerGaugeSeries(StandardBaseModel):
    """One gauge series sampled over a benchmark's measurement window."""

    labels: dict[str, str] = Field(
        description="Labels identifying the series, such as model_name or engine.",
    )
    summary: DistributionSummary = Field(
        description="Distribution of the values scraped within the window.",
    )
    samples: list[tuple[float, float]] = Field(
        description="Scraped (timestamp, value) pairs within the window.",
    )


class ServerCounterSeries(StandardBaseModel):
    """One counter series over a benchmark's measurement window."""

    labels: dict[str, str] = Field(
        description="Labels identifying the series, such as model_name or engine.",
    )
    increase: float = Field(
        description=(
            "Increase over the window, interpolated at its edges and corrected "
            "for counter resets."
        ),
    )
    rate: float = Field(
        description="Average increase per second over the window.",
    )


class ServerHistogramSeries(StandardBaseModel):
    """One histogram series over a benchmark's measurement window."""

    labels: dict[str, str] = Field(
        description="Labels identifying the series, excluding the bucket label.",
    )
    count: float = Field(
        description="Observations recorded within the window.",
    )
    sum: float = Field(
        description="Sum of the observations recorded within the window.",
    )
    mean: float | None = Field(
        description="Mean observation within the window, or None without any.",
    )
    quantiles: dict[str, float] = Field(
        description=(
            "Quantile estimates within the window, keyed p50, p90, p95 and p99. "
            "They are interpolated within histogram buckets, so their precision "
            "is limited by the server's bucket boundaries."
        ),
    )


class ServerMetricsSummary(StandardBaseModel):
    """Server metrics scraped from one source during one benchmark."""

    source: str = Field(
        description="The scraped endpoint, such as a metrics URL.",
    )
    scrapes: int = Field(
        description="Successful scrapes during the benchmark run.",
    )
    scrape_errors: int = Field(
        description="Failed scrapes during the benchmark run.",
    )
    gauges: dict[str, list[ServerGaugeSeries]] = Field(
        default_factory=dict,
        description="Gauge series by metric family name.",
    )
    counters: dict[str, list[ServerCounterSeries]] = Field(
        default_factory=dict,
        description="Counter series by metric family name.",
    )
    histograms: dict[str, list[ServerHistogramSeries]] = Field(
        default_factory=dict,
        description="Histogram series by metric family name.",
    )
