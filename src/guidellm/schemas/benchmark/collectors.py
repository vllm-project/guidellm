"""
Collector arguments and the Prometheus collector's per-benchmark summary.

Collectors gather data from outside GuideLLM, such as a model server's
Prometheus endpoint, while each benchmark runs. Every collector reports one
result per benchmark, so server-side signals can be read alongside the
client-side results.
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
    "CollectorArgs",
    "PrometheusCollectorArgs",
    "PrometheusCounterSeries",
    "PrometheusGaugeSeries",
    "PrometheusHistogramSeries",
    "PrometheusSummary",
    "VLLMServerInfoCollectorArgs",
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


class CollectorArgs(PydanticClassRegistryMixin["CollectorArgs"], ABC):
    """Base class for collector arguments.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[CollectorArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base CollectorArgs class for schema validation
        """
        if cls.__name__ == "CollectorArgs":
            return cls

        return CollectorArgs

    kind: str = Field(
        description="The kind of collector to run.",
    )


@CollectorArgs.register("prometheus")
class PrometheusCollectorArgs(CollectorArgs):
    """Scrape a Prometheus text-format endpoint, such as vLLM's ``/metrics``.

    Each collector takes its own URL rather than the backend's target, since the
    backend may point at a gateway or load balancer. Configure one collector per
    server to scrape several servers.
    """

    kind: Literal["prometheus"] = Field(
        default="prometheus",
        description="The kind of collector to run.",
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
                f"Prometheus collector url must be an http or https URL, got {url!r}"
            )
        return url


@CollectorArgs.register("vllm_server_info")
class VLLMServerInfoCollectorArgs(CollectorArgs):
    """Capture structured vLLM configuration after each benchmark."""

    kind: Literal["vllm_server_info"] = "vllm_server_info"
    url: str = Field(
        description="Explicit URL of the vLLM server-info endpoint.",
        examples=["http://localhost:8000/server_info"],
    )
    redact_hosts: bool = Field(
        default=True,
        description=(
            "Redact remote host fields and URL authorities in collected data. "
            "Credential filtering remains enabled regardless of this setting."
        ),
    )

    @field_validator("url")
    @classmethod
    def _validate_url(cls, url: str) -> str:
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Server-info url must be HTTP(S) without credentials, query "
                "parameters or a fragment"
            )
        # Accessing port also rejects malformed ports before starting a process.
        _ = parsed.port
        return url


class PrometheusGaugeSeries(StandardBaseModel):
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


class PrometheusCounterSeries(StandardBaseModel):
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


class PrometheusHistogramSeries(StandardBaseModel):
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


class PrometheusSummary(StandardBaseModel):
    """Metrics scraped from one Prometheus endpoint during one benchmark."""

    source: str = Field(
        description="The scraped endpoint, such as a metrics URL.",
    )
    scrapes: int = Field(
        description="Successful scrapes during the benchmark run.",
    )
    scrape_errors: int = Field(
        description="Failed scrapes during the benchmark run.",
    )
    gauges: dict[str, list[PrometheusGaugeSeries]] = Field(
        default_factory=dict,
        description="Gauge series by metric family name.",
    )
    counters: dict[str, list[PrometheusCounterSeries]] = Field(
        default_factory=dict,
        description="Counter series by metric family name.",
    )
    histograms: dict[str, list[PrometheusHistogramSeries]] = Field(
        default_factory=dict,
        description="Histogram series by metric family name.",
    )
