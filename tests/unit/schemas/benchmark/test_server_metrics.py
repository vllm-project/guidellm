"""Unit tests for server metrics source arguments."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.schemas.benchmark import (
    DEFAULT_VLLM_METRICS,
    BenchmarkArgs,
    PrometheusServerMetricsArgs,
    ServerMetricsArgs,
)

BASE_ARGS = {
    "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
    "data": [{"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}],
    "profile": {"kind": "synchronous"},
}


class TestPrometheusServerMetricsArgs:
    @pytest.mark.smoke
    def test_defaults(self):
        """Default to one-second scrapes of vLLM's metrics, old and new names.

        ## WRITTEN BY AI ##
        """
        args = ServerMetricsArgs.model_validate(
            {"kind": "prometheus", "url": "http://localhost:8000/metrics"}
        )

        assert isinstance(args, PrometheusServerMetricsArgs)
        assert args.interval == 1.0
        assert args.timeout == 5.0
        assert args.metrics == DEFAULT_VLLM_METRICS
        assert "vllm:kv_cache_usage_perc" in args.metrics
        assert "vllm:gpu_cache_usage_perc" in args.metrics

    @pytest.mark.sanity
    @pytest.mark.parametrize("field", ["interval", "timeout"])
    def test_rejects_non_positive_durations(self, field: str):
        """Reject a zero scrape interval or timeout.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            PrometheusServerMetricsArgs(
                url="http://localhost:8000/metrics", **{field: 0}
            )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "url", ["localhost:8000/metrics", "ftp://server/metrics", "http://"]
    )
    def test_rejects_non_http_urls(self, url: str):
        """Reject URLs that could never be scraped over HTTP.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="http or https"):
            PrometheusServerMetricsArgs(url=url)

    @pytest.mark.sanity
    def test_requires_url(self):
        """Reject a Prometheus source without a URL.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            ServerMetricsArgs.model_validate({"kind": "prometheus"})


class TestBenchmarkArgsServerMetrics:
    @pytest.mark.smoke
    def test_defaults_to_no_sources(self):
        """Scrape nothing unless a source is configured.

        ## WRITTEN BY AI ##
        """
        assert BenchmarkArgs.model_validate(BASE_ARGS).server_metrics == []

    @pytest.mark.smoke
    def test_accepts_multiple_sources(self):
        """Accept several sources, such as one per server replica.

        ## WRITTEN BY AI ##
        """
        args = BenchmarkArgs.model_validate(
            {
                **BASE_ARGS,
                "server_metrics": [
                    {"kind": "prometheus", "url": "http://replica-0:8000/metrics"},
                    {
                        "kind": "prometheus",
                        "url": "http://replica-1:8000/metrics",
                        "interval": 5.0,
                        "metrics": ["vllm:num_requests_running"],
                    },
                ],
            }
        )

        assert [source.url for source in args.server_metrics] == [
            "http://replica-0:8000/metrics",
            "http://replica-1:8000/metrics",
        ]
        assert args.server_metrics[1].interval == 5.0
        assert args.server_metrics[1].metrics == ["vllm:num_requests_running"]
