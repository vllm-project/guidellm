"""Unit tests for the mock server's Prometheus metrics."""

from __future__ import annotations

import math

import pytest

from guidellm.benchmark.server_metrics import parse_prometheus_text
from guidellm.mock_server.metrics import MockServerMetrics

FAMILIES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:request_success",
    "vllm:e2e_request_latency_seconds",
]
LABELS = (("model_name", "test-model"),)


@pytest.mark.smoke
def test_render_tracks_requests_in_vllm_format():
    """Report running, waiting and finished requests under vLLM's metric names.

    ## WRITTEN BY AI ##
    """
    metrics = MockServerMetrics("test-model")
    metrics.request_queued()
    metrics.request_dequeued()
    finished = metrics.request_started()
    metrics.request_started()
    metrics.request_finished(finished, success=True)
    failed = metrics.request_started()
    metrics.request_finished(failed, success=False)

    snapshot = parse_prometheus_text(metrics.render(), FAMILIES, 0.0)

    assert snapshot.gauges[("vllm:num_requests_running", LABELS)] == 1.0
    assert snapshot.gauges[("vllm:num_requests_waiting", LABELS)] == 0.0
    success_labels = (("finished_reason", "stop"), ("model_name", "test-model"))
    assert snapshot.counters[("vllm:request_success", success_labels)] == 1.0
    latency = snapshot.histograms[("vllm:e2e_request_latency_seconds", LABELS)]
    assert latency.count == 1.0
    assert latency.buckets[math.inf] == 1.0


@pytest.mark.sanity
def test_queued_requests_are_reported_as_waiting():
    """Report requests waiting for a concurrency slot.

    ## WRITTEN BY AI ##
    """
    metrics = MockServerMetrics("test-model")
    metrics.request_queued()
    metrics.request_queued()

    snapshot = parse_prometheus_text(metrics.render(), FAMILIES, 0.0)

    assert snapshot.gauges[("vllm:num_requests_waiting", LABELS)] == 2.0
