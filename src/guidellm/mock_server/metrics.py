"""
Prometheus metrics for the mock server.

Exposes a small vLLM-compatible subset of metrics so server metrics collection
can be exercised without a real model server: in-flight and queued generation
requests, completed requests, and end-to-end request latency.
"""

from __future__ import annotations

import math
import time

__all__ = ["MockServerMetrics"]

_LATENCY_BUCKETS: tuple[float, ...] = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    math.inf,
)


class MockServerMetrics:
    """
    Track generation requests and render them in Prometheus text format.

    Metric names follow vLLM's, so the same collector configuration works
    against the mock server and a real vLLM server.

    :param model: Model name reported in the ``model_name`` label
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.running = 0
        self.waiting = 0
        self.successes = 0
        self.latency_buckets = [0] * len(_LATENCY_BUCKETS)
        self.latency_sum = 0.0
        self.latency_count = 0

    def request_queued(self) -> None:
        """Record a request waiting for a concurrency slot."""
        self.waiting += 1

    def request_dequeued(self) -> None:
        """Record a queued request acquiring a concurrency slot."""
        self.waiting -= 1

    def request_started(self) -> float:
        """
        Record a request starting generation.

        :return: Start time to pass to :meth:`request_finished`
        """
        self.running += 1
        return time.monotonic()

    def request_finished(self, started_at: float, success: bool) -> None:
        """
        Record a request finishing, including the end of a streamed response.

        :param started_at: Start time returned by :meth:`request_started`
        :param success: Whether the request completed without an error
        """
        self.running -= 1
        if not success:
            return
        latency = time.monotonic() - started_at
        self.successes += 1
        self.latency_sum += latency
        self.latency_count += 1
        for index, bound in enumerate(_LATENCY_BUCKETS):
            if latency <= bound:
                self.latency_buckets[index] += 1

    def render(self) -> str:
        """
        Render the metrics in Prometheus text exposition format.

        :return: The exposition text
        """
        labels = f'model_name="{self.model}"'
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels}}} {self.running}",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{{{labels}}} {self.waiting}",
            "# TYPE vllm:request_success_total counter",
            f'vllm:request_success_total{{finished_reason="stop",{labels}}} '
            f"{self.successes}",
            "# TYPE vllm:e2e_request_latency_seconds histogram",
        ]
        for bound, count in zip(_LATENCY_BUCKETS, self.latency_buckets, strict=True):
            bound_text = "+Inf" if math.isinf(bound) else repr(bound)
            lines.append(
                f'vllm:e2e_request_latency_seconds_bucket{{le="{bound_text}",'
                f"{labels}}} {count}"
            )
        lines.append(
            f"vllm:e2e_request_latency_seconds_count{{{labels}}} {self.latency_count}"
        )
        lines.append(
            f"vllm:e2e_request_latency_seconds_sum{{{labels}}} {self.latency_sum}"
        )

        return "\n".join(lines) + "\n"
