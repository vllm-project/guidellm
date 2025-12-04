"""
Generative AI benchmark schemas for performance measurement and analysis.

This module provides the complete schema ecosystem for executing, tracking, and
analyzing generative AI benchmarks. It encompasses configuration entrypoints for
benchmark setup, real-time metric accumulators for execution monitoring,
comprehensive result containers with statistical summaries, and multi-benchmark
reporting capabilities. The schemas support domain-specific metrics for text,
image, video, and audio generation tasks, enabling detailed performance analysis
including throughput, latency distributions, concurrency patterns, and scheduler
behavior tracking across successful, incomplete, and errored requests.
"""

from __future__ import annotations

from .accumulator import (
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarkTimings,
    GenerativeMetricsAccumulator,
    GenerativeRequestsAccumulator,
    RunningMetricStats,
    SchedulerMetricsAccumulator,
)
from .benchmark import GenerativeBenchmark
from .entrypoints import BenchmarkGenerativeTextArgs
from .metrics import (
    GenerativeAudioMetricsSummary,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeTextMetricsSummary,
    GenerativeVideoMetricsSummary,
    SchedulerMetrics,
)
from .report import GenerativeBenchmarkMetadata, GenerativeBenchmarksReport

__all__ = [
    "BenchmarkGenerativeTextArgs",
    "GenerativeAudioMetricsSummary",
    "GenerativeBenchmark",
    "GenerativeBenchmarkAccumulator",
    "GenerativeBenchmarkMetadata",
    "GenerativeBenchmarkTimings",
    "GenerativeBenchmarksReport",
    "GenerativeImageMetricsSummary",
    "GenerativeMetrics",
    "GenerativeMetricsAccumulator",
    "GenerativeMetricsSummary",
    "GenerativeRequestsAccumulator",
    "GenerativeTextMetricsSummary",
    "GenerativeVideoMetricsSummary",
    "RunningMetricStats",
    "SchedulerMetrics",
    "SchedulerMetricsAccumulator",
]
