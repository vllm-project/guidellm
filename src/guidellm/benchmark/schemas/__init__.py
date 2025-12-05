"""
Benchmark schemas for performance measurement and result analysis.

This module consolidates the complete benchmark schema ecosystem, providing both
base abstractions for benchmark execution and domain-specific implementations
for generative AI tasks. It exports core configuration objects, accumulator
interfaces for real-time metric collection, benchmark result containers with
statistical summaries, and reporting utilities. The schemas support flexible
scheduling strategies, comprehensive metric tracking including latency and
throughput distributions, and multi-modal generative benchmarks for text, image,
video, and audio generation tasks.
"""

from __future__ import annotations

from .base import (
    Benchmark,
    BenchmarkAccumulator,
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
)
from .generative import (
    BenchmarkGenerativeTextArgs,
    GenerativeAudioMetricsSummary,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarkMetadata,
    GenerativeBenchmarksReport,
    GenerativeBenchmarkTimings,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsAccumulator,
    GenerativeMetricsSummary,
    GenerativeRequestsAccumulator,
    GenerativeTextMetricsSummary,
    GenerativeVideoMetricsSummary,
    RunningMetricStats,
    SchedulerMetrics,
    SchedulerMetricsAccumulator,
)

__all__ = [
    "Benchmark",
    "BenchmarkAccumulator",
    "BenchmarkAccumulatorT",
    "BenchmarkConfig",
    "BenchmarkGenerativeTextArgs",
    "BenchmarkT",
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
