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

from .accumulator import (
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarkTimings,
    GenerativeMetricsAccumulator,
    GenerativeRequestsAccumulator,
    RunningMetricStats,
    SchedulerMetricsAccumulator,
)
from .base import (
    Benchmark,
    BenchmarkAccumulator,
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkT,
    TransientPhaseConfig,
)
from .benchmark import GenerativeBenchmark
from .entrypoints import (
    BenchmarkArgs,
    BenchmarkMetadata,
    BenchmarkScenario,
    GenerativeMetricsArgs,
    MetricsArgs,
)
from .metrics import (
    GenerativeAudioMetricsSummary,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeTextMetricsSummary,
    GenerativeVideoMetricsSummary,
    SchedulerMetrics,
)
from .output import BenchmarkOutputArgs
from .profiles import ProfileArgs
from .random import RandomArgs, StaticRandomArgs
from .report import GenerativeBenchmarkMetadata, GenerativeBenchmarksReport

__all__ = [
    "Benchmark",
    "BenchmarkAccumulator",
    "BenchmarkAccumulatorT",
    "BenchmarkArgs",
    "BenchmarkConfig",
    "BenchmarkMetadata",
    "BenchmarkOutputArgs",
    "BenchmarkScenario",
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
    "GenerativeMetricsArgs",
    "GenerativeMetricsSummary",
    "GenerativeRequestsAccumulator",
    "GenerativeTextMetricsSummary",
    "GenerativeVideoMetricsSummary",
    "MetricsArgs",
    "ProfileArgs",
    "RandomArgs",
    "RunningMetricStats",
    "SchedulerMetrics",
    "SchedulerMetricsAccumulator",
    "StaticRandomArgs",
    "TransientPhaseConfig",
]
