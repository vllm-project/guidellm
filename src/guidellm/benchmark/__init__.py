"""
Benchmark execution and performance analysis framework.

Provides comprehensive benchmarking capabilities for LLM inference workloads,
including profile-based execution strategies, metrics collection and aggregation,
progress tracking, and multi-format output generation. Supports synchronous,
asynchronous, concurrent, sweep, and throughput-based benchmarking profiles for
evaluating model performance under various load conditions.
"""

from __future__ import annotations

from .benchmarker import Benchmarker
from .entrypoints import benchmark_generative_text, reimport_benchmarks_report
from .outputs import (
    GenerativeBenchmarkerConsole,
    GenerativeBenchmarkerCSV,
    GenerativeBenchmarkerHTML,
    GenerativeBenchmarkerOutput,
)
from .profiles import (
    AsyncProfile,
    ConcurrentProfile,
    Profile,
    ProfileType,
    SweepProfile,
    SynchronousProfile,
    ThroughputProfile,
)
from .progress import BenchmarkerProgress, GenerativeConsoleBenchmarkerProgress
from .scenarios import get_builtin_scenarios
from .schemas import (
    Benchmark,
    BenchmarkAccumulator,
    BenchmarkAccumulatorT,
    BenchmarkConfig,
    BenchmarkGenerativeTextArgs,
    BenchmarkT,
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
    "AsyncProfile",
    "Benchmark",
    "BenchmarkAccumulator",
    "BenchmarkAccumulatorT",
    "BenchmarkConfig",
    "BenchmarkGenerativeTextArgs",
    "BenchmarkT",
    "Benchmarker",
    "BenchmarkerProgress",
    "ConcurrentProfile",
    "GenerativeAudioMetricsSummary",
    "GenerativeBenchmark",
    "GenerativeBenchmarkAccumulator",
    "GenerativeBenchmarkMetadata",
    "GenerativeBenchmarkTimings",
    "GenerativeBenchmarkerCSV",
    "GenerativeBenchmarkerConsole",
    "GenerativeBenchmarkerHTML",
    "GenerativeBenchmarkerOutput",
    "GenerativeBenchmarksReport",
    "GenerativeConsoleBenchmarkerProgress",
    "GenerativeImageMetricsSummary",
    "GenerativeMetrics",
    "GenerativeMetricsAccumulator",
    "GenerativeMetricsSummary",
    "GenerativeRequestsAccumulator",
    "GenerativeTextMetricsSummary",
    "GenerativeVideoMetricsSummary",
    "Profile",
    "ProfileType",
    "RunningMetricStats",
    "SchedulerMetrics",
    "SchedulerMetricsAccumulator",
    "SweepProfile",
    "SynchronousProfile",
    "ThroughputProfile",
    "benchmark_generative_text",
    "get_builtin_scenarios",
    "reimport_benchmarks_report",
]
