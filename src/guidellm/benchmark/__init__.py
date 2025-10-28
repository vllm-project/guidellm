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
from .output import (
    GenerativeBenchmarkerConsole,
    GenerativeBenchmarkerCSV,
    GenerativeBenchmarkerHTML,
    GenerativeBenchmarkerOutput,
)
from .profile import (
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
    BenchmarkerArgs,
    BenchmarkerDict,
    BenchmarkGenerativeTextArgs,
    BenchmarkSchedulerStats,
    EstimatedBenchmarkState,
    GenerativeAudioMetricsSummary,
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
    GenerativeImageMetricsSummary,
    GenerativeMetrics,
    GenerativeMetricsSummary,
    GenerativeVideoMetricsSummary,
    SchedulerDict,
)

__all__ = [
    "AsyncProfile",
    "Benchmark",
    "BenchmarkGenerativeTextArgs",
    "BenchmarkSchedulerStats",
    "Benchmarker",
    "BenchmarkerArgs",
    "BenchmarkerDict",
    "BenchmarkerProgress",
    "ConcurrentProfile",
    "EstimatedBenchmarkState",
    "GenerativeAudioMetricsSummary",
    "GenerativeBenchmark",
    "GenerativeBenchmarkerCSV",
    "GenerativeBenchmarkerConsole",
    "GenerativeBenchmarkerHTML",
    "GenerativeBenchmarkerOutput",
    "GenerativeBenchmarksReport",
    "GenerativeConsoleBenchmarkerProgress",
    "GenerativeImageMetricsSummary",
    "GenerativeMetrics",
    "GenerativeMetricsSummary",
    "GenerativeVideoMetricsSummary",
    "Profile",
    "ProfileType",
    "SchedulerDict",
    "SweepProfile",
    "SynchronousProfile",
    "ThroughputProfile",
    "benchmark_generative_text",
    "get_builtin_scenarios",
    "reimport_benchmarks_report",
]
