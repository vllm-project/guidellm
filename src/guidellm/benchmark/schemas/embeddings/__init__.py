"""
Embeddings benchmark schemas for performance measurement and analysis.

This module provides the complete schema ecosystem for executing, tracking,
and analyzing embeddings benchmarks. It encompasses configuration entrypoints
for benchmark setup, real-time metric accumulators for execution monitoring,
comprehensive result containers with statistical summaries, and multi-benchmark
reporting capabilities.
"""

from __future__ import annotations

from .accumulator import (
    EmbeddingsBenchmarkAccumulator,
    EmbeddingsBenchmarkTimings,
    EmbeddingsMetricsAccumulator,
    EmbeddingsRequestsAccumulator,
    RunningMetricStats,
    SchedulerMetricsAccumulator,
)
from .benchmark import EmbeddingsBenchmark
from .entrypoints import BenchmarkEmbeddingsArgs
from .metrics import (
    EmbeddingsMetrics,
    SchedulerMetrics,
)
from .report import EmbeddingsBenchmarkMetadata, EmbeddingsBenchmarksReport

__all__ = [
    "BenchmarkEmbeddingsArgs",
    "EmbeddingsBenchmark",
    "EmbeddingsBenchmarkAccumulator",
    "EmbeddingsBenchmarkMetadata",
    "EmbeddingsBenchmarkTimings",
    "EmbeddingsBenchmarksReport",
    "EmbeddingsMetrics",
    "EmbeddingsMetricsAccumulator",
    "EmbeddingsRequestsAccumulator",
    "RunningMetricStats",
    "SchedulerMetrics",
    "SchedulerMetricsAccumulator",
]
