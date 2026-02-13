"""
Embeddings benchmark schemas for performance measurement and analysis.

This module provides the complete schema ecosystem for executing, tracking, and
analyzing embeddings benchmarks. It encompasses configuration entrypoints for
benchmark setup, real-time metric accumulators for execution monitoring,
comprehensive result containers with statistical summaries, multi-benchmark
reporting capabilities, and optional quality validation metrics including cosine
similarity and MTEB benchmarks.
"""

from __future__ import annotations

from .accumulator import (
    EmbeddingsBenchmarkAccumulator,
    EmbeddingsBenchmarkTimings,
    EmbeddingsMetricsAccumulator,
    EmbeddingsQualityMetricsAccumulator,
    EmbeddingsRequestsAccumulator,
    RunningMetricStats,
    SchedulerMetricsAccumulator,
)
from .benchmark import EmbeddingsBenchmark
from .entrypoints import BenchmarkEmbeddingsArgs
from .metrics import (
    EmbeddingsMetrics,
    EmbeddingsQualityMetrics,
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
    "EmbeddingsQualityMetrics",
    "EmbeddingsQualityMetricsAccumulator",
    "EmbeddingsRequestsAccumulator",
    "RunningMetricStats",
    "SchedulerMetrics",
    "SchedulerMetricsAccumulator",
]
