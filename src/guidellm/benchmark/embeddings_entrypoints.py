"""
Primary interface for executing embeddings benchmarks.

This module orchestrates embeddings benchmarking workflows by coordinating
backend initialization, data loading, profile configuration, and output
generation. Provides the main entry point `benchmark_embeddings` for
executing new embeddings benchmarks with comprehensive metric tracking.
"""

from __future__ import annotations

from typing import Any

from guidellm.benchmark.entrypoints_utils import run_benchmark_workflow
from guidellm.benchmark.outputs import (
    EmbeddingsBenchmarkerConsole,
    EmbeddingsBenchmarkerOutput,
)
from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress
from guidellm.benchmark.schemas.embeddings import (
    BenchmarkEmbeddingsArgs,
    EmbeddingsBenchmark,
    EmbeddingsBenchmarkAccumulator,
    EmbeddingsBenchmarksReport,
)
from guidellm.scheduler import ConstraintInitializer
from guidellm.utils.console import Console

__all__ = ["benchmark_embeddings"]


# resolve_embeddings_output_formats function removed - now uses
# resolve_output_formats_generic from entrypoints_utils


async def benchmark_embeddings(
    args: BenchmarkEmbeddingsArgs,
    progress: GenerativeConsoleBenchmarkerProgress | None = None,
    console: Console | None = None,
    **constraints: str | ConstraintInitializer | Any,
) -> tuple[EmbeddingsBenchmarksReport, dict[str, Any]]:
    """
    Execute a comprehensive embeddings benchmarking workflow.

    Orchestrates the full embeddings benchmarking pipeline by resolving all
    components from provided arguments, executing benchmark runs across
    configured profiles, and finalizing results in specified output formats.

    :param args: Configuration arguments for the embeddings benchmark
        execution
    :param progress: Progress tracker for benchmark execution, or None for
        no tracking
    :param console: Console instance for status reporting, or None for
        silent operation
    :param constraints: Additional constraint initializers for benchmark
        limits
    :return: Tuple of EmbeddingsBenchmarksReport and dictionary of output
        format results

    Example:
    ::
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            data=["dataset.json"]
        )
        report, outputs = await benchmark_embeddings(args)
    """

    def setup_backend_kwargs(
        backend_kwargs: dict[str, Any], args: BenchmarkEmbeddingsArgs
    ) -> dict[str, Any]:
        """Configure backend for embeddings requests."""
        # Set request_format for embeddings endpoint
        if "request_format" not in backend_kwargs:
            backend_kwargs["request_format"] = args.request_format or "/v1/embeddings"
        # Add encoding_format to backend extras
        extras = backend_kwargs.get("extras", {})
        if isinstance(extras, dict):
            extras["encoding_format"] = args.encoding_format
            backend_kwargs["extras"] = extras
        return backend_kwargs

    def setup_profile_kwargs(
        profile_kwargs: dict[str, Any], args: BenchmarkEmbeddingsArgs
    ) -> dict[str, Any]:
        """Configure profile kwargs for embeddings (no rampup, uses
        max_duration)."""
        profile_kwargs["rampup"] = 0.0  # Embeddings don't use rampup
        profile_kwargs["max_seconds"] = args.max_duration
        # Embeddings don't use these constraints
        profile_kwargs["max_error_rate"] = None
        profile_kwargs["max_global_error_rate"] = None
        profile_kwargs["over_saturation"] = None
        return profile_kwargs

    return await run_benchmark_workflow(
        args=args,
        accumulator_class=EmbeddingsBenchmarkAccumulator,
        benchmark_class=EmbeddingsBenchmark,
        benchmarks_report_class=EmbeddingsBenchmarksReport,
        output_handler_class=EmbeddingsBenchmarkerOutput,  # type: ignore[type-abstract]
        console_handler_class=EmbeddingsBenchmarkerConsole,
        progress=progress,
        console=console,
        backend_extras_modifier=setup_backend_kwargs,  # type: ignore[arg-type]
        resolve_profile_kwargs_modifier=setup_profile_kwargs,  # type: ignore[arg-type]
        benchmarker_kwargs={
            "sample_requests": False,  # Embeddings don't need sampling
            "prefer_response_metrics": True,  # Prefer API metrics
        },
        **constraints,
    )
