"""
Primary interface for executing embeddings benchmarks.

This module orchestrates embeddings benchmarking workflows by coordinating backend
initialization, data loading, profile configuration, optional quality validation,
and output generation. Provides the main entry point `benchmark_embeddings` for
executing new embeddings benchmarks with comprehensive metric tracking.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.entrypoints import (
    resolve_backend,
    resolve_processor,
    resolve_profile,
    resolve_request_loader,
)
from guidellm.benchmark.outputs import (
    EmbeddingsBenchmarkerConsole,
    EmbeddingsBenchmarkerOutput,
)
from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress
from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.benchmark.schemas.embeddings import (
    BenchmarkEmbeddingsArgs,
    EmbeddingsBenchmark,
    EmbeddingsBenchmarkAccumulator,
    EmbeddingsBenchmarksReport,
)
from guidellm.scheduler import ConstraintInitializer, NonDistributedEnvironment
from guidellm.schemas import GenerationRequest, GenerationResponse
from guidellm.utils import Console

__all__ = ["benchmark_embeddings"]


async def resolve_embeddings_output_formats(
    outputs: list[str] | tuple[str],
    output_dir: str | Path | None,
    console: Console | None = None,
) -> dict[str, EmbeddingsBenchmarkerOutput]:
    """
    Resolve output format specifications into configured embeddings output handler instances.

    :param outputs: Specification of desired output files/types
    :param output_dir: Base path for output file generation, or None for default
    :param console: Console instance for progress reporting, or None
    :return: Dictionary mapping format names to configured output handler instances
    """
    console_step = (
        console.print_update_step(title="Resolving output formats") if console else None
    )

    resolved = EmbeddingsBenchmarkerOutput.resolve(
        outputs=outputs, output_dir=output_dir
    )

    if console_step:
        console_step.finish(
            title="Output formats resolved",
            details={key: str(val) for key, val in resolved.items()},
            status_level="success",
        )

    return resolved


async def benchmark_embeddings(
    args: BenchmarkEmbeddingsArgs,
    progress: GenerativeConsoleBenchmarkerProgress | None = None,
    console: Console | None = None,
    **constraints: str | ConstraintInitializer | Any,
) -> tuple[EmbeddingsBenchmarksReport, dict[str, Any]]:
    """
    Execute a comprehensive embeddings benchmarking workflow.

    Orchestrates the full embeddings benchmarking pipeline by resolving all components
    from provided arguments, executing benchmark runs across configured profiles, and
    finalizing results in specified output formats. Optionally performs quality
    validation using cosine similarity and MTEB benchmarks.

    :param args: Configuration arguments for the embeddings benchmark execution
    :param progress: Progress tracker for benchmark execution, or None for no tracking
    :param console: Console instance for status reporting, or None for silent operation
    :param constraints: Additional constraint initializers for benchmark limits
    :return: Tuple of EmbeddingsBenchmarksReport and dictionary of output format results

    Example:
    ::
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            data=["dataset.json"],
            enable_quality_validation=True,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        report, outputs = await benchmark_embeddings(args)
    """
    # Resolve backend
    backend, model = await resolve_backend(
        backend=args.backend,
        target=args.target,
        model=args.model,
        request_format=args.request_format or "/v1/embeddings",
        console=console,
        **(args.backend_kwargs or {}),
    )

    # Resolve processor (tokenizer)
    processor = await resolve_processor(
        processor=args.processor, model=model, console=console
    )

    # Resolve request loader for embeddings data
    request_loader = await resolve_request_loader(
        data=args.data,
        model=model,
        data_args=args.data_args,
        data_samples=args.data_samples,
        processor=processor,
        processor_args=args.processor_args,
        data_column_mapper=args.data_column_mapper,
        data_preprocessors=args.data_preprocessors,
        data_preprocessors_kwargs=args.data_preprocessors_kwargs,
        data_finalizer=args.data_finalizer,
        data_collator=args.data_collator,
        data_sampler=args.data_sampler,
        data_num_workers=args.data_num_workers,
        random_seed=args.random_seed,
        console=console,
        **(args.dataloader_kwargs or {}),
    )

    # Resolve transient phases
    warmup = TransientPhaseConfig.create_from_value(args.warmup)
    cooldown = TransientPhaseConfig.create_from_value(args.cooldown)
    if console:
        console.print_update(
            title="Resolved transient phase configurations",
            details="\n".join(
                [
                    f"Warmup: {warmup}",
                    f"Cooldown: {cooldown}",
                ]
            ),
            status="success",
        )

    # Resolve profile
    profile = await resolve_profile(
        profile=args.profile,
        rate=args.rate,
        random_seed=args.random_seed,
        rampup=0.0,  # Embeddings typically don't use rampup
        constraints=constraints,
        max_seconds=args.max_duration,
        max_requests=args.max_requests,
        max_errors=args.max_errors,
        max_error_rate=None,
        max_global_error_rate=None,
        over_saturation=None,
        console=console,
    )

    # Resolve output formats
    output_formats = await resolve_embeddings_output_formats(
        outputs=args.outputs, output_dir=args.output_dir, console=console
    )

    # Initialize quality validation if requested
    quality_validator = None
    if args.enable_quality_validation:
        if console:
            console.print_update(
                title="Initializing quality validation",
                details=f"Baseline model: {args.baseline_model or model}",
                status="info",
            )

        try:
            from guidellm.benchmark.quality import EmbeddingsQualityValidator

            quality_validator = EmbeddingsQualityValidator(
                baseline_model=args.baseline_model or model,
                tolerance=args.quality_tolerance,
            )

            if console:
                console.print_update(
                    title="Quality validation initialized",
                    details=f"Tolerance: {args.quality_tolerance}",
                    status="success",
                )
        except ImportError as e:
            if console:
                console.print_update(
                    title="Quality validation unavailable",
                    details=(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    ),
                    status="warning",
                )

    # Run MTEB evaluation if requested (before main benchmark)
    mteb_results = None
    if args.enable_mteb:
        if console:
            console.print_update(
                title="Running MTEB evaluation",
                details=f"Tasks: {args.mteb_tasks or 'default'}",
                status="info",
            )

        try:
            from guidellm.benchmark.quality import MTEBValidator

            mteb_validator = MTEBValidator(
                model_name=args.baseline_model or model,
                task_names=args.mteb_tasks,
            )
            mteb_results = mteb_validator.run_evaluation()

            if console:
                console.print_update(
                    title="MTEB evaluation complete",
                    details=f"Main score: {mteb_results['mteb_main_score']:.4f}",
                    status="success",
                )
        except ImportError as e:
            if console:
                console.print_update(
                    title="MTEB evaluation unavailable",
                    details="mteb not installed. Install with: pip install mteb",
                    status="warning",
                )

    # Create report
    report = EmbeddingsBenchmarksReport(args=args)

    if console:
        console.print_update(
            title="Setup complete, starting embeddings benchmarks...", status="success"
        )
        console.print("\n\n")

    # Run benchmarks
    benchmarker: Benchmarker[
        EmbeddingsBenchmark, GenerationRequest, GenerationResponse
    ] = Benchmarker()

    async for benchmark in benchmarker.run(
        accumulator_class=EmbeddingsBenchmarkAccumulator,
        benchmark_class=EmbeddingsBenchmark,
        requests=request_loader,
        backend=backend,
        profile=profile,
        environment=NonDistributedEnvironment(),
        progress=progress,
        sample_requests=False,  # Embeddings don't need request sampling
        warmup=warmup,
        cooldown=cooldown,
        prefer_response_metrics=True,  # Prefer API-provided metrics
    ):
        if benchmark:
            # Inject MTEB results if available
            if mteb_results and benchmark.metrics.quality:
                benchmark.metrics.quality.mteb_main_score = mteb_results[
                    "mteb_main_score"
                ]
                benchmark.metrics.quality.mteb_task_scores = mteb_results[
                    "mteb_task_scores"
                ]

            report.benchmarks.append(benchmark)

    # Finalize outputs
    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    # Print console output
    if console:
        await EmbeddingsBenchmarkerConsole(console=console).finalize(report)
        console.print("\n\n")
        console.print_update(
            title=(
                "Embeddings benchmarking complete, generated "
                f"{len(report.benchmarks)} benchmark(s)"
            ),
            status="success",
        )
        for key, value in output_format_results.items():
            console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
