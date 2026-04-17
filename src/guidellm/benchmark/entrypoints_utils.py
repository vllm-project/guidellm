"""
Shared utility functions for benchmark entrypoints.

Provides common functionality used across different benchmark types
(generative, embeddings, etc.) to reduce code duplication and maintain
consistency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from guidellm.benchmark.schemas.base import TransientPhaseConfig
from guidellm.utils.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

    from guidellm.benchmark.outputs.output import (
        EmbeddingsBenchmarkerOutput,
        GenerativeBenchmarkerOutput,
    )
    from guidellm.benchmark.progress import (
        GenerativeConsoleBenchmarkerProgress,
    )
    from guidellm.benchmark.schemas.base_args import BaseBenchmarkArgs
    from guidellm.scheduler import ConstraintInitializer

__all__ = [
    "resolve_output_formats_generic",
    "resolve_transient_phases",
    "run_benchmark_workflow",
]

OutputHandlerT = TypeVar(
    "OutputHandlerT",
    "GenerativeBenchmarkerOutput",
    "EmbeddingsBenchmarkerOutput",
)

ReportT = TypeVar("ReportT")
ConsoleHandlerT = TypeVar("ConsoleHandlerT")


async def resolve_output_formats_generic(
    outputs: list[str] | tuple[str],
    output_dir: str | Path | None,
    output_handler_class: type[OutputHandlerT],
    console: Console | None = None,
) -> dict[str, OutputHandlerT]:
    """
    Resolve output format specifications into configured output handler
    instances.

    Generic version that works with any BenchmarkerOutput subclass.

    :param outputs: Specification of desired output files/types
    :param output_dir: Base path for output file generation, or None for
        default
    :param output_handler_class: The specific output handler class to use
        (e.g., GenerativeBenchmarkerOutput,
        EmbeddingsBenchmarkerOutput)
    :param console: Console instance for progress reporting, or None
    :return: Dictionary mapping format names to configured output handler
        instances
    """
    console_step = (
        console.print_update_step(title="Resolving output formats") if console else None
    )

    resolved = output_handler_class.resolve(outputs=outputs, output_dir=output_dir)

    if console_step:
        console_step.finish(
            title="Output formats resolved",
            details={key: str(val) for key, val in resolved.items()},
            status_level="success",
        )

    return resolved


async def resolve_transient_phases(
    warmup_value: TransientPhaseConfig | float | int | dict | None,
    cooldown_value: TransientPhaseConfig | float | int | dict | None,
    console: Console | None = None,
    extra_details: list[str] | None = None,
) -> tuple[TransientPhaseConfig, TransientPhaseConfig]:
    """
    Resolve and validate warmup/cooldown phase configurations.

    :param warmup_value: Warmup phase configuration value
    :param cooldown_value: Cooldown phase configuration value
    :param console: Console instance for progress reporting, or None
    :param extra_details: Additional detail lines to include in console output
    :return: Tuple of (warmup_config, cooldown_config)
    """
    warmup = TransientPhaseConfig.create_from_value(warmup_value)
    cooldown = TransientPhaseConfig.create_from_value(cooldown_value)

    if console:
        details_lines = [
            f"Warmup: {warmup}",
            f"Cooldown: {cooldown}",
        ]
        if extra_details:
            details_lines.extend(extra_details)

        console.print_update(
            title="Resolved transient phase configurations",
            details="\n".join(details_lines),
            status="success",
        )

    return warmup, cooldown


async def run_benchmark_workflow(  # noqa: C901,PLR0912
    args: BaseBenchmarkArgs,
    accumulator_class: type,
    benchmark_class: type,
    benchmarks_report_class: type[ReportT],
    output_handler_class: type[OutputHandlerT],
    console_handler_class: type[ConsoleHandlerT],
    progress: GenerativeConsoleBenchmarkerProgress | None = None,
    console: Console | None = None,
    backend_extras_modifier: (
        Callable[[dict[str, Any], BaseBenchmarkArgs], dict[str, Any]] | None
    ) = None,
    resolve_profile_kwargs_modifier: (
        Callable[[dict[str, Any], BaseBenchmarkArgs], dict[str, Any]] | None
    ) = None,
    benchmarker_kwargs: dict[str, Any] | None = None,
    **constraints: str | ConstraintInitializer | Any,
) -> tuple[ReportT, dict[str, Any]]:
    """
    Execute a comprehensive benchmark workflow with unified orchestration.

    Generic benchmark orchestration that works for both embeddings and
    generative benchmarks. Handles the complete workflow from backend
    resolution through output finalization.

    :param args: Benchmark configuration arguments
    :param accumulator_class: Accumulator class for metrics
    :param benchmark_class: Benchmark class for results
    :param benchmarks_report_class: Report class for final results
    :param output_handler_class: Output handler for formatting results
    :param console_handler_class: Console handler for terminal output
    :param progress: Progress tracker, or None for no tracking
    :param console: Console instance, or None for silent operation
    :param backend_extras_modifier: Optional function to modify backend
        kwargs based on args
    :param resolve_profile_kwargs_modifier: Optional function to modify
        profile resolution kwargs
    :param benchmarker_kwargs: Additional kwargs for benchmarker.run()
    :param constraints: Additional constraint initializers
    :return: Tuple of report and output format results
    """
    # Import here to avoid circular dependencies
    from guidellm.benchmark.benchmarker import Benchmarker
    from guidellm.benchmark.entrypoints import (
        resolve_backend,
        resolve_processor,
        resolve_profile,
        resolve_request_loader,
    )
    from guidellm.scheduler import NonDistributedEnvironment

    # Step 1: Resolve backend
    # Convert BackendArgs to dict, excluding fields passed separately
    if hasattr(args.backend_kwargs, "model_dump"):
        # BackendArgs object - serialize and exclude top-level fields
        backend_kwargs = args.backend_kwargs.model_dump(
            exclude={"target", "model", "request_format"}
        )
    else:
        # Legacy dict support
        backend_kwargs = dict(args.backend_kwargs or {})
        # Remove fields that will be passed separately to avoid conflicts
        backend_kwargs.pop("target", None)
        backend_kwargs.pop("model", None)

    if args.request_format is not None:
        backend_kwargs["request_format"] = args.request_format
    if backend_extras_modifier:
        backend_kwargs = backend_extras_modifier(backend_kwargs, args)

    backend, model = await resolve_backend(
        backend=args.backend,
        target=args.target,
        model=args.model,
        console=console,
        **backend_kwargs,
    )

    # Step 2: Resolve processor (tokenizer)
    processor = await resolve_processor(
        processor=args.processor, model=model, console=console
    )

    # Step 3: Resolve request loader
    request_loader = await resolve_request_loader(
        data=args.data,
        model=model,
        data_args=args.data_args,
        data_samples=args.data_samples,
        processor=processor,
        processor_args=args.processor_args,
        data_column_mapper=args.data_column_mapper,  # type: ignore[arg-type]
        data_preprocessors=args.data_preprocessors,
        data_preprocessors_kwargs=args.data_preprocessors_kwargs,
        data_finalizer=args.data_finalizer,
        data_collator=args.data_collator,  # type: ignore[arg-type]
        data_sampler=args.data_sampler,
        data_num_workers=args.data_num_workers,
        random_seed=args.random_seed,
        console=console,
        **(args.dataloader_kwargs or {}),
    )

    # Step 4: Resolve transient phases
    warmup, cooldown = await resolve_transient_phases(
        warmup_value=args.warmup,
        cooldown_value=args.cooldown,
        console=console,
    )

    # Step 5: Resolve profile
    profile_kwargs = {
        "profile": args.profile,
        "rate": args.rate,
        "random_seed": args.random_seed,
        "constraints": constraints,
        "max_requests": args.max_requests,
        "max_errors": args.max_errors,
        "console": console,
    }
    if resolve_profile_kwargs_modifier:
        profile_kwargs = resolve_profile_kwargs_modifier(profile_kwargs, args)

    profile = await resolve_profile(**profile_kwargs)  # type: ignore[arg-type]

    # Step 6: Resolve output formats
    output_formats = await resolve_output_formats_generic(
        outputs=args.outputs,
        output_dir=args.output_dir,
        output_handler_class=output_handler_class,
        console=console,
    )

    # Step 7: Create report
    report = benchmarks_report_class(args=args)  # type: ignore[call-arg]
    if hasattr(report, "metadata") and hasattr(report.metadata, "start_time"):
        report.metadata.start_time = time.time()

    if console:
        console.print_update(
            title="Setup complete, starting benchmarks...",
            status="success",
        )
        console.print("\n\n")

    # Step 8: Run benchmarks
    benchmarker: Benchmarker = Benchmarker()

    run_kwargs = {
        "accumulator_class": accumulator_class,
        "benchmark_class": benchmark_class,
        "requests": request_loader,
        "backend": backend,
        "profile": profile,
        "environment": NonDistributedEnvironment(),
        "progress": progress,
        "warmup": warmup,
        "cooldown": cooldown,
    }
    if benchmarker_kwargs:
        run_kwargs.update(benchmarker_kwargs)

    async for benchmark in benchmarker.run(**run_kwargs):  # type: ignore[arg-type]
        if benchmark:
            report.benchmarks.append(benchmark)  # type: ignore[attr-defined]

    # Step 9: Finalize outputs
    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)  # type: ignore[arg-type]
        output_format_results[key] = output_result

    # Step 10: Print console output and completion
    if console:
        if "console" not in output_formats:
            await console_handler_class(console=console).finalize(report)  # type: ignore[call-arg,attr-defined]

        console.print("\n\n")
        console.print_update(
            title=(
                f"Benchmarking complete, generated "
                f"{len(report.benchmarks)} benchmark(s)"  # type: ignore[attr-defined]
            ),
            status="success",
        )
        for key, value in output_format_results.items():
            console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
