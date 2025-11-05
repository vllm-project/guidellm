"""
High-level entry points for executing generative text benchmarks.

This module provides the primary interface for running generative text benchmarks
through the `benchmark_generative_text` function and re-importing existing benchmark
reports via `reimport_benchmarks_report`. It orchestrates the initialization and
coordination of backends, data loaders, profiles, and output formats to execute
comprehensive benchmarking workflows. The module handles all resolution logic for
converting user-provided arguments into fully configured components ready for
benchmarking execution.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase
from typing_extensions import TypeAliasType

from guidellm.backends import Backend, BackendType
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkGenerativeTextArgs,
    GenerativeBenchmark,
    GenerativeBenchmarksReport,
)
from guidellm.data import (
    DataLoader,
    DatasetPreprocessor,
    GenerativeRequestCollator,
    PreprocessorRegistry,
    ProcessorFactory,
)
from guidellm.data.preprocessors import GenerativeColumnMapper
from guidellm.scheduler import (
    ConstraintInitializer,
    NonDistributedEnvironment,
    StrategyType,
)
from guidellm.schemas import GenerationRequest, GenerationResponse
from guidellm.utils import Console, InfoMixin

__all__ = [
    "benchmark_generative_text",
    "reimport_benchmarks_report",
]


# Helper Functions

OutputFormatT = TypeAliasType(
    "OutputFormatT",
    tuple[str, ...]
    | list[str]
    | dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
    | None,
)

ProcessorInputT = TypeAliasType("ProcessorInputT", str | Path | PreTrainedTokenizerBase)


async def resolve_backend(
    backend: BackendType | Backend,
    target: str,
    model: str | None,
    console: Console | None = None,
    **backend_kwargs: dict[str, Any],
) -> tuple[Backend, str | None]:
    """
    Initialize and validate a backend instance for benchmarking.

    :param backend: Backend type identifier or pre-configured Backend instance
    :param target: Target endpoint URL or connection string for the backend
    :param model: Model identifier to use with the backend, or None to use default
    :param console: Console instance for progress reporting, or None
    :param backend_kwargs: Additional keyword arguments passed to backend initialization
    :return: Tuple of initialized Backend instance and resolved model identifier
    """
    console_step = (
        console.print_update_step(title=f"Initializing backend {backend}")
        if console
        else None
    )
    backend = (
        Backend.create(backend, target=target, model=model, **(backend_kwargs or {}))
        if not isinstance(backend, Backend)
        else backend
    )

    if console_step:
        console_step.update(f"{backend.__class__.__name__} backend initialized")

    await backend.process_startup()
    await backend.validate()

    if model is None:
        if console_step:
            console_step.update(
                title="Resolving default model from backend.default_model",
                status_level="info",
            )
        model = await backend.default_model()

    await backend.process_shutdown()

    if console_step:
        console_step.finish(
            title=(
                f"{backend.__class__.__name__} backend validated with model {model}"
            ),
            details=backend.info,
            status_level="success",
        )

    return backend, model


async def resolve_processor(
    processor: ProcessorInputT | None,
    model: str | None,
    console: Console | None = None,
) -> ProcessorInputT | None:
    """
    Resolve the processor for tokenization, defaulting to model if not provided.

    :param processor: Processor identifier, path, tokenizer instance, or None
    :param model: Model identifier to use as fallback processor
    :param console: Console instance for progress reporting, or None
    :return: Resolved processor or None if neither processor nor model provided
    """
    console_step = (
        console.print_update_step(title=f"Resolving processor {processor}")
        if console
        else None
    )

    if processor is not None:
        if console_step:
            console_step.finish(
                title="Processor resolved",
                details=f"Using processor '{processor}'",
                status_level="success",
            )
    else:
        processor = model
        if console_step:
            console_step.finish(
                title="Processor resolved",
                details=f"Using model '{processor}' as processor",
                status_level="success",
            )

    return processor


async def resolve_request_loader(
    data: list[Any],
    model: str | None,
    data_args: list[dict[str, Any]] | None,
    data_samples: int,
    processor: ProcessorInputT | None,
    processor_args: dict[str, Any] | None,
    data_column_mapper: (
        DatasetPreprocessor | dict[str, str] | Literal["generative_column_mapper"]
    ),
    data_request_formatter: (DatasetPreprocessor | dict[str, str] | str),
    data_collator: Callable | Literal["generative"] | None,
    data_sampler: Sampler[int] | Literal["shuffle"] | None,
    data_num_workers: int | None,
    random_seed: int,
    console: Console | None = None,
    **dataloader_kwargs: dict[str, Any] | None,
) -> DataLoader[GenerationRequest]:
    """
    Construct a DataLoader for GenerationRequest objects from raw data inputs.

    :param data: List of data sources to load requests from
    :param model: Model identifier for request formatting
    :param data_args: Arguments for each data source in the data list
    :param data_samples: Number of samples to draw from the dataset
    :param processor: Processor for tokenization operations
    :param processor_args: Arguments for processor initialization
    :param data_column_mapper: Preprocessor or mapping for standardizing column names
    :param data_request_formatter: Preprocessor or config for formatting requests
    :param data_collator: Collation function or type for batching requests
    :param data_sampler: Sampler instance or type for data sampling
    :param data_num_workers: Number of worker processes for data loading
    :param random_seed: Seed for reproducible random operations
    :param console: Console instance for progress reporting, or None
    :param dataloader_kwargs: Additional arguments passed to DataLoader initialization
    :return: Configured DataLoader instance for GenerationRequest objects
    """
    console_step = (
        console.print_update_step(title=f"Initializing request loader from {data}")
        if console
        else None
    )

    if not isinstance(data_column_mapper, DatasetPreprocessor):
        column_mappings = (
            data_column_mapper if isinstance(data_column_mapper, dict) else None
        )
        data_column_mapper = GenerativeColumnMapper(
            column_mappings=column_mappings,
        )
    if not isinstance(data_request_formatter, DatasetPreprocessor):
        request_type = (
            data_request_formatter
            if isinstance(data_request_formatter, str)
            else data_request_formatter.pop("request_type", "chat_completions")
        )
        data_request_formatter = PreprocessorRegistry.get_registered_object(
            request_type
        )(
            model=model,
            **(
                data_request_formatter
                if isinstance(data_request_formatter, dict)
                else {}
            ),
        )

    request_loader = DataLoader(
        data=data,
        data_args=data_args,
        data_samples=data_samples,
        processor_factory=ProcessorFactory(
            processor=processor, processor_args=processor_args
        ),
        preprocessors=[data_column_mapper, data_request_formatter],
        collator=(
            data_collator if callable(data_collator) else GenerativeRequestCollator()
        ),
        sampler=data_sampler,
        num_workers=data_num_workers,
        random_seed=random_seed,
        **(dataloader_kwargs or {}),
    )

    if console_step:
        console_step.finish(
            title=(
                f"Request loader initialized with "
                f"{data_samples if data_samples > 0 else 'inf'} "
                f"unique requests from {data}"
            ),
            details=InfoMixin.extract_from_obj(request_loader),
            status_level="success",
        )

    return request_loader


async def resolve_profile(
    profile: StrategyType | ProfileType | Profile,
    rate: float | list[float] | None,
    random_seed: int,
    constraints: dict[str, ConstraintInitializer | Any],
    max_seconds: int | float | None,
    max_requests: int | None,
    max_errors: int | None,
    max_error_rate: float | None,
    max_global_error_rate: float | None,
    console: Console | None = None,
) -> Profile:
    """
    Resolve and configure a benchmark profile with rate and constraint settings.

    :param profile: Profile type identifier or pre-configured Profile instance
    :param rate: Request rate(s) for the benchmark execution
    :param random_seed: Seed for reproducible random operations
    :param constraints: Dictionary of constraint initializers for benchmark limits
    :param max_seconds: Maximum duration in seconds for the benchmark
    :param max_requests: Maximum number of requests to process
    :param max_errors: Maximum number of errors before stopping
    :param max_error_rate: Maximum error rate threshold before stopping
    :param max_global_error_rate: Maximum global error rate threshold before stopping
    :param console: Console instance for progress reporting, or None
    :return: Configured Profile instance ready for benchmarking
    :raises ValueError: If constraints are provided with a pre-configured Profile
    """
    console_step = (
        console.print_update_step(title=f"Resolving profile {profile}")
        if console
        else None
    )

    for key, val in {
        "max_seconds": max_seconds,
        "max_requests": max_requests,
        "max_errors": max_errors,
        "max_error_rate": max_error_rate,
        "max_global_error_rate": max_global_error_rate,
    }.items():
        if val is not None:
            constraints[key] = val
    if not isinstance(profile, Profile):
        profile = Profile.create(
            rate_type=profile,
            rate=rate,
            random_seed=random_seed,
            constraints={**constraints},
        )
    elif constraints:
        raise ValueError(
            "Constraints must be empty when providing a Profile instance. "
            f"Provided constraints: {constraints} ; provided profile: {profile}"
        )

    if console_step:
        console_step.finish(
            title=f"{profile.__class__.__name__} profile resolved",
            details=InfoMixin.extract_from_obj(profile),
            status_level="success",
        )

    return profile


async def resolve_output_formats(
    output_formats: OutputFormatT,
    output_path: str | Path | None,
    console: Console | None = None,
) -> dict[str, GenerativeBenchmarkerOutput]:
    """
    Resolve output format specifications into configured output handler instances.

    :param output_formats: Specification of desired output formats
    :param output_path: Base path for output file generation, or None for default
    :param console: Console instance for progress reporting, or None
    :return: Dictionary mapping format names to configured output handler instances
    """
    console_step = (
        console.print_update_step(title="Resolving output formats") if console else None
    )

    resolved = GenerativeBenchmarkerOutput.resolve(
        output_formats=output_formats, output_path=output_path
    )

    if console_step:
        console_step.finish(
            title="Output formats resolved",
            details={key: str(val) for key, val in resolved.items()},
            status_level="success",
        )

    return resolved


# Main Entrypoints Functions


async def benchmark_generative_text(
    args: BenchmarkGenerativeTextArgs,
    progress: GenerativeConsoleBenchmarkerProgress | None = None,
    console: Console | None = None,
    **constraints: dict[str, ConstraintInitializer | Any],
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    """
    Execute a comprehensive generative text benchmarking workflow.

    Orchestrates the full benchmarking pipeline by resolving all components (backend,
    data loader, profile, outputs) from provided arguments, executing the benchmark
    runs, and finalizing results in the specified output formats.

    :param args: Configuration arguments for the benchmark execution
    :param progress: Progress tracker for benchmark execution, or None for no tracking
    :param console: Console instance for status reporting, or None for silent operation
    :param constraints: Additional constraint initializers for benchmark limits
    :return: Tuple of GenerativeBenchmarksReport and dictionary of output format results
    """
    backend, model = await resolve_backend(
        backend=args.backend,
        target=args.target,
        model=args.model,
        console=console,
        **(args.backend_kwargs or {}),
    )
    processor = await resolve_processor(
        processor=args.processor, model=model, console=console
    )
    request_loader = await resolve_request_loader(
        data=args.data,
        model=model,
        data_args=args.data_args,
        data_samples=args.data_samples,
        processor=processor,
        processor_args=args.processor_args,
        data_column_mapper=args.data_column_mapper,
        data_request_formatter=args.data_request_formatter,
        data_collator=args.data_collator,
        data_sampler=args.data_sampler,
        data_num_workers=args.data_num_workers,
        random_seed=args.random_seed,
        console=console,
        **(args.dataloader_kwargs or {}),
    )
    profile = await resolve_profile(
        profile=args.profile,
        rate=args.rate,
        random_seed=args.random_seed,
        constraints=constraints,
        max_seconds=args.max_seconds,
        max_requests=args.max_requests,
        max_errors=args.max_errors,
        max_error_rate=args.max_error_rate,
        max_global_error_rate=args.max_global_error_rate,
        console=console,
    )
    output_formats = await resolve_output_formats(
        output_formats=args.output_formats,
        output_path=args.output_path,
        console=console,
    )

    report = GenerativeBenchmarksReport(args=args)
    if console:
        console.print_update(
            title="Setup complete, starting benchmarks...", status="success"
        )
        console.print("\n\n")

    benchmarker: Benchmarker[
        GenerativeBenchmark, GenerationRequest, GenerationResponse
    ] = Benchmarker()
    async for benchmark in benchmarker.run(
        benchmark_class=args.benchmark_cls,
        requests=request_loader,
        backend=backend,
        profile=profile,
        environment=NonDistributedEnvironment(),
        data=args.data,
        progress=progress,
        sample_requests=args.sample_requests,
        warmup=args.warmup,
        cooldown=args.cooldown,
        prefer_response_metrics=args.prefer_response_metrics,
    ):
        if benchmark:
            report.benchmarks.append(benchmark)

    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    if console:
        console.print("\n\n")
        console.print_update(
            title=(
                "Benchmarking complete, generated "
                f"{len(report.benchmarks)} benchmark(s)"
            ),
            status="success",
        )
        for key, value in output_format_results.items():
            console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results


async def reimport_benchmarks_report(
    file: Path,
    output_path: Path | None,
    output_formats: OutputFormatT = ("console", "json", "html", "csv"),
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    """
    Load and re-export an existing benchmarks report in specified formats.

    :param file: Path to the existing benchmark report file to load
    :param output_path: Base path for output file generation, or None for default
    :param output_formats: Specification of desired output formats for the report
    :return: Tuple of loaded GenerativeBenchmarksReport and dictionary of output results
    """
    console = Console()

    with console.print_update_step(
        title=f"Loading benchmarks from {file}..."
    ) as console_step:
        report = GenerativeBenchmarksReport.load_file(file)
        console_step.finish(
            "Import of old benchmarks complete;"
            f" loaded {len(report.benchmarks)} benchmark(s)"
        )

    output_formats = await resolve_output_formats(
        output_formats, output_path, console=console
    )
    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    for key, value in output_format_results.items():
        console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
