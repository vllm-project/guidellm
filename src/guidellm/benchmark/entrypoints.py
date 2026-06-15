"""
Primary interface for executing and re-importing generative text benchmarks.

This module orchestrates comprehensive benchmarking workflows by coordinating backend
initialization, data loading, profile configuration, and output generation. It provides
two main entry points: `benchmark_generative_text` for executing new benchmarks and
`reimport_benchmarks_report` for re-exporting existing results. The resolution functions
convert user-provided arguments into fully configured components, handling backend
validation, data preprocessing, profile constraints, and output format specifications.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from typing_extensions import TypeAliasType

from guidellm.backends import Backend, BackendArgs
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.outputs import (
    GenerativeBenchmarkerConsole,
    GenerativeBenchmarkerOutput,
)
from guidellm.benchmark.profiles import Profile, ProfileFactory
from guidellm.benchmark.progress import GenerativeConsoleBenchmarkerProgress
from guidellm.benchmark.schemas import (
    BenchmarkArgs,
    BenchmarkOutputArgs,
    BenchmarkScenario,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarksReport,
    ProfileArgs,
)
from guidellm.data import (
    DataLoader,
    create_data_loader,
)
from guidellm.scheduler import (
    ConstraintInitializer,
    ConstraintsInitializerFactory,
    NonDistributedEnvironment,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
)
from guidellm.utils.console import Console
from guidellm.utils.mixins import InfoMixin
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "benchmark_generative_text",
    "reimport_benchmarks_report",
]


# Type Aliases

OutputFormatT = TypeAliasType(
    "OutputFormatT",
    tuple[str, ...]
    | list[str]
    | Mapping[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
    | None,
)
"""Output format specification as strings, mappings, or configured output instances"""


# Helper Functions


async def resolve_backend(
    backend_args: BackendArgs,
    console: Console | None = None,
) -> tuple[Backend, str]:
    """
    Initialize and validate a backend instance for benchmarking execution.

    Handles backend creation from type identifiers or pre-configured instances,
    performs startup validation, and resolves the default model if not specified.
    The backend is shut down after validation to ensure clean state for subsequent
    benchmark execution.

    All backend-specific options (target, model, request_format, etc.) are passed
    through ``backend_kwargs``. Options with a ``None`` value are filtered out so
    that backends which do not accept a given parameter are not sent unexpected
    keyword arguments.

    :param backend: Backend type identifier or pre-configured Backend instance
    :param console: Console instance for progress reporting, or None
    :param backend_kwargs: Keyword arguments passed to backend initialization
        (e.g. target, model, request_format)
    :return: Tuple of initialized Backend instance and resolved model identifier
    """
    console_step = (
        console.print_update_step(title=f"Initializing backend {backend_args.kind}")
        if console
        else None
    )

    backend_instance = Backend.create(backend_args)

    if console_step:
        console_step.update(
            f"{backend_instance.__class__.__name__} backend initialized"
        )

    await backend_instance.process_startup()
    await backend_instance.validate()

    if console_step:
        console_step.update(
            title="Resolving default model from backend.default_model",
            status_level="info",
        )
    model = await backend_instance.default_model()

    await backend_instance.process_shutdown()

    if console_step:
        console_step.finish(
            title=(
                f"{backend_instance.__class__.__name__} backend validated "
                f"with model {model}"
            ),
            details=backend_instance.info,
            status_level="success",
        )

    return backend_instance, model


async def resolve_tokenizer(
    args: BenchmarkArgs,
    model: str | None,
    console: Console | None = None,
) -> None:
    """
    Resolve the tokenization processor, defaulting to model if not provided.

    :param args: BenchmarkArgs containing tokenizer configuration
    :param model: Resolved model identifier from the backend, used as default if
        tokenizer model is not specified
    :param console: Console instance for progress reporting, or None
    :return: None (the tokenizer model is set in-place on args.tokenizer)
    """
    console_step = (
        console.print_update_step(title=f"Resolving tokenizer {args.tokenizer}")
        if console
        else None
    )

    if args.tokenizer.model is not None:
        if console_step:
            console_step.finish(
                title="Processor resolved",
                details=f"Using tokenizer '{args.tokenizer.model}' from arguments",
                status_level="success",
            )
    else:
        args.tokenizer.model = model
        if console_step:
            console_step.finish(
                title="Processor resolved",
                details=f"Using model '{model}' as tokenizer",
                status_level="success",
            )


BaseTypeT = TypeVar("BaseTypeT")


def resolve_item_from_registry(
    base_type: type[BaseTypeT],
    registry: type[RegistryMixin],
    item: Any,
    extras: dict[str, Any] | None = None,
) -> BaseTypeT:
    """
    Resolve an item from a registry, instantiating it if necessary.

    :param base_type: The expected base type of the item
    :param item: The item to resolve, either an instance or a string identifier
    :param registry: The registry to use for resolving string identifiers
    :param extras: Additional keyword arguments to pass during instantiation
    :return: The resolved item as an instance of the base type
    :raises ValueError: If the item cannot be resolved from the registry
    :raises TypeError: If the resolved item is not of the expected base type
    """
    if isinstance(item, base_type):
        return item
    else:
        kwargs: dict[str, Any] = extras.copy() if extras is not None else {}
        if isinstance(item, str):
            item_type = item
        else:
            item_dict = dict(item)
            item_type = item_dict.pop("type", None)
            if item_type is None:
                raise ValueError(
                    f"Item dictionary must contain a 'type' key to resolve from "
                    f"{registry.__class__.__name__}."
                )
            kwargs.update(item_dict)

        if (item_class := registry.get_registered_object(item_type)) is None:
            raise ValueError(
                f"Item type '{item_type}' is not registered in the "
                f"{registry.__class__.__name__}."
            )
        if not issubclass(item_class, base_type):
            raise TypeError(
                f"Resolved item type '{item_type}' is not a subclass of "
                f"{base_type.__name__}."
            )
        return item_class(**kwargs)


async def resolve_request_loader(
    args: BenchmarkArgs,
    console: Console | None = None,
) -> DataLoader[GenerationRequest]:
    """
    Construct a DataLoader for GenerationRequest objects from raw data inputs.

    Initializes and configures the data pipeline including column mapping, request
    formatting, collation, and sampling. Resolves string-based preprocessor identifiers
    from the PreprocessorRegistry and creates appropriate instances with provided
    configurations.

    :param args: BenchmarkArgs containing data loading configuration
    :param console: Console instance for progress reporting, or None
    :return: Configured DataLoader instance yielding GenerationRequest objects
    """
    console_step = (
        console.print_update_step(title=f"Initializing request loader from {args.data}")
        if console
        else None
    )

    request_loader: DataLoader[GenerationRequest] = create_data_loader(
        loader_config=args.data_loader,
        data_config=args.data,
        tokenizer_config=args.tokenizer,
        column_mapper_config=args.data_column_mapper,
        preprocessors_config=args.data_preprocessors,
        finalizer_config=args.data_finalizer,
        random_seed=args.seed.value,  # type: ignore[attr-defined]
    )

    if console_step:
        samples = args.data_loader.samples if args.data_loader.samples > 0 else "inf"
        console_step.finish(
            title=(f"Request loader initialized with {samples} unique requests"),
            details=InfoMixin.extract_from_obj(request_loader),
            status_level="success",
        )

    return request_loader


def resolve_constraints(
    args: BenchmarkArgs,
    **extra_constraints: ConstraintInitializer | Any,
) -> dict[str, Any]:
    """
    Resolve all constraint sources into a unified constraints dictionary.

    Resolves the explicit ``constraints`` list from args into a single dict keyed
    by constraint registry names. Also merges any programmatically provided extra
    constraints.

    :param args: Benchmark configuration containing constraint fields
    :param extra_constraints: Additional constraint initializers passed programmatically
    :return: Dictionary mapping constraint keys to initializers or raw values
    """
    resolved: dict[str, Any] = {}

    for constraint_arg in args.constraints:
        resolved[constraint_arg.constraint_key] = ConstraintsInitializerFactory.create(
            constraint_arg
        )

    for key, val in extra_constraints.items():
        if isinstance(val, dict) and "type_" in val:
            resolved[key] = ConstraintsInitializerFactory.deserialize(
                initializer_dict=val
            )
        else:
            resolved[key] = val

    return resolved


async def resolve_profile(
    profile: ProfileArgs | Profile,
    constraints: dict[str, Any] | None,
    console: Console | None = None,
    random_seed: int = 42,
    **profile_kwargs: Any,
) -> Profile:
    """
    Resolve and configure a benchmark profile with rate and constraint settings.

    Constructs a Profile instance from type identifiers or validates pre-configured
    profiles.

    :param profile: Profile type identifier or pre-configured Profile instance
    :param constraints: Pre-resolved constraints dictionary for benchmark limits
    :param console: Console instance for progress reporting, or None
    :param random_seed: Seed for reproducible random operations in profile strategies
    :param profile_kwargs: Additional profile-specific arguments such as data and
        data_samples, used by some profiles and ignored by others.
    :return: Configured Profile instance ready for benchmarking
    :raises ValueError: If constraints are provided with a pre-configured Profile
    """
    console_step = (
        console.print_update_step(title=f"Resolving profile {profile}")
        if console
        else None
    )

    if not isinstance(profile, Profile):
        profile = ProfileFactory.create(
            profile, random_seed, constraints, **profile_kwargs
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
    outputs: list[str] | tuple[str],
    output_dir: str | Path | None,
    console: Console | None = None,
) -> dict[str, GenerativeBenchmarkerOutput]:
    """
    Resolve output format specifications into configured output handler instances.

    :param outputs: Specification of desired output files/types
    :param output_dir: Base path for output file generation, or None for default
    :param console: Console instance for progress reporting, or None
    :return: Dictionary mapping format names to configured output handler instances
    """
    console_step = (
        console.print_update_step(title="Resolving output formats") if console else None
    )

    resolved = GenerativeBenchmarkerOutput.resolve(
        outputs=outputs, output_dir=output_dir
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
    args: BenchmarkScenario,
    progress: GenerativeConsoleBenchmarkerProgress | None = None,
    console: Console | None = None,
    **constraints: str | ConstraintInitializer | Any,
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    """
    Execute a comprehensive generative text benchmarking workflow.

    Orchestrates the full benchmarking pipeline by resolving all components from
    provided arguments, executing benchmark runs across configured profiles, and
    finalizing results in specified output formats. Components include backend
    initialization, data loading, profile configuration, and output generation.

    :param args: Scenario configuration for the benchmark execution
    :param progress: Progress tracker for benchmark execution, or None for no tracking
    :param console: Console instance for status reporting, or None for silent operation
    :param constraints: Additional constraint initializers for benchmark limits
    :return: Tuple of GenerativeBenchmarksReport and dictionary of output format
        results
    """
    benchmark_args = args.get_benchmarks()[0]

    backend, model = await resolve_backend(
        backend_args=benchmark_args.backend,
        console=console,
    )
    await resolve_tokenizer(args=benchmark_args, model=model, console=console)
    request_loader = await resolve_request_loader(
        args=benchmark_args,
        console=console,
    )

    warmup = benchmark_args.profile.warmup
    cooldown = benchmark_args.profile.cooldown
    if console:
        console.print_update(
            title="Resolved transient phase configurations",
            details="\n".join(
                [
                    f"Warmup: {warmup}",
                    f"Cooldown: {cooldown}",
                    "Rampup (Throughput/Concurrent): "
                    f"{benchmark_args.profile.rampup_duration}",
                ]
            ),
            status="success",
        )

    constraints = resolve_constraints(benchmark_args, **constraints)
    profile = await resolve_profile(
        profile=benchmark_args.profile,
        constraints=constraints,
        console=console,
        random_seed=benchmark_args.seed.value,  # type: ignore[attr-defined]
        data=benchmark_args.data,
        data_samples=request_loader.info.get("data_samples", -1),
    )
    output_formats = await resolve_output_formats(
        outputs=args.outputs, output_dir=args.output_dir, console=console
    )

    report = GenerativeBenchmarksReport(config=benchmark_args)
    if console:
        console.print_update(
            title="Setup complete, starting benchmarks...", status="success"
        )
        console.print("\n\n")

    benchmarker: Benchmarker[
        GenerativeBenchmark, GenerationRequest, GenerationResponse
    ] = Benchmarker()
    async for benchmark in benchmarker.run(
        accumulator_class=GenerativeBenchmarkAccumulator,
        benchmark_class=GenerativeBenchmark,
        requests=request_loader,  # type: ignore[arg-type]
        backend=backend,
        profile=profile,
        environment=NonDistributedEnvironment(),
        progress=progress,
        sample_requests=None,
        warmup=warmup,
        cooldown=cooldown,
        prefer_response_metrics=True,
    ):
        if benchmark:
            report.benchmarks.append(benchmark)

    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    if console:
        await GenerativeBenchmarkerConsole(console=console).finalize(report)
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
    Load and re-export an existing benchmarks report in specified output formats.

    :param file: Path to the existing benchmark report file to load
    :param output_path: Base path for output file generation, or None for default
    :param output_formats: Specification of desired output formats for the report
    :return: Tuple of loaded GenerativeBenchmarksReport and dictionary of output
        results
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

    resolved_output_formats = await resolve_output_formats(
        output_formats,  # type: ignore[arg-type]
        output_path,
        console=console,
    )
    output_format_results = {}
    for key, output in resolved_output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    for key, value in output_format_results.items():
        console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
