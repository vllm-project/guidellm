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

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

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
    GenerativeMetricsArgs,
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
        model = args.tokenizer.model
        if console_step:
            console_step.finish(
                title=f"Tokenizer resolved, Using tokenizer '{model}' from arguments",
                details=args.tokenizer.model_dump(mode="json"),
                status_level="success",
            )
    else:
        args.tokenizer.model = model
        if console_step:
            console_step.finish(
                title=f"Tokenizer resolved, using model '{model}' as tokenizer",
                details=args.tokenizer.model_dump(mode="json"),
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
    outputs: list[BenchmarkOutputArgs],
    console: Console | None = None,
) -> dict[str, GenerativeBenchmarkerOutput]:
    """
    Resolve output format specifications into configured output handler instances.

    :param outputs: List of BenchmarkOutputArgs specifying output kind and path
    :param console: Console instance for progress reporting, or None
    :return: Dictionary mapping format names to configured output handler instances
    """
    console_step = (
        console.print_update_step(title="Resolving output formats") if console else None
    )

    resolved: dict[str, GenerativeBenchmarkerOutput] = {}
    for output_arg in outputs:
        resolved[output_arg.kind] = GenerativeBenchmarkerOutput.resolve(output_arg)

    if console_step:
        console_step.finish(
            title="Output formats resolved",
            details=[args.model_dump(mode="json") for args in outputs],
            status_level="success",
        )

    return resolved


_SKIP_FIELDS: frozenset[str] = frozenset({"profile", "constraints"})
_PROFILE_RATE_FIELDS: frozenset[str] = frozenset({"rate", "streams"})
_CONSTRAINT_LIST_FIELDS: frozenset[str] = frozenset(
    {
        "seconds",
        "count",
        "rate",
    }
)


def _assert_fields_equal(
    objects: list[Any],
    field_names: Iterable[str],
    skip: frozenset[str],
    context: str,
) -> None:
    """
    Raise :class:`NotImplementedError` if any field in *field_names* differs
    across *objects* (skipping names in *skip*).

    :param objects: Homogeneous list of Pydantic model instances to compare
    :param field_names: Field names to check
    :param skip: Field names to exclude from comparison
    :param context: Label used in the error message (e.g. ``"benchmark"``)
    """
    base = objects[0]
    for field_name in field_names:
        if field_name in skip:
            continue
        base_val = base.__dict__[field_name]
        for other in objects[1:]:
            if other.__dict__[field_name] != base_val:
                raise NotImplementedError(
                    f"Differing {context} field '{field_name}' cannot be merged. "
                    "All sub-benchmarks must share the same value for this field."
                )


def resolve_to_single_benchmark(benchmarks: list[BenchmarkArgs]) -> BenchmarkArgs:
    """
    Collapse multiple benchmark argument sets into a single instance.

    NOTE: This is a temporary workaround to support the new config format with the
    existing initization flow. This method will be removed once we support arbitrarily
    overriding any fields in spec (or at least handle supported overrides better).

    Temporary adapter that bridges the new multi-benchmark configuration format
    to the old single-benchmark internal pipeline.  All fields must be equal
    across the provided benchmarks except for recognised list-capable fields
    (see ``_PROFILE_RATE_FIELDS`` and ``_CONSTRAINT_LIST_FIELDS``), which are
    flattened into a single list.

    :param benchmarks: One or more benchmark argument instances to merge
    :return: A single, merged ``BenchmarkArgs`` ready for execution
    :raises NotImplementedError: If any non-mergeable field differs across benchmarks
    """
    if len(benchmarks) == 1:
        return benchmarks[0]

    # Use this for determining correct field kinds
    # `kind` should not chnage between benchmarks
    base = benchmarks[0]

    _assert_fields_equal(
        benchmarks, BenchmarkArgs.model_fields, _SKIP_FIELDS, "benchmark"
    )

    # BEGIN: profile hack
    profiles = [b.profile for b in benchmarks]
    profile_cls = type(base.profile)
    _assert_fields_equal(
        profiles, profile_cls.model_fields, _PROFILE_RATE_FIELDS, "profile"
    )

    # Get the correct "rate" field for the profile
    rate_field = next(
        (f for f in _PROFILE_RATE_FIELDS if f in profile_cls.model_fields),
        None,
    )
    merged_profile = base.profile
    if rate_field is not None:
        merged_rates: list[Any] = []
        for bench in benchmarks:
            val = bench.profile.__dict__[rate_field]
            if isinstance(val, list | tuple):
                merged_rates.extend(val)
            else:
                merged_rates.append(val)

        profile_dump = base.profile.model_dump()
        profile_dump[rate_field] = merged_rates
        merged_profile = profile_cls.model_validate(profile_dump)
    # END: profile hack

    # BEGIN: constraints hack
    merged_constraints: list[Any] = []
    for idx in range(len(base.constraints)):
        constraints_at_idx = [b.constraints[idx] for b in benchmarks]
        constraint_cls = type(constraints_at_idx[0])
        _assert_fields_equal(
            constraints_at_idx,
            constraint_cls.model_fields,
            _CONSTRAINT_LIST_FIELDS,
            "constraint",
        )

        list_field = next(
            (f for f in _CONSTRAINT_LIST_FIELDS if f in constraint_cls.model_fields),
            None,
        )
        if list_field is None:
            merged_constraints.append(constraints_at_idx[0])
            continue

        merged_values: list[Any] = []
        for constraint in constraints_at_idx:
            val = constraint.__dict__[list_field]
            if isinstance(val, list | tuple):
                merged_values.extend(val)
            else:
                merged_values.append(val)

        constraint_dump = constraints_at_idx[0].model_dump()
        constraint_dump[list_field] = merged_values
        merged_constraints.append(constraint_cls.model_validate(constraint_dump))
    # END: constraints hack

    return base.model_copy(
        update={"profile": merged_profile, "constraints": merged_constraints}
    )


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
    benchmark_args = resolve_to_single_benchmark(args.get_benchmarks())

    metrics_args = benchmark_args.metrics
    if not isinstance(metrics_args, GenerativeMetricsArgs):
        raise TypeError(
            f"Expected GenerativeMetricsArgs for generative text benchmark, "
            f"got {type(metrics_args).__name__}"
        )

    backend, model = await resolve_backend(
        backend_args=benchmark_args.backend,
        console=console,
    )
    await resolve_tokenizer(args=benchmark_args, model=model, console=console)
    request_loader: DataLoader[GenerationRequest] = await create_data_loader(
        loader_config=benchmark_args.data_loader,
        data_config=benchmark_args.data,
        tokenizer_config=benchmark_args.tokenizer,
        column_mapper_config=benchmark_args.data_column_mapper,
        preprocessors_config=benchmark_args.data_preprocessors,
        finalizer_config=benchmark_args.data_finalizer,
        random_seed=benchmark_args.seed.value,  # type: ignore[attr-defined]
        console=console,
    )

    warmup = benchmark_args.profile.warmup
    cooldown = benchmark_args.profile.cooldown

    constraints = resolve_constraints(benchmark_args, **constraints)
    profile = await resolve_profile(
        profile=benchmark_args.profile,
        constraints=constraints,
        console=console,
        random_seed=benchmark_args.seed.value,  # type: ignore[attr-defined]
    )
    output_formats = await resolve_output_formats(
        outputs=benchmark_args.outputs, console=console
    )

    report = GenerativeBenchmarksReport(config=args)
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
        sample_size=metrics_args.sample_size,
        warmup=warmup,
        cooldown=cooldown,
        prefer_response_metrics=metrics_args.prefer_response_metrics,
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
    output_formats: tuple[str, ...] | list[str] = ("console", "json", "csv"),
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    """
    Load and re-export an existing benchmarks report in specified output formats.

    :param file: Path to the existing benchmark report file to load
    :param output_path: Base path for output file generation, or None for default
    :param output_formats: Output format kind strings to resolve and finalize
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

    base_path = Path(output_path) if output_path else Path.cwd()
    output_args: list[BenchmarkOutputArgs] = []
    for fmt in output_formats:
        data: dict[str, Any] = {"kind": fmt}
        data["path"] = base_path / f"benchmarks.{fmt}"
        output_args.append(BenchmarkOutputArgs.model_validate(data))

    output_format_results: dict[str, Any] = {}
    for args in output_args:
        output = GenerativeBenchmarkerOutput.resolve(args)
        output_format_results[args.kind] = await output.finalize(report)

    for key, value in output_format_results.items():
        console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
