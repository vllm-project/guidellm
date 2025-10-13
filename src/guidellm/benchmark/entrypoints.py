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
from guidellm.benchmark.progress import BenchmarkerProgress, BenchmarkerProgressGroup
from guidellm.benchmark.schemas import GenerativeBenchmark, GenerativeBenchmarksReport
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


# Helper Variables

_CURRENT_WORKING_DIR = Path.cwd()

OutputFormatT = TypeAliasType(
    "OutputFormatT",
    tuple[str, ...]
    | list[str]
    | dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
    | None,
)

ProcessorInputT = TypeAliasType("ProcessorInputT", str | Path | PreTrainedTokenizerBase)

ProgressInputT = TypeAliasType(
    "ProgressInputT", tuple[str, ...] | list[str] | list[BenchmarkerProgress]
)


# Helper Functions


async def resolve_backend(
    backend: BackendType | Backend,
    target: str,
    model: str | None,
    console: Console | None = None,
    **backend_kwargs: dict[str, Any],
) -> tuple[Backend, str | None]:
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


# @validate_call(config={"arbitrary_types_allowed": True})
async def benchmark_generative_text(  # noqa: C901, PLR0915, PLR0912
    # Required
    target: str,
    data: list[Any],
    # Benchmark configuration
    profile: StrategyType | ProfileType | Profile = "sweep",
    rate: float | list[float] | None = None,
    # Backend configuration
    backend: BackendType | Backend = "openai_http",
    backend_kwargs: dict[str, Any] | None = None,
    model: str | None = None,
    # Data configuration
    processor: ProcessorInputT | None = None,
    processor_args: dict[str, Any] | None = None,
    data_args: list[dict[str, Any]] | None = None,
    data_samples: int = -1,
    data_column_mapper: (
        DatasetPreprocessor | dict[str, str] | Literal["generative_column_mapper"]
    ) = "generative_column_mapper",
    data_request_formatter: (
        DatasetPreprocessor | dict[str, str] | str
    ) = "chat_completions",
    data_collator: Callable | Literal["generative"] | None = "generative",
    data_sampler: Sampler[int] | Literal["shuffle"] | None = None,
    data_num_workers: int | None = 1,
    dataloader_kwargs: dict[str, Any] | None = None,
    random_seed: int = 42,
    # Output configuration
    output_path: str | Path | None = _CURRENT_WORKING_DIR,
    output_formats: (
        tuple[str, ...]
        | list[str]
        | dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
        | None
    ) = ("console", "json", "html", "csv"),
    # Updates configuration
    progress: ProgressInputT | None = None,
    print_updates: bool = False,
    # Benchmarker configuration
    benchmark_cls: type[GenerativeBenchmark] = GenerativeBenchmark,
    sample_requests: int | None = 10,
    warmup: float | None = None,
    cooldown: float | None = None,
    # Constraints configuration
    max_seconds: int | float | None = None,
    max_requests: int | None = None,
    max_errors: int | None = None,
    max_error_rate: float | None = None,
    max_global_error_rate: float | None = None,
    **constraints: dict[str, ConstraintInitializer | Any],
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    console = Console(quiet=not print_updates)
    backend, model = await resolve_backend(
        backend=backend,
        target=target,
        model=model,
        console=console,
        **(backend_kwargs or {}),
    )
    processor = await resolve_processor(
        processor=processor, model=model, console=console
    )
    request_loader = await resolve_request_loader(
        data=data,
        model=model,
        data_args=data_args,
        data_samples=data_samples,
        processor=processor,
        processor_args=processor_args,
        data_column_mapper=data_column_mapper,
        data_request_formatter=data_request_formatter,
        data_collator=data_collator,
        data_sampler=data_sampler,
        data_num_workers=data_num_workers,
        random_seed=random_seed,
        console=console,
        **(dataloader_kwargs or {}),
    )
    profile = await resolve_profile(
        profile=profile,
        rate=rate,
        random_seed=random_seed,
        constraints=constraints,
        max_seconds=max_seconds,
        max_requests=max_requests,
        max_errors=max_errors,
        max_error_rate=max_error_rate,
        max_global_error_rate=max_global_error_rate,
        console=console,
    )
    output_formats = await resolve_output_formats(
        output_formats=output_formats, output_path=output_path, console=console
    )

    progress_group = BenchmarkerProgressGroup(
        instances=progress or [], enabled=bool(progress)
    )
    report = GenerativeBenchmarksReport()
    console.print_update(
        title="Setup complete, starting benchmarks...", status="success"
    )
    console.print("\n\n")

    async for (
        _aggregator_update,
        benchmark,
        _strategy,
        _scheduler_state,
    ) in progress_group(
        profile,
        Benchmarker[
            GenerativeBenchmark,
            GenerationRequest,
            GenerationResponse,
        ]().run(
            benchmark_class=benchmark_cls,
            requests=request_loader,
            backend=backend,
            profile=profile,
            environment=NonDistributedEnvironment(),
            sample_requests=sample_requests,
            warmup=warmup,
            cooldown=cooldown,
            prefer_response_metrics=True,
        ),
    ):
        if benchmark:
            report.benchmarks.append(benchmark)

    output_format_results = {}
    for key, output in output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result

    console.print("\n\n")
    console.print_update(
        title=f"Benchmarking complete, generated {len(report.benchmarks)} benchmark(s)",
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
    The command-line entry point for re-importing and displaying an
    existing benchmarks report. Can also specify
    Assumes the file provided exists.
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
