from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from guidellm.backends import (
    Backend,
    BackendType,
    GenerationRequest,
    GenerationResponse,
)
from guidellm.benchmark.aggregator import (
    GenerativeRequestsAggregator,
    GenerativeStatsProgressAggregator,
    SchedulerStatsAggregator,
    SerializableAggregator,
)
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.objects import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.benchmark.output import (
    GenerativeBenchmarkerOutput,
)
from guidellm.benchmark.profile import Profile, ProfileType
from guidellm.benchmark.progress import BenchmarkerProgressGroup
from guidellm.benchmark.scenario import enable_scenarios
from guidellm.benchmark.types import (
    OutputFormatType,
    DataInputType,
    ProcessorInputType,
    ProgressInputType,
    AggregatorInputType
)
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import (
    ConstraintInitializer,
    NonDistributedEnvironment,
    StrategyType,
)
from guidellm.utils import Console, InfoMixin

__all__ = [
    "benchmark_generative_text",
    "reimport_benchmarks_report",
]


_CURRENT_WORKING_DIR = Path.cwd()


# Helper functions

async def initialize_backend(
    backend: BackendType | Backend,
    target: str,
    model: str | None,
    backend_kwargs: dict[str, Any] | None,
) -> Backend:
    backend = (
        Backend.create(
            backend, target=target, model=model, **(backend_kwargs or {})
        )
        if not isinstance(backend, Backend)
        else backend
    )
    await backend.process_startup()
    await backend.validate()
    return backend


async def resolve_profile(
    constraint_inputs: dict[str, int | float],
    profile: Profile | str | None,
    rate: list[float] | None,
    random_seed: int,
    constraints: dict[str, ConstraintInitializer | Any],
):
    for key, val in constraint_inputs.items():
        if val is not None:
            constraints[key] = val
    if not isinstance(profile, Profile):
        if isinstance(profile, str):
            profile = Profile.create(
                rate_type=profile,
                rate=rate,
                random_seed=random_seed,
                constraints={**constraints},
            )
        else:
            raise ValueError(f"Expected string for profile; got {type(profile)}")

    elif constraints:
        raise ValueError(
            "Constraints must be empty when providing a Profile instance. "
            f"Provided constraints: {constraints} ; provided profile: {profile}"
        )
    return profile

async def resolve_output_formats(
    output_formats: OutputFormatType,
    output_path: str | Path | None,
) -> dict[str, GenerativeBenchmarkerOutput]:
    output_formats = GenerativeBenchmarkerOutput.resolve(
        output_formats=(output_formats or {}), output_path=output_path
    )
    return output_formats

async def finalize_outputs(
    report: GenerativeBenchmarksReport,
    resolved_output_formats: dict[str, GenerativeBenchmarkerOutput]
):
    output_format_results = {}
    for key, output in resolved_output_formats.items():
        output_result = await output.finalize(report)
        output_format_results[key] = output_result
    return output_format_results


# Complete entrypoints


# @validate_call(config={"arbitrary_types_allowed": True})
@enable_scenarios
async def benchmark_generative_text(  # noqa: C901
    target: str,
    data: DataInputType,
    profile: StrategyType | ProfileType | Profile,
    rate: list[float] | None = None,
    random_seed: int = 42,
    # Backend configuration
    backend: BackendType | Backend = "openai_http",
    backend_kwargs: dict[str, Any] | None = None,
    model: str | None = None,
    # Data configuration
    processor: ProcessorInputType | None = None,
    processor_args: dict[str, Any] | None = None,
    data_args: dict[str, Any] | None = None,
    data_sampler: Literal["random"] | None = None,
    # Output configuration
    output_path: str | Path | None = _CURRENT_WORKING_DIR,
    output_formats: OutputFormatType = ("console", "json", "html", "csv"),
    # Updates configuration
    progress: ProgressInputType | None = None,
    print_updates: bool = False,
    # Aggregators configuration
    add_aggregators: AggregatorInputType | None = None,
    warmup: float | None = None,
    cooldown: float | None = None,
    request_samples: int | None = 20,
    # Constraints configuration
    max_seconds: int | float | None = None,
    max_requests: int | None = None,
    max_errors: int | None = None,
    max_error_rate: float | None = None,
    max_global_error_rate: float | None = None,
    **constraints: dict[str, ConstraintInitializer | Any],
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    console = Console(quiet=not print_updates)

    with console.print_update_step(
        title=f"Initializing backend {backend}"
    ) as console_step:
        backend = await initialize_backend(backend, target, model, backend_kwargs)
        console_step.finish(
            title=f"{backend.__class__.__name__} backend initialized",
            details=backend.info,
            status_level="success",
        )

    with console.print_update_step(title="Resolving processor") as console_step:
        if processor is not None:
            console_step.finish(
                title="Processor resolved",
                details=f"Using processor '{processor}'",
                status_level="success",
            )
        elif model is not None:
            console_step.finish(
                title="Processor resolved",
                details=f"Using model '{model}' as processor",
                status_level="success",
            )
            processor = model
        else:
            console_step.update(
                title="Resolving processor from backend.default_model",
                status_level="info",
            )
            processor = await backend.default_model()
            console_step.finish(
                title="Processor resolved",
                details=(
                    f"Using model '{processor}' from backend "
                    f"{backend.__class__.__name__} as processor"
                ),
                status_level="success",
            )
        await backend.process_shutdown()

    with console.print_update_step(
        title=f"Initializing request loader from {data}"
    ) as console_step:
        request_loader = GenerativeRequestLoader(
            data=data,
            data_args=data_args,
            processor=processor,
            processor_args=processor_args,
            shuffle=data_sampler == "random",
            random_seed=random_seed,
        )
        unique_requests = request_loader.num_unique_items(raise_err=False)
        console_step.finish(
            title=(
                f"Request loader initialized with {unique_requests} unique requests "
                f"from {data}"
            ),
            details=InfoMixin.extract_from_obj(request_loader),
            status_level="success",
        )

    with console.print_update_step(
        title=f"Resolving profile {profile}"
    ) as console_step:
        profile = await resolve_profile(
            {
                "max_seconds": max_seconds,
                "max_requests": max_requests,
                "max_errors": max_errors,
                "max_error_rate": max_error_rate,
                "max_global_error_rate": max_global_error_rate,
            },
            profile,
            rate,
            random_seed,
            constraints,
        )
        console_step.finish(
            title=f"{profile.__class__.__name__} profile resolved",
            details=InfoMixin.extract_from_obj(profile),
            status_level="success",
        )

    with console.print_update_step(
        title="Creating benchmark aggregators"
    ) as console_step:
        aggregators = {
            "scheduler_stats": SchedulerStatsAggregator(),
            "requests_progress": GenerativeStatsProgressAggregator(),
            "requests": GenerativeRequestsAggregator(
                request_samples=request_samples,
                warmup=warmup,
                cooldown=cooldown,
            ),
            **SerializableAggregator.resolve(add_aggregators or {}),
        }
        console_step.finish(
            title="Benchmark aggregators created",
            details={key: str(val) for key, val in aggregators.items()},
            status_level="success",
        )

    with console.print_update_step(title="Resolving output formats") as console_step:
        resolved_output_formats = await resolve_output_formats(output_formats, output_path)
        console_step.finish(
            title="Output formats resolved",
            details={key: str(val) for key, val in resolved_output_formats.items()},
            status_level="success",
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
            requests=request_loader,
            backend=backend,
            profile=profile,
            environment=NonDistributedEnvironment(),
            benchmark_aggregators=aggregators,
            benchmark_class=GenerativeBenchmark,
        ),
    ):
        if benchmark:
            report.benchmarks.append(benchmark)

    output_format_results = await finalize_outputs(report, resolved_output_formats)

    console.print("\n\n")
    console.print_update(
        title=f"Benchmarking complete; generated {len(report.benchmarks)} benchmark(s)",
        status="success",
    )
    for key, value in output_format_results.items():
        console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results


async def reimport_benchmarks_report(
    file: Path,
    output_path: Path | None,
    output_formats: OutputFormatType = ("console", "json", "html", "csv"),
) -> tuple[GenerativeBenchmarksReport, dict[str, Any]]:
    """
    The command-line entry point for re-importing and displaying an
    existing benchmarks report. Can also specify an output format.
    Assumes the file provided exists.
    """
    console = Console()
    with console.print_update_step(
        title=f"Loading benchmarks from {file}"
    ) as console_step:
        report = GenerativeBenchmarksReport.load_file(file)
        console_step.finish(f"Import of old benchmarks complete; loaded {len(report.benchmarks)} benchmark(s)")

    with console.print_update_step(title="Resolving output formats") as console_step:
        resolved_output_formats = await resolve_output_formats(output_formats, output_path)
        console_step.finish(
            title="Output formats resolved",
            details={key: str(val) for key, val in resolved_output_formats.items()},
            status_level="success",
        )

    output_format_results = await finalize_outputs(report, resolved_output_formats)

    for key, value in output_format_results.items():
        console.print_update(title=f"  {key:<8}: {value}", status="debug")

    return report, output_format_results
