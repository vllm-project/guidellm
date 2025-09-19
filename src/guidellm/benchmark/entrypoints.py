from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.backend import Backend, BackendType
from guidellm.benchmark.benchmarker import GenerativeBenchmarker
from guidellm.benchmark.output import (
    GenerativeBenchmarksConsole,
    GenerativeBenchmarksReport,
)
from guidellm.benchmark.profile import ProfileType, create_profile
from guidellm.benchmark.progress import GenerativeTextBenchmarkerProgressDisplay
from guidellm.benchmark.scenario import GenerativeTextScenario, Scenario
from guidellm.request import GenerativeRequestLoader
from guidellm.scheduler import StrategyType


async def benchmark_with_scenario(scenario: Scenario, **kwargs):
    """
    Run a benchmark using a scenario and specify any extra arguments
    """

    if isinstance(scenario, GenerativeTextScenario):
        return await benchmark_generative_text(**vars(scenario), **kwargs)
    else:
        raise ValueError(f"Unsupported Scenario type {type(scenario)}")


async def benchmark_generative_text(
    target: str,
    backend_type: BackendType,
    backend_args: Optional[dict[str, Any]],
    model: Optional[str],
    processor: Optional[Optional[Union[str, Path, PreTrainedTokenizerBase]]],
    processor_args: Optional[dict[str, Any]],
    data: Union[
        str,
        Path,
        Iterable[Union[str, dict[str, Any]]],
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
    ],
    data_args: Optional[dict[str, Any]],
    data_sampler: Optional[Literal["random"]],
    rate_type: Union[StrategyType, ProfileType],
    rate: Optional[Union[float, list[float]]],
    max_seconds: Optional[float],
    max_requests: Optional[int],
    warmup_percent: Optional[float],
    cooldown_percent: Optional[float],
    output_path: Optional[Union[str, Path]],
    output_extras: Optional[dict[str, Any]],
    output_sampling: Optional[int],
    random_seed: int,
    show_progress: bool = True,
    show_progress_scheduler_stats: bool = False,
    output_console: bool = True,
) -> tuple[GenerativeBenchmarksReport, Optional[Path]]:
    console = GenerativeBenchmarksConsole(enabled=show_progress)
    console.print_line("Creating backend...")
    backend = Backend.create(
        backend_type, target=target, model=model, **(backend_args or {})
    )
    await backend.validate()
    console.print_line(
        f"Backend {backend_type} connected to {target} for model {backend.model}."
    )

    if processor is None:
        processor = backend.model

    console.print_line("Creating request loader...")
    request_loader = GenerativeRequestLoader(
        data=data,
        data_args=data_args,
        processor=processor,
        processor_args=processor_args,
        shuffle=data_sampler == "random",
        iter_type=(
            "finite"  # assume a finite dataset is our limit
            if max_requests is None and max_seconds is None
            else "infinite"  # default to infinite so we don't run out of data
        ),
        random_seed=random_seed,
    )
    unique_requests = request_loader.num_unique_items(raise_err=False)
    console.print_line(
        f"Created loader with {unique_requests} unique requests from {data}.\n\n"
        if unique_requests > 0
        else f"Created loader with unknown number unique requests from {data}.\n\n"
    )

    profile = create_profile(rate_type=rate_type, rate=rate)
    benchmarker = GenerativeBenchmarker(
        backend=backend,
        request_loader=request_loader,
        request_loader_description=request_loader.description,
        benchmark_save_extras=output_extras,
        processor=processor,
        processor_args=processor_args,
    )
    progress = (
        GenerativeTextBenchmarkerProgressDisplay(
            display_scheduler_stats=show_progress_scheduler_stats
        )
        if show_progress
        else None
    )
    report = GenerativeBenchmarksReport()

    async for result in benchmarker.run(
        profile=profile,
        max_number_per_strategy=max_requests,
        max_duration_per_strategy=max_seconds,
        warmup_percent_per_strategy=warmup_percent,
        cooldown_percent_per_strategy=cooldown_percent,
    ):
        if progress:
            progress.update(result)

        if result.type_ == "benchmark_compiled":
            if result.current_benchmark is None:
                raise ValueError("Current benchmark is None")
            report.benchmarks.append(
                result.current_benchmark.set_sample_size(output_sampling)
            )

    if output_console:
        console.benchmarks = report.benchmarks
        console.print_full_report()

    if output_path:
        console.print_line("\nSaving benchmarks report...")
        saved_path = report.save_file(output_path)
        console.print_line(f"Benchmarks report saved to {saved_path}")
    else:
        saved_path = None

    console.print_line("\nBenchmarking complete.")

    return report, saved_path


def reimport_benchmarks_report(file: Path, output_path: Optional[Path]) -> None:
    """
    The command-line entry point for re-importing and displaying an
    existing benchmarks report. Can also specify
    Assumes the file provided exists.
    """
    console = GenerativeBenchmarksConsole(enabled=True)
    report = GenerativeBenchmarksReport.load_file(file)
    console.benchmarks = report.benchmarks
    console.print_full_report()

    if output_path:
        console.print_line("\nSaving benchmarks report...")
        saved_path = report.save_file(output_path)
        console.print_line(f"Benchmarks report saved to {saved_path}")
