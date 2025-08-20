import asyncio
import codecs
from pathlib import Path
from typing import Union

import click

from guidellm.backend import BackendType
from guidellm.benchmark import (
    GenerativeConsoleBenchmarkerProgress,
    InjectExtrasAggregator,
    ProfileType,
    benchmark_generative_text,
    reimport_benchmarks_report,
)
from guidellm.benchmark.scenario import (
    GenerativeTextScenario,
)
from guidellm.preprocess.dataset import ShortPromptStrategy, process_dataset
from guidellm.scheduler import StrategyType
from guidellm.settings import print_config
from guidellm.utils import DefaultGroupHandler, get_literal_vals
from guidellm.utils import cli as cli_tools

STRATEGY_PROFILE_CHOICES = list(get_literal_vals(Union[ProfileType, StrategyType]))


@click.group()
@click.version_option(package_name="guidellm", message="guidellm version: %(version)s")
def cli():
    pass


@cli.group(
    help="Commands to run a new benchmark or load a prior one.",
    cls=DefaultGroupHandler,
    default="run",
)
def benchmark():
    pass


@benchmark.command(
    "run",
    help="Run a benchmark against a generative model using the specified arguments.",
    context_settings={"auto_envvar_prefix": "GUIDELLM"},
)
@click.option(
    "--target",
    type=str,
    help="The target path for the backend to run benchmarks against. For example, http://localhost:8000",
)
@click.option(
    "--data",
    type=str,
    help=(
        "The HuggingFace dataset ID, a path to a HuggingFace dataset, "
        "a path to a data file csv, json, jsonl, or txt, "
        "or a synthetic data config as a json or key=value string."
    ),
)
@click.option(
    "--profile",
    "--rate-type",  # legacy alias
    "profile",
    type=click.Choice(STRATEGY_PROFILE_CHOICES),
    help=(
        "The type of benchmark to run. "
        f"Supported types {', '.join(STRATEGY_PROFILE_CHOICES)}. "
    ),
)
@click.option(
    "--rate",
    default=None,
    help=(
        "The rates to run the benchmark at. "
        "Can be a single number or a comma-separated list of numbers. "
        "For rate-type=sweep, this is the number of benchmarks it runs in the sweep. "
        "For rate-type=concurrent, this is the number of concurrent requests. "
        "For rate-type=async,constant,poisson, this is the rate requests per second. "
        "For rate-type=synchronous,throughput, this must not be set."
    ),
)
@click.option(
    "--random-seed",
    default=GenerativeTextScenario.get_default("random_seed"),
    type=int,
    help="The random seed to use for benchmarking to ensure reproducibility.",
)
# Backend configuration
@click.option(
    "--backend",
    "--backend-type",  # legacy alias
    "backend",
    type=click.Choice(list(get_literal_vals(BackendType))),
    help=(
        "The type of backend to use to run requests against. Defaults to 'openai_http'."
        f" Supported types: {', '.join(get_literal_vals(BackendType))}"
    ),
    default="openai_http",
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=cli_tools.parse_json,
    default=None,
    help=(
        "A JSON string containing any arguments to pass to the backend as a "
        "dict with **kwargs. Headers can be removed by setting their value to "
        "null. For example: "
        """'{"headers": {"Authorization": null, "Custom-Header": "Custom-Value"}}'"""
    ),
)
@click.option(
    "--model",
    default=None,
    type=str,
    help=(
        "The ID of the model to benchmark within the backend. "
        "If None provided (default), then it will use the first model available."
    ),
)
# Data configuration
@click.option(
    "--processor",
    default=None,
    type=str,
    help=(
        "The processor or tokenizer to use to calculate token counts for statistics "
        "and synthetic data generation. If None provided (default), will load "
        "using the model arg, if needed."
    ),
)
@click.option(
    "--processor-args",
    default=None,
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the processor constructor "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--data-args",
    default=None,
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the dataset creation "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--data-sampler",
    default=None,
    type=click.Choice(["random"]),
    help=(
        "The data sampler type to use. 'random' will add a random shuffle on the data. "
        "Defaults to None"
    ),
)
# Output configuration
@click.option(
    "--output-path",
    type=click.Path(),
    default=Path.cwd(),
    help=(
        "The path to save the output formats to, if the format is a file type. "
        "If it is a directory, it will save all output formats selected under it. "
        "If it is a file, it will save the corresponding output format to that file. "
        "Any output formats that were given that do not match the file extension will "
        "be saved in the parent directory of the file path. "
        "Defaults to the current working directory. "
    ),
)
@click.option(
    "--output-formats",
    multiple=True,
    type=str,
    default=("console", "json"),  # ("console", "json", "html", "csv")
    help=(
        "The output formats to use for the benchmark results. "
        "Defaults to console, json, html, and csv where the file formats "
        "will be saved at the specified output path."
    ),
)
@click.option(
    "--disable-console-outputs",
    is_flag=True,
    help="Set this flag to disable console output",
)
# Updates configuration
@click.option(
    "--disable-progress",
    is_flag=True,
    help="Set this flag to disable progress updates to the console",
)
@click.option(
    "--display-scheduler-stats",
    is_flag=True,
    help="Set this flag to display stats for the processes running the benchmarks",
)
# Aggregators configuration
@click.option(
    "--output-extras",
    callback=cli_tools.parse_json,
    help="A JSON string of extra data to save with the output benchmarks",
)
@click.option(
    "--warmup",
    "--warmup-percent",  # legacy alias
    "warmup",
    type=float,
    default=None,
    help=(
        "The specification around the number of requests to run before benchmarking. "
        "If within (0, 1), then the percent of requests/time to use for warmup. "
        "If >=1, then the number of requests or seconds to use for warmup."
        "Whether it's requests/time used is dependent on which constraint is active. "
        "Default None for no warmup."
    ),
)
@click.option(
    "--cooldown",
    "--cooldown-percent",  # legacy alias
    "cooldown",
    type=float,
    default=GenerativeTextScenario.get_default("cooldown_percent"),
    help=(
        "The specification around the number of requests to run after benchmarking. "
        "If within (0, 1), then the percent of requests/time to use for cooldown. "
        "If >=1, then the number of requests or seconds to use for cooldown."
        "Whether it's requests/time used is dependent on which constraint is active. "
        "Default None for no cooldown."
    ),
)
@click.option(
    "--request-samples",
    "--output-sampling",  # legacy alias
    "request_samples",
    type=int,
    help=(
        "The number of samples for each request status and each benchmark to save "
        "in the output file. If None (default), will save all samples. "
        "Defaults to 20."
    ),
    default=20,
)
# Constraints configuration
@click.option(
    "--max-seconds",
    type=float,
    default=None,
    help=(
        "The maximum number of seconds each benchmark can run for. "
        "If None, will run until max_requests or the data is exhausted."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=None,
    help=(
        "The maximum number of requests each benchmark can run for. "
        "If None, will run until max_seconds or the data is exhausted."
    ),
)
@click.option("--max-errors", type=int, default=None, help="")
@click.option("--max-error-rate", type=float, default=None, help="")
@click.option("--max-global-error-rate", type=float, default=None, help="")
def run(
    target,
    data,
    profile,
    rate,
    random_seed,
    # Backend Configuration
    backend,
    backend_kwargs,
    model,
    # Data configuration
    processor,
    processor_args,
    data_args,
    data_sampler,
    # Output configuration
    output_path,
    output_formats,
    # Updates configuration
    disable_console_outputs,
    disable_progress,
    display_scheduler_stats,
    # Aggregators configuration
    output_extras,
    warmup,
    cooldown,
    request_samples,
    # Constraints configuration
    max_seconds,
    max_requests,
    max_errors,
    max_error_rate,
    max_global_error_rate,
):
    asyncio.run(
        benchmark_generative_text(
            target=target,
            data=data,
            profile=profile,
            rate=rate,
            random_seed=random_seed,
            # Backend configuration
            backend=backend,
            backend_kwargs=backend_kwargs,
            model=model,
            # Data configuration
            processor=processor,
            processor_args=processor_args,
            data_args=data_args,
            data_sampler=data_sampler,
            # Output configuration
            output_path=output_path,
            output_formats=[
                fmt
                for fmt in output_formats
                if not disable_console_outputs or fmt != "console"
            ],
            # Updates configuration
            progress=(
                [
                    GenerativeConsoleBenchmarkerProgress(
                        display_scheduler_stats=display_scheduler_stats
                    )
                ]
                if not disable_progress
                else None
            ),
            print_updates=not disable_console_outputs,
            # Aggregators configuration
            add_aggregators={"extras": InjectExtrasAggregator(extras=output_extras)},
            warmup=warmup,
            cooldown=cooldown,
            request_samples=request_samples,
            # Constraints configuration
            max_seconds=max_seconds,
            max_requests=max_requests,
            max_errors=max_errors,
            max_error_rate=max_error_rate,
            max_global_error_rate=max_global_error_rate,
        )
    )


@benchmark.command("from-file", help="Load a saved benchmark report.")
@click.argument(
    "path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    default=Path.cwd() / "benchmarks.json",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=True, dir_okay=True, exists=False),
    default=None,
    is_flag=False,
    flag_value=Path.cwd() / "benchmarks_reexported.json",
    help=(
        "Allows re-exporting the benchmarks to another format. "
        "The path to save the output to. If it is a directory, "
        "it will save benchmarks.json under it. "
        "Otherwise, json, yaml, or csv files are supported for output types "
        "which will be read from the extension for the file path. "
        "This input is optional. If the output path flag is not provided, "
        "the benchmarks will not be reexported. If the flag is present but "
        "no value is specified, it will default to the current directory "
        "with the file name `benchmarks_reexported.json`."
    ),
)
def from_file(path, output_path):
    reimport_benchmarks_report(path, output_path)


def decode_escaped_str(_ctx, _param, value):
    """
    Click auto adds characters. For example, when using --pad-char "\n",
    it parses it as "\\n". This method decodes the string to handle escape
    sequences correctly.
    """
    if value is None:
        return None
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception as e:
        raise click.BadParameter(f"Could not decode escape sequences: {e}") from e


@cli.command(
    short_help="Prints environment variable settings.",
    help=(
        "Print out the available configuration settings that can be set "
        "through environment variables."
    ),
)
def config():
    print_config()


@cli.group(help="General preprocessing tools and utilities.")
def preprocess():
    pass


@preprocess.command(
    help=(
        "Convert a dataset to have specific prompt and output token sizes.\n"
        "DATA: Path to the input dataset or dataset ID.\n"
        "OUTPUT_PATH: Path to save the converted dataset, including file suffix."
    ),
    context_settings={"auto_envvar_prefix": "GUIDELLM"},
)
@click.argument(
    "data",
    type=str,
    required=True,
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True,
)
@click.option(
    "--processor",
    type=str,
    required=True,
    help=(
        "The processor or tokenizer to use to calculate token counts for statistics "
        "and synthetic data generation."
    ),
)
@click.option(
    "--processor-args",
    default=None,
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the processor constructor "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--data-args",
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the dataset creation "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--short-prompt-strategy",
    type=click.Choice([s.value for s in ShortPromptStrategy]),
    default=ShortPromptStrategy.IGNORE.value,
    show_default=True,
    help="Strategy to handle prompts shorter than the target length. ",
)
@click.option(
    "--pad-char",
    type=str,
    default="",
    callback=decode_escaped_str,
    help="The token to pad short prompts with when using the 'pad' strategy.",
)
@click.option(
    "--concat-delimiter",
    type=str,
    default="",
    help=(
        "The delimiter to use when concatenating prompts that are too short."
        " Used when strategy is 'concatenate'."
    ),
)
@click.option(
    "--prompt-tokens",
    type=str,
    default=None,
    help="Prompt tokens config (JSON, YAML file or key=value string)",
)
@click.option(
    "--output-tokens",
    type=str,
    default=None,
    help="Output tokens config (JSON, YAML file or key=value string)",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Set this flag to push the converted dataset to the Hugging Face Hub.",
)
@click.option(
    "--hub-dataset-id",
    type=str,
    default=None,
    help="The Hugging Face Hub dataset ID to push to. "
    "Required if --push-to-hub is used.",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for prompt token sampling and output tokens sampling.",
)
def dataset(
    data,
    output_path,
    processor,
    processor_args,
    data_args,
    short_prompt_strategy,
    pad_char,
    concat_delimiter,
    prompt_tokens,
    output_tokens,
    push_to_hub,
    hub_dataset_id,
    random_seed,
):
    process_dataset(
        data=data,
        output_path=output_path,
        processor=processor,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        processor_args=processor_args,
        data_args=data_args,
        short_prompt_strategy=short_prompt_strategy,
        pad_char=pad_char,
        concat_delimiter=concat_delimiter,
        push_to_hub=push_to_hub,
        hub_dataset_id=hub_dataset_id,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    cli()
