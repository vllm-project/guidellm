"""
GuideLLM command-line interface providing benchmarking, dataset preprocessing, and
mock server functionality.

This module serves as the primary entry point for the GuideLLM CLI application,
offering a comprehensive suite of tools for language model evaluation and testing.
It provides three main command groups: benchmark operations for performance testing
against generative models, dataset preprocessing utilities for data preparation and
transformation, and a mock server for testing and development scenarios. The CLI
supports various backends, output formats, and configuration options to accommodate
different benchmarking needs and deployment environments.

Example:
::
    # Run a benchmark against a model
    guidellm benchmark run --target http://localhost:8000 --data dataset.json \\
        --profile sweep

    # Preprocess a dataset
    guidellm preprocess dataset input.json output.json --processor gpt2

    # Start a mock server for testing
    guidellm mock-server --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import codecs
from pathlib import Path
from typing import Annotated, Union

import click

try:
    import uvloop

    HAS_UVLOOP: Annotated[
        bool, "Flag indicating if uvloop is available for event loop optimization"
    ] = True
except ImportError:
    uvloop = None

    HAS_UVLOOP: Annotated[
        bool, "Flag indicating if uvloop is available for event loop optimization"
    ] = False

from guidellm.backends import BackendType
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
from guidellm.mock_server import MockServer, MockServerConfig
from guidellm.preprocess.dataset import ShortPromptStrategy, process_dataset
from guidellm.scheduler import StrategyType
from guidellm.settings import print_config
from guidellm.utils import Console, DefaultGroupHandler, get_literal_vals
from guidellm.utils import cli as cli_tools

__all__ = [
    "STRATEGY_PROFILE_CHOICES",
    "benchmark",
    "cli",
    "config",
    "dataset",
    "decode_escaped_str",
    "from_file",
    "mock_server",
    "preprocess",
    "run",
]

STRATEGY_PROFILE_CHOICES: Annotated[
    list[str], "Available strategy and profile choices for benchmark execution types"
] = list(get_literal_vals(Union[ProfileType, StrategyType]))


def decode_escaped_str(_ctx, _param, value):
    """
    Decode escape sequences in Click option values.

    Click automatically escapes characters in option values, converting sequences
    like "\\n" to "\\\\n". This function properly decodes these escape sequences
    to their intended characters for use in CLI options.

    :param _ctx: Click context (unused)
    :param _param: Click parameter (unused)
    :param value: String value to decode escape sequences from
    :return: Decoded string with proper escape sequences
    :raises click.BadParameter: When escape sequence decoding fails
    """
    if value is None:
        return None
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception as e:
        raise click.BadParameter(f"Could not decode escape sequences: {e}") from e


@click.group()
def cli():
    """
    Main entry point for the GuideLLM command-line interface.

    This is the root command group that organizes all GuideLLM CLI functionality
    into logical subgroups for benchmarking, preprocessing, configuration, and
    mock server operations.
    """


@cli.group(
    help="Commands to run a new benchmark or load a prior one.",
    cls=DefaultGroupHandler,
    default="run",
)
def benchmark():
    """
    Benchmark command group for running and managing performance tests.

    This command group provides functionality to execute new benchmarks against
    generative models and load previously saved benchmark reports for analysis.
    Supports various benchmarking strategies, output formats, and backend types.
    """


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
@click.option(
    "--max-errors",
    type=int,
    default=None,
    help="Maximum number of errors allowed before stopping the benchmark",
)
@click.option(
    "--max-error-rate",
    type=float,
    default=None,
    help="Maximum error rate allowed before stopping the benchmark",
)
@click.option(
    "--max-global-error-rate",
    type=float,
    default=None,
    help="Maximum global error rate allowed across all benchmarks",
)
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
    """
    Execute a generative text benchmark against a target model backend.

    Runs comprehensive performance testing using various strategies and profiles,
    collecting metrics on latency, throughput, error rates, and resource usage.
    Supports multiple backends, data sources, output formats, and constraint types
    for flexible benchmark configuration.
    """
    if HAS_UVLOOP:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
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
    """
    Load and optionally re-export a previously saved benchmark report.

    Imports benchmark results from a saved file and provides optional conversion
    to different output formats. Supports JSON, YAML, and CSV export formats
    based on the output file extension.
    """
    reimport_benchmarks_report(path, output_path)


@cli.command(
    short_help="Prints environment variable settings.",
    help=(
        "Print out the available configuration settings that can be set "
        "through environment variables."
    ),
)
def config():
    """
    Display available GuideLLM configuration environment variables.

    Prints a comprehensive list of all environment variables that can be used
    to configure GuideLLM behavior, including their current values, defaults,
    and descriptions.
    """
    print_config()


@cli.group(help="General preprocessing tools and utilities.")
def preprocess():
    """
    Preprocessing command group for dataset preparation and transformation.

    This command group provides utilities for converting, processing, and
    optimizing datasets for use in GuideLLM benchmarks. Includes functionality
    for token count adjustments, format conversions, and data validation.
    """


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
    """
    Convert and process datasets for specific prompt and output token requirements.

    Transforms datasets to meet target token length specifications using various
    strategies for handling short prompts and output length adjustments. Supports
    multiple input formats and can optionally push results to Hugging Face Hub.
    """
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


@cli.command(help="Start the GuideLLM mock OpenAI/vLLM server for testing.")
@click.option("--host", default="127.0.0.1", help="Host to bind the server to")
@click.option("--port", default=8000, type=int, help="Port to bind the server to")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
@click.option(
    "--model", default="llama-3.1-8b-instruct", help="The name of the model to mock"
)
@click.option("--processor", default=None, help="The processor to use for requests")
@click.option(
    "--request-latency",
    default=3,
    type=float,
    help="Request latency in seconds for non-streaming requests",
)
@click.option(
    "--request-latency-std",
    default=0,
    type=float,
    help=(
        "Request latency standard deviation (normal distribution) "
        "in seconds for non-streaming requests"
    ),
)
@click.option(
    "--ttft-ms",
    default=150,
    type=float,
    help="Time to first token in milliseconds for streaming requests",
)
@click.option(
    "--ttft-ms-std",
    default=0,
    type=float,
    help=(
        "Time to first token standard deviation (normal distribution) in milliseconds"
    ),
)
@click.option(
    "--itl-ms",
    default=10,
    type=float,
    help="Inter token latency in milliseconds for streaming requests",
)
@click.option(
    "--itl-ms-std",
    default=0,
    type=float,
    help=(
        "Inter token latency standard deviation (normal distribution) "
        "in milliseconds for streaming requests"
    ),
)
@click.option(
    "--output-tokens",
    default=128,
    type=int,
    help="Output tokens for streaming requests",
)
@click.option(
    "--output-tokens-std",
    default=0,
    type=float,
    help=(
        "Output tokens standard deviation (normal distribution) for streaming requests"
    ),
)
def mock_server(
    host: str,
    port: int,
    workers: int,
    model: str,
    processor: str | None,
    request_latency: float,
    request_latency_std: float,
    ttft_ms: float,
    ttft_ms_std: float,
    itl_ms: float,
    itl_ms_std: float,
    output_tokens: int,
    output_tokens_std: float,
):
    """
    Start a GuideLLM mock OpenAI/vLLM-compatible server for testing and development.

    Launches a mock server that simulates model inference with configurable latency
    characteristics, token generation patterns, and response timing. Useful for
    testing GuideLLM benchmarks without requiring actual model deployment or for
    development scenarios requiring predictable server behavior.
    """

    config = MockServerConfig(
        host=host,
        port=port,
        workers=workers,
        model=model,
        processor=processor,
        request_latency=request_latency,
        request_latency_std=request_latency_std,
        ttft_ms=ttft_ms,
        ttft_ms_std=ttft_ms_std,
        itl_ms=itl_ms,
        itl_ms_std=itl_ms_std,
        output_tokens=output_tokens,
        output_tokens_std=output_tokens_std,
    )

    server = MockServer(config)
    console = Console()
    console.print_update(
        title="GuideLLM mock server starting...",
        details=f"Listening on http://{host}:{port} for model {model}",
        status="success",
    )
    server.run()


if __name__ == "__main__":
    cli()
