"""
GuideLLM command-line interface entry point.

Primary CLI application providing benchmark execution, dataset preprocessing, and
mock server functionality for language model evaluation. Organizes commands into
three main groups: benchmark operations for performance testing, preprocessing
utilities for data transformation, and mock server capabilities for development
and testing. Supports multiple backends, output formats, and flexible configuration
through CLI options and environment variables.

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

import click
from pydantic import ValidationError

try:
    import uvloop
except ImportError:
    uvloop = None # type: ignore[assignment] # Optional dependency

from guidellm.backends import BackendType
from guidellm.benchmark import (
    BenchmarkGenerativeTextArgs,
    GenerativeConsoleBenchmarkerProgress,
    ProfileType,
    benchmark_generative_text,
    get_builtin_scenarios,
    reimport_benchmarks_report,
)
from guidellm.mock_server import MockServer, MockServerConfig
from guidellm.preprocess.dataset import ShortPromptStrategy, process_dataset
from guidellm.scheduler import StrategyType
from guidellm.schemas import GenerativeRequestType
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

STRATEGY_PROFILE_CHOICES: list[str] = list(get_literal_vals(ProfileType | StrategyType))
"""Available strategy and profile type choices for benchmark execution."""


def decode_escaped_str(_ctx, _param, value):
    """
    Decode escape sequences in Click option values.

    Click automatically escapes characters converting sequences like "\\n" to
    "\\\\n". This function decodes these sequences to their intended characters.

    :param _ctx: Click context (unused)
    :param _param: Click parameter (unused)
    :param value: String value to decode
    :return: Decoded string with proper escape sequences, or None if input is None
    :raises click.BadParameter: When escape sequence decoding fails
    """
    if value is None:
        return None
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception as e:
        raise click.BadParameter(f"Could not decode escape sequences: {e}") from e


@click.group()
@click.version_option(package_name="guidellm", message="guidellm version: %(version)s")
def cli():
    """GuideLLM CLI for benchmarking, preprocessing, and testing language models."""


@cli.group(
    help="Run a benchmark or load a previously saved benchmark report.",
    cls=DefaultGroupHandler,
    default="run",
)
def benchmark():
    """Benchmark commands for performance testing generative models."""


@benchmark.command(
    "run",
    help=(
        "Run a benchmark against a generative model. "
        "Supports multiple backends, data sources, strategies, and output formats. "
        "Configuration can be loaded from a scenario file or specified via options."
    ),
    context_settings={"auto_envvar_prefix": "GUIDELLM"},
)
@click.option(
    "--scenario",
    type=cli_tools.Union(
        click.Path(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            path_type=Path,
        ),
        click.Choice(tuple(get_builtin_scenarios().keys())),
    ),
    default=None,
    help=(
        "Builtin scenario name or path to config file. "
        "CLI options override scenario settings."
    ),
)
@click.option(
    "--target",
    type=str,
    help="Target backend URL (e.g., http://localhost:8000).",
)
@click.option(
    "--data",
    type=str,
    multiple=True,
    help=(
        "HuggingFace dataset ID, path to dataset, path to data file "
        "(csv/json/jsonl/txt), or synthetic data config (json/key=value)."
    ),
)
@click.option(
    "--profile",
    "--rate-type",  # legacy alias
    "profile",
    default=BenchmarkGenerativeTextArgs.get_default("profile"),
    type=click.Choice(STRATEGY_PROFILE_CHOICES),
    help=f"Benchmark profile type. Options: {', '.join(STRATEGY_PROFILE_CHOICES)}.",
)
@click.option(
    "--rate",
    type=str,
    callback=cli_tools.parse_list_floats,
    multiple=False,
    default=BenchmarkGenerativeTextArgs.get_default("rate"),
    help=(
        "Benchmark rate(s) to test. Meaning depends on profile: "
        "sweep=number of benchmarks, concurrent=concurrent requests, "
        "async/constant/poisson=requests per second."
    ),
)
# Backend configuration
@click.option(
    "--backend",
    "--backend-type",  # legacy alias
    "backend",
    type=click.Choice(list(get_literal_vals(BackendType))),
    default=BenchmarkGenerativeTextArgs.get_default("backend"),
    help=f"Backend type. Options: {', '.join(get_literal_vals(BackendType))}.",
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=cli_tools.parse_json,
    default=BenchmarkGenerativeTextArgs.get_default("backend_kwargs"),
    help="JSON string of arguments to pass to the backend.",
)
@click.option(
    "--model",
    default=BenchmarkGenerativeTextArgs.get_default("model"),
    type=str,
    help="Model ID to benchmark. If not provided, uses first available model.",
)
# Data configuration
@click.option(
    "--request-type",
    default=BenchmarkGenerativeTextArgs.get_default("data_request_formatter"),
    type=click.Choice(list(get_literal_vals(GenerativeRequestType))),
    help=(
        f"Request type to create for each data sample. "
        f"Options: {', '.join(get_literal_vals(GenerativeRequestType))}."
    ),
)
@click.option(
    "--request-formatter-kwargs",
    default=None,
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the request formatter.",
)
@click.option(
    "--processor",
    default=BenchmarkGenerativeTextArgs.get_default("processor"),
    type=str,
    help=(
        "Processor or tokenizer for token count calculations. "
        "If not provided, loads from model."
    ),
)
@click.option(
    "--processor-args",
    default=BenchmarkGenerativeTextArgs.get_default("processor_args"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("data_args"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--data-samples",
    default=BenchmarkGenerativeTextArgs.get_default("data_samples"),
    type=int,
    help=(
        "Number of samples from dataset. -1 (default) uses all samples "
        "and dynamically generates more."
    ),
)
@click.option(
    "--data-column-mapper",
    default=BenchmarkGenerativeTextArgs.get_default("data_column_mapper"),
    callback=cli_tools.parse_json,
    help="JSON string of column mappings to apply to the dataset.",
)
@click.option(
    "--data-sampler",
    default=BenchmarkGenerativeTextArgs.get_default("data_sampler"),
    type=click.Choice(["shuffle"]),
    help="Data sampler type.",
)
@click.option(
    "--data-num-workers",
    default=BenchmarkGenerativeTextArgs.get_default("data_num_workers"),
    type=int,
    help="Number of worker processes for data loading.",
)
@click.option(
    "--dataloader_kwargs",
    default=BenchmarkGenerativeTextArgs.get_default("dataloader_kwargs"),
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the dataloader constructor.",
)
@click.option(
    "--random-seed",
    default=BenchmarkGenerativeTextArgs.get_default("random_seed"),
    type=int,
    help="Random seed for reproducibility.",
)
# Output configuration
@click.option(
    "--output-path",
    type=click.Path(),
    default=BenchmarkGenerativeTextArgs.get_default("output_path"),
    help=(
        "Path to save output files. Can be a directory or file. "
        "If a file, saves that format; mismatched formats save to parent directory."
    ),
)
@click.option(
    "--output-formats",
    multiple=True,
    type=str,
    default=BenchmarkGenerativeTextArgs.get_default("output_formats"),
    help="Output formats for results (e.g., console, json, html, csv).",
)
@click.option(
    "--disable-console-outputs",
    is_flag=True,
    help="Disable console output.",
)
# Updates configuration
@click.option(
    "--disable-progress",
    is_flag=True,
    help="Disable progress updates to the console.",
)
@click.option(
    "--display-scheduler-stats",
    is_flag=True,
    help="Display scheduler process statistics.",
)
# Aggregators configuration
@click.option(
    "--warmup",
    "--warmup-percent",  # legacy alias
    "warmup",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("warmup"),
    help=(
        "Warmup specification: if in (0,1) = percent, if >=1 = number of "
        "requests/seconds (depends on active constraint)."
    ),
)
@click.option(
    "--cooldown",
    "--cooldown-percent",  # legacy alias
    "cooldown",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("cooldown"),
    help=(
        "Cooldown specification: if in (0,1) = percent, if >=1 = number of "
        "requests/seconds (depends on active constraint)."
    ),
)
@click.option(
    "--sample-requests",
    "--output-sampling",  # legacy alias
    "sample_requests",
    type=int,
    help=(
        "Number of sample requests per status to save. "
        "None (default) saves all, recommended: 20."
    ),
)
# Constraints configuration
@click.option(
    "--max-seconds",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("max_seconds"),
    help=(
        "Maximum seconds per benchmark. "
        "If None, runs until max_requests or data exhaustion."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=BenchmarkGenerativeTextArgs.get_default("max_requests"),
    help=(
        "Maximum requests per benchmark. "
        "If None, runs until max_seconds or data exhaustion."
    ),
)
@click.option(
    "--max-errors",
    type=int,
    default=BenchmarkGenerativeTextArgs.get_default("max_errors"),
    help="Maximum errors before stopping the benchmark.",
)
@click.option(
    "--max-error-rate",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("max_error_rate"),
    help="Maximum error rate before stopping the benchmark.",
)
@click.option(
    "--max-global-error-rate",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("max_global_error_rate"),
    help="Maximum global error rate across all benchmarks.",
)
def run(**kwargs):
    request_type = kwargs.pop("request_type", None)
    request_formatter_kwargs = kwargs.pop("request_formatter_kwargs", None)
    kwargs["data_request_formatter"] = (
        request_type
        if not request_formatter_kwargs
        else {"request_type": request_type, **request_formatter_kwargs}
    )
    kwargs["data"] = cli_tools.format_list_arg(
        kwargs.get("data"), default=[], simplify_single=False
    )
    kwargs["data_args"] = cli_tools.format_list_arg(
        kwargs.get("data_args"), default=[], simplify_single=False
    )
    kwargs["rate"] = cli_tools.format_list_arg(
        kwargs.get("rate"), default=None, simplify_single=False
    )

    disable_console_outputs = kwargs.pop("disable_console_outputs", False)
    display_scheduler_stats = kwargs.pop("display_scheduler_stats", False)
    disable_progress = kwargs.pop("disable_progress", False)

    try:
        args = BenchmarkGenerativeTextArgs.create(
            scenario=kwargs.pop("scenario", None), **kwargs
        )
    except ValidationError as err:
        # Translate pydantic valdation error to click argument error
        errs = err.errors(include_url=False, include_context=True, include_input=True)
        param_name = "--" + str(errs[0]["loc"][0]).replace("_", "-")
        raise click.BadParameter(
            errs[0]["msg"], ctx=click.get_current_context(), param_hint=param_name
        ) from err

    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(
        benchmark_generative_text(
            args=args,
            progress=(
                GenerativeConsoleBenchmarkerProgress(
                    display_scheduler_stats=display_scheduler_stats
                )
                if not disable_progress
                else None
            ),
            console=Console() if not disable_console_outputs else None,
        )
    )


@benchmark.command(
    "from-file",
    help=(
        "Load a saved benchmark report and optionally re-export to other formats. "
        "PATH: Path to the saved benchmark report file (default: ./benchmarks.json)."
    ),
)
@click.argument(
    "path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    default=Path.cwd() / "benchmarks.json",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=Path.cwd(),
    help=(
        "Directory or file path to save re-exported benchmark results. "
        "If a directory, all output formats will be saved there. "
        "If a file, the matching format will be saved to that file."
    ),
)
@click.option(
    "--output-formats",
    multiple=True,
    type=str,
    default=("console", "json"),  # ("console", "json", "html", "csv")
    help="Output formats for benchmark results (e.g., console, json, html, csv).",
)
def from_file(path, output_path, output_formats):
    asyncio.run(reimport_benchmarks_report(path, output_path, output_formats))


@cli.command(
    short_help="Show configuration settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def config():
    print_config()


@cli.group(help="Tools for preprocessing datasets for use in benchmarks.")
def preprocess():
    """Dataset preprocessing utilities."""


@preprocess.command(
    "dataset",
    help=(
        "Process a dataset to have specific prompt and output token sizes. "
        "Supports multiple strategies for handling prompts and optional "
        "Hugging Face Hub upload.\n\n"
        "DATA: Path to the input dataset or dataset ID.\n\n"
        "OUTPUT_PATH: Path to save the processed dataset, including file suffix."
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
    help="Processor or tokenizer name for calculating token counts.",
)
@click.option(
    "--processor-args",
    default=None,
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    callback=cli_tools.parse_json,
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--short-prompt-strategy",
    type=click.Choice([s.value for s in ShortPromptStrategy]),
    default=ShortPromptStrategy.IGNORE.value,
    show_default=True,
    help="Strategy for handling prompts shorter than target length.",
)
@click.option(
    "--pad-char",
    type=str,
    default="",
    callback=decode_escaped_str,
    help="Character to pad short prompts with when using 'pad' strategy.",
)
@click.option(
    "--concat-delimiter",
    type=str,
    default="",
    help=(
        "Delimiter for concatenating short prompts (used with 'concatenate' strategy)."
    ),
)
@click.option(
    "--prompt-tokens",
    type=str,
    default=None,
    help="Prompt tokens configuration (JSON, YAML file, or key=value string).",
)
@click.option(
    "--output-tokens",
    type=str,
    default=None,
    help="Output tokens configuration (JSON, YAML file, or key=value string).",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Push the processed dataset to Hugging Face Hub.",
)
@click.option(
    "--hub-dataset-id",
    type=str,
    default=None,
    help=("Hugging Face Hub dataset ID for upload (required if --push-to-hub is set)."),
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible token sampling.",
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


@cli.command(
    "mock-server",
    help=(
        "Start a mock OpenAI/vLLM-compatible server for testing. "
        "Simulates model inference with configurable latency and token generation."
    ),
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host address to bind the server to.",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port number to bind the server to.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes.",
)
@click.option(
    "--model",
    default="llama-3.1-8b-instruct",
    help="Name of the model to mock.",
)
@click.option(
    "--processor",
    default=None,
    help="Processor or tokenizer to use for requests.",
)
@click.option(
    "--request-latency",
    default=3,
    type=float,
    help="Request latency in seconds for non-streaming requests.",
)
@click.option(
    "--request-latency-std",
    default=0,
    type=float,
    help="Request latency standard deviation in seconds (normal distribution).",
)
@click.option(
    "--ttft-ms",
    default=150,
    type=float,
    help="Time to first token in milliseconds for streaming requests.",
)
@click.option(
    "--ttft-ms-std",
    default=0,
    type=float,
    help="Time to first token standard deviation in milliseconds.",
)
@click.option(
    "--itl-ms",
    default=10,
    type=float,
    help="Inter-token latency in milliseconds for streaming requests.",
)
@click.option(
    "--itl-ms-std",
    default=0,
    type=float,
    help="Inter-token latency standard deviation in milliseconds.",
)
@click.option(
    "--output-tokens",
    default=128,
    type=int,
    help="Number of output tokens for streaming requests.",
)
@click.option(
    "--output-tokens-std",
    default=0,
    type=float,
    help="Output tokens standard deviation (normal distribution).",
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
