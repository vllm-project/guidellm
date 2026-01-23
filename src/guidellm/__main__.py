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

from guidellm.data import ShortPromptStrategy, process_dataset

try:
    import uvloop
except ImportError:
    uvloop = None  # type: ignore[assignment] # Optional dependency

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
from guidellm.scheduler import StrategyType
from guidellm.schemas import GenerativeRequestType
from guidellm.settings import print_config
from guidellm.utils import Console, DefaultGroupHandler, get_literal_vals
from guidellm.utils import cli as cli_tools

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
    "-c",
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
    callback=cli_tools.parse_list_floats,
    multiple=True,
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
    "--dataloader-kwargs",
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
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=BenchmarkGenerativeTextArgs.get_default("output_dir"),
    help="The directory path to save file output types in",
)
@click.option(
    "--outputs",
    callback=cli_tools.parse_list,
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("outputs"),
    help=(
        "The filename.ext for each of the outputs to create or the "
        "alises (json, csv, html) for the output files to create with "
        "their default file names (benchmark.[EXT])"
    ),
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help=(
        "Legacy parameter for the output path to save the output result to. "
        "Resolves to fill in output-dir and outputs based on input path."
    ),
)
@click.option(
    "--disable-console",
    "--disable-console-outputs",  # legacy alias
    "disable_console",
    is_flag=True,
    help=(
        "Disable all outputs to the console (updates, interactive progress, results)."
    ),
)
@click.option(
    "--disable-console-interactive",
    "--disable-progress",  # legacy alias
    "disable_console_interactive",
    is_flag=True,
    help="Disable interactive console progress updates.",
)
# Aggregators configuration
@click.option(
    "--warmup",
    "--warmup-percent",  # legacy alias
    "warmup",
    default=BenchmarkGenerativeTextArgs.get_default("warmup"),
    callback=cli_tools.parse_json,
    help=(
        "Warmup specification: int, float, or dict as string "
        "(json or key=value). "
        "Controls time or requests before measurement starts. "
        "Numeric in (0, 1): percent of duration or request count. "
        "Numeric >=1: duration in seconds or request count. "
        "Advanced config: see TransientPhaseConfig schema."
    ),
)
@click.option(
    "--cooldown",
    "--cooldown-percent",  # legacy alias
    "cooldown",
    default=BenchmarkGenerativeTextArgs.get_default("cooldown"),
    callback=cli_tools.parse_json,
    help=(
        "Cooldown specification: int, float, or dict as string "
        "(json or key=value). "
        "Controls time or requests after measurement ends. "
        "Numeric in (0, 1): percent of duration or request count. "
        "Numeric >=1: duration in seconds or request count. "
        "Advanced config: see TransientPhaseConfig schema."
    ),
)
@click.option(
    "--rampup",
    type=float,
    default=BenchmarkGenerativeTextArgs.get_default("rampup"),
    help=(
        "The time, in seconds, to ramp up the request rate over. "
        "Applicable for Throughput, Concurrent, and Constant strategies"
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
@click.option(
    "--over-saturation",
    "over_saturation",
    callback=cli_tools.parse_json,
    default=None,
    help=(
        "Enable over-saturation detection. "
        "Pass a JSON dict with configuration "
        '(e.g., \'{"enabled": true, "min_seconds": 30}\'). '
        "Defaults to None (disabled)."
    ),
)
@click.option(
    "--detect-saturation",
    "--default-over-saturation",
    "over_saturation",
    callback=cli_tools.parse_json,
    flag_value='{"enabled": true}',
    help="Enable over-saturation detection with default settings.",
)
def run(**kwargs):  # noqa: C901
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(click.get_current_context(), **kwargs)

    # Handle remapping for request params
    request_type = kwargs.pop("request_type", None)
    request_formatter_kwargs = kwargs.pop("request_formatter_kwargs", None)
    if request_type is not None:
        kwargs["data_request_formatter"] = (
            request_type
            if not request_formatter_kwargs
            else {"request_type": request_type, **request_formatter_kwargs}
        )
    elif request_formatter_kwargs is not None:
        kwargs["data_request_formatter"] = request_formatter_kwargs

    # Handle output path remapping
    if (output_path := kwargs.pop("output_path", None)) is not None:
        if kwargs.get("outputs_dir", None) is not None:
            raise click.BadParameter("Cannot use --output-path with --output-dir.")
        path = Path(output_path)
        if path.is_dir():
            kwargs["output_dir"] = path
        else:
            kwargs["output_dir"] = path.parent
            kwargs["outputs"] = (path.name,)

    # Handle console options
    disable_console = kwargs.pop("disable_console", False)
    disable_console_interactive = (
        kwargs.pop("disable_console_interactive", False) or disable_console
    )
    console = Console() if not disable_console else None
    envs = cli_tools.list_set_env()
    if console and envs:
        console.print_update(
            title=(
                "Note: the following environment variables "
                "are set and **may** affect configuration"
            ),
            details=", ".join(envs),
            status="warning",
        )

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
                GenerativeConsoleBenchmarkerProgress()
                if not disable_console_interactive
                else None
            ),
            console=console,
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
    "--config",
    type=str,
    required=True,
    help=(
        "PreprocessDatasetConfig as JSON string, key=value pairs, "
        "or file path (.json, .yaml, .yml, .config). "
        "Example: 'prompt_tokens=100,output_tokens=50,prefix_tokens_max=10'"
        ' or \'{"prompt_tokens": 100, "output_tokens": 50, '
        '"prefix_tokens_max": 10}\''
    ),
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
    "--data-column-mapper",
    default=None,
    callback=cli_tools.parse_json,
    help="JSON string of column mappings to apply to the dataset.",
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
    "--include-prefix-in-token-count",
    is_flag=True,
    default=False,
    help="Include prefix tokens in prompt token count calculation.",
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
    config,
    processor_args,
    data_args,
    data_column_mapper,
    short_prompt_strategy,
    pad_char,
    concat_delimiter,
    include_prefix_in_token_count,
    push_to_hub,
    hub_dataset_id,
    random_seed,
):
    process_dataset(
        data=data,
        output_path=output_path,
        processor=processor,
        config=config,
        processor_args=processor_args,
        data_args=data_args,
        data_column_mapper=data_column_mapper,
        short_prompt_strategy=short_prompt_strategy,
        pad_char=pad_char,
        concat_delimiter=concat_delimiter,
        include_prefix_in_token_count=include_prefix_in_token_count,
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
