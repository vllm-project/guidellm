import asyncio
import codecs
from pathlib import Path
from typing import Annotated, Union

import click
from pydantic import ValidationError

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
    get_builtin_scenarios,
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
    "--scenario",
    type=cli_tools.Union(
        click.Path(
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            path_type=Path,
        ),
        click.Choice(get_builtin_scenarios()),
    ),
    default=None,
    help=(
        "The name of a builtin scenario or path to a config file. "
        "Missing values from the config will use defaults. "
        "Options specified on the commandline will override the scenario."
    ),
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
    default=GenerativeTextScenario.get_default("rate"),
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
    default=GenerativeTextScenario.get_default("backend"),
    help=(
        "The type of backend to use to run requests against. Defaults to 'openai_http'."
        f" Supported types: {', '.join(get_literal_vals(BackendType))}"
    ),
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=cli_tools.parse_json,
    default=GenerativeTextScenario.get_default("backend_kwargs"),
    help=(
        "A JSON string containing any arguments to pass to the backend as a "
        "dict with **kwargs. Headers can be removed by setting their value to "
        "null. For example: "
        """'{"headers": {"Authorization": null, "Custom-Header": "Custom-Value"}}'"""
    ),
)
@click.option(
    "--model",
    default=GenerativeTextScenario.get_default("model"),
    type=str,
    help=(
        "The ID of the model to benchmark within the backend. "
        "If None provided (default), then it will use the first model available."
    ),
)
# Data configuration
@click.option(
    "--processor",
    default=GenerativeTextScenario.get_default("processor"),
    type=str,
    help=(
        "The processor or tokenizer to use to calculate token counts for statistics "
        "and synthetic data generation. If None provided (default), will load "
        "using the model arg, if needed."
    ),
)
@click.option(
    "--processor-args",
    default=GenerativeTextScenario.get_default("processor_args"),
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the processor constructor "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--data-args",
    default=GenerativeTextScenario.get_default("data_args"),
    callback=cli_tools.parse_json,
    help=(
        "A JSON string containing any arguments to pass to the dataset creation "
        "as a dict with **kwargs."
    ),
)
@click.option(
    "--data-sampler",
    default=GenerativeTextScenario.get_default("data_sampler"),
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
    default=GenerativeTextScenario.get_default("warmup"),
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
    default=GenerativeTextScenario.get_default("cooldown"),
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
    default=GenerativeTextScenario.get_default("request_samples"),
    type=int,
    help=(
        "The number of samples for each request status and each benchmark to save "
        "in the output file. If None (default), will save all samples. "
        "Defaults to 20."
    ),
)
# Constraints configuration
@click.option(
    "--max-seconds",
    type=float,
    default=GenerativeTextScenario.get_default("max_seconds"),
    help=(
        "The maximum number of seconds each benchmark can run for. "
        "If None, will run until max_requests or the data is exhausted."
    ),
)
@click.option(
    "--max-requests",
    type=int,
    default=GenerativeTextScenario.get_default("max_requests"),
    help=(
        "The maximum number of requests each benchmark can run for. "
        "If None, will run until max_seconds or the data is exhausted."
    ),
)
@click.option(
    "--max-errors",
    type=int,
    default=GenerativeTextScenario.get_default("max_errors"),
    help="Maximum number of errors allowed before stopping the benchmark",
)
@click.option(
    "--max-error-rate",
    type=float,
    default=GenerativeTextScenario.get_default("max_error_rate"),
    help="Maximum error rate allowed before stopping the benchmark",
)
@click.option(
    "--max-global-error-rate",
    type=float,
    default=GenerativeTextScenario.get_default("max_global_error_rate"),
    help="Maximum global error rate allowed across all benchmarks",
)
def run(**kwargs):
    """
    Execute a generative text benchmark against a target model backend.

    Runs comprehensive performance testing using various strategies and profiles,
    collecting metrics on latency, throughput, error rates, and resource usage.
    Supports multiple backends, data sources, output formats, and constraint types
    for flexible benchmark configuration.
    """
    scenario = kwargs.pop("scenario")
    click_ctx = click.get_current_context()
    overrides = cli_tools.set_if_not_default(click_ctx, **kwargs)

    try:
        # If a scenario file was specified read from it
        if scenario is None:
            _scenario = GenerativeTextScenario.model_validate(overrides)
        elif isinstance(scenario, Path):
            _scenario = GenerativeTextScenario.from_file(scenario, overrides)
        else:  # Only builtins can make it here; click will catch anything else
            _scenario = GenerativeTextScenario.from_builtin(scenario, overrides)
    except ValidationError as e:
        # Translate pydantic valdation error to click argument error
        errs = e.errors(include_url=False, include_context=True, include_input=True)
        param_name = "--" + str(errs[0]["loc"][0]).replace("_", "-")
        raise click.BadParameter(
            errs[0]["msg"], ctx=click_ctx, param_hint=param_name
        ) from e

    if HAS_UVLOOP:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(
        benchmark_generative_text(
            scenario=_scenario,
            # Output configuration
            output_path=kwargs["output_path"],
            output_formats=[
                fmt
                for fmt in kwargs["output_formats"]
                if not kwargs["disable_console_outputs"] or fmt != "console"
            ],
            # Updates configuration
            progress=(
                [
                    GenerativeConsoleBenchmarkerProgress(
                        display_scheduler_stats=kwargs["display_scheduler_stats"]
                    )
                ]
                if not kwargs["disable_progress"]
                else None
            ),
            print_updates=not kwargs["disable_console_outputs"],
            # Aggregators configuration
            add_aggregators={
                "extras": InjectExtrasAggregator(extras=kwargs["output_extras"])
            },
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
