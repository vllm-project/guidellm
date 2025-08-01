import asyncio
import codecs
from pathlib import Path
from typing import get_args

import click
from pydantic import ValidationError

from guidellm.backend import BackendType
from guidellm.benchmark import (
    ProfileType,
    reimport_benchmarks_report,
)
from guidellm.benchmark.entrypoints import benchmark_with_scenario
from guidellm.benchmark.scenario import GenerativeTextScenario, get_builtin_scenarios
from guidellm.config import print_config
from guidellm.preprocess.dataset import ShortPromptStrategy, process_dataset
from guidellm.scheduler import StrategyType
from guidellm.utils import DefaultGroupHandler
from guidellm.utils import cli as cli_tools

STRATEGY_PROFILE_CHOICES = list(
    set(list(get_args(ProfileType)) + list(get_args(StrategyType)))
)


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
    "--backend-type",
    type=click.Choice(list(get_args(BackendType))),
    help=(
        "The type of backend to use to run requests against. Defaults to 'openai_http'."
        f" Supported types: {', '.join(get_args(BackendType))}"
    ),
    default=GenerativeTextScenario.get_default("backend_type"),
)
@click.option(
    "--backend-args",
    callback=cli_tools.parse_json,
    default=GenerativeTextScenario.get_default("backend_args"),
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
    "--data",
    type=str,
    help=(
        "The HuggingFace dataset ID, a path to a HuggingFace dataset, "
        "a path to a data file csv, json, jsonl, or txt, "
        "or a synthetic data config as a json or key=value string."
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
@click.option(
    "--rate-type",
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
    "--max-error",
    type=float,
    help=(
        "The maximum error after which a benchmark will stop. "
        "Can either be a rate i.e 0 < rate < 1 or constant number. "
        "If rate is given and rate_type is 'constant' and 'max_seconds' exists "
        "then the rate will be calculated as part of the total expected "
        "requests count i.e rate * duration. If rate is given and number"
        "of requests is not pre-determined than a context window "
        "of the last requests will be looked at. Context window size"
        "is configurable under GUIDELLM__ERROR_CHECK_WINDOW_SIZE."
        "If a number above 1 is given than we just count the total"
        "number of error and check if it's above the threshold."
    ),
)
@click.option(
    "--warmup-percent",
    type=float,
    default=GenerativeTextScenario.get_default("warmup_percent"),
    help=(
        "The percent of the benchmark (based on max-seconds, max-requests, "
        "or lenth of dataset) to run as a warmup and not include in the final results. "
        "Defaults to None."
    ),
)
@click.option(
    "--cooldown-percent",
    type=float,
    default=GenerativeTextScenario.get_default("cooldown_percent"),
    help=(
        "The percent of the benchmark (based on max-seconds, max-requests, or length "
        "of dataset) to run as a cooldown and not include in the final results. "
        "Defaults to None."
    ),
)
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
@click.option(
    "--disable-console-outputs",
    is_flag=True,
    help="Set this flag to disable console output",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=Path.cwd() / "benchmarks.json",
    help=(
        "The path to save the output to. If it is a directory, "
        "it will save benchmarks.json under it. "
        "Otherwise, json, yaml, csv, or html files are supported for output types "
        "which will be read from the extension for the file path."
    ),
)
@click.option(
    "--output-extras",
    callback=cli_tools.parse_json,
    help="A JSON string of extra data to save with the output benchmarks",
)
@click.option(
    "--output-sampling",
    type=int,
    help=(
        "The number of samples to save in the output file. "
        "If None (default), will save all samples."
    ),
    default=GenerativeTextScenario.get_default("output_sampling"),
)
@click.option(
    "--random-seed",
    default=GenerativeTextScenario.get_default("random_seed"),
    type=int,
    help="The random seed to use for benchmarking to ensure reproducibility.",
)
def run(
    scenario,
    target,
    backend_type,
    backend_args,
    model,
    processor,
    processor_args,
    data,
    data_args,
    data_sampler,
    rate_type,
    rate,
    max_seconds,
    max_requests,
    max_error,
    warmup_percent,
    cooldown_percent,
    disable_progress,
    display_scheduler_stats,
    disable_console_outputs,
    output_path,
    output_extras,
    output_sampling,
    random_seed,
):
    click_ctx = click.get_current_context()

    overrides = cli_tools.set_if_not_default(
        click_ctx,
        target=target,
        backend_type=backend_type,
        backend_args=backend_args,
        model=model,
        processor=processor,
        processor_args=processor_args,
        data=data,
        data_args=data_args,
        data_sampler=data_sampler,
        rate_type=rate_type,
        rate=rate,
        max_seconds=max_seconds,
        max_requests=max_requests,
        max_error=max_error,
        warmup_percent=warmup_percent,
        cooldown_percent=cooldown_percent,
        output_sampling=output_sampling,
        random_seed=random_seed,
    )

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

    asyncio.run(
        benchmark_with_scenario(
            scenario=_scenario,
            show_progress=not disable_progress,
            show_progress_scheduler_stats=display_scheduler_stats,
            output_console=not disable_console_outputs,
            output_path=output_path,
            output_extras=output_extras,
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
