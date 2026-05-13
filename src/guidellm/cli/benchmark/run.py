"""Benchmark run command."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.benchmark import (
    BenchmarkGenerativeTextArgs,
    GenerativeConsoleBenchmarkerProgress,
    ProfileType,
    benchmark_generative_text,
    get_builtin_scenarios,
)
from guidellm.scheduler import StrategyType
from guidellm.utils.console import Console
from guidellm.utils.env_validator import validate_env_vars
from guidellm.utils.typing import get_literal_vals

__all__ = ["STRATEGY_PROFILE_CHOICES", "run"]

STRATEGY_PROFILE_CHOICES: list[str] = list(get_literal_vals(ProfileType | StrategyType))
"""Available strategy and profile type choices for benchmark execution."""


@click.command(
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
)
@click.option(
    "--backend-kwargs",
    "--backend-args",  # legacy alias
    "backend_kwargs",
    callback=cli_tools.parse_arguments,
    help=(
        "JSON string of arguments to pass to the backend. E.g., "
        '\'{"api_key": "apikey-*", "verify": false}\''
    ),
)
@click.option(
    "--model",
    type=str,
    help="Model ID to benchmark. If not provided, uses first available model.",
)
# Data configuration
@click.option(
    "--request-format",
    "--request-type",
    help=(
        "Format to use for requests. Options depend on backend. "
        "For vLLM backend: plain (no chat template, text appending only), "
        "default-template (use tokenizer default), or a file path / single-line "
        "template per vLLM docs. Default: default-template"
        "For openai backend: http endpoint path (/v1/chat/completions, "
        "/v1/completions, /v1/audio/transcriptions, /v1/audio/translations) or "
        "alias (e.g. chat_completions); default /v1/chat/completions."
    ),
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
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("processor_args"),
    help="JSON string of arguments to pass to the processor constructor.",
)
@click.option(
    "--data-args",
    multiple=True,
    default=BenchmarkGenerativeTextArgs.get_default("data_args"),
    help="JSON string of arguments to pass to dataset creation.",
)
@click.option(
    "--data-samples",
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("data_samples"),
    type=int,
    help=(
        "Number of samples from dataset. -1 (default) uses all samples "
        "and dynamically generates more."
    ),
)
@click.option(
    "--data-column-mapper",
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("data_column_mapper"),
    help=(
        "JSON string of column mappings to apply to the dataset. "
        'E.g., \'{"column_mappings": {"text_column": "article", '
        '"output_tokens_count_column": "output_tokens"}}\''
    ),
)
@click.option(
    "--data-preprocessors",
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("data_preprocessors"),
    multiple=True,
    help=(
        "List of of preprocessors to apply to the dataset. E.g., "
        "'encode_media,my_custom_preprocessor'"
    ),
)
@click.option(
    "--data-preprocessors-kwargs",
    callback=cli_tools.parse_arguments,
    help="JSON string of arguments to pass to all preprocessors.",
)
@click.option(
    "--data-finalizer",
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("data_finalizer"),
    help=(
        "JSON string of finalizer to convert dataset rows to requests."
        " E.g., 'generative' or '{\"type\": \"generative\"}'"
    ),
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
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("dataloader_kwargs"),
    help="JSON string of arguments to pass to the dataloader constructor.",
)
@click.option(
    "--random-seed",
    default=BenchmarkGenerativeTextArgs.get_default("random_seed"),
    type=int,
    help="Random seed for reproducibility.",
)
@click.option(
    "--images-per-request",
    type=str,
    default=None,
    help=(
        "Number of images per request for vision benchmarks. "
        "Single value (e.g., '2') or comma-separated list (e.g., '1,2,5'). "
        "When a list is provided, runs sequential benchmarks for each count."
    ),
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
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("warmup"),
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
    callback=cli_tools.parse_arguments,
    default=BenchmarkGenerativeTextArgs.get_default("cooldown"),
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
    callback=cli_tools.parse_arguments,
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
    callback=cli_tools.parse_arguments,
    flag_value='{"enabled": true}',
    help="Enable over-saturation detection with default settings.",
)
def run(**kwargs):  # noqa: C901
    ctx = click.get_current_context()
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(ctx, **kwargs)

    # Handle images_per_request parameter
    images_per_request = kwargs.pop("images_per_request", None)
    if images_per_request:
        # Parse as single value or comma-separated list
        try:
            parts = [int(x.strip()) for x in images_per_request.split(",")]
            kwargs["images_per_request"] = parts if len(parts) > 1 else parts[0]
        except ValueError:
            raise click.BadParameter(
                f"--images-per-request must be an integer or comma-separated integers, got '{images_per_request}'"
            )

    # Handle output path remapping
    if (output_path := kwargs.pop("output_path", None)) is not None:
        if kwargs.get("output_dir", None) is not None:
            raise click.BadParameter("Cannot use --output-path with --output-dir.")
        path = Path(output_path)
        if path.is_dir():
            kwargs["output_dir"] = path
        else:
            kwargs["output_dir"] = path.parent
            kwargs["outputs"] = (path.name,)

    # Map top-level CLI options to backend_kwargs
    backend_kwargs = kwargs.pop("backend_kwargs", {})
    backend_type = kwargs.pop("backend", "openai_http")
    backend_kwargs["type"] = backend_type
    for alias in ("target", "model", "request_format"):
        with contextlib.suppress(KeyError):
            backend_kwargs[alias] = kwargs.pop(alias)
    kwargs["backend_kwargs"] = backend_kwargs

    # Handle console options
    disable_console = kwargs.pop("disable_console", False)
    disable_console_interactive = (
        kwargs.pop("disable_console_interactive", False) or disable_console
    )
    console = Console() if not disable_console else None

    if console:
        invalid_set_envs, valid_set_envs = validate_env_vars(ctx)

        if valid_set_envs:
            console.print_update(
                title=(
                    "The following environment variables are set and will be used "
                    "by GuideLLM (if not overridden by CLI arguments/config)."
                ),
                details=", ".join(valid_set_envs),
                status="info",
            )
        if invalid_set_envs:
            console.print_update(
                title=(
                    "The following environment variables are set "
                    "but not recognized by GuideLLM. Please verify "
                    "that the benchmark is configured correctly."
                ),
                details=", ".join(invalid_set_envs),
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
            errs[0]["msg"], ctx=ctx, param_hint=param_name
        ) from err

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
