from __future__ import annotations

import asyncio
from pathlib import Path

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.benchmark import (
    BenchmarkArgs,
    BenchmarkScenario,
    GenerativeConsoleBenchmarkerProgress,
    benchmark_generative_text,
    get_builtin_scenarios,
)
from guidellm.settings import Settings
from guidellm.utils.click_pydantic import (
    format_validation_errors,
    registry_options_from_model,
)
from guidellm.utils.console import Console
from guidellm.utils.env_validator import validate_env_vars

__all__ = [
    "run",
]


@click.command(
    "run",
    help=(
        "Run a benchmark against a generative model. "
        "Supports multiple backends, data sources, strategies, and output formats. "
        "Configuration can be loaded from a scenario file or specified via options."
    ),
)
@click.option(
    "--config",
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
    help=(
        "Builtin scenario name or path to config file. "
        "CLI options override scenario settings."
    ),
)
@click.option(
    "--label",
    "-l",
    "labels",
    multiple=True,
    callback=cli_tools.parse_kv_str,
    help=(
        "Define a labels in key-value pair for the run. "
        "Example: `--label timestamp=1999-09-12@12:00:00 --label env=staging`"
        "  [repeatable]"
    ),
)
@registry_options_from_model(model=BenchmarkArgs, group_key="spec")
@click.option(
    "--override",
    "benchmarks",
    nargs=2,
    callback=cli_tools.parse_overrides,
    multiple=True,
    help=(
        "Define overrides for each sub-benchmark. "
        "Currently this only supports `profile.streams` or `profile.rate`. "
        "Example: `--profile kind=concurrent --override 'profile.streams' 1,2,4,8,16`"
        "  [repeatable]"
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
def run(**kwargs):  # noqa: C901, PLR0915
    ctx = click.get_current_context()
    # Only set CLI args that differ from click defaults
    kwargs = cli_tools.set_if_not_default(ctx, **kwargs)

    disable_console = kwargs.pop("disable_console", False)
    disable_console_interactive = (
        kwargs.pop("disable_console_interactive", False) or disable_console
    )
    console = Console() if not disable_console else None

    if console:
        invalid_set_envs, valid_set_envs = validate_env_vars(
            Settings, BenchmarkScenario
        )

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
        args = BenchmarkScenario.create(
            spec=kwargs.get("spec", {}),
            benchmarks=kwargs.get("benchmarks", []),
            metadata={"labels": dict(kwargs.get("labels", []))},
            scenario=kwargs.get("config"),
        )
    except ValidationError as err:
        # Translate pydantic validation error to click argument error
        raise format_validation_errors(ctx, err, base_class=BenchmarkScenario) from err

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
