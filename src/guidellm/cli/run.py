from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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
from guidellm.utils.typing import BLANK

__all__ = [
    "run",
]


def _parse_append_payloads(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> dict[str, Any] | None:
    """Parse and validate structured content payload fields from the CLI."""
    if value is None:
        return None
    parsed = cli_tools.parse_arguments(ctx, param, value)
    if not isinstance(parsed, dict):
        raise click.BadParameter(
            "must be a JSON, YAML, or key=value object",
            ctx=ctx,
            param=param,
        )
    return parsed


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
        "Define a label as a key-value pair for the run. "
        "Example: `--label timestamp=1999-09-12@12:00:00 --label env=staging` "
        " [repeatable]"
    ),
)
@registry_options_from_model(model=BenchmarkArgs, group_key="spec")
@click.option(
    "--append-payloads",
    callback=_parse_append_payloads,
    help=(
        "Append key-value fields to structured content objects in chat completion "
        "requests. Values override matching metadata from custom datasets. "
        'Example: `--append-payloads \'{"key":"value"}\'`'
    ),
)
@click.option(
    "--override",
    "benchmarks",
    nargs=2,
    callback=cli_tools.parse_overrides,
    multiple=True,
    help=(
        "Define overrides for each sub-benchmark. "
        "Currently this only supports `profile.streams` or `profile.rate`. "
        "Example: `--profile kind=concurrent --override 'profile.streams' 1,2,4,8,16` "
        " [repeatable]"
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
    append_payloads = kwargs.pop("append_payloads", None)
    if append_payloads is not None:
        spec = kwargs.setdefault("spec", {})
        backend = spec.setdefault("backend", {})
        backend.setdefault("kind", "openai_http")
        backend["append_payloads"] = append_payloads
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
            benchmarks=kwargs.get("benchmarks") or BLANK,
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
