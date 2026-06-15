from __future__ import annotations

import asyncio

import click
from pydantic import ValidationError

import guidellm.utils.cli as cli_tools
from guidellm.benchmark import (
    BenchmarkArgs,
    BenchmarkScenario,
    GenerativeConsoleBenchmarkerProgress,
    benchmark_generative_text,
)
from guidellm.utils.click_pydantic import (
    format_validation_errors,
    registry_options_from_model,
)
from guidellm.utils.console import Console

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
@registry_options_from_model(model=BenchmarkArgs, group_key="spec")
@click.option(
    "--scenario",
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help=(
        "Path to a benchmark scenario file (YAML or JSON) "
        "that defines the benchmark configuration"
    ),
)
@click.option(
    "--override",
    "benchmarks",
    nargs=2,
    callback=cli_tools.parse_overrides,
    multiple=True,
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

    try:
        args = BenchmarkScenario.create(
            spec=kwargs.get("spec", {}),
            benchmarks=kwargs.get("benchmarks", []),
            scenario=kwargs.get("scenario"),
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
