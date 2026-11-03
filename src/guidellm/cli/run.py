from __future__ import annotations

import asyncio

import click
from pydantic import ValidationError

import guidellm.entrypoints as entry
import guidellm.utils.cli as cli_tools
from guidellm.schemas.benchmark import BenchmarkScenario
from guidellm.settings import Settings
from guidellm.utils.click_pydantic import RegistryAwareCommand, format_validation_errors
from guidellm.utils.console import Console
from guidellm.utils.env_validator import validate_env_vars
from guidellm.utils.typing import BLANK

from .common import common_run_options

__all__ = [
    "common_run_options",
    "run",
]


@click.command(
    "run",
    cls=RegistryAwareCommand,
    help=(
        "Run a benchmark against a generative model. "
        "Supports multiple backends, data sources, strategies, and output formats. "
        "Configuration can be loaded from a scenario file or specified via options."
    ),
)
@common_run_options
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
            benchmarks=kwargs.get("benchmarks") or BLANK,
            metadata={"labels": dict(kwargs.get("labels", []))},
            scenario=kwargs.get("config"),
        )
    except ValidationError as err:
        # Translate pydantic validation error to click argument error
        raise format_validation_errors(ctx, err, base_class=BenchmarkScenario) from err

    asyncio.run(
        entry.benchmark_generative_text(
            args=args,
            progress=not disable_console_interactive,
            console=console,
        )
    )
