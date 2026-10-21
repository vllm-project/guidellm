"""Shared Click configuration for benchmark commands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

import guidellm.utils.cli as cli_tools
from guidellm.schemas.benchmark import BenchmarkArgs, get_builtin_scenarios
from guidellm.utils.click_pydantic import registry_options_from_model

__all__ = ["common_run_options"]


def common_run_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Add the options shared by the ``run`` and ``config`` commands.

    :param func: Command function to decorate
    :return: Command function with the common benchmark options applied
    """
    # Apply options in reverse help order to preserve the existing ``run`` CLI.
    func = click.option(
        "--disable-console-interactive",
        "--disable-progress",  # legacy alias
        "disable_console_interactive",
        is_flag=True,
        help="Disable interactive console progress updates.",
    )(func)
    func = click.option(
        "--disable-console",
        "--disable-console-outputs",  # legacy alias
        "disable_console",
        is_flag=True,
        help=(
            "Disable all outputs to the console (updates, interactive progress, "
            "results)."
        ),
    )(func)
    func = click.option(
        "--override",
        "benchmarks",
        nargs=2,
        callback=cli_tools.parse_overrides,
        multiple=True,
        help=(
            "Define overrides for each sub-benchmark. "
            "Currently this only supports `profile.streams` or `profile.rate`. "
            "Example: `--profile kind=concurrent --override 'profile.streams' "
            "1,2,4,8,16` [repeatable]"
        ),
    )(func)
    func = registry_options_from_model(model=BenchmarkArgs, group_key="spec")(func)
    func = click.option(
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
    )(func)
    return click.option(
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
        multiple=True,
        help=(
            "Builtin scenario name or path to config file. "
            "CLI options override scenario settings."
        ),
    )(func)
