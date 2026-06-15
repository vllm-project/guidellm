"""Benchmark command group."""

from __future__ import annotations

import click

from guidellm.utils.default_group import DefaultGroupHandler

from .from_file import from_file

__all__ = ["benchmark"]


@click.group(
    help="Run a benchmark or load a previously saved benchmark report.",
    cls=DefaultGroupHandler,
    default="run",
)
def benchmark():
    """Benchmark commands for performance testing generative models."""


# Register subcommands
benchmark.add_command(from_file)
