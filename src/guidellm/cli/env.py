"""Environment configuration display command."""

from __future__ import annotations

import click

from guidellm.settings import print_config

__all__ = ["env"]


@click.command(
    short_help="Show environment variable settings.",
    help="Display environment variables for configuring GuideLLM behavior.",
)
def env():
    print_config()
