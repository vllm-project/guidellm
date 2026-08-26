"""Preprocess command group."""

from __future__ import annotations

import click

from .dataset import dataset

__all__ = ["preprocess"]


@click.group(help="Tools for preprocessing datasets for use in benchmarks.")
def preprocess():
    """Dataset preprocessing utilities."""


# Register subcommands
preprocess.add_command(dataset)
