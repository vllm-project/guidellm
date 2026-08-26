"""Preprocess command group."""

from __future__ import annotations

import click

# Ensure registry subclasses are imported before schema reload.
# FIXME: Move to guidellms.schemas.cli
from guidellm.cli.preprocess.args import PreprocessDatasetArgs
from guidellm.schemas.data import PreprocessStrategyArgs

from .dataset import dataset

__all__ = ["preprocess"]

# FIXME: Is this really nessessary?
# Rebuild schemas to ensure all registry subclasses are known
PreprocessStrategyArgs.reload_schema()
PreprocessDatasetArgs.reload_schema()


@click.group(help="Tools for preprocessing datasets for use in benchmarks.")
def preprocess():
    """Dataset preprocessing utilities."""


# Register subcommands
preprocess.add_command(dataset)
