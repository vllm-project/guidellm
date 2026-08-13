"""Preprocess command group."""

from __future__ import annotations

import click

# Ensure registry subclasses are imported before schema reload.
import guidellm.data.loaders  # noqa: F401
import guidellm.data.preprocessors  # noqa: F401
import guidellm.data.tokenizers  # noqa: F401
from guidellm.cli.preprocess.args import PreprocessDatasetArgs
from guidellm.data.schemas import PreprocessStrategyArgs

from .dataset import dataset

__all__ = ["preprocess"]

# Rebuild schemas to ensure all registry subclasses are known
PreprocessStrategyArgs.reload_schema()
PreprocessDatasetArgs.reload_schema()


@click.group(help="Tools for preprocessing datasets for use in benchmarks.")
def preprocess():
    """Dataset preprocessing utilities."""


# Register subcommands
preprocess.add_command(dataset)
