"""
GuideLLM command-line interface entry point.

Primary CLI application providing benchmark execution, dataset preprocessing, and
mock server functionality for language model evaluation. Organizes commands into
three main groups: benchmark operations for performance testing, preprocessing
utilities for data transformation, and mock server capabilities for development
and testing. Supports multiple backends, output formats, and flexible configuration
through CLI options and environment variables.

Example:
::
    # Run a benchmark against a model
    guidellm benchmark run --target http://localhost:8000 --data dataset.json \\
        --profile sweep

    # Preprocess a dataset
    guidellm preprocess dataset input.json output.json --processor gpt2

    # Start a mock server for testing
    guidellm mock-server --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import click

from .benchmark import benchmark
from .env import env
from .mock_server import mock_server
from .preprocess import preprocess
from .run import run

__all__ = ["cli"]


@click.group()
@click.version_option(package_name="guidellm", message="guidellm version: %(version)s")
def cli():
    """GuideLLM CLI for benchmarking, preprocessing, and testing language models."""


# Register all commands and groups
cli.add_command(env)
cli.add_command(mock_server)
cli.add_command(preprocess)
cli.add_command(run)
cli.add_command(benchmark)
