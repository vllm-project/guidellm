"""Benchmark from-file command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from guidellm.benchmark import reimport_benchmarks_report

__all__ = ["from_file"]


@click.command(
    "from-file",
    help=(
        "Load a saved benchmark report and optionally re-export to other formats. "
        "PATH: Path to the saved benchmark report file (default: ./benchmarks.json)."
    ),
)
@click.argument(
    "path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    default=Path.cwd() / "benchmarks.json",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=Path.cwd(),
    help=(
        "Directory or file path where the re-exported benchmark results will be saved. "
        "If a directory, default filenames are used. "
        "If a file path, the suffix is used directly when generating a "
        "single format, or replaced by each format's extension when generating "
        "multiple formats."
    ),
)
@click.option(
    "--output-formats",
    multiple=True,
    type=str,
    default=("console", "json"),  # ("console", "json", "html", "csv")
    help="Output formats for benchmark results (e.g., console, json, html, csv).",
)
def from_file(path, output_path, output_formats):
    asyncio.run(reimport_benchmarks_report(path, output_path, output_formats))
