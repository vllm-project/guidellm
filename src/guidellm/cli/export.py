"""Export command for loading and re-exporting saved benchmark reports."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from guidellm.benchmark import reimport_benchmarks_report
from guidellm.benchmark.schemas import BenchmarkOutputArgs
from guidellm.utils.click_pydantic import RegistryAwareCommand, registry_option

__all__ = ["export"]


@click.command(
    "export",
    cls=RegistryAwareCommand,
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
@registry_option(
    "--output",
    "outputs",
    registry=BenchmarkOutputArgs,
    multiple=True,
    default=[{"kind": "console"}, {"kind": "json"}, {"kind": "html"}, {"kind": "csv"}],
    help="Output formats for the report (default: console, json, html, csv).",
)
def export(path, outputs):
    asyncio.run(reimport_benchmarks_report(path, outputs))
