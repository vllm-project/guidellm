from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from guidellm.schemas.benchmark.outputs.output import BenchmarkOutputArgs
from guidellm.settings import settings

__all__ = ["HTMLBenchmarkOutputArgs"]


@BenchmarkOutputArgs.register("html")
class HTMLBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for HTML benchmark output arguments."""

    kind: Literal["html"] = Field(
        default="html",
        description="The kind of output.",
    )
    path: Path = Field(
        default_factory=lambda: settings.default_results_dir / "benchmarks.html",
        description="The file to save the output to.",
    )
