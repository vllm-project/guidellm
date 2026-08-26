from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from guidellm.schemas.benchmark.outputs.output import BenchmarkOutputArgs
from guidellm.settings import settings

__all__ = [
    "JSONBenchmarkOutputArgs",
    "YAMLBenchmarkOutputArgs",
]


@BenchmarkOutputArgs.register("json")
class JSONBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for JSON benchmark output arguments."""

    kind: Literal["json"] = Field(
        default="json",
        description="The kind of output.",
        examples=["json"],
    )
    path: Path = Field(
        default_factory=lambda: settings.default_results_dir / "benchmarks.json",
        description="The file to save the output to.",
        examples=["./benchmarks.json"],
    )


@BenchmarkOutputArgs.register("yaml")
class YAMLBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Model for YAML benchmark output arguments."""

    kind: Literal["yaml"] = Field(
        default="yaml",
        description="The kind of output.",
        examples=["yaml"],
    )
    path: Path = Field(
        default_factory=lambda: settings.default_results_dir / "benchmarks.yaml",
        description="The file to save the output to.",
        examples=["./benchmarks.yaml"],
    )
