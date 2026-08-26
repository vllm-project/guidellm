from __future__ import annotations

from typing import Literal

from pydantic import Field

from guidellm.schemas.benchmark.outputs.output import BenchmarkOutputArgs

__all__ = ["ConsoleBenchmarkOutputArgs"]


@BenchmarkOutputArgs.register("console")
class ConsoleBenchmarkOutputArgs(BenchmarkOutputArgs):
    """Base class for console benchmark output arguments."""

    kind: Literal["console"] = Field(
        default="console",
        description="The kind of output.",
    )
