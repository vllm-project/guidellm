from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase  # type: ignore[import]
from typing_extensions import TypeAliasType

from guidellm.benchmark.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.progress import BenchmarkerProgress

__all__ = [
    "AggregatorInputT",
    "DataInputT",
    "OutputFormatT",
    "ProcessorInputT",
    "ProgressInputT",
]


OutputFormatT = TypeAliasType(
    "OutputFormatT",
    tuple[str, ...]
    | list[str]
    | dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
    | None,
)

ProcessorInputT = TypeAliasType("ProcessorInputT", str | Path | PreTrainedTokenizerBase)

ProgressInputT = TypeAliasType(
    "ProgressInputT", tuple[str, ...] | list[str] | list[BenchmarkerProgress]
)
