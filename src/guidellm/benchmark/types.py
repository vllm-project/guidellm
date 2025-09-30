from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAliasType

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from transformers import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.benchmark.aggregator import (
    Aggregator,
    CompilableAggregator,
)
from guidellm.benchmark.output import (
    GenerativeBenchmarkerOutput,
)
from guidellm.benchmark.progress import BenchmarkerProgress

__all__ = [
    "AggregatorInputT",
    "DataInputT",
    "OutputFormatT",
    "ProcessorInputT",
    "ProgressInputT",
]


DataInputT = TypeAliasType(
    "DataInputT",
    Iterable[str]
    | Iterable[dict[str, Any]]
    | Dataset
    | DatasetDict
    | IterableDataset
    | IterableDatasetDict
    | str
    | Path,
)

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

AggregatorInputT = TypeAliasType(
    "AggregatorInputT",
    dict[str, str | dict[str, Any] | Aggregator | CompilableAggregator],
)
