from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from pathlib import Path
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from guidellm.benchmark.output import (
    GenerativeBenchmarkerOutput,
)

from transformers import (  # type: ignore[import]
    PreTrainedTokenizerBase,
)

from guidellm.benchmark.progress import BenchmarkerProgress

from guidellm.benchmark.aggregator import (
    Aggregator,
    CompilableAggregator,
)


DataInputType = (
    Iterable[str]
    | Iterable[dict[str, Any]]
    | Dataset
    | DatasetDict
    | IterableDataset
    | IterableDatasetDict
    | str
    | Path
)

OutputFormatType = (
    tuple[str, ...]
    | list[str]
    | dict[str, str | dict[str, Any] | GenerativeBenchmarkerOutput]
    | None
)

ProcessorInputType = str | Path | PreTrainedTokenizerBase

ProgressInputType = tuple[str, ...] | list[str] | list[BenchmarkerProgress]

AggregatorInputType = dict[str, str | dict[str, Any] | Aggregator | CompilableAggregator]
