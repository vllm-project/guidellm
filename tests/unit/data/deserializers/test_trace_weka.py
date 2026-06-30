from collections.abc import Callable
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.deserializers.trace_weka import WEKATraceFormatArgs


def _mock_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is one token."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{i}" for i, _ in enumerate(tokens)
    )
    return proc


def _write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


@dataclasses.dataclass
class TraceColumnGenerator:
    name: str
    # Function with row index as the one argument
    data_generator: Callable[[int], Any]


def _generate_trace(num_rows: int, columns: list[TraceColumnGenerator]) -> str:
    return "\n".join(
        "{"
        + ", ".join(f'"{col.name}": {col.data_generator(idx)}' for col in columns)
        + "}"
        for idx in range(num_rows)
    )


def _get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


class TestWEKATraceFormat:
    @pytest.mark.regression
    def test_format_registered_with_deserializer(self, tmp_path: Path):
        ...

    @pytest.fixture
    def deserializer(self) -> TraceDatasetDeserializer:
        return TraceDatasetDeserializer()
