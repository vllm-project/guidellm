import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from datasets import IterableDataset
from faker import Faker
from pydantic import ValidationError

from guidellm.data.deserializers import DataNotSupportedError
from guidellm.data.deserializers.trace_common import (
    TraceDataArgs,
    TraceDatasetDeserializer,
    TraceFormatRegistry,
    decode_prompt,
    generate_token_ids,
)
from guidellm.data.deserializers.trace_minimal import MinimalTraceFormatArgs


def _mock_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is one token."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{t}" for t in tokens
    )
    return proc


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], ""),
        ([0], "tok0"),
        ([1, 1], "tok1 tok1"),
        ([0, 2, 3, 2], "tok0 tok2 tok3 tok2"),
    ],
)
def test_decode_prompt(token_ids, expected):
    proc = _mock_processor()
    assert decode_prompt(proc, token_ids) == expected


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [
        (0, []),
        (1, [0]),
        (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (1000, list(range(1000))),
    ],
)
def test_generate_token_ids(token_count, expected):
    proc = _mock_processor()
    faker = Faker()
    res = generate_token_ids(token_count, proc, faker)
    assert len(res) == len(expected)
    assert res == expected


class TestTraceFormatRegistry:
    def test_unknown_kind_raises(self, tmp_path: Path):
        config = TraceDataArgs(kind="unknown_kind", path=tmp_path)
        with pytest.raises(DataNotSupportedError, match="not registered"):
            TraceFormatRegistry.dispatch(config)


@dataclasses.dataclass
class TraceColumnGenerator:
    name: str
    # Function with row index as the one argument
    data_generator: Callable[[int], Any]


def _write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


def _generate_trace(num_rows: int, columns: list[TraceColumnGenerator]) -> str:
    return "\n".join(
        "{"
        + ", ".join(f'"{col.name}": {col.data_generator(idx)}' for col in columns)
        + "}"
        for idx in range(num_rows)
    )


def _get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


class TestTraceDatasetDeserializer:
    @pytest.fixture
    def deserializer(self) -> TraceDatasetDeserializer:
        return TraceDatasetDeserializer()

    def _deserialize(self, deserializer, data, **kwargs):
        col_kwargs = _get_from_kwargs(
            (
                "timestamp_column",
                "prompt_tokens_column",
                "output_tokens_column",
            ),
            kwargs,
        )
        config = MinimalTraceFormatArgs(path=data, **col_kwargs)
        return deserializer(
            config=config,
            processor_factory=_mock_processor,
            random_seed=42,
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "suffix",
        [".json", ".jsonl"],
    )
    def test_loads_json(self, tmp_path: Path, deserializer, suffix):
        trace = _write_trace(
            tmp_path,
            '{"timestamp": 1, "input_length": 10, "output_length": 1}\n'
            '{"timestamp": 2, "input_length": 20, "output_length": 2}\n',
            suffix=suffix,
        )
        ds = self._deserialize(deserializer, trace)
        for i, row in enumerate(ds):
            assert row["relative_timestamp"] == i
            assert row["prompt_tokens_count"] == (i + 1) * 10
            assert row["output_tokens_count"] == i + 1

    @pytest.mark.sanity
    def test_loads_csv(self, tmp_path: Path, deserializer):
        trace = _write_trace(
            tmp_path,
            "timestamp,input_length,output_length\n1,10,1\n2,20,2\n",
            suffix=".csv",
        )
        ds = self._deserialize(deserializer, trace)
        for i, row in enumerate(ds):
            assert row["relative_timestamp"] == i
            assert row["prompt_tokens_count"] == (i + 1) * 10
            assert row["output_tokens_count"] == i + 1

    @pytest.mark.smoke
    def test_loads_sorted_rows_and_keeps_token_columns_aligned(
        self, tmp_path: Path, deserializer
    ):
        n_rows = 10
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: n_rows - i),
                    TraceColumnGenerator("input_length", lambda i: n_rows - i),
                    TraceColumnGenerator("output_length", lambda i: (n_rows - i) * 10),
                ],
            ),
        )
        ds = self._deserialize(deserializer, trace)
        assert isinstance(ds, IterableDataset)
        proc = _mock_processor()
        for i, row in enumerate(ds):
            assert row["prompt_tokens_count"] == i + 1
            assert row["output_tokens_count"] == (i + 1) * 10
            assert len(proc.encode(row["prompt"])) == row["prompt_tokens_count"]

    @pytest.mark.smoke
    def test_emits_relative_timestamp_column_sorted_from_trace(
        self, tmp_path: Path, deserializer
    ):
        n_rows = 5
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i + 3),
                    TraceColumnGenerator("input_length", lambda i: i),
                    TraceColumnGenerator("output_length", lambda i: i),
                ],
            ),
        )
        ds = self._deserialize(deserializer, trace)
        for i, row in enumerate(ds):
            assert row["relative_timestamp"] == i

    @pytest.mark.smoke
    def test_rejects_invalid_path(self, deserializer):
        with pytest.raises(ValidationError, match="not a valid path"):
            self._deserialize(deserializer, 123)
        with pytest.raises(DataNotSupportedError, match="file not found"):
            self._deserialize(deserializer, "bad_path.jsonl")
        with pytest.raises(DataNotSupportedError, match="not a file"):
            self._deserialize(deserializer, Path.cwd())

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("content", "kwargs", "match"),
        [
            ("", {}, "empty"),
            (
                '{"ts": 0, "input_length": 10, "output_length": 5}\n',
                {},
                "timestamp",
            ),
            (
                '{"timestamp": 0, "input_length": 10}\n',
                {},
                "output_length",
            ),
            (
                '{"timestamp": 0, "prompt_tokens": 10, "output_length": 5}\n',
                {
                    "prompt_tokens_column": "prompt_tokens",
                    "output_tokens_column": "out",
                },
                "out",
            ),
            (
                '{"timestamp": 0, "input_length": -1, "output_length": 5}\n',
                {},
                "non-negative",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": -1}\n',
                {},
                "non-negative",
            ),
            (
                '{"timestamp": "bad", "input_length": 10, "output_length": 5}\n',
                {},
                "scalar of type float",
            ),
            (
                '{"timestamp": 0, "input_length": "bad", "output_length": 5}\n',
                {},
                "scalar of type int32",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": null}\n',
                {},
                "NoneType",
            ),
        ],
    )
    def test_trace_validation_raises(
        self, tmp_path: Path, deserializer, content, kwargs, match
    ):
        trace = _write_trace(tmp_path, content)
        with pytest.raises(DataNotSupportedError, match=match):
            self._deserialize(deserializer, trace, **kwargs)

    @pytest.mark.sanity
    def test_unsupported_file_suffix_raises(self, tmp_path: Path, deserializer):
        trace = _write_trace(
            tmp_path,
            '{"timestamp": 0, "input_length": 10, "output_length": 5, '
            '"hash_ids": [0]}\n',
            suffix=".txt",
        )
        with pytest.raises(DataNotSupportedError, match=r"Unsupported.*\.txt"):
            self._deserialize(deserializer, trace)
