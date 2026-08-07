import dataclasses
import json
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
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)
from guidellm.utils.hf_datasets import load_dataset_from_file


def mock_processor() -> Mock:
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
    proc = mock_processor()
    assert decode_prompt(proc, token_ids) == expected


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [
        (0, ()),
        (1, (0,)),
        (10, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
        (1000, tuple(range(1000))),
    ],
)
def test_generate_token_ids(token_count, expected):
    proc = mock_processor()
    faker = Faker()
    res = generate_token_ids(token_count, proc, faker)
    assert len(res) == len(expected)
    assert res == expected


class TestTraceFormatRegistry:
    def test_unknown_kind_raises(self, tmp_path: Path):
        trace = write_trace(
            tmp_path,
            '{"timestamp": 1, "input_length": 10, "output_length": 1}\n',
        )
        config = TraceDataArgs(kind="unknown_kind", path=tmp_path)
        dataset = load_dataset_from_file(trace)
        with pytest.raises(DataNotSupportedError, match="not registered"):
            TraceFormatRegistry.dispatch(config, dataset)


@dataclasses.dataclass
class TraceColumnGenerator:
    name: str
    # Function with row index as the one argument
    data_generator: Callable[[int], Any]


def write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


def generate_trace(num_rows: int, columns: list[TraceColumnGenerator]) -> str:
    """Returns valid JSON lines."""
    return "\n".join(
        "{"
        + ", ".join(
            f'"{col.name}": {col.data_generator(idx)}' for col in columns
        ).replace("'", '"')
        + "}"
        for idx in range(num_rows)
    )


def load_graph(row: dict) -> ConversationGraphData:
    return ConversationGraphData.model_validate(json.loads(row["conversation_turns"]))


def load_graph_turns(row: dict) -> list[ConversationTurnData]:
    return load_graph(row).turns


def get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


class TestTraceDatasetDeserializer:
    @pytest.fixture
    def deserializer(self) -> TraceDatasetDeserializer:
        return TraceDatasetDeserializer()

    def deserialize(self, deserializer, data, **kwargs):
        col_kwargs = get_from_kwargs(
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
            processor_factory=mock_processor,
            random_seed=42,
        )

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "suffix",
        [".json", ".jsonl"],
    )
    def test_loads_json(self, tmp_path: Path, deserializer, suffix):
        trace = write_trace(
            tmp_path,
            '{"timestamp": 1, "input_length": 10, "output_length": 1}\n'
            '{"timestamp": 2, "input_length": 20, "output_length": 2}\n',
            suffix=suffix,
        )
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        for i, turn in enumerate(conv):
            assert turn.columns["relative_timestamp_column"][0] == i
            assert turn.columns["prompt_tokens_count_column"][0] == (i + 1) * 10
            assert turn.columns["output_tokens_count_column"][0] == i + 1

    @pytest.mark.sanity
    def test_loads_csv(self, tmp_path: Path, deserializer):
        trace = write_trace(
            tmp_path,
            "timestamp,input_length,output_length\n1,10,1\n2,20,2\n",
            suffix=".csv",
        )
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        for i, turn in enumerate(conv):
            assert turn.columns["relative_timestamp_column"][0] == i
            assert turn.columns["prompt_tokens_count_column"][0] == (i + 1) * 10
            assert turn.columns["output_tokens_count_column"][0] == i + 1

    @pytest.mark.smoke
    def test_loads_sorted_rows_and_keeps_token_columns_aligned(
        self, tmp_path: Path, deserializer
    ):
        n_rows = 10
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: n_rows - i),
                    TraceColumnGenerator("input_length", lambda i: n_rows - i),
                    TraceColumnGenerator("output_length", lambda i: (n_rows - i) * 10),
                ],
            ),
        )
        ds = self.deserialize(deserializer, trace)
        assert isinstance(ds, IterableDataset)
        conv = load_graph(next(iter(ds)))
        proc = mock_processor()
        assert len(conv.turns) == n_rows
        for i, turn in enumerate(conv.turns):
            assert turn.node_id == f"main_{i}"
            if i > 0:
                assert turn.parents[0].parent_node_id == f"main_{i - 1}"
            n_in = turn.columns["prompt_tokens_count_column"][0]
            assert n_in == i + 1
            assert turn.columns["output_tokens_count_column"][0] == (i + 1) * 10
            assert len(proc.encode(turn.columns["text_column"][0])) == n_in

    @pytest.mark.smoke
    def test_emits_relative_timestamp_column_column_sorted_from_trace(
        self, tmp_path: Path, deserializer
    ):
        n_rows = 5
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i + 3),
                    TraceColumnGenerator("input_length", lambda i: i),
                    TraceColumnGenerator("output_length", lambda i: i),
                ],
            ),
        )
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        for i, turn in enumerate(conv):
            assert turn.columns["relative_timestamp_column"][0] == i

    @pytest.mark.smoke
    def test_rejects_invalid_path(self, deserializer):
        with pytest.raises(ValidationError, match="not a valid path"):
            self.deserialize(deserializer, 123)
        with pytest.raises(DataNotSupportedError, match="file not found"):
            self.deserialize(deserializer, "bad_path.jsonl")
        with pytest.raises(DataNotSupportedError, match="not a file"):
            self.deserialize(deserializer, Path.cwd())

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
                "Missing column values",
            ),
        ],
    )
    def test_trace_validation_raises(
        self, tmp_path: Path, deserializer, content, kwargs, match
    ):
        trace = write_trace(tmp_path, content)
        with pytest.raises(DataNotSupportedError, match=match):
            self.deserialize(deserializer, trace, **kwargs)

    @pytest.mark.sanity
    def test_unsupported_file_suffix_raises(self, tmp_path: Path, deserializer):
        trace = write_trace(
            tmp_path,
            '{"timestamp": 0, "input_length": 10, "output_length": 5, '
            '"hash_ids": [0]}\n',
            suffix=".txt",
        )
        with pytest.raises(DataNotSupportedError, match=r"Unsupported.*\.txt"):
            self.deserialize(deserializer, trace)
