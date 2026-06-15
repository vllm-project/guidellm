import copy
import dataclasses
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from datasets import IterableDataset
from pydantic import ValidationError

from guidellm.data.deserializers.trace_mooncake import (
    TraceMooncakeDataArgs,
    TraceMooncakeDatasetDeserializer,
)
from guidellm.data.schemas import DataNotSupportedError


def _ascending_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is assigned a token
    in ascending order starting from 0. This is incompatible with most mooncake
    traces as there is no way to generate distinct token blocks for sibling
    nodes."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{i}" for i, _ in enumerate(tokens)
    )
    return proc


def _compatible_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is assigned a token
    selected from a range of random integers. This is compatible with most
    mooncake traces as there is a way to generate distinct token blocks for
    sibling nodes."""
    random.seed(0)
    proc = Mock()
    proc.encode.side_effect = lambda text: [
        random.randint(0, 1000) for _ in range(len(text.split()))
    ]
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{t}" for t in tokens
    )
    return proc


def _write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


def _make_valid_hash_ids(
    n_rows: int, prompt_lengths: list[int], block_size: int
) -> list[list[int]]:
    """The final token block of every row may be less than the hash id block
    size due to the prompt length not being divisible by it. Use this
    when testing large trace prompts to avoid including token blocks with
    less than the block size in the middle of later rows."""
    tail_hash_ids = []
    original_prompt_positions = dict(zip(prompt_lengths, range(n_rows), strict=False))
    sorted_lengths = copy.deepcopy(prompt_lengths)
    sorted_lengths.sort()
    hash_ids = [None for _ in range(n_rows)]
    for length in sorted_lengths:
        original_position = original_prompt_positions[length]
        n_blocks = math.ceil(length / block_size)
        n_to_generate = n_blocks + len(tail_hash_ids)
        hash_ids[original_position] = [
            i for i in range(n_to_generate) if i not in tail_hash_ids
        ]
        tail_hash_ids.append(hash_ids[original_position][-1])
    return hash_ids


def _all_equal(items: list):
    return len(set(items)) == 1


def _all_distinct(items: list):
    seen = set()
    return not any(i in seen or seen.add(i) for i in items)


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


class TestTraceMooncakeDatasetDeserializer:
    @pytest.fixture
    def deserializer(self) -> TraceMooncakeDatasetDeserializer:
        return TraceMooncakeDatasetDeserializer()

    def _deserialize(self, deserializer, data, **kwargs):
        field_names = (
            "timestamp_column",
            "prompt_tokens_column",
            "output_tokens_column",
            "hash_ids_column",
            "hash_id_block_size",
        )
        col_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        config = TraceMooncakeDataArgs(path=data, **col_kwargs)
        return deserializer(
            config=config,
            processor_factory=_ascending_processor,
            random_seed=42,
        )

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
                    TraceColumnGenerator("hash_ids", lambda i: [n_rows - i]),
                ],
            ),
        )
        ds = self._deserialize(deserializer, trace)
        assert isinstance(ds, IterableDataset)
        proc = _ascending_processor()
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
                    TraceColumnGenerator("input_length", lambda i: i + 1),
                    TraceColumnGenerator("output_length", lambda i: (i + 1) * 10),
                    TraceColumnGenerator("hash_ids", lambda i: [i]),
                ],
            ),
        )
        ds = self._deserialize(deserializer, trace)
        for i, row in enumerate(ds):
            assert row["relative_timestamp"] == i

    @pytest.mark.smoke
    def test_honors_custom_column_names(self, tmp_path: Path, deserializer):
        n_rows = 3
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("ts", lambda i: i),
                    TraceColumnGenerator("input_tokens", lambda i: i + 1),
                    TraceColumnGenerator("generated_tokens", lambda i: (i + 1) * 10),
                    TraceColumnGenerator("ids", lambda i: [i]),
                ],
            ),
        )
        ds = self._deserialize(
            deserializer,
            trace,
            timestamp_column="ts",
            prompt_tokens_column="input_tokens",
            output_tokens_column="generated_tokens",
            hash_ids_column="ids",
        )
        for i, row in enumerate(ds):
            assert row["prompt_tokens_count"] == i + 1
            assert row["output_tokens_count"] == (i + 1) * 10

    @pytest.mark.smoke
    def test_custom_hash_id_block_size(self, tmp_path: Path, deserializer):
        n_rows = 1
        n_in = 1000
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda _: n_in),
                    TraceColumnGenerator("output_length", lambda i: i + 1),
                    # Would throw a DataNotSupportedError with default block size 512
                    # See row validation in trace_mooncake.py
                    TraceColumnGenerator("hash_ids", lambda _: [0, 1, 2, 3, 4]),
                ],
            ),
        )
        self._deserialize(deserializer, trace, hash_id_block_size=n_in / 5)

    @pytest.mark.smoke
    def test_generates_large_trace_prompts(self, tmp_path: Path, deserializer):
        random.seed(0)
        n_rows = 25
        prompt_lengths = [random.randint(2000, 100000) for _ in range(n_rows)]
        output_lengths = [random.randint(3, 800) for _ in range(n_rows)]
        times = [0.0, 0.5, 1.0, 2.0]
        timestamps = [times[int(i / n_rows * len(times))] for i in range(n_rows)]
        block_size = TraceMooncakeDataArgs(path=tmp_path).hash_id_block_size
        hash_ids = _make_valid_hash_ids(n_rows, prompt_lengths, block_size)
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: timestamps[i]),
                    TraceColumnGenerator("input_length", lambda i: prompt_lengths[i]),
                    TraceColumnGenerator("output_length", lambda i: output_lengths[i]),
                    TraceColumnGenerator("hash_ids", lambda i: hash_ids[i]),
                ],
            ),
        )
        processor = _ascending_processor()
        config = TraceMooncakeDataArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: processor,
            random_seed=42,
        )

        for i, row in enumerate(ds):
            in_cnt = row["prompt_tokens_count"]
            assert in_cnt == prompt_lengths[i]
            assert row["output_tokens_count"] == output_lengths[i]

            actual_prompt_length = len(processor.encode(row["prompt"]))
            if actual_prompt_length != in_cnt:
                pytest.fail(f"{actual_prompt_length} != {in_cnt}")

    @pytest.mark.smoke
    def test_rejects_invalid_path(self, deserializer):
        with pytest.raises(ValidationError, match="not a valid path"):
            self._deserialize(deserializer, 123)
        with pytest.raises(DataNotSupportedError, match="path to a trace file"):
            self._deserialize(deserializer, "bad_path.jsonl")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("content", "kwargs", "match"),
        [
            ("", {}, "empty"),
            (
                '{"ts": 0, "input_length": 10, "output_length": 5, "hash_ids": [0]}\n',
                {},
                "timestamp",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "hash_ids": [0]}\n',
                {},
                "output_length",
            ),
            (
                '{"timestamp": 0, "prompt_tokens": 10, "output_length": 5, '
                '"hash_ids": [0]}\n',
                {
                    "prompt_tokens_column": "prompt_tokens",
                    "output_tokens_column": "out",
                },
                "out",
            ),
            (
                '{"timestamp": "bad", "input_length": 10, "output_length": 5, '
                '"hash_ids": [0]}\n',
                {},
                "scalar of type float",
            ),
            (
                '{"timestamp": 0, "input_length": "bad", "output_length": 5, '
                '"hash_ids": [0]}\n',
                {},
                "scalar of type int32",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": null, '
                '"hash_ids": [0]}\n',
                {},
                "NoneType",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": 5, '
                '"hash_ids": [0]}\nnot-json\n',
                {},
                "generating the dataset",
            ),
            (
                '{"timestamp": 0, "input_length": 1024, "output_length": 5, '
                '"hash_ids": [0]}\n',
                {},
                "given 1 blocks",
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
            suffix=".json",
        )
        with pytest.raises(DataNotSupportedError, match=r"Unsupported.*\.json"):
            self._deserialize(deserializer, trace)

    @pytest.mark.sanity
    def test_incompatible_encoding_raises(self, tmp_path: Path, deserializer):
        n_rows = 2
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda _: 1024),
                    TraceColumnGenerator("output_length", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [0, i + 1]),
                ],
            ),
        )
        config = TraceMooncakeDataArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: _ascending_processor(),
            random_seed=42,
        )
        with pytest.raises(ValueError, match="generate distinct"):
            for _ in ds:
                ...

    @pytest.mark.smoke
    def test_token_block_distinctness(self, tmp_path: Path, deserializer):
        n_rows = 4
        n_in = 1024
        trace = _write_trace(
            tmp_path,
            _generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda _: n_in),
                    TraceColumnGenerator("output_length", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [0, i + 1]),
                ],
            ),
        )
        config = TraceMooncakeDataArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: _compatible_processor(),
            random_seed=42,
        )
        root_blocks, sibling_blocks = zip(
            *[(row["prompt"][: n_in // 2], row["prompt"][n_in // 2 :]) for row in ds],
            strict=False,
        )
        assert _all_equal(root_blocks)
        assert _all_distinct(sibling_blocks)
