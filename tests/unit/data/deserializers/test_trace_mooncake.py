import copy
import dataclasses
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.deserializers.trace_mooncake import MooncakeTraceFormatArgs
from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)


def ascending_processor() -> Mock:
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


def compatible_processor() -> Mock:
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


def write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


def make_valid_hash_ids(prompt_lengths: list[int], block_size: int) -> list[list[int]]:
    """The final token block of every row may be less than the hash id block
    size due to the prompt length not being divisible by it. Use this
    when testing large trace prompt to avoid including token blocks with
    less than the block size in the middle of later rows."""
    tail_hash_ids = []
    n_rows = len(prompt_lengths)
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


def all_equal(items: list):
    return len(set(items)) == 1


def all_distinct(items: list):
    seen = set()
    return not any(i in seen or seen.add(i) for i in items)


@dataclasses.dataclass
class TraceColumnGenerator:
    name: str
    # Function with row index as the one argument
    data_generator: Callable[[int], Any]


def generate_trace(num_rows: int, columns: list[TraceColumnGenerator]) -> str:
    return "\n".join(
        "{"
        + ", ".join(f'"{col.name}": {col.data_generator(idx)}' for col in columns)
        + "}"
        for idx in range(num_rows)
    )


def load_graph_turns(row: dict) -> list[ConversationTurnData]:
    graph = ConversationGraphData.model_validate(json.loads(row["conversation_turns"]))
    return graph.turns


def get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


class TestMooncakeTraceFormat:
    @pytest.mark.regression
    def test_format_registered_with_deserializer(self, tmp_path: Path):
        trace = write_trace(
            tmp_path,
            '{"timestamp": 0.0, "input_length": 10, "output_length": 5, '
            '"hash_ids": [0]}\n',
        )
        DatasetDeserializerFactory.deserialize(
            config=MooncakeTraceFormatArgs(path=trace),
            processor_factory=ascending_processor,
            random_seed=42,
        )

    @pytest.fixture
    def default_block_size(self, tmp_path: Path) -> int:
        return MooncakeTraceFormatArgs(path=tmp_path).hash_id_block_size

    @pytest.fixture
    def deserializer(self) -> TraceDatasetDeserializer:
        return TraceDatasetDeserializer()

    def deserialize(self, deserializer, data, **kwargs):
        col_kwargs = get_from_kwargs(
            (
                "timestamp_column",
                "prompt_tokens_column",
                "output_tokens_column",
                "hash_ids_column",
                "hash_id_block_size",
            ),
            kwargs,
        )
        config = MooncakeTraceFormatArgs(path=data, **col_kwargs)
        return deserializer(
            config=config,
            processor_factory=ascending_processor,
            random_seed=42,
        )

    @pytest.mark.smoke
    def test_honors_custom_column_names(self, tmp_path: Path, deserializer):
        n_rows = 3
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("ts", lambda i: i),
                    TraceColumnGenerator("input_tokens", lambda i: i + 1),
                    TraceColumnGenerator("generated_tokens", lambda i: (i + 1) * 10),
                    TraceColumnGenerator("ids", lambda i: [i]),
                ],
            ),
        )
        self.deserialize(
            deserializer,
            trace,
            timestamp_column="ts",
            prompt_tokens_column="input_tokens",
            output_tokens_column="generated_tokens",
            hash_ids_column="ids",
        )

    @pytest.mark.smoke
    def test_custom_hash_id_block_size(self, tmp_path: Path, deserializer):
        n_rows = 1
        n_in = 1000
        trace = write_trace(
            tmp_path,
            generate_trace(
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
        self.deserialize(deserializer, trace, hash_id_block_size=n_in / 5)

    @pytest.mark.smoke
    def test_generates_large_trace_prompts(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        random.seed(0)
        n_rows = 25
        prompt_lengths = [random.randint(2000, 100000) for _ in range(n_rows)]
        output_lengths = [random.randint(3, 800) for _ in range(n_rows)]
        times = [0.0, 0.5, 1.0, 2.0]
        timestamps = [times[int(i / n_rows * len(times))] for i in range(n_rows)]
        hash_ids = make_valid_hash_ids(prompt_lengths, default_block_size)
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: timestamps[i]),
                    TraceColumnGenerator("input_length", lambda i: prompt_lengths[i]),
                    TraceColumnGenerator("output_length", lambda i: output_lengths[i]),
                    TraceColumnGenerator("hash_ids", lambda i: hash_ids[i]),
                ],
            ),
        )
        processor = ascending_processor()
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        for i, turn in enumerate(conv):
            n_in = turn.columns["prompt_tokens_count_column"][0]
            assert n_in == prompt_lengths[i]
            assert turn.columns["output_tokens_count_column"][0] == output_lengths[i]

            actual_length = len(processor.encode(turn.columns["text_column"][0]))
            if actual_length != n_in:
                pytest.fail(f"{actual_length} != {n_in}")

    @pytest.mark.smoke
    def test_prompt_matching_or_bordering_block_size(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 3
        n_in = list(range(default_block_size - 1, default_block_size + 2))
        hash_ids = make_valid_hash_ids(n_in, default_block_size)
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda i: n_in[i]),
                    TraceColumnGenerator("output_length", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: hash_ids[i]),
                ],
            ),
        )
        processor = compatible_processor()
        ds = deserializer(
            config=MooncakeTraceFormatArgs(path=trace),
            processor_factory=lambda: processor,
            random_seed=42,
        )
        conv = load_graph_turns(next(iter(ds)))
        for turn in conv:
            in_cnt = turn.columns["prompt_tokens_count_column"][0]
            actual_length = len(processor.encode(turn.columns["text_column"][0]))
            if actual_length != in_cnt:
                pytest.fail(f"{actual_length} != {in_cnt}")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (
                '{"timestamp": 0, "input_length": 10, "output_length": 5, '
                '"hash_ids": [-1]}\n',
                "non-negative",
            ),
            (
                '{"timestamp": 0, "input_length": 1024, "output_length": 5, '
                '"hash_ids": [0]}\n',
                "given 1 blocks",
            ),
        ],
    )
    def test_trace_validation_raises(
        self, tmp_path: Path, deserializer, content, match
    ):
        trace = write_trace(tmp_path, content)
        with pytest.raises(DataNotSupportedError, match=match):
            self.deserialize(deserializer, trace)

    @pytest.mark.sanity
    def test_incompatible_encoding_raises(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 2
        n_in = default_block_size * 2
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda _: n_in),
                    TraceColumnGenerator("output_length", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [0, i + 1]),
                ],
            ),
        )
        config = MooncakeTraceFormatArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=ascending_processor,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="generate distinct"):
            load_graph_turns(next(iter(ds)))

    @pytest.mark.smoke
    def test_token_block_distinctness(self, tmp_path: Path, deserializer):
        n_rows = 4
        n_in = 1024
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_length", lambda _: n_in),
                    TraceColumnGenerator("output_length", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [0, i + 1]),
                ],
            ),
        )
        ds = deserializer(
            config=MooncakeTraceFormatArgs(path=trace),
            processor_factory=compatible_processor,
            random_seed=42,
        )
        conv = load_graph_turns(next(iter(ds)))
        root_blocks, sibling_blocks = zip(
            *[
                (
                    turn.columns["text_column"][0][: n_in // 2],
                    turn.columns["text_column"][0][n_in // 2 :],
                )
                for turn in conv
            ],
            strict=False,
        )
        assert all_equal(root_blocks)
        assert all_distinct(sibling_blocks)
