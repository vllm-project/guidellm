import dataclasses
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.schemas import DataNotSupportedError
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)
from guidellm.schemas.data import WEKATraceFormatArgs


def ascending_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is assigned a token
    in ascending order starting from 0. This is incompatible with most WEKA
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
    WEKA traces as there is a way to generate distinct token blocks for
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
    """Hash IDs in WEKA format start at 1, and only create IDs for full token
    blocks."""
    hash_ids = []
    for length in prompt_lengths:
        n_ids = math.floor(length / block_size)
        hash_ids.append(list(range(1, n_ids + 1)))
    return hash_ids


@dataclasses.dataclass
class TraceColumnGenerator:
    name: str
    # Function with row index as the one argument
    data_generator: Callable[[int], Any]


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


def generate_weka_trace(
    num_rows: int,
    num_virtual_rows: int,
    non_wrapper_cols: list[TraceColumnGenerator],
    virtual_cols: list[TraceColumnGenerator],
) -> str:
    trace_rows = generate_trace(num_virtual_rows, virtual_cols)
    json_data = [json.loads(s) for s in trace_rows.split("\n")]
    cols = non_wrapper_cols + [TraceColumnGenerator("requests", lambda _: json_data)]
    return generate_trace(num_rows, cols)


def load_graph_turns(row: dict) -> list[ConversationTurnData]:
    graph = ConversationGraphData.model_validate(json.loads(row["conversation_turns"]))
    return graph.turns


def get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


def all_equal(items: list):
    return len(set(items)) == 1


def all_distinct(items: list):
    seen = set()
    return not any(i in seen or seen.add(i) for i in items)


class TestWEKATraceFormat:
    @pytest.mark.regression
    def test_format_registered_with_deserializer(self, tmp_path: Path):
        trace = write_trace(
            tmp_path,
            '{"id": "conv0", "requests": [{"t": 0, "in": 10,'
            '"out": 5, "hash_ids": []}]}\n',
        )
        DatasetDeserializerFactory.deserialize(
            config=WEKATraceFormatArgs(path=trace),
            processor_factory=ascending_processor,
            random_seed=42,
        )

    @pytest.fixture
    def default_block_size(self, tmp_path: Path) -> int:
        return WEKATraceFormatArgs(path=tmp_path).hash_id_block_size

    @pytest.fixture
    def deserializer(self) -> TraceDatasetDeserializer:
        return TraceDatasetDeserializer()

    def deserialize(self, deserializer, data, **kwargs):
        col_kwargs = get_from_kwargs(
            (
                "conversation_id_column",
                "timestamp_column",
                "prompt_tokens_column",
                "output_tokens_column",
                "hash_ids_column",
                "hash_id_block_size",
            ),
            kwargs,
        )
        config = WEKATraceFormatArgs(path=data, **col_kwargs)
        return deserializer(
            config=config,
            processor_factory=ascending_processor,
            random_seed=42,
        )

    @pytest.mark.smoke
    def test_honors_custom_column_names(self, tmp_path: Path, deserializer):
        n_rows = 1
        n_virtual_rows = 3
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("conv_id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("timestamp", lambda i: i),
                    TraceColumnGenerator("input_tokens", lambda i: i + 1),
                    TraceColumnGenerator("generated_tokens", lambda i: (i + 1) * 10),
                    TraceColumnGenerator("ids", lambda _: []),
                ],
            ),
        )
        self.deserialize(
            deserializer,
            trace,
            conversation_id_column="conv_id",
            timestamp_column="timestamp",
            prompt_tokens_column="input_tokens",
            output_tokens_column="generated_tokens",
            hash_ids_column="ids",
        )

    @pytest.mark.smoke
    def test_custom_hash_id_block_size(self, tmp_path: Path, deserializer):
        n_rows = 1
        n_virtual_rows = 1
        n_in = 1000
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: n_in),
                    TraceColumnGenerator("out", lambda i: i + 1),
                    # Would throw a DataNotSupportedError with default block size 64
                    # See row validation in trace_weka.py
                    TraceColumnGenerator("hash_ids", lambda _: [1, 2, 3, 4, 5]),
                ],
            ),
        )
        self.deserialize(deserializer, trace, hash_id_block_size=n_in / 5)

    @pytest.mark.smoke
    def test_generates_large_trace_prompts(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        random.seed(0)
        n_rows = 1
        n_virtual_rows = 25
        prompt_lengths = [random.randint(2000, 100000) for _ in range(n_virtual_rows)]
        output_lengths = [random.randint(3, 800) for _ in range(n_virtual_rows)]
        times = [0.0, 0.5, 1.0, 2.0]
        timestamps = [
            times[int(i / n_virtual_rows * len(times))] for i in range(n_virtual_rows)
        ]
        hash_ids = make_valid_hash_ids(prompt_lengths, default_block_size)
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: timestamps[i]),
                    TraceColumnGenerator("in", lambda i: prompt_lengths[i]),
                    TraceColumnGenerator("out", lambda i: output_lengths[i]),
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
        n_rows = 1
        n_virtual_rows = 3
        n_in = list(range(default_block_size - 1, default_block_size + 2))
        hash_ids = make_valid_hash_ids(n_in, default_block_size)
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda i: n_in[i]),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: hash_ids[i]),
                ],
            ),
        )
        processor = ascending_processor()
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        for turn in conv:
            in_cnt = turn.columns["prompt_tokens_count_column"][0]
            actual_length = len(processor.encode(turn.columns["text_column"][0]))
            if actual_length != in_cnt:
                pytest.fail(f"{actual_length} != {in_cnt}")

    @pytest.mark.sanity
    def test_removes_partially_filled_hash_ids(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 1
        n_virtual_rows = 2
        n_in = default_block_size + 1
        hash_ids = make_valid_hash_ids([n_in], default_block_size)[0]
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: n_in),
                    TraceColumnGenerator("out", lambda i: i),
                    TraceColumnGenerator("hash_ids", lambda i: hash_ids + [i + 2]),
                ],
            ),
        )
        processor = ascending_processor()
        ds = self.deserialize(deserializer, trace)
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
                '{"id": "conv0", "requests": [{"t": 0, "in": 10,'
                '"out": 5, "hash_ids": [-1]}]}\n',
                "non-negative",
            ),
            (
                '{"id": "conv0", "requests": [{"t": 0, "in": 1024,'
                '"out": 5, "hash_ids": [1]}]}\n',
                "given 1 blocks",
            ),
            (
                '{"id": "conv0", "requests": [[{"t": 0, "in": 10,'
                '"out": 5, "hash_ids": []}]]}\n',
                "Failed to find requests",
            ),
            (
                '{"id": "conv0", "requests": []}\n',
                "requests was empty",
            ),
            (
                '{"id": "conv0", "requests": [{"t": 0, "in": 10, "out": 5,'
                '"hash_ids": [], "model": "a", "ttft": 0.5}]}\n'
                '{"id": "conv1", "requests": [{"t": 0, "in": 10, "out": 5,'
                '"model": "a"}]}\n',
                "Missing column values in hash_ids",
            ),
        ],
    )
    def test_trace_validation_raises(
        self, tmp_path: Path, deserializer, content, match
    ):
        trace = write_trace(tmp_path, content)
        with pytest.raises(DataNotSupportedError, match=match):
            self.deserialize(deserializer, trace)

    @pytest.mark.regression
    def test_rejects_empty_conversation_after_first_row(
        self, tmp_path: Path, deserializer
    ):
        """Reject an empty nested conversation during deserialization.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            '{"id": "conv0", "requests": [{"t": 0, "in": 10, '
            '"out": 5, "hash_ids": []}]}\n'
            '{"id": "conv1", "requests": []}\n',
        )
        with pytest.raises(DataNotSupportedError, match="conversation is empty"):
            self.deserialize(deserializer, trace)

    @pytest.mark.regression
    def test_accepts_spec_metadata_on_request_records(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        """Accept WEKA request records carrying the fields the specification defines.

        Those are model, type, input_types, output_types and stop on every
        record, plus api_time, ttft and think_time, which are optional.

        ## WRITTEN BY AI ##
        """
        first_in = 2 * default_block_size
        second_in = 3 * default_block_size
        trace = write_trace(
            tmp_path,
            '{"id": "conv0", "hash_id_scope": "local", "requests": ['
            f'{{"t": 0.0, "type": "s", "model": "a", "in": {first_in}, "out": 205, '
            '"hash_ids": [1, 2], "input_types": ["text"], '
            '"output_types": ["thinking", "tool_use"], "stop": "tool_use", '
            '"api_time": 4.87, "ttft": 0.5, "think_time": 0.0}, '
            f'{{"t": 17.0, "type": "n", "model": "a", "in": {second_in}, "out": 82, '
            '"hash_ids": [1, 2, 3], "input_types": ["tool_result"], '
            '"output_types": ["tool_use"], "stop": "tool_use", '
            '"api_time": 2.87}]}\n',
        )
        ds = self.deserialize(deserializer, trace)
        turns = load_graph_turns(next(iter(ds)))
        assert len(turns) == 2
        assert [turn.columns["prompt_tokens_count_column"][0] for turn in turns] == [
            first_in,
            second_in,
        ]
        assert [turn.columns["output_tokens_count_column"][0] for turn in turns] == [
            205,
            82,
        ]
        assert [turn.columns["relative_timestamp_column"][0] for turn in turns] == [
            0.0,
            17.0,
        ]

    @pytest.mark.sanity
    def test_incompatible_encoding_raises(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 1
        n_virtual_rows = 2
        n_in = default_block_size * 2
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: n_in),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [1, i + 2]),
                ],
            ),
        )
        ds = deserializer(
            config=WEKATraceFormatArgs(path=trace),
            processor_factory=ascending_processor,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="generate distinct"):
            load_graph_turns(next(iter(ds)))

    @pytest.mark.smoke
    def test_token_block_distinctness(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 1
        n_virtual_rows = 4
        n_in = default_block_size * 2
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: n_in),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [1, i + 2]),
                ],
            ),
        )
        ds = deserializer(
            config=WEKATraceFormatArgs(path=trace),
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

    @pytest.mark.smoke
    def test_multi_conversation_resets_relative_timestamp(
        self, tmp_path: Path, deserializer
    ):
        n_rows = 2
        n_virtual_rows = 3
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: 10),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda _: []),
                ],
            ),
        )
        ds = self.deserialize(deserializer, trace)
        ds_iter = iter(ds)
        conv1 = load_graph_turns(next(ds_iter))
        conv2 = load_graph_turns(next(ds_iter))
        timestamps1 = [turn.columns["relative_timestamp_column"][0] for turn in conv1]
        timestamps2 = [turn.columns["relative_timestamp_column"][0] for turn in conv2]
        assert timestamps1[0] == 0.0
        assert timestamps1[1] != 0.0
        assert timestamps2[0] == 0.0

    @pytest.mark.sanity
    def test_multi_conversation_resets_hash_id_state(
        self, tmp_path: Path, deserializer, default_block_size
    ):
        n_rows = 2
        n_virtual_rows = 2
        n_in = default_block_size * 2
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                n_rows,
                n_virtual_rows,
                [TraceColumnGenerator("id", lambda i: f'"conv{i}"')],
                [
                    TraceColumnGenerator("t", lambda i: i),
                    TraceColumnGenerator("in", lambda _: n_in),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda i: [1, i + 2]),
                ],
            ),
        )
        ds = deserializer(
            config=WEKATraceFormatArgs(path=trace),
            processor_factory=compatible_processor,
            random_seed=42,
        )
        ds_iter = iter(ds)
        conv1 = load_graph_turns(next(ds_iter))
        conv2 = load_graph_turns(next(ds_iter))
        prompts1 = [turn.columns["text_column"][0] for turn in conv1]
        prompts2 = [turn.columns["text_column"][0] for turn in conv2]
        assert prompts1[0] != prompts2[0]

    @pytest.mark.sanity
    def test_zero_prompt_tokens_empty_hash_ids(self, tmp_path: Path, deserializer):
        trace = write_trace(
            tmp_path,
            generate_weka_trace(
                1,
                1,
                [TraceColumnGenerator("id", lambda _: '"conv0"')],
                [
                    TraceColumnGenerator("t", lambda _: 0.0),
                    TraceColumnGenerator("in", lambda _: 0),
                    TraceColumnGenerator("out", lambda _: 5),
                    TraceColumnGenerator("hash_ids", lambda _: []),
                ],
            ),
        )
        ds = self.deserialize(deserializer, trace)
        conv = load_graph_turns(next(iter(ds)))
        turns = list(conv)
        assert turns[0].columns["prompt_tokens_count_column"][0] == 0
        assert turns[0].columns["output_tokens_count_column"][0] == 5
        assert turns[0].node_id == "main_0"
        assert turns[0].agent_id == "default"

    @pytest.mark.smoke
    def test_subagent_spawns_and_joins_parent(self, tmp_path: Path, deserializer):
        """A declared subagent chain spawns from the preceding parent and
        joins the following parent turn.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "explore",
                            "requests": [
                                {"t": 0.0, "in": 8, "out": 2, "hash_ids": []},
                                {"t": 0.5, "in": 8, "out": 2, "hash_ids": []},
                            ],
                        },
                        {"t": 10.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        turns = {turn.node_id: turn for turn in load_graph_turns(next(iter(ds)))}
        assert set(turns) == {"main_0", "sa_0_0", "sa_0_1", "main_1"}
        assert turns["main_0"].agent_id == "default"
        assert turns["sa_0_0"].agent_id == "explore"
        assert turns["sa_0_1"].agent_id == "explore"
        assert turns["sa_0_0"].parents[0].parent_node_id == "main_0"
        assert turns["sa_0_0"].parents[0].history_context == "new"
        assert turns["sa_0_1"].parents[0].parent_node_id == "sa_0_0"
        assert turns["sa_0_1"].parents[0].history_context == "full"
        main_1_parents = {
            parent.parent_node_id: parent.history_context
            for parent in turns["main_1"].parents
        }
        assert main_1_parents == {"main_0": "full", "sa_0_1": "last"}

    @pytest.mark.smoke
    def test_parallel_subagents_share_spawn_and_join(
        self, tmp_path: Path, deserializer
    ):
        """Adjacent subagents between the same parent turns run in parallel.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "agentA",
                            "requests": [
                                {"t": 0.0, "in": 8, "out": 2, "hash_ids": []},
                                {"t": 0.4, "in": 8, "out": 2, "hash_ids": []},
                            ],
                        },
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "agentB",
                            "requests": [
                                {"t": 0.0, "in": 8, "out": 2, "hash_ids": []},
                            ],
                        },
                        {"t": 10.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        turns = {turn.node_id: turn for turn in load_graph_turns(next(iter(ds)))}
        assert turns["sa_0_0"].agent_id == "agentA"
        assert turns["sa_1_0"].agent_id == "agentB"
        assert turns["sa_0_0"].parents[0].parent_node_id == "main_0"
        assert turns["sa_0_0"].parents[0].history_context == "new"
        assert turns["sa_1_0"].parents[0].parent_node_id == "main_0"
        assert turns["sa_1_0"].parents[0].history_context == "new"
        sa_parent_ids = {
            parent.parent_node_id for parent in turns["sa_0_0"].parents
        } | {parent.parent_node_id for parent in turns["sa_1_0"].parents}
        assert "sa_0_0" not in {
            parent.parent_node_id for parent in turns["sa_1_0"].parents
        }
        assert "sa_1_0" not in {
            parent.parent_node_id for parent in turns["sa_0_0"].parents
        }
        assert sa_parent_ids == {"main_0"}
        main_1_parents = {
            parent.parent_node_id: parent.history_context
            for parent in turns["main_1"].parents
        }
        assert main_1_parents == {
            "main_0": "full",
            "sa_0_1": "last",
            "sa_1_0": "last",
        }

    @pytest.mark.sanity
    def test_inner_timestamps_relative_to_spawn(self, tmp_path: Path, deserializer):
        """Inner t smaller than spawn t is treated as relative to spawn.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 100.0,
                            "type": "subagent",
                            "agent_id": "explore",
                            "requests": [
                                {"t": 0.0, "in": 8, "out": 2, "hash_ids": []},
                                {"t": 3.0, "in": 8, "out": 2, "hash_ids": []},
                            ],
                        },
                        {"t": 200.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        turns = {turn.node_id: turn for turn in load_graph_turns(next(iter(ds)))}
        assert turns["main_0"].columns["relative_timestamp_column"][0] == 0.0
        assert turns["sa_0_0"].columns["relative_timestamp_column"][0] == 100.0
        assert turns["sa_0_1"].columns["relative_timestamp_column"][0] == 103.0
        assert turns["main_1"].columns["relative_timestamp_column"][0] == 200.0

    @pytest.mark.sanity
    def test_inner_timestamps_absolute_when_not_before_spawn(
        self, tmp_path: Path, deserializer
    ):
        """Inner t on the parent timeline is left as absolute.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 100.0,
                            "type": "subagent",
                            "agent_id": "explore",
                            "requests": [
                                {"t": 100.0, "in": 8, "out": 2, "hash_ids": []},
                                {"t": 103.0, "in": 8, "out": 2, "hash_ids": []},
                            ],
                        },
                        {"t": 200.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        turns = {turn.node_id: turn for turn in load_graph_turns(next(iter(ds)))}
        assert turns["sa_0_0"].columns["relative_timestamp_column"][0] == 100.0
        assert turns["sa_0_1"].columns["relative_timestamp_column"][0] == 103.0

    @pytest.mark.sanity
    def test_subagent_entries_do_not_fail_validation(
        self, tmp_path: Path, deserializer
    ):
        """Subagent marker rows lack in/out/hash_ids and must still load.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "explore",
                            "requests": [{"t": 0.0, "in": 8, "out": 2, "hash_ids": []}],
                        },
                        {"t": 2.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        turns = load_graph_turns(next(iter(ds)))
        assert len(turns) == 3

    @pytest.mark.sanity
    @patch("guidellm.data.deserializers.trace_weka.logger")
    def test_overlap_warns_on_same_chain(
        self, mock_logger, tmp_path: Path, deserializer
    ):
        """Consecutive turns of one agent that overlap in time are logged at debug.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {
                            "t": 0.0,
                            "in": 10,
                            "out": 5,
                            "hash_ids": [],
                            "api_time": 5.0,
                        },
                        {"t": 1.0, "in": 10, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        load_graph_turns(next(iter(ds)))
        messages = [
            call.args[0].format(*call.args[1:]) if call.args else ""
            for call in mock_logger.debug.call_args_list
        ]
        assert any("overlapping requests" in message for message in messages)
        assert any("will run until t=" in message for message in messages)
        assert any("default" in message for message in messages)

    @pytest.mark.sanity
    @patch("guidellm.data.deserializers.trace_weka.logger")
    def test_overlap_does_not_warn_for_parallel_subagents(
        self, mock_logger, tmp_path: Path, deserializer
    ):
        """Parallel subagents may share timestamps without an overlap debug log.

        ## WRITTEN BY AI ##
        """
        trace = write_trace(
            tmp_path,
            json.dumps(
                {
                    "id": "conv0",
                    "requests": [
                        {"t": 0.0, "in": 10, "out": 5, "hash_ids": []},
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "agentA",
                            "requests": [{"t": 0.0, "in": 8, "out": 2, "hash_ids": []}],
                        },
                        {
                            "t": 1.0,
                            "type": "subagent",
                            "agent_id": "agentB",
                            "requests": [{"t": 0.0, "in": 8, "out": 2, "hash_ids": []}],
                        },
                        {"t": 10.0, "in": 12, "out": 5, "hash_ids": []},
                    ],
                }
            ),
        )
        ds = self.deserialize(deserializer, trace)
        load_graph_turns(next(iter(ds)))
        messages = [
            call.args[0].format(*call.args[1:]) if call.args else ""
            for call in mock_logger.debug.call_args_list
        ]
        assert not any("overlapping requests" in message for message in messages)
