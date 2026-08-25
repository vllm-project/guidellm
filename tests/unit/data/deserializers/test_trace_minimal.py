from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.schemas.conversation_graph_data import (
    ConversationGraphData,
    ConversationTurnData,
)
from guidellm.schemas.data import MinimalTraceFormatArgs


def mock_processor() -> Mock:
    """Tokenizer where each whitespace-delimited word is one token."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(len(text.split())))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        f"tok{i}" for i, _ in enumerate(tokens)
    )
    return proc


def write_trace(tmp_path: Path, content: str, suffix: str = ".jsonl") -> Path:
    path = tmp_path / f"trace{suffix}"
    path.write_text(content)
    return path


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


class TestMinimalTraceFormat:
    @pytest.mark.regression
    def test_format_registered_with_deserializer(self, tmp_path: Path):
        trace = write_trace(
            tmp_path,
            '{"timestamp": 0.0, "input_length": 10, "output_length": 5}\n',
        )
        DatasetDeserializerFactory.deserialize(
            config=MinimalTraceFormatArgs(path=trace),
            processor_factory=mock_processor,
            random_seed=42,
        )

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

    @pytest.mark.smoke
    def test_honors_custom_column_names(self, tmp_path: Path, deserializer):
        trace = write_trace(
            tmp_path,
            '{"ts": 3.0, "input_tokens": 4, "generated_tokens": 40}\n'
            '{"ts": 1.0, "input_tokens": 2, "generated_tokens": 20}\n',
        )

        ds = self.deserialize(
            deserializer,
            trace,
            timestamp_column="ts",
            prompt_tokens_column="input_tokens",
            output_tokens_column="generated_tokens",
        )
        conv = load_graph_turns(next(iter(ds)))

        prompt_counts = [2, 4]
        output_counts = [20, 40]
        for i, turn in enumerate(conv):
            assert turn.columns["prompt_tokens_count_column"][0] == prompt_counts[i]
            assert turn.columns["output_tokens_count_column"][0] == output_counts[i]

    @pytest.mark.smoke
    def test_generates_large_trace_prompts_from_reusable_base(
        self, tmp_path: Path, deserializer
    ):
        random.seed(0)
        n_rows = 25
        prompt_lengths = [random.randint(2000, 100000) for _ in range(n_rows)]
        output_lengths = [random.randint(3, 800) for _ in range(n_rows)]
        times = [0.0, 0.5, 1.0, 2.0]
        timestamps = [times[int(i / n_rows * len(times))] for i in range(n_rows)]
        trace = write_trace(
            tmp_path,
            generate_trace(
                n_rows,
                [
                    TraceColumnGenerator("timestamp", lambda i: timestamps[i]),
                    TraceColumnGenerator("input_length", lambda i: prompt_lengths[i]),
                    TraceColumnGenerator("output_length", lambda i: output_lengths[i]),
                ],
            ),
        )
        processor = mock_processor()
        config = MinimalTraceFormatArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: processor,
            random_seed=42,
        )
        conv = load_graph_turns(next(iter(ds)))

        assert processor.encode.call_count <= len(prompt_lengths) + 4
        for i, turn in enumerate(conv):
            n_in = turn.columns["prompt_tokens_count_column"][0]
            assert n_in == prompt_lengths[i]
            assert turn.columns["output_tokens_count_column"][0] == output_lengths[i]

            actual_length = len(processor.encode(turn.columns["text_column"][0]))
            if actual_length != n_in:
                pytest.fail(f"{actual_length} != {n_in}")
