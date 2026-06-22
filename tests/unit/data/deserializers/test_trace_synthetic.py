from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from guidellm.data.deserializers.trace_common import TraceDatasetDeserializer
from guidellm.data.deserializers.trace_synthetic import MinimalTraceFormatArgs


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


def _get_from_kwargs(keys, kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in keys}


class TestMinimalTraceFormat:
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
    
    @pytest.mark.smoke
    def test_honors_custom_column_names(self, tmp_path: Path, deserializer):
        trace = _write_trace(
            tmp_path,
            '{"ts": 3.0, "input_tokens": 4, "generated_tokens": 40}\n'
            '{"ts": 1.0, "input_tokens": 2, "generated_tokens": 20}\n',
        )

        ds = self._deserialize(
            deserializer,
            trace,
            timestamp_column="ts",
            prompt_tokens_column="input_tokens",
            output_tokens_column="generated_tokens",
        )

        expected_prompt_count = [2, 4]
        expected_output_count = [20, 40]
        for i, row in enumerate(ds):
            assert row["prompt_tokens_count"] == expected_prompt_count[i]
            assert row["output_tokens_count"] == expected_output_count[i]

    @pytest.mark.smoke
    def test_generates_large_trace_prompts_from_reusable_base(
        self, tmp_path: Path, deserializer
    ):
        prompt_lengths = [
            6755,
            7319,
            7234,
            2287,
            9013,
            6506,
            4824,
            3119,
            23090,
            3135,
            26874,
            10487,
            17448,
            6253,
            6725,
            13538,
            87162,
            6166,
            6320,
            2007,
            3174,
            3131,
            3159,
            6820,
            3154,
            9416,
            7460,
        ]
        output_lengths = [
            500,
            490,
            794,
            316,
            3,
            3,
            173,
            20,
            453,
            19,
            458,
            402,
            610,
            3,
            32,
            71,
            402,
            24,
            548,
            354,
            19,
            23,
            20,
            26,
            21,
            145,
            3,
        ]
        timestamps = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
        trace = _write_trace(
            tmp_path,
            "\n".join(
                (
                    f'{{"timestamp": {timestamp}, '
                    f'"input_length": {prompt_length}, '
                    f'"output_length": {output_length}}}'
                )
                for timestamp, prompt_length, output_length in zip(
                    timestamps, prompt_lengths, output_lengths, strict=True
                )
            ),
        )
        processor = _mock_processor()

        config = MinimalTraceFormatArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: processor,
            random_seed=42,
        )

        assert processor.encode.call_count <= len(prompt_lengths) + 4
        for i, row in enumerate(ds):
            in_cnt = row["prompt_tokens_count"]
            assert in_cnt == prompt_lengths[i]
            assert row["output_tokens_count"] == output_lengths[i]

            actual_prompt_length = len(processor.encode(row["prompt"]))
            if actual_prompt_length != in_cnt:
                pytest.fail(f"{actual_prompt_length} != {in_cnt}")
