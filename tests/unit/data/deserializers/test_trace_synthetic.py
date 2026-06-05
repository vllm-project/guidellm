from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import Dataset
from pydantic import ValidationError

from guidellm.data.deserializers.trace_synthetic import (
    TraceSyntheticDataArgs,
    TraceSyntheticDatasetDeserializer,
)
from guidellm.data.schemas import DataNotSupportedError


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


class TestTraceSyntheticDatasetDeserializer:
    @pytest.fixture
    def deserializer(self) -> TraceSyntheticDatasetDeserializer:
        return TraceSyntheticDatasetDeserializer()

    def _deserialize(self, deserializer, data, **kwargs):
        col_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("timestamp_column", "prompt_tokens_column", "output_tokens_column")
        }
        try:
            config = TraceSyntheticDataArgs(path=data, **col_kwargs)
        except ValidationError as e:
            raise DataNotSupportedError(
                f"Expected a path to a trace file, got {data!r}"
            ) from e
        return deserializer(
            config=config,
            processor_factory=_mock_processor,
            random_seed=42,
        )

    @pytest.mark.smoke
    def test_loads_sorted_rows_and_keeps_token_columns_aligned(
        self, tmp_path: Path, deserializer
    ):
        trace = _write_trace(
            tmp_path,
            '{"timestamp": 5.0, "input_length": 3, "output_length": 30}\n'
            '{"timestamp": 2.0, "input_length": 1, "output_length": 10}\n'
            '{"timestamp": 2.0, "input_length": 2, "output_length": 20}\n'
            '{"timestamp": 8.0, "input_length": 0, "output_length": 40}\n',
        )

        ds = self._deserialize(deserializer, trace)

        assert isinstance(ds, Dataset)
        assert ds["prompt_tokens_count"] == [1, 2, 3, 0]
        assert ds["output_tokens_count"] == [10, 20, 30, 40]
        for prompt, token_count in zip(
            ds["prompt"], ds["prompt_tokens_count"], strict=True
        ):
            assert len(_mock_processor().encode(prompt)) == token_count

    @pytest.mark.smoke
    def test_emits_relative_timestamp_column_sorted_from_trace(
        self, tmp_path: Path, deserializer
    ):
        """Each row gets offset seconds from the earliest sorted timestamp.

        ### WRITTEN BY AI ###
        """
        trace = _write_trace(
            tmp_path,
            '{"timestamp": 5.0, "input_length": 1, "output_length": 10}\n'
            '{"timestamp": 2.0, "input_length": 2, "output_length": 20}\n'
            '{"timestamp": 8.0, "input_length": 3, "output_length": 30}\n'
            '{"timestamp": 2.0, "input_length": 4, "output_length": 40}\n'
            '{"timestamp": 5.0, "input_length": 5, "output_length": 50}\n',
        )

        ds = self._deserialize(deserializer, trace)

        assert ds["relative_timestamp"] == pytest.approx(
            [0.0, 0.0, 3.0, 3.0, 6.0], abs=1e-9
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

        assert ds["prompt_tokens_count"] == [2, 4]
        assert ds["output_tokens_count"] == [20, 40]

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

        config = TraceSyntheticDataArgs(path=trace)
        ds = deserializer(
            config=config,
            processor_factory=lambda: processor,
            random_seed=42,
        )

        assert ds["prompt_tokens_count"] == prompt_lengths
        assert ds["output_tokens_count"] == output_lengths
        assert processor.encode.call_count <= len(prompt_lengths) + 4
        for prompt, token_count in zip(
            ds["prompt"], ds["prompt_tokens_count"], strict=True
        ):
            assert len(_mock_processor().encode(prompt)) == token_count

    @pytest.mark.smoke
    def test_rejects_invalid_data(self, deserializer):
        with pytest.raises(DataNotSupportedError, match="path to a trace file"):
            self._deserialize(deserializer, 123)

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
                '{"timestamp": "bad", "input_length": 10, "output_length": 5}\n',
                {},
                "could not convert",
            ),
            (
                '{"timestamp": 0, "input_length": "bad", "output_length": 5}\n',
                {},
                "invalid literal",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": null}\n',
                {},
                "NoneType",
            ),
            (
                '{"timestamp": 0, "input_length": 10, "output_length": 5}\nnot-json\n',
                {},
                "generating the dataset",
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
            '{"timestamp": 0, "input_length": 10, "output_length": 5}\n',
            suffix=".json",
        )

        with pytest.raises(DataNotSupportedError, match=r"Unsupported.*\.json"):
            self._deserialize(deserializer, trace)
