## WRITTEN BY AI ##

"""
Unit tests for TraceSyntheticDatasetDeserializer.

Ensures trace file is loaded and synthetic prompts are generated with exact
input_length.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import Dataset

from guidellm.data.deserializers.trace_synthetic import (
    TraceSyntheticDatasetDeserializer,
)
from guidellm.data.schemas import DataNotSupportedError


def _mock_processor():
    """Tokenizer that returns token count = number of words in text."""
    proc = Mock()
    proc.encode.side_effect = lambda text: list(range(max(1, len(text.split()))))
    proc.decode.side_effect = lambda tokens, skip_special_tokens=False: " ".join(
        "x" for _ in range(len(tokens))
    )
    return proc


def _deserialize(deserializer, data, **kwargs):
    defaults = {
        "processor_factory": _mock_processor,
        "random_seed": 42,
    }
    return deserializer(**{**defaults, "data": data, **kwargs})


class TestTraceSyntheticDatasetDeserializer:
    """Tests for TraceSyntheticDatasetDeserializer."""

    @pytest.fixture
    def deserializer(self):
        return TraceSyntheticDatasetDeserializer()

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            # Basic small counts
            (
                '{"timestamp": 0, "input_length": 50, "output_length": 20}\n'
                '{"timestamp": 0.5, "input_length": 100, "output_length": 30}\n',
                [(50, 20), (100, 30)],
            ),
            # Production-scale token counts (4K-128K contexts)
            (
                '{"timestamp": 0, "input_length": 4096, "output_length": 512}\n'
                '{"timestamp": 1.0, "input_length": 8192, "output_length": 1024}\n'
                '{"timestamp": 2.0, "input_length": 32768, "output_length": 4096}\n'
                '{"timestamp": 3.0, "input_length": 131072, "output_length": 8192}\n',
                [(4096, 512), (8192, 1024), (32768, 4096), (131072, 8192)],
            ),
            # Mixed high/low alternating (edge cases)
            (
                '{"timestamp": 0, "input_length": 10, "output_length": 5}\n'
                '{"timestamp": 0.1, "input_length": 65536, "output_length": 16384}\n'
                '{"timestamp": 0.2, "input_length": 20, "output_length": 10}\n'
                '{"timestamp": 0.3, "input_length": 131072, "output_length": 32768}\n',
                [(10, 5), (65536, 16384), (20, 10), (131072, 32768)],
            ),
            # Unsorted timestamps with duplicates (sorts by timestamp)
            (
                '{"timestamp": 5.0, "input_length": 100, "output_length": 10}\n'
                '{"timestamp": 2.0, "input_length": 200, "output_length": 20}\n'
                '{"timestamp": 8.0, "input_length": 300, "output_length": 30}\n'
                '{"timestamp": 2.0, "input_length": 400, "output_length": 40}\n',
                [(200, 20), (400, 40), (100, 10), (300, 30)],
            ),
            # Concurrent burst (5 requests at same timestamp)
            (
                '{"timestamp": 1.0, "input_length": 100, "output_length": 10}\n'
                '{"timestamp": 1.0, "input_length": 200, "output_length": 20}\n'
                '{"timestamp": 1.0, "input_length": 300, "output_length": 30}\n'
                '{"timestamp": 1.0, "input_length": 400, "output_length": 40}\n'
                '{"timestamp": 1.0, "input_length": 500, "output_length": 50}\n',
                [(100, 10), (200, 20), (300, 30), (400, 40), (500, 50)],
            ),
        ],
    )
    def test_load_jsonl_various_scenarios(
        self, tmp_path: Path, deserializer, content, expected
    ):
        """Trace JSONL yields exact token counts (small, large, mixed, unsorted,
        duplicates)."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text(content)
        ds = _deserialize(deserializer, str(trace), type_="trace_synthetic")
        assert isinstance(ds, Dataset)
        assert len(ds) == len(expected)
        assert set(ds.column_names) >= {
            "prompt",
            "prompt_tokens_count",
            "output_tokens_count",
        }
        for row, (in_len, out_len) in zip(ds, expected, strict=True):
            assert row["prompt_tokens_count"] == in_len
            assert row["output_tokens_count"] == out_len

    @pytest.mark.smoke
    def test_rejects_invalid_data(self, deserializer):
        """Non-path data raises DataNotSupportedError."""
        with pytest.raises(DataNotSupportedError, match="path to a trace file"):
            _deserialize(deserializer, 123)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("", "empty"),
            ('{"ts": 0, "input_length": 10, "output_length": 5}\n', "timestamp"),
        ],
    )
    def test_trace_validation_raises(
        self, tmp_path: Path, deserializer, content, match
    ):
        """Empty trace or missing required column raises DataNotSupportedError."""
        trace = tmp_path / "trace.jsonl"
        trace.write_text(content)
        with pytest.raises(DataNotSupportedError, match=match):
            _deserialize(deserializer, str(trace))
