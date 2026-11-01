"""Tests for deterministic synthetic audio datasets."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from datasets import IterableDataset
from pydantic import ValidationError

from guidellm.data.deserializers import DatasetDeserializerFactory
from guidellm.schemas.data import DataArgs, SyntheticAudioDataArgs


@pytest.mark.regression
def test_synthetic_audio_factory_yields_canonical_rows(monkeypatch):
    """Dispatch the typed config and yield indexed canonical audio records.

    :param monkeypatch: Pytest monkeypatch fixture for isolating audio encoding.
    :return: None.
    ## WRITTEN BY AI ##
    """
    calls: list[dict[str, object]] = []

    def fake_synthesize_audio(
        *,
        duration_seconds: float,
        sample_rate: int,
        content: str,
        audio_format: str,
        seed: int,
        row_index: int,
    ) -> dict[str, object]:
        """Capture options and construct one canonical synthetic audio record.

        :param duration_seconds: Requested audio clip duration.
        :param sample_rate: Requested sample rate in hertz.
        :param content: Waveform content mode.
        :param audio_format: Encoded audio format.
        :param seed: Seed used for deterministic synthesis.
        :param row_index: Index of the current dataset row.
        :return: Canonical audio-file record for this row.
        """
        calls.append(
            {
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "content": content,
                "audio_format": audio_format,
                "seed": seed,
                "row_index": row_index,
            }
        )
        return {
            "type": "audio_file",
            "audio": bytes([row_index]),
            "file_name": f"synthetic_{row_index}.wav",
            "format": audio_format,
            "mimetype": f"audio/{audio_format}",
            "audio_samples": 800,
            "audio_seconds": 0.1,
            "audio_bytes": 1,
        }

    monkeypatch.setattr(
        "guidellm.data.deserializers.synthetic_audio.synthesize_audio",
        fake_synthesize_audio,
    )
    config = DataArgs.model_validate(
        {
            "kind": "synthetic_audio",
            "duration_seconds": 0.1,
            "sample_rate": 8000,
            "content": "noise",
            "format": "wav",
            "seed": 21,
        }
    )

    dataset = DatasetDeserializerFactory.deserialize(
        config=config,
        processor_factory=Mock,
        random_seed=99,
    )
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(3)]

    assert isinstance(dataset, IterableDataset)
    assert [row["audio"]["audio"] for row in rows] == [b"\x00", b"\x01", b"\x02"]
    assert [call["row_index"] for call in calls] == [0, 1, 2]
    assert all(call["seed"] == 21 for call in calls)
    assert all(call["sample_rate"] == 8000 for call in calls)


@pytest.mark.regression
@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"duration_seconds": 0}, "duration_seconds"),
        ({"sample_rate": 0}, "sample_rate"),
    ],
)
def test_synthetic_audio_config_requires_positive_layout(options, field):
    """Reject non-positive audio duration and sampling rate.

    :param options: Invalid config values under test.
    :param field: Config field expected in the validation error.
    :return: None.
    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError, match=field):
        SyntheticAudioDataArgs(**options)
