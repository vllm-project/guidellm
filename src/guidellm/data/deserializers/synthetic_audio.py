"""Synthetic audio deserializer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.schemas.data.deserializers import SyntheticAudioDataArgs
from guidellm.utils.audio import synthesize_audio

__all__ = [
    "SyntheticAudioDataset",
    "SyntheticAudioDatasetDeserializer",
]


class _SyntheticAudioExamplesIterable(_BaseExamplesIterable):
    """Examples iterable that yields rows of synthetic audio."""

    def __init__(self, config: SyntheticAudioDataArgs):
        super().__init__()
        self.config = config
        self._state_dict: dict

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        row_index = self._state_dict["row_index"] if self._state_dict is not None else 0
        while True:
            row = {
                "audio": synthesize_audio(
                    duration_seconds=self.config.duration_seconds,
                    sample_rate=self.config.sample_rate,
                    content=self.config.content,
                    audio_format=self.config.format,
                    seed=self.config.seed,
                    row_index=row_index,
                )
            }
            if self._state_dict is not None:
                self._state_dict["row_index"] = row_index + 1
            yield row_index, row
            row_index += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features(
            {
                "audio": {
                    "type": Value("string"),
                    "audio": Value("binary"),
                    "file_name": Value("string"),
                    "format": Value("string"),
                    "mimetype": Value("string"),
                    "audio_samples": Value("int64"),
                    "audio_seconds": Value("float64"),
                    "audio_bytes": Value("int64"),
                }
            }
        )

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: Any,  # noqa: ARG002
    ) -> _SyntheticAudioExamplesIterable:
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticAudioExamplesIterable:
        return self

    def _init_state_dict(self) -> dict:
        self._state_dict = {"row_index": 0}
        return self._state_dict


class SyntheticAudioDataset(IterableDataset):
    def __init__(self, config: SyntheticAudioDataArgs):
        self.config = config

        synth_audio_iterable = _SyntheticAudioExamplesIterable(config)
        super().__init__(
            ex_iterable=synth_audio_iterable,
            info=DatasetInfo(
                description="Synthetic audio dataset generator",
                features=synth_audio_iterable.features,
            ),
        )


@DatasetDeserializerFactory.register("synthetic_audio")
class SyntheticAudioDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticAudioDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        _ = processor_factory, random_seed
        return SyntheticAudioDataset(config)
