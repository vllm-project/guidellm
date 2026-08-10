"""Synthetic video dataset deserializer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.schemas.data.deserializers import SyntheticVideoDataArgs
from guidellm.utils.random import IntegerRangeSampler
from guidellm.utils.vision import synthesize_video

__all__ = [
    "SyntheticVideoDataset",
    "SyntheticVideoDatasetDeserializer",
]


_DESERIALIZER_TYPE = "synthetic_video"


class _SyntheticVideoExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        config: SyntheticVideoDataArgs,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.random_seed = random_seed
        self.iteration_count = 0

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        iter_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        output_tokens_sampler = (
            iter(
                IntegerRangeSampler(
                    average=self.config.output_tokens,
                    variance=self.config.output_tokens_stdev,
                    min_value=self.config.output_tokens_min,
                    max_value=self.config.output_tokens_max,
                    random_seed=iter_seed + 1,
                )
            )
            if self.config.output_tokens is not None
            else None
        )

        row_index = 0
        while True:
            output_token_count = (
                next(output_tokens_sampler)
                if output_tokens_sampler is not None
                else None
            )
            width = self.config.width
            height = self.config.height
            if width is None or height is None:
                raise RuntimeError("Synthetic video dimensions were not resolved.")

            row: dict[str, Any] = {
                "video": synthesize_video(
                    width=width,
                    height=height,
                    frames=int(self.config.frames),
                    fps=float(self.config.fps),
                    content=self.config.content,
                    video_format=self.config.format,
                    video_bitrate=self.config.video_bitrate,
                    seed=self.config.seed,
                    row_index=row_index,
                ),
            }
            if output_token_count is not None:
                row["output_tokens_count_0"] = output_token_count

            yield row_index, row
            row_index += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        features: dict[str, Any] = {
            "video": {
                "type": Value("string"),
                "video": Value("string"),
                "video_frames": Value("int64"),
                "video_seconds": Value("float64"),
                "video_bytes": Value("int64"),
            },
        }
        if self.config.output_tokens is not None:
            features["output_tokens_count_0"] = Value("int32")
        return Features(features)

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticVideoExamplesIterable:
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticVideoExamplesIterable:
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class SyntheticVideoDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticVideoDataArgs,
        random_seed: int = 42,
    ):
        self.config = config
        self.random_seed = random_seed

        ex_iterable = _SyntheticVideoExamplesIterable(
            config=config,
            random_seed=random_seed,
        )
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic video dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        if isinstance(self._ex_iterable, _SyntheticVideoExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register(_DESERIALIZER_TYPE)
class SyntheticVideoDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticVideoDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        _ = processor_factory
        return SyntheticVideoDataset(
            config=config,
            random_seed=random_seed,
        )
