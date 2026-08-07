"""Synthetic image dataset deserializer."""

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
from guidellm.schemas.data.deserializers import SyntheticImageDataArgs
from guidellm.utils.random import IntegerRangeSampler
from guidellm.utils.vision import synthesize_image

__all__ = [
    "SyntheticImageDataArgs",
    "SyntheticImageDataset",
    "SyntheticImageDatasetDeserializer",
]


_DESERIALIZER_TYPE = "synthetic_image"


class _SyntheticImageExamplesIterable(_BaseExamplesIterable):
    """Examples iterable that yields rows of synthetic images."""

    def __init__(
        self,
        config: SyntheticImageDataArgs,
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

            row: dict[str, Any] = {}
            if output_token_count is not None:
                row["output_tokens_count_0"] = output_token_count

            width = self.config.width
            height = self.config.height
            if width is None or height is None:
                raise RuntimeError("Synthetic image dimensions were not resolved.")

            for img_idx in range(self.config.images_per_request):
                encoded = synthesize_image(
                    width=width,
                    height=height,
                    content=self.config.content,
                    image_format=self.config.format,
                    jpeg_quality=self.config.jpeg_quality,
                    seed=self.config.seed,
                    row_index=row_index * self.config.images_per_request + img_idx,
                )
                if self.config.images_per_request == 1:
                    row["image"] = encoded
                else:
                    row[f"image_{img_idx}"] = encoded

            yield row_index, row
            row_index += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        features: dict[str, Any] = {}
        if self.config.output_tokens is not None:
            features["output_tokens_count_0"] = Value("int32")
        image_struct = {
            "type": Value("string"),
            "image": Value("string"),
            "image_pixels": Value("int64"),
            "image_bytes": Value("int64"),
        }
        if self.config.images_per_request == 1:
            features["image"] = image_struct
        else:
            for img_idx in range(self.config.images_per_request):
                features[f"image_{img_idx}"] = image_struct
        return Features(features)

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticImageExamplesIterable:
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticImageExamplesIterable:
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class SyntheticImageDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticImageDataArgs,
        random_seed: int = 42,
    ):
        self.config = config
        self.random_seed = random_seed

        ex_iterable = _SyntheticImageExamplesIterable(
            config=config,
            random_seed=random_seed,
        )
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic image dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        if isinstance(self._ex_iterable, _SyntheticImageExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register(_DESERIALIZER_TYPE)
class SyntheticImageDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticImageDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        _ = processor_factory
        return SyntheticImageDataset(
            config=config,
            random_seed=random_seed,
        )
