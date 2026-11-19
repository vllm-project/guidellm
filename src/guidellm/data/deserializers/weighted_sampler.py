from __future__ import annotations

from collections.abc import Callable, Iterator
from decimal import Decimal
from itertools import count
from math import gcd
from numbers import Real
from random import Random
from typing import Any

import numpy as np
from datasets import Dataset, Features, IterableDataset
from datasets.iterable_dataset import _BaseExamplesIterable
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataNotSupportedError
from guidellm.schemas.data.deserializers import WeightedSamplerDataArgs

__all__ = ["WeightedSamplerDataset", "WeightedSamplerDatasetDeserializer"]


def _build_pool(source: Dataset, weight_column: str) -> list[int]:
    if weight_column not in source.column_names:
        raise DataNotSupportedError(f"Missing weight column '{weight_column}'.")
    if not len(source):
        raise DataNotSupportedError(
            f"Cannot sample weight column '{weight_column}' from an empty dataset."
        )

    weights = []
    common_divisor = 0
    for row_index, value in enumerate(source[weight_column]):
        error = (
            f"Invalid weight in column '{weight_column}' at row {row_index}: "
            f"{value!r}; expected a non-negative integer-valued number."
        )
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise DataNotSupportedError(error)
        try:
            # Real's typing stub does not declare the numeric __int__ conversion.
            weight = int(value)  # type: ignore[arg-type]
        except (ValueError, OverflowError) as exc:
            raise DataNotSupportedError(error) from exc
        if weight < 0 or weight != value:
            raise DataNotSupportedError(error)
        weights.append(weight)
        common_divisor = gcd(common_divisor, weight)

    if not common_divisor:
        raise DataNotSupportedError(
            f"Weight column '{weight_column}' must contain a positive weight."
        )

    return [
        index
        for index, weight in enumerate(weights)
        for _ in range(weight // common_divisor)
    ]


class _WeightedSamplerExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self, source: Dataset, pool: list[int], replacement: bool, random_seed: int
    ):
        super().__init__()
        self.source = source
        self.pool = pool
        self.replacement = replacement
        self.random_seed = random_seed
        self.iteration_count = 0

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        rand = Random(self.random_seed + self.iteration_count)
        self.iteration_count += 1
        keys = count()
        if self.replacement:
            while True:
                yield next(keys), self.source[rand.choice(self.pool)]
        else:
            pool = self.pool.copy()
            while True:
                rand.shuffle(pool)
                for index in pool:
                    yield next(keys), self.source[index]

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return self.source.features

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _WeightedSamplerExamplesIterable:
        """Keep the single source; sampling already randomizes row selection.

        :param generator: Unused source-shuffling generator.
        :return: This examples iterable.
        """
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _WeightedSamplerExamplesIterable:
        """Keep the sequence shared for the loader's position-based partitioning.

        :param num_shards: Unused number of source shards.
        :param index: Unused source shard index.
        :param contiguous: Unused source shard layout.
        :return: This examples iterable.
        """
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the iteration seed offset.

        :param state_dict: State containing the iteration count.
        """
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class WeightedSamplerDataset(IterableDataset):
    """Infinite sampling backed by a GCD-reduced pool of source row indices.

    Large relatively prime weights can require a large pool even after reduction.
    Row payloads remain in the source dataset rather than being duplicated.

    :param source: Finite, indexable source dataset.
    :param config: Weight column and replacement settings.
    :param random_seed: Base seed for reproducible sampling.
    """

    def __init__(
        self,
        source: Dataset,
        config: WeightedSamplerDataArgs,
        random_seed: int = 42,
    ):
        self.source = source
        self.config = config
        self.random_seed = random_seed
        ex_iterable = _WeightedSamplerExamplesIterable(
            source=source,
            pool=_build_pool(source, config.weight_column),
            replacement=config.replacement,
            random_seed=random_seed,
        )
        super().__init__(
            ex_iterable=ex_iterable, info=source.info.copy(), split=source.split
        )

    def _iter_pytorch(self) -> Iterator[dict[str, Any]]:
        # GuideLLM's DatasetsIterator partitions by position. Hugging Face's
        # additional source sharding would stop all but one worker for this
        # single-source iterable, dropping those workers' assigned positions.
        for _, example in self._prepare_ex_iterable_for_iteration():
            yield example

    def set_epoch(self, epoch: int) -> None:
        """Set the seed offset for the next iteration, as for synthetic datasets.

        :param epoch: Epoch used to offset the base random seed.
        """
        super().set_epoch(epoch)
        if isinstance(self._ex_iterable, _WeightedSamplerExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("weighted_sampler")
class WeightedSamplerDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: WeightedSamplerDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> WeightedSamplerDataset:
        source = DatasetDeserializerFactory.deserialize(
            config=config.source,
            processor_factory=processor_factory,
            random_seed=random_seed,
        )
        if not isinstance(source, Dataset):
            raise DataNotSupportedError(
                f"Weighted sampling of column '{config.weight_column}' requires a "
                "finite, indexable Dataset; iterable/streaming sources are unsupported."
            )
        return WeightedSamplerDataset(source, config, random_seed)
