from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from random import Random
from typing import Any

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from transformers import PreTrainedTokenizerBase

from guidellm.data.config import load_config
from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import SyntheticTextDatasetConfig
from guidellm.utils import IntegerRangeSampler

__all__ = [
    "SyntheticTextDataset",
    "SyntheticTextDatasetDeserializer",
]


class _SyntheticTextExamplesIterable(_BaseExamplesIterable):
    """Custom examples iterable for synthetic text generation."""

    def __init__(
        self,
        config: SyntheticTextDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        self.iteration_count = 0

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        iter_random_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_random_seed)
        prompt_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.prompt_tokens,
                variance=self.config.prompt_tokens_stdev,
                min_value=self.config.prompt_tokens_min,
                max_value=self.config.prompt_tokens_max,
                random_seed=iter_random_seed,
            )
        )
        output_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.output_tokens,
                variance=self.config.output_tokens_stdev,
                min_value=self.config.output_tokens_min,
                max_value=self.config.output_tokens_max,
                random_seed=iter_random_seed + 1,  # ensure diff dist from prompts
            )
        )

        # Create a shared prefix if specified
        rand = Random(iter_random_seed + 3)
        prefix_iter = self._create_prefix_iter(faker, rand)
        samples_count = 0

        while True:
            prompt_tokens_count = next(prompt_tokens_sampler)
            output_tokens_count = next(output_tokens_sampler)

            yield (
                samples_count,
                {
                    "prefix": next(prefix_iter),
                    "prompt": self._create_prompt(
                        prompt_tokens_count,
                        faker,
                        f"{self.iteration_count} {samples_count} ",
                    ),
                    "prompt_tokens_count": prompt_tokens_count,
                    "output_tokens_count": output_tokens_count,
                },
            )
            samples_count += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        return Features(
            {
                "prefix": Value("string"),
                "prompt": Value("string"),
                "prompt_tokens_count": Value("int32"),
                "output_tokens_count": Value("int32"),
            }
        )

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data doesn't have fixed sources to shuffle."""
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticTextExamplesIterable:
        """Return self since synthetic data generation is infinite and stateless."""
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state from a state dict."""
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        """Initialize the state dict for the iterable."""
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict

    def _create_prompt(
        self, prompt_tokens_count: int, faker: Faker, unique: str = ""
    ) -> str:
        prompt_token_ids: list[int] = []
        avg_chars_per_token = 5
        margin_of_safety = 1.5
        attempts = 0

        while len(prompt_token_ids) < prompt_tokens_count:
            attempts += 1
            num_chars = int(
                prompt_tokens_count * avg_chars_per_token * margin_of_safety * attempts
            )
            text = unique + faker.text(max_nb_chars=num_chars)
            prompt_token_ids = self.processor.encode(text)

        return self.processor.decode(
            prompt_token_ids[:prompt_tokens_count], skip_special_tokens=True
        )

    def _create_prefix_iter(self, faker: Faker, rand: Random) -> Iterator[str]:
        if not self.config.prefix_buckets:
            while True:
                yield ""

        # Increase weights to ensure an integer number of samples per per-prefix
        least_common_prefix_count = math.lcm(
            *(bucket.prefix_count for bucket in self.config.prefix_buckets)
        )
        unnorm_weights = [
            least_common_prefix_count * bucket.bucket_weight // bucket.prefix_count
            for bucket in self.config.prefix_buckets
        ]
        # Use GCD to reduce the weights to smallest integer ratio
        common_divisor = math.gcd(*unnorm_weights)

        # Create prefix list maintaining the correct distribution
        prefixes = []
        for bucket, weight in zip(
            self.config.prefix_buckets, unnorm_weights, strict=False
        ):
            bucket_prefixes = [
                self._create_prompt(bucket.prefix_tokens, faker)
                for _ in range(bucket.prefix_count)
            ]
            sample_count = weight // common_divisor
            prefixes.extend(bucket_prefixes * sample_count)

        while True:
            yield rand.choice(prefixes)


class SyntheticTextDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticTextDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

        # Create the examples iterable
        ex_iterable = _SyntheticTextExamplesIterable(
            config=config,
            processor=processor,
            random_seed=random_seed,
        )

        # Initialize parent with proper ex_iterable
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic text dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset iteration."""
        if isinstance(self._ex_iterable, _SyntheticTextExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register("synthetic_text")
class SyntheticTextDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> IterableDataset:
        # Config file and string pathways; deserialize and call self again
        if (config := load_config(data, SyntheticTextDatasetConfig)) is not None:
            return self(config, processor_factory, random_seed, **data_kwargs)

        if not isinstance(data, SyntheticTextDatasetConfig):
            raise DataNotSupportedError(
                "Unsupported data for SyntheticTextDatasetDeserializer, "
                "expected SyntheticTextDatasetConfig, str or Path to a config file, "
                f"got {data}"
            )

        return SyntheticTextDataset(
            config=data,
            processor=processor_factory(),
            random_seed=random_seed,
        )
