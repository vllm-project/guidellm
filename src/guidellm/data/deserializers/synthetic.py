from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from random import Random
from typing import Any

import yaml
from datasets import Features, IterableDataset, Value
from faker import Faker
from pydantic import ConfigDict, Field, model_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DataNotSupportedError,
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.utils import IntegerRangeSampler, StandardBaseModel

__all__ = [
    "SyntheticTextDatasetConfig",
    "SyntheticTextDatasetDeserializer",
    "SyntheticTextGenerator",
    "SyntheticTextPrefixBucketConfig",
]


class SyntheticTextPrefixBucketConfig(StandardBaseModel):
    bucket_weight: int = Field(
        description="Weight of this bucket in the overall distribution.",
        gt=0,
        default=100,
    )
    prefix_count: int = Field(
        description="The number of unique prefixes to generate for this bucket.",
        ge=1,
        default=1,
    )
    prefix_tokens: int = Field(
        description="The number of prefix tokens per-prompt for this bucket.",
        ge=0,
        default=0,
    )


class SyntheticTextDatasetConfig(StandardBaseModel):
    model_config = ConfigDict(
        extra="allow",
    )

    prefix_buckets: list[SyntheticTextPrefixBucketConfig] | None = Field(
        description="Buckets for the prefix tokens distribution.",
        default=None,
    )
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts.",
        gt=0,
    )
    prompt_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens generated for outputs.",
        gt=0,
    )
    output_tokens_stdev: int | None = Field(
        description="The standard deviation of the tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="The minimum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="The maximum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    source: str = Field(
        description="The source of the text data to be used for generation.",
        default="data:prideandprejudice.txt.gz",
    )

    @model_validator(mode="after")
    def check_prefix_options(self) -> SyntheticTextDatasetConfig:
        if self.__pydantic_extra__ is not None:
            prefix_count = self.__pydantic_extra__.get("prefix_count", None)  # type: ignore[attr-defined]
            prefix_tokens = self.__pydantic_extra__.get("prefix_tokens", None)  # type: ignore[attr-defined]

            if prefix_count is not None or prefix_tokens is not None:
                if self.prefix_buckets:
                    raise ValueError(
                        "prefix_buckets is mutually exclusive"
                        " with prefix_count and prefix_tokens"
                    )

                self.prefix_buckets = [
                    SyntheticTextPrefixBucketConfig(
                        prefix_count=prefix_count or 1,
                        prefix_tokens=prefix_tokens or 0,
                    )
                ]

        return self


class SyntheticTextGenerator:
    def __init__(
        self,
        config: SyntheticTextDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        samples_generated = 0

        faker = Faker()
        faker.seed_instance(self.random_seed)
        prompt_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.prompt_tokens,
                variance=self.config.prompt_tokens_stdev,
                min_value=self.config.prompt_tokens_min,
                max_value=self.config.prompt_tokens_max,
                random_seed=self.random_seed,
            )
        )
        output_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.output_tokens,
                variance=self.config.output_tokens_stdev,
                min_value=self.config.output_tokens_min,
                max_value=self.config.output_tokens_max,
                random_seed=self.random_seed + 1,  # ensure diff dist from prompts
            )
        )

        # Create a shared prefix if specified
        rand = Random(self.random_seed + 3)
        prefix_iter = self._create_prefix_iter(faker, rand)

        while True:
            prompt_tokens_count = next(prompt_tokens_sampler)
            output_tokens_count = next(output_tokens_sampler)

            yield {
                "prefix": next(prefix_iter),
                "prompt": self._create_prompt(
                    prompt_tokens_count, faker, f"{samples_generated} "
                ),
                "prompt_tokens_count": prompt_tokens_count,
                "output_tokens_count": output_tokens_count,
            }
            samples_generated += 1

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


@DatasetDeserializerFactory.register("synthetic_text")
class SyntheticTextDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> IterableDataset:
        # Config file pathways, deserialize and call self again
        if (config := self._load_config_file(data)) is not None:
            return self(config, processor_factory, random_seed, **data_kwargs)

        # Config str pathways, deserialize and call self again
        if (config := self._load_config_str(data)) is not None:
            return self(config, processor_factory, random_seed, **data_kwargs)

        if not isinstance(data, SyntheticTextDatasetConfig):
            raise DataNotSupportedError(
                "Unsupported data for SyntheticTextDatasetDeserializer, "
                "expected SyntheticTextDatasetConfig, str or Path to a config file, "
                f"got {data}"
            )

        return IterableDataset.from_generator(
            SyntheticTextGenerator,
            gen_kwargs={
                "config": data,
                "processor": processor_factory(),
                "random_seed": random_seed,
            },
            features=Features(
                {
                    "prefix": Value("string"),
                    "prompt": Value("string"),
                    "prompt_tokens_count": Value("int32"),
                    "output_tokens_count": Value("int32"),
                }
            ),
        )

    def _load_config_file(self, data: Any) -> SyntheticTextDatasetConfig | None:
        if (not isinstance(data, str) and not isinstance(data, Path)) or (
            not Path(data).is_file()
        ):
            return None

        data_path = Path(data) if isinstance(data, str) else data
        error = None

        if Path(data).is_file() and data_path.suffix.lower() == ".json":
            try:
                return SyntheticTextDatasetConfig.model_validate_json(
                    data_path.read_text()
                )
            except Exception as err:  # noqa: BLE001
                error = err

        if Path(data).is_file() and data_path.suffix.lower() in {
            ".yaml",
            ".yml",
            ".config",
        }:
            try:
                return SyntheticTextDatasetConfig.model_validate(
                    yaml.safe_load(data_path.read_text())
                )
            except Exception as err:  # noqa: BLE001
                error = err

        err_message = (
            f"Unsupported file {data_path} for "
            f"SyntheticTextDatasetDeserializer, expected .json, "
            f".yaml, .yml, or .config"
        )

        if error is not None:
            err_message += f" with error: {error}"
            raise DataNotSupportedError(err_message) from error
        raise DataNotSupportedError(err_message)

    def _load_config_str(self, data: str) -> SyntheticTextDatasetConfig | None:
        if not isinstance(data, str):
            return None

        data_str = data.strip()
        error = None

        if (data_str.startswith("{") and data_str.endswith("}")) or (
            data_str.startswith("[") and data_str.endswith("]")
        ):
            try:
                return SyntheticTextDatasetConfig.model_validate_json(data_str)
            except Exception as err:  # noqa: BLE001
                error = err

        if data_str.count("=") > 1:
            # key=value pairs separated by commas
            try:
                config_dict = {}
                items = data_str.split(",")
                for item in items:
                    key, value = item.split("=")
                    config_dict[key.strip()] = (
                        int(value.strip())
                        if value.strip().isnumeric()
                        else value.strip()
                    )

                return SyntheticTextDatasetConfig.model_validate(config_dict)
            except Exception as err:  # noqa: BLE001
                error = err

        err_message = (
            "Unsupported string data for SyntheticTextDatasetDeserializer, "
            f"expected JSON or key-value pairs, got {data}"
        )
        if error is not None:
            err_message += f" with error: {error}"
            raise DataNotSupportedError(err_message) from error
        raise DataNotSupportedError(err_message)
