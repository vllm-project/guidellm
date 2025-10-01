from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import yaml
from datasets import Features, IterableDataset, Value
from faker import Faker
from pydantic import Field
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
]


class SyntheticTextDatasetConfig(StandardBaseModel):
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

        while True:
            prompt_tokens_count = next(prompt_tokens_sampler)
            output_tokens_count = next(output_tokens_sampler)

            yield {
                "prompt": self._create_prompt(
                    prompt_tokens_count, samples_generated, faker
                ),
                "prompt_tokens_count": prompt_tokens_count,
                "output_tokens_count": output_tokens_count,
            }
            samples_generated += 1

    def _create_prompt(self, prompt_tokens_count: int, index: int, faker: Faker) -> str:
        prompt_token_ids = []
        avg_chars_per_token = 5
        margin_of_safety = 1.5
        attempts = 0

        while len(prompt_token_ids) < prompt_tokens_count:
            attempts += 1
            num_chars = (
                prompt_tokens_count * avg_chars_per_token * margin_of_safety * attempts
            )
            text = f"{index} " + faker.text(max_nb_chars=num_chars)
            prompt_token_ids = self.processor.encode(text)

        return self.processor.decode(
            prompt_token_ids[:prompt_tokens_count], skip_special_tokens=True
        )


@DatasetDeserializerFactory.register("synthetic_text")
class SyntheticTextDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        data_kwargs: dict[str, Any],
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        # Config file pathways, deserialize and call self again
        if (config := self._load_config_file(data)) is not None:
            return self(config, data_kwargs, processor_factory, random_seed)

        # Config str pathways, deserialize and call self again
        if (config := self._load_config_str(data)) is not None:
            return self(config, data_kwargs, processor_factory, random_seed)

        if not isinstance(data, SyntheticTextDatasetConfig):
            raise DataNotSupportedError(
                "Unsupported data for SyntheticTextDatasetDeserializer, "
                "expected SyntheticTextDatasetConfig, str or Path to a config file, "
                f"got {data}"
            )

        return IterableDataset.from_generator(
            lambda: SyntheticTextGenerator(
                config=data, processor=processor_factory(), random_seed=random_seed
            ),
            features=Features(
                {
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
