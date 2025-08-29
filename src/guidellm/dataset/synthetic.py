import json
import random
from collections.abc import Iterable, Iterator
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase  # type: ignore[import]

from guidellm.dataset.creator import ColumnInputTypes, DatasetCreator
from guidellm.utils import EndlessTextCreator, IntegerRangeSampler, check_load_processor

__all__ = [
    "SyntheticDatasetConfig",
    "SyntheticDatasetCreator",
    "SyntheticTextItemsGenerator",
]


class SyntheticDatasetConfig(BaseModel):
    prefix_tokens: int = Field(
        description="The number of shared prefix tokens to prepend to each prompt.",
        ge=0,
        default=0,
    )
    prompt_tokens: int = Field(
        description="The average number of text tokens generated for prompts.",
        gt=0,
    )
    prompt_tokens_stdev: Optional[int] = Field(
        description="The standard deviation of the tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_min: Optional[int] = Field(
        description="The minimum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    prompt_tokens_max: Optional[int] = Field(
        description="The maximum number of text tokens generated for prompts.",
        gt=0,
        default=None,
    )
    output_tokens: int = Field(
        description="The average number of text tokens generated for outputs.",
        gt=0,
    )
    output_tokens_stdev: Optional[int] = Field(
        description="The standard deviation of the tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_min: Optional[int] = Field(
        description="The minimum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    output_tokens_max: Optional[int] = Field(
        description="The maximum number of text tokens generated for outputs.",
        gt=0,
        default=None,
    )
    samples: int = Field(
        description="The number of samples to generate for the dataset.",
        gt=0,
        default=1000,
    )
    source: str = Field(
        description="The source of the text data to be used for generation.",
        default="data:prideandprejudice.txt.gz",
    )

    @staticmethod
    def parse_str(data: Union[str, Path]) -> "SyntheticDatasetConfig":
        if (
            isinstance(data, Path)
            or data.strip().endswith(".config")
            or data.strip().endswith(".yaml")
        ):
            return SyntheticDatasetConfig.parse_config_file(data)

        if data.strip().startswith("{"):
            return SyntheticDatasetConfig.parse_json(data)

        if data.count("=") > 1:
            return SyntheticDatasetConfig.parse_key_value_pairs(data)

        raise ValueError(
            f"Unsupported data format. Expected JSON or key-value pairs, got {data}"
        )

    @staticmethod
    def parse_json(data: str) -> "SyntheticDatasetConfig":
        config_dict = json.loads(data.strip())

        return SyntheticDatasetConfig(**config_dict)

    @staticmethod
    def parse_key_value_pairs(data: str) -> "SyntheticDatasetConfig":
        config_dict = {}
        items = data.strip().split(",")
        for item in items:
            key, value = item.split("=")
            config_dict[key.strip()] = (
                int(value.strip()) if value.strip().isnumeric() else value.strip()
            )

        return SyntheticDatasetConfig(**config_dict)  # type: ignore[arg-type]

    @staticmethod
    def parse_config_file(data: Union[str, Path]) -> "SyntheticDatasetConfig":
        with Path(data).open("r") as file:
            config_dict = yaml.safe_load(file)

        return SyntheticDatasetConfig(**config_dict)


class SyntheticTextItemsGenerator(
    Iterable[
        dict[
            Literal["prompt", "prompt_tokens_count", "output_tokens_count"],
            Union[str, int],
        ]
    ]
):
    def __init__(
        self,
        config: SyntheticDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        self.text_creator = EndlessTextCreator(
            data=config.source,
        )

    def __iter__(
        self,
    ) -> Iterator[
        dict[
            Literal["prompt", "prompt_tokens_count", "output_tokens_count"],
            Union[str, int],
        ]
    ]:
        prompt_tokens_sampler = IntegerRangeSampler(
            average=self.config.prompt_tokens,
            variance=self.config.prompt_tokens_stdev,
            min_value=self.config.prompt_tokens_min,
            max_value=self.config.prompt_tokens_max,
            random_seed=self.random_seed,
        )
        output_tokens_sampler = IntegerRangeSampler(
            average=self.config.output_tokens,
            variance=self.config.output_tokens_stdev,
            min_value=self.config.output_tokens_min,
            max_value=self.config.output_tokens_max,
            random_seed=self.random_seed + 1,  # ensure diff dist from prompts
        )
        # ensure diff distribution from output tokens
        rand = random.Random(self.random_seed + 2)  # noqa: S311
        unique_prefix_iter = cycle(self.processor.get_vocab().values())

        prefix_index = rand.randint(0, len(self.text_creator.words))
        prefix_tokens = self._create_prompt(self.config.prefix_tokens, prefix_index)

        for _, prompt_tokens, output_tokens in zip(
            range(self.config.samples),
            prompt_tokens_sampler,
            output_tokens_sampler,
        ):
            start_index = rand.randint(0, len(self.text_creator.words))
            prompt_text = self.processor.decode(
                prefix_tokens
                + self._create_prompt(
                    prompt_tokens, start_index, next(unique_prefix_iter)
                ),
                skip_special_tokens=True,
            )
            yield {
                "prompt": prompt_text,
                "prompt_tokens_count": self.config.prefix_tokens + prompt_tokens,
                "output_tokens_count": output_tokens,
            }

    def _create_prompt(
        self, prompt_tokens: int, start_index: int, unique_prefix: Optional[int] = None
    ) -> list[int]:
        if prompt_tokens <= 0:
            return []

        left = start_index
        right = start_index + 4 * prompt_tokens
        start_tokens = [unique_prefix] if unique_prefix else []

        while left < right:
            mid = (left + right) // 2
            test_prompt = self.text_creator.create_text(start_index, mid - start_index)
            test_tokens = start_tokens + self.processor.encode(test_prompt)

            if len(test_tokens) == prompt_tokens:
                return test_tokens
            elif len(test_tokens) < prompt_tokens:
                left = mid + 1
            else:
                right = mid

        final_text = self.text_creator.create_text(start_index, left - start_index)
        return start_tokens + self.processor.encode(final_text)


class SyntheticDatasetCreator(DatasetCreator):
    @classmethod
    def is_supported(
        cls,
        data: Any,
        data_args: Optional[dict[str, Any]],  # noqa: ARG003
    ) -> bool:
        if (
            isinstance(data, Path)
            and data.exists()
            and data.suffix in {".config", ".yaml"}
        ):
            return True

        if isinstance(data, str):
            data_str: str = data.strip()
            if (
                data_str.startswith("{")
                or data_str.count("=") > 1
                or data_str.endswith((".config", ".yaml"))
            ):
                return True

        return False

    @classmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[dict[str, Any]],
        random_seed: int,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        processor = check_load_processor(
            processor,
            processor_args,
            error_msg=(
                "Processor/tokenizer required for synthetic dataset generation."
            ),
        )

        config = SyntheticDatasetConfig.parse_str(data)
        generator = SyntheticTextItemsGenerator(config, processor, random_seed)
        items = list(generator)

        return Dataset.from_list(items, **(data_args or {}))

    @classmethod
    def extract_args_column_mappings(
        cls,
        data_args: Optional[dict[str, Any]],
    ) -> dict[ColumnInputTypes, str]:
        data_args_columns = super().extract_args_column_mappings(data_args)

        if data_args_columns:
            raise ValueError(
                f"Column mappings are not supported for synthetic datasets. "
                f"Got {data_args_columns}"
            )

        return {
            "prompt_column": "prompt",
            "prompt_tokens_count_column": "prompt_tokens_count",
            "output_tokens_count_column": "output_tokens_count",
        }
