import json
import os
import random
import re
import hashlib
import time
from collections.abc import Iterable, Iterator
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
import numpy as np  # type: ignore[import]
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from loguru import logger
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
        # Pre-tokenize entire source once and cache per (tokenizer, source)
        start_time = time.perf_counter()
        self._cached_tokens: list[int] = self._load_or_build_token_cache()
        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Synthetic: token cache ready | tokens={} | took_ms={:.2f}",
            len(self._cached_tokens),
            elapsed,
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

        prefix_index = rand.randint(0, max(len(self._cached_tokens) - 1, 0))
        prefix_tokens = self._create_prompt(self.config.prefix_tokens, prefix_index)

        sample_start_time = time.perf_counter()
        for _, prompt_tokens, output_tokens in zip(
            range(self.config.samples),
            prompt_tokens_sampler,
            output_tokens_sampler,
        ):
            start_index = rand.randint(0, max(len(self._cached_tokens) - 1, 0))
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
        elapsed_samples = (time.perf_counter() - sample_start_time) * 1000.0
        logger.info(
            "Synthetic: generated_samples={} | took_ms={:.2f} | avg_ms_per_sample={:.4f}",
            self.config.samples,
            elapsed_samples,
            elapsed_samples / max(self.config.samples, 1),
        )

    def _create_prompt(
        self, prompt_tokens: int, start_index: int, unique_prefix: Optional[int] = None
    ) -> list[int]:
        if prompt_tokens <= 0:
            return []

        # Determine how many tokens to take from cache, accounting for optional unique prefix
        remaining = prompt_tokens - (1 if unique_prefix is not None else 0)
        if remaining < 0:
            remaining = 0

        sampled = self._take_tokens(start_index, remaining)
        if unique_prefix is not None:
            return [unique_prefix] + sampled
        return sampled

    def _take_tokens(self, start_index: int, count: int) -> list[int]:
        if count <= 0:
            return []
        tokens = self._cached_tokens
        n = len(tokens)
        if n == 0:
            return []
        # Wrap-around contiguous sampling
        result: list[int] = []
        base = start_index % n
        for offset in range(count):
            result.append(tokens[(base + offset) % n])
        return result

    def _load_or_build_token_cache(self) -> list[int]:
        # Create cache directory
        cache_dir = Path(
            os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        ) / "guidellm" / "synthetic_tokens"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Compute a stable tokenizer identifier and source digest
        tokenizer_id = self._tokenizer_identifier(self.processor)
        source_digest = hashlib.sha1(
            self.text_creator.filtered_text.encode("utf-8", errors="ignore")
        ).hexdigest()

        safe_tokenizer_id = re.sub(r"[^A-Za-z0-9_.-]", "_", tokenizer_id)
        cache_path = cache_dir / f"{safe_tokenizer_id}-{source_digest}.npy"

        if cache_path.exists():
            try:
                arr = np.load(cache_path)
                # Ensure 1-D integer array
                arr = np.asarray(arr, dtype=np.int64).reshape(-1)
                logger.debug(
                    "Synthetic: loaded token cache from {} | tokens={}",
                    str(cache_path),
                    arr.size,
                )
                return arr.astype(int).tolist()
            except Exception:
                # If loading fails, rebuild below
                pass

        # Build tokens once from full filtered text
        # Avoid adding special tokens so spans don't include BOS/EOS markers repeatedly
        build_start = time.perf_counter()
        full_tokens = self.processor.encode(
            self.text_creator.filtered_text,
            add_special_tokens=False,
        )
        build_elapsed = (time.perf_counter() - build_start) * 1000.0
        logger.info(
            "Synthetic: built token cache in {:.2f} ms | tokens={}",
            build_elapsed,
            len(full_tokens),
        )

        # Persist to cache
        try:
            np.save(cache_path, np.asarray(full_tokens, dtype=np.int32))
            logger.debug(
                "Synthetic: saved token cache to {} | bytes≈{}",
                str(cache_path),
                int(np.asarray(full_tokens, dtype=np.int32).nbytes),
            )
        except Exception:
            # Best effort; ignore cache write failures
            pass

        return full_tokens

    @staticmethod
    def _tokenizer_identifier(tokenizer: PreTrainedTokenizerBase) -> str:
        name_or_path = getattr(tokenizer, "name_or_path", None) or "unknown"
        vocab_size = getattr(tokenizer, "vocab_size", None)
        cls_name = tokenizer.__class__.__name__
        return f"{cls_name}-{name_or_path}-{vocab_size}"


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
