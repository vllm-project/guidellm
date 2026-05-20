"""Synthetic image dataset deserializer.

Generates fully encoded image rows (data URLs + metric fields) without
requiring an external dataset. The emitted dict matches the canonical shape
that :func:`guidellm.extras.vision.encode_image` returns, so it flows
through the column mapper and finalizer unchanged. The ``MediaEncoder``
preprocessor is intentionally bypassed for synthetic rows since the payload
is already encoded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from random import Random
from typing import Any

import numpy as np
import yaml
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
from guidellm.data.schemas import SyntheticImageDatasetConfig
from guidellm.extras.vision import synthesize_image
from guidellm.utils import arg_string
from guidellm.utils.random import IntegerRangeSampler

__all__ = [
    "SyntheticImageDataset",
    "SyntheticImageDatasetDeserializer",
]


_DESERIALIZER_TYPE = "synthetic_image"


class _SyntheticImageExamplesIterable(_BaseExamplesIterable):
    """Examples iterable that yields rows of synthetic images + text."""

    def __init__(
        self,
        config: SyntheticImageDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.random_seed = random_seed
        self.iteration_count = 0

    @staticmethod
    def _build_prompt(
        token_count: int,
        processor: PreTrainedTokenizerBase,
        faker: Faker,
        unique: str,
    ) -> str:
        token_ids: list[int] = []
        avg_chars_per_token = 5
        margin_of_safety = 1.5
        attempts = 0
        while len(token_ids) < token_count:
            attempts += 1
            num_chars = int(
                token_count * avg_chars_per_token * margin_of_safety * attempts
            )
            text = unique + faker.text(max_nb_chars=num_chars)
            token_ids = processor.encode(text)
        decoded = processor.decode(token_ids[:token_count], skip_special_tokens=True)
        if isinstance(decoded, str):
            return decoded
        raise RuntimeError(
            "Processor.decode returned a non-string value while generating "
            "synthetic image prompt text."
        )

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        iter_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_seed)
        Random(iter_seed)

        text_tokens_sampler = iter(
            IntegerRangeSampler(
                average=self.config.text_tokens,
                variance=self.config.text_tokens_stdev,
                min_value=self.config.text_tokens_min,
                max_value=self.config.text_tokens_max,
                random_seed=iter_seed,
            )
        )
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
            text_token_count = next(text_tokens_sampler)
            output_token_count = (
                next(output_tokens_sampler)
                if output_tokens_sampler is not None
                else None
            )
            prompt = self._build_prompt(
                text_token_count,
                self.processor,
                faker,
                f"{self.iteration_count} {row_index} ",
            )

            row: dict[str, Any] = {
                "prefix": "",
                "prompt_0": prompt,
                "prompt_tokens_count_0": text_token_count,
            }
            if output_token_count is not None:
                row["output_tokens_count_0"] = output_token_count

            # The column mapper looks for canonical names like "image" /
            # "image_0" / "image_1". When images_per_request > 1 we emit
            # each as image_<idx>.
            for img_idx in range(self.config.images_per_request):
                encoded = synthesize_image(
                    width=int(self.config.width),  # type: ignore[arg-type]
                    height=int(self.config.height),  # type: ignore[arg-type]
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
        features: dict[str, Any] = {
            "prefix": Value("string"),
            "prompt_0": Value("string"),
            "prompt_tokens_count_0": Value("int32"),
        }
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
        config: SyntheticImageDatasetConfig,
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

        ex_iterable = _SyntheticImageExamplesIterable(
            config=config,
            processor=processor,
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


def _peek_type(data: Any) -> str | None:
    """Return the value of a ``type`` key if data is a recognizable config.

    Returns ``None`` when ``type`` is unset or the input shape isn't one we
    can peek into without consuming it.
    """
    if isinstance(data, dict):
        value = data.get("type")
        return value if isinstance(value, str) else None
    if isinstance(data, str):
        try:
            parsed = yaml.safe_load(data)
        except yaml.YAMLError:
            parsed = data
        if parsed == data:
            try:
                parsed = arg_string.loads(parsed)
            except arg_string.ArgStringParseError:
                return None
        if isinstance(parsed, dict):
            value = parsed.get("type")
            return value if isinstance(value, str) else None
    return None


@DatasetDeserializerFactory.register(_DESERIALIZER_TYPE)
class SyntheticImageDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        data: Any,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
        **data_kwargs: dict[str, Any],
    ) -> IterableDataset:
        # Bail early if the user explicitly asked for a different type.
        peeked_type = _peek_type(data)
        if peeked_type is not None and peeked_type != _DESERIALIZER_TYPE:
            raise DataNotSupportedError(
                f"SyntheticImageDatasetDeserializer requires "
                f"type='{_DESERIALIZER_TYPE}' (got '{peeked_type}')."
            )

        if (config := load_config(data, SyntheticImageDatasetConfig)) is not None:
            return self(config, processor_factory, random_seed, **data_kwargs)

        if not isinstance(data, SyntheticImageDatasetConfig):
            raise DataNotSupportedError(
                "Unsupported data for SyntheticImageDatasetDeserializer, "
                "expected SyntheticImageDatasetConfig, str, or Path to a "
                f"config file, got {data}"
            )

        return SyntheticImageDataset(
            config=data,
            processor=processor_factory(),
            random_seed=random_seed,
        )
