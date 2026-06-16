"""Multi-image synthetic data deserializer for vision benchmarking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from datasets import IterableDataset
from pydantic import Field, field_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.synthetic import (
    SyntheticTextDataArgs,
    SyntheticTextDataset,
    SyntheticTextDatasetDeserializer,
)
from guidellm.data.generators.multi_image import ImageSize
from guidellm.data.schemas import DataArgs

__all__ = [
    "MultiImageDataArgs",
    "MultiImageDatasetDeserializer",
]

_VALID_IMAGE_SIZES = sorted(ImageSize.SIZES.keys())


@DataArgs.register("multi_image")
class MultiImageDataArgs(SyntheticTextDataArgs):
    """
    Data args for generating synthetic multi-image prompts.

    Extends SyntheticTextDataArgs with image count and resolution fields.
    """

    kind: Literal["multi_image"] = Field(  # type: ignore[assignment]
        default="multi_image",
        description="Type identifier for the multi-image dataset configuration.",
    )
    images_per_request: int = Field(
        description="Number of images to include per request.",
        ge=1,
        le=10,
        default=1,
    )
    image_size: str = Field(
        description=(
            f"Standard image resolution key. Valid values: {_VALID_IMAGE_SIZES}."
        ),
        default="720p",
    )

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: str) -> str:
        if value not in ImageSize.SIZES:
            raise ValueError(
                f"Invalid image_size {value!r}. Valid options: {_VALID_IMAGE_SIZES}"
            )
        return value


# Keep the old name as an alias for backwards compatibility within this PR.
MultiImageDatasetConfig = MultiImageDataArgs


@DatasetDeserializerFactory.register("multi_image")
class MultiImageDatasetDeserializer(SyntheticTextDatasetDeserializer):
    def __call__(
        self,
        config: MultiImageDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        return SyntheticTextDataset(
            config=config,
            processor=processor_factory(),
            random_seed=random_seed,
        )
