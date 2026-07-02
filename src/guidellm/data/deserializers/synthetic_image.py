"""Synthetic image dataset deserializer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from pydantic import Field, model_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.random import IntegerRangeSampler
from guidellm.utils.vision import synthesize_image

__all__ = [
    "SyntheticImageDataArgs",
    "SyntheticImageDataset",
    "SyntheticImageDatasetDeserializer",
]


_DESERIALIZER_TYPE = "synthetic_image"
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}


def parse_aspect_ratio(aspect: str) -> float:
    """Parse 'W:H' or 'W/H' into a float ratio."""
    sep = ":" if ":" in aspect else "/"
    try:
        w, h = aspect.split(sep)
        return float(w) / float(h)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid aspect_ratio '{aspect}', expected 'W:H' or 'W/H'"
        ) from exc


class SyntheticVisionDataArgs(DataArgs):
    output_tokens: int | None = Field(
        description="The average number of output tokens to request.",
        gt=0,
        default=None,
    )
    output_tokens_stdev: int | None = Field(
        description="Standard deviation of output-token counts per request.",
        gt=0,
        default=None,
    )
    output_tokens_min: int | None = Field(
        description="Minimum number of output tokens per request.",
        gt=0,
        default=None,
    )
    output_tokens_max: int | None = Field(
        description="Maximum number of output tokens per request.",
        gt=0,
        default=None,
    )
    seed: int = Field(
        description="Base random seed for reproducible synthetic payloads.",
        default=42,
    )


@DataArgs.register(_DESERIALIZER_TYPE)
class SyntheticImageDataArgs(SyntheticVisionDataArgs):
    """Model for synthetic image dataset deserializer arguments."""

    kind: Literal["synthetic_image"] = Field(  # type: ignore[assignment]
        default="synthetic_image",
        description="Type identifier for the synthetic image dataset configuration.",
    )
    width: int | None = Field(
        description="Image width in pixels.",
        gt=0,
        default=None,
    )
    height: int | None = Field(
        description="Image height in pixels.",
        gt=0,
        default=None,
    )
    resolution: str | None = Field(
        description="Resolution shortcut such as '720p' or '1080p'.",
        default=None,
    )
    aspect_ratio: str | None = Field(
        description="Aspect ratio override, e.g. '16:9' or '4:3'.",
        default=None,
    )
    format: Literal["jpeg", "png"] = Field(
        description="Encoded image format.",
        default="jpeg",
    )
    jpeg_quality: int = Field(
        description="JPEG quality 1..100. Ignored when format='png'.",
        ge=1,
        le=100,
        default=85,
    )
    content: Literal["gradient", "noise", "solid", "checkerboard"] = Field(
        description="Pixel content to synthesize.",
        default="gradient",
    )
    images_per_request: int = Field(
        description="Number of images per emitted row.",
        ge=1,
        default=1,
    )

    @model_validator(mode="after")
    def _resolve_dimensions(self) -> SyntheticImageDataArgs:
        w = self.width
        h = self.height
        if self.resolution is not None:
            preset = RESOLUTION_PRESETS.get(self.resolution.lower())
            if preset is None:
                raise ValueError(
                    f"Unknown resolution '{self.resolution}'. Known: "
                    f"{sorted(RESOLUTION_PRESETS)}"
                )
            preset_w, preset_h = preset
            if h is None:
                h = preset_h
            if w is None:
                w = (
                    int(round(h * parse_aspect_ratio(self.aspect_ratio)))
                    if self.aspect_ratio is not None
                    else preset_w
                )
        elif self.aspect_ratio is not None:
            if h is not None and w is None:
                w = int(round(h * parse_aspect_ratio(self.aspect_ratio)))
            elif w is not None and h is None:
                h = int(round(w / parse_aspect_ratio(self.aspect_ratio)))

        if w is None or h is None:
            raise ValueError(
                "synthetic_image config requires width and height, either "
                "explicitly or via resolution/aspect_ratio."
            )
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved image dims must be positive, got {self.width}x{self.height}"
            )
        return self


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
