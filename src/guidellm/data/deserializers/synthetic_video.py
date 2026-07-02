"""Synthetic video dataset deserializer."""

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
from guidellm.data.deserializers.synthetic_image import (
    RESOLUTION_PRESETS,
    SyntheticVisionDataArgs,
    parse_aspect_ratio,
)
from guidellm.data.schemas import DataArgs
from guidellm.utils.random import IntegerRangeSampler
from guidellm.utils.vision import synthesize_video

__all__ = [
    "SyntheticVideoDataArgs",
    "SyntheticVideoDataset",
    "SyntheticVideoDatasetDeserializer",
]


_DESERIALIZER_TYPE = "synthetic_video"


@DataArgs.register(_DESERIALIZER_TYPE)
class SyntheticVideoDataArgs(SyntheticVisionDataArgs):
    """Model for synthetic video dataset deserializer arguments."""

    kind: Literal["synthetic_video"] = Field(  # type: ignore[assignment]
        default="synthetic_video",
        description="Type identifier for the synthetic video dataset configuration.",
    )
    width: int | None = Field(
        description="Frame width in pixels.",
        gt=0,
        default=None,
    )
    height: int | None = Field(
        description="Frame height in pixels.",
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
    frames: int = Field(
        description="Number of frames in the clip.",
        ge=1,
    )
    fps: float = Field(
        description="Frames per second.",
        gt=0.0,
        default=1.0,
    )
    format: Literal["mp4"] = Field(
        description="Container / codec. Only mp4 (h264, yuv420p) in v1.",
        default="mp4",
    )
    video_bitrate: str | None = Field(
        description="Optional libx264 bitrate string, e.g. '500k'.",
        default=None,
    )
    content: Literal["gradient", "noise"] = Field(
        description="Frame content to synthesize.",
        default="gradient",
    )

    @model_validator(mode="after")
    def _resolve_dimensions(self) -> SyntheticVideoDataArgs:
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
                "synthetic_video config requires width and height, either "
                "explicitly or via resolution/aspect_ratio."
            )
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved video dims must be positive, got {self.width}x{self.height}"
            )
        return self


class _SyntheticVideoExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        config: SyntheticVideoDataArgs,
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
            width = self.config.width
            height = self.config.height
            if width is None or height is None:
                raise RuntimeError("Synthetic video dimensions were not resolved.")

            row: dict[str, Any] = {
                "video": synthesize_video(
                    width=width,
                    height=height,
                    frames=int(self.config.frames),
                    fps=float(self.config.fps),
                    content=self.config.content,
                    video_format=self.config.format,
                    video_bitrate=self.config.video_bitrate,
                    seed=self.config.seed,
                    row_index=row_index,
                ),
            }
            if output_token_count is not None:
                row["output_tokens_count_0"] = output_token_count

            yield row_index, row
            row_index += 1

    @property
    def is_typed(self) -> bool:
        return True

    @property
    def features(self) -> Features:
        features: dict[str, Any] = {
            "video": {
                "type": Value("string"),
                "video": Value("string"),
                "video_frames": Value("int64"),
                "video_seconds": Value("float64"),
                "video_bytes": Value("int64"),
            },
        }
        if self.config.output_tokens is not None:
            features["output_tokens_count_0"] = Value("int32")
        return Features(features)

    @property
    def num_shards(self) -> int:
        return 1

    def shuffle_data_sources(
        self,
        generator: np.random.Generator,  # noqa: ARG002
    ) -> _SyntheticVideoExamplesIterable:
        return self

    def shard_data_sources(
        self,
        num_shards: int,  # noqa: ARG002
        index: int,  # noqa: ARG002
        contiguous: bool = True,  # noqa: ARG002
    ) -> _SyntheticVideoExamplesIterable:
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration_count = state_dict.get("iteration_count", 0)

    def _init_state_dict(self) -> dict:
        self._state_dict = {"iteration_count": self.iteration_count}
        return self._state_dict


class SyntheticVideoDataset(IterableDataset):
    def __init__(
        self,
        config: SyntheticVideoDataArgs,
        random_seed: int = 42,
    ):
        self.config = config
        self.random_seed = random_seed

        ex_iterable = _SyntheticVideoExamplesIterable(
            config=config,
            random_seed=random_seed,
        )
        super().__init__(
            ex_iterable=ex_iterable,
            info=DatasetInfo(
                description="Synthetic video dataset generator",
                features=ex_iterable.features,
            ),
        )

    def set_epoch(self, epoch: int):
        if isinstance(self._ex_iterable, _SyntheticVideoExamplesIterable):
            self._ex_iterable.iteration_count = epoch


@DatasetDeserializerFactory.register(_DESERIALIZER_TYPE)
class SyntheticVideoDatasetDeserializer(DatasetDeserializer):
    def __call__(
        self,
        config: SyntheticVideoDataArgs,
        processor_factory: Callable[[], PreTrainedTokenizerBase],
        random_seed: int,
    ) -> IterableDataset:
        _ = processor_factory
        return SyntheticVideoDataset(
            config=config,
            random_seed=random_seed,
        )
