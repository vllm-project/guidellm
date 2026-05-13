"""Synthetic video dataset deserializer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
from datasets import DatasetInfo, Features, IterableDataset, Value
from datasets.iterable_dataset import _BaseExamplesIterable
from faker import Faker
from pydantic import Field, model_validator
from transformers import PreTrainedTokenizerBase

from guidellm.data.deserializers.deserializer import (
    DatasetDeserializer,
    DatasetDeserializerFactory,
)
from guidellm.data.deserializers.synthetic_image import (
    _RESOLUTION_PRESETS,
    _SyntheticVisionTextMixin,
    _parse_aspect_ratio,
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
class SyntheticVideoDataArgs(_SyntheticVisionTextMixin):
    """Model for synthetic video dataset deserializer arguments."""

    kind: Literal["synthetic_video"] = Field(  # type: ignore[assignment]
        default=_DESERIALIZER_TYPE,
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
            preset = _RESOLUTION_PRESETS.get(self.resolution.lower())
            if preset is None:
                raise ValueError(
                    f"Unknown resolution '{self.resolution}'. Known: "
                    f"{sorted(_RESOLUTION_PRESETS)}"
                )
            preset_w, preset_h = preset
            if h is None:
                h = preset_h
            if w is None:
                w = (
                    int(round(h * _parse_aspect_ratio(self.aspect_ratio)))
                    if self.aspect_ratio is not None
                    else preset_w
                )
        elif self.aspect_ratio is not None:
            if h is not None and w is None:
                w = int(round(h * _parse_aspect_ratio(self.aspect_ratio)))
            elif w is not None and h is None:
                h = int(round(w / _parse_aspect_ratio(self.aspect_ratio)))

        if w is None or h is None:
            raise ValueError(
                "synthetic_video config requires width and height, either "
                "explicitly or via resolution/aspect_ratio."
            )
        self.width = int(w) - (int(w) % 2)
        self.height = int(h) - (int(h) % 2)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Resolved video dims must be positive, got "
                f"{self.width}x{self.height}"
            )
        return self


class _SyntheticVideoExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        config: SyntheticVideoDataArgs,
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
        raise RuntimeError("Processor returned unexpected prompt decode type.")

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        iter_seed = self.random_seed + self.iteration_count
        self.iteration_count += 1

        faker = Faker()
        faker.seed_instance(iter_seed)

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
                "video": synthesize_video(
                    width=int(self.config.width),
                    height=int(self.config.height),
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
            "prefix": Value("string"),
            "prompt_0": Value("string"),
            "prompt_tokens_count_0": Value("int32"),
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
        features["video"] = {
            "type": Value("string"),
            "video": Value("string"),
            "video_frames": Value("int64"),
            "video_seconds": Value("float64"),
            "video_bytes": Value("int64"),
        }
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
        processor: PreTrainedTokenizerBase,
        random_seed: int = 42,
    ):
        self.config = config
        self.processor = processor
        self.random_seed = random_seed

        ex_iterable = _SyntheticVideoExamplesIterable(
            config=config,
            processor=processor,
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
        return SyntheticVideoDataset(
            config=config,
            processor=processor_factory(),
            random_seed=random_seed,
        )
