from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import DataPreprocessorArgs
from guidellm.utils import audio as guidellm_audio
from guidellm.utils import vision as guidellm_vision

__all__ = ["MediaEncoder"]


@DataPreprocessorArgs.register("encode_media")
class MediaEncoderArgs(DataPreprocessorArgs):
    kind: Literal["encode_media"] = Field(
        default="encode_media",
        description="Type identifier for the media encoder preprocessor.",
    )
    audio_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for audio encoding.",
    )
    image_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for image encoding.",
    )
    video_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for video encoding.",
    )


@PreprocessorRegistry.register("encode_media")
class MediaEncoder(DatasetPreprocessor):
    def __init__(
        self,
        config: MediaEncoderArgs,
    ) -> None:
        self.config = config

    def __call__(self, items: list[dict[str, list[Any]]]) -> list[dict[str, list[Any]]]:
        return [self.encode_turn(item) for item in items]

    def encode_turn(self, columns: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if columns.get("audio_column"):
            encoded_audio = []
            for audio in columns["audio_column"]:
                if not audio:
                    continue

                encoded_audio.append(
                    guidellm_audio.encode_audio(audio, **self.config.audio_kwargs)
                )
            columns["audio_column"] = encoded_audio

        if columns.get("image_column"):
            encoded_images = []
            for image in columns["image_column"]:
                if not image:
                    continue

                encoded_images.append(
                    guidellm_vision.encode_image(image, **self.config.image_kwargs)
                )
            columns["image_column"] = encoded_images

        if columns.get("video_column"):
            encoded_videos = []
            for video in columns["video_column"]:
                if not video:
                    continue

                encoded_videos.append(
                    guidellm_vision.encode_video(video, **self.config.video_kwargs)
                )
            columns["video_column"] = encoded_videos

        return columns
