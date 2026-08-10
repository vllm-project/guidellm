from __future__ import annotations

from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.schemas.data.preprocessors import MediaEncoderArgs
from guidellm.utils import audio as guidellm_audio
from guidellm.utils import vision as guidellm_vision

__all__ = ["MediaEncoder"]


def _is_encoded_image(item: Any) -> bool:
    return isinstance(item, dict) and "image" in item and "type" in item


def _is_encoded_video(item: Any) -> bool:
    return isinstance(item, dict) and "video" in item and "type" in item


@PreprocessorRegistry.register("encode_media")
class MediaEncoder(DatasetPreprocessor):
    def __init__(
        self,
        config: MediaEncoderArgs,
    ) -> None:
        self.config = config

    def __call__(self, items: list[dict[str, list[Any]]]) -> list[dict[str, list[Any]]]:
        return [self.encode_turn(item) for item in items]

    def _encode_images(self, images: list[Any]) -> list[Any]:
        encoded_images = []
        for image in images:
            if not image:
                continue
            if _is_encoded_image(image):
                encoded_images.append(image)
                continue

            encoded_images.append(
                guidellm_vision.encode_image(image, **self.config.image_kwargs)
            )
        return encoded_images

    def _encode_videos(self, videos: list[Any]) -> list[Any]:
        encoded_videos = []
        for video in videos:
            if not video:
                continue
            if _is_encoded_video(video):
                encoded_videos.append(video)
                continue

            encoded_videos.append(
                guidellm_vision.encode_video(video, **self.config.video_kwargs)
            )
        return encoded_videos

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
            columns["image_column"] = self._encode_images(columns["image_column"])

        if columns.get("video_column"):
            columns["video_column"] = self._encode_videos(columns["video_column"])

        return columns
