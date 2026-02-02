from __future__ import annotations

from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)

__all__ = ["MediaEncoder"]


@PreprocessorRegistry.register("encode_media")
class MediaEncoder(DatasetPreprocessor):
    def __init__(
        self,
        encode_kwargs: dict[str, Any] | None = None,
        **_: Any,  # Ignore global kwargs
    ) -> None:
        self.encode_audio_kwargs = (
            encode_kwargs.get("audio", {}) if encode_kwargs else {}
        )
        self.encode_image_kwargs = (
            encode_kwargs.get("image", {}) if encode_kwargs else {}
        )
        self.encode_video_kwargs = (
            encode_kwargs.get("video", {}) if encode_kwargs else {}
        )

    @staticmethod
    def encode_audio(*args, **kwargs):
        from guidellm.extras.audio import encode_audio

        return encode_audio(*args, **kwargs)

    @staticmethod
    def encode_image(*args, **kwargs):
        from guidellm.extras.vision import encode_image

        return encode_image(*args, **kwargs)

    @staticmethod
    def encode_video(*args, **kwargs):
        from guidellm.extras.vision import encode_video

        return encode_video(*args, **kwargs)

    def __call__(self, columns: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if columns.get("audio_column"):
            encoded_audio = []
            for audio in columns["audio_column"]:
                if not audio:
                    continue

                encoded_audio.append(
                    self.encode_audio(audio, b64encode=True, **self.encode_audio_kwargs)
                )
            columns["audio_column"] = encoded_audio

        if columns.get("image_column"):
            encoded_images = []
            for image in columns["image_column"]:
                if not image:
                    continue

                encoded_images.append(
                    self.encode_image(image, **self.encode_image_kwargs)
                )
            columns["image_column"] = encoded_images

        if columns.get("video_column"):
            encoded_videos = []
            for video in columns["video_column"]:
                if not video:
                    continue

                encoded_videos.append(
                    self.encode_video(video, **self.encode_video_kwargs)
                )
            columns["video_column"] = encoded_videos

        return columns
