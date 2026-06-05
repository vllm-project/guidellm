from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field

from guidellm.data.finalizers.finalizer import DatasetFinalizer, FinalizerRegistry
from guidellm.data.schemas import DataFinalizerArgs
from guidellm.schemas import GenerationRequest, RequestSettings, UsageMetrics

__all__ = [
    "GenerativeRequestFinalizer",
    "GenerativeRequestFinalizerArgs",
]


@DataFinalizerArgs.register("generative")
class GenerativeRequestFinalizerArgs(DataFinalizerArgs):
    """
    Configuration for the GenerativeRequestFinalizer.
    """

    kind: Literal["generative"] = Field(
        default="generative",
        description="Type identifier for the generative request finalizer.",
    )


@FinalizerRegistry.register("generative")
class GenerativeRequestFinalizer(DatasetFinalizer[Iterable[GenerationRequest]]):
    """
    Finalizer that converts dataset rows into GenerationRequest objects,
    aggregating usage metrics from the provided columns.
    """

    def __init__(self, config: GenerativeRequestFinalizerArgs) -> None:
        self.config = config

    def __call__(self, items: list[dict[str, Any]]) -> list[GenerationRequest]:
        return [self.finalize_turn(item) for item in items]

    def finalize_turn(  # noqa: C901 PLR0912
        self, columns: dict[str, Any]
    ) -> GenerationRequest:
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Sum prompt token column
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Sum output token column
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens

        # Count words in prefixes
        for prefix in columns.get("prefix_column", []):
            if not prefix:
                continue

            input_metrics.add_text_metrics(prefix)

        # Count words in text prompts
        for text in columns.get("text_column", []):
            if not text:
                continue

            input_metrics.add_text_metrics(text)

        # Count pixels and bytes in images
        for image in columns.get("image_column", []):
            if not image:
                continue

            if (image_pixels := image.get("image_pixels")) is not None:
                input_metrics.image_pixels = (
                    input_metrics.image_pixels or 0
                ) + image_pixels
            if (image_bytes := image.get("image_bytes")) is not None:
                input_metrics.image_bytes = (
                    input_metrics.image_bytes or 0
                ) + image_bytes

        # Count frames, seconds, and bytes in videos
        for video in columns.get("video_column", []):
            if not video:
                continue

            if (video_frames := video.get("video_frames")) is not None:
                input_metrics.video_frames = (
                    input_metrics.video_frames or 0
                ) + video_frames
            if (video_seconds := video.get("video_seconds")) is not None:
                input_metrics.video_seconds = (
                    input_metrics.video_seconds or 0.0
                ) + video_seconds
            if (video_bytes := video.get("video_bytes")) is not None:
                input_metrics.video_bytes = (
                    input_metrics.video_bytes or 0
                ) + video_bytes

        # Count samples, seconds, and bytes in audio
        for audio in columns.get("audio_column", []):
            if not audio:
                continue

            if (audio_samples := audio.get("audio_samples")) is not None:
                input_metrics.audio_samples = (
                    input_metrics.audio_samples or 0
                ) + audio_samples
            if (audio_seconds := audio.get("audio_seconds")) is not None:
                input_metrics.audio_seconds = (
                    input_metrics.audio_seconds or 0.0
                ) + audio_seconds
            if (audio_bytes := audio.get("audio_bytes")) is not None:
                input_metrics.audio_bytes = (
                    input_metrics.audio_bytes or 0
                ) + audio_bytes

        # A turn expects a tool call if it has tool definitions.
        # Which turns carry tools_column is controlled by the data pipeline
        # (synthetic generator or dataset columns).
        expects_tool_call = bool(columns.get("tools_column"))

        return GenerationRequest(
            columns=columns,
            expects_tool_call=expects_tool_call,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            settings=self._request_settings_from_columns(columns),
        )

    def _request_settings_from_columns(
        self, columns: dict[str, Any]
    ) -> RequestSettings:
        relative_values = columns.get("relative_timestamp_column", [])
        if relative_values and relative_values[0] is not None:
            return RequestSettings(relative_timestamp=float(relative_values[0]))
        return RequestSettings()
