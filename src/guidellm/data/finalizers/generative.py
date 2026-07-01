from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field

from guidellm.data.finalizers.finalizer import DatasetFinalizer, FinalizerRegistry
from guidellm.data.schemas import DataFinalizerArgs
from guidellm.schemas import (
    GenerationRequest,
    RequestSettings,
    TurnType,
    UsageMetrics,
)

__all__ = [
    "GenerativeRequestFinalizer",
    "GenerativeRequestFinalizerArgs",
]


@DataFinalizerArgs.register("generative")
class GenerativeRequestFinalizerArgs(DataFinalizerArgs):
    """Model for generative request finalizer arguments."""

    kind: Literal["generative"] = Field(
        default="generative",
        description="Type identifier for the generative request finalizer.",
    )
    tool_call_mode: Literal["client", "server"] = Field(
        default="client",
        description="How to handle turns with tool definitions. "
        "'client' (default) creates client_tool_call + injection turns. "
        "'server' creates server_tool_call turns (no injection, "
        "tools are server-managed).",
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
        results: list[GenerationRequest] = []
        for item in items:
            request = self.finalize_turn(item)
            if request.turn_type == "client_tool_call":
                # Split tool-calling turns: the tool_response_column moves
                # to a separate injection turn that follows the tool call.
                tool_response_data = request.columns.pop("tool_response_column", [])
                injection_columns: dict[str, list[Any]] = {}
                if tool_response_data:
                    injection_columns["tool_response_column"] = tool_response_data
                # Move output metrics to next turn
                metrics_config = request.output_metrics
                request.output_metrics = UsageMetrics()
                results.append(request)
                results.append(
                    GenerationRequest(
                        columns=injection_columns,
                        turn_type="tool_response_injection",
                        output_metrics=metrics_config,
                    )
                )
            else:
                results.append(request)
        return results

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

        # Resolve turn type with priority:
        # 1. Explicit turn_type_column (from synthetic server_tool_call_turns
        #    or hand-crafted datasets)
        # 2. tools_column presence + tool_call_mode config
        # 3. Default to "standard"
        turn_type_values = columns.get("turn_type_column", [])
        turn_type: TurnType
        if turn_type_values and turn_type_values[0]:
            turn_type = turn_type_values[0]
        elif columns.get("tools_column"):
            if self.config.tool_call_mode == "server":
                turn_type = "server_tool_call"
                # Tools are server-managed; strip data-provided definitions
                # so the request handler doesn't inject them.
                columns.pop("tools_column", None)
                columns.pop("tool_response_column", None)
            else:
                turn_type = "client_tool_call"
        else:
            turn_type = "standard"

        return GenerationRequest(
            columns=columns,
            turn_type=turn_type,
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
