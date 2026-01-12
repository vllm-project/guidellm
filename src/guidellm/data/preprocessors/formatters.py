from __future__ import annotations

from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.schemas import GenerationRequest, GenerationRequestArguments, UsageMetrics

__all__ = [
    "GenerativeAudioTranscriptionRequestFormatter",
    "GenerativeAudioTranslationRequestFormatter",
    "GenerativeChatCompletionsRequestFormatter",
    "GenerativeTextCompletionsRequestFormatter",
    "RequestFormatter",
]


class RequestFormatter(DatasetPreprocessor):
    def __init__(self, model: str, **_kwargs):
        self.model = model

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


@PreprocessorRegistry.register("text_completions")
class GenerativeTextCompletionsRequestFormatter(RequestFormatter):
    def __init__(
        self,
        model: str,
        extras: dict[str, Any] | GenerationRequestArguments | None = None,
        stream: bool = True,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ):
        self.model: str = model
        self.extras = (
            GenerationRequestArguments(**extras)
            if extras and isinstance(extras, dict)
            else extras
        )
        self.stream: bool = stream
        self.max_tokens: int | None = max_tokens or max_completion_tokens

    def __call__(self, columns: dict[str, list[Any]]) -> GenerationRequest:
        """
        :param columns: A dict of GenerativeDatasetColumnType to Any
        """
        arguments: GenerationRequestArguments = GenerationRequestArguments()
        arguments.body = {}  # The type checker works better setting this field here
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            arguments.body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            arguments.body["stream"] = True
            arguments.body["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }

        # Handle output tokens
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens
            arguments.body["max_tokens"] = output_tokens
            arguments.body["stop"] = None
            arguments.body["ignore_eos"] = True
        elif self.max_tokens is not None:
            arguments.body["max_tokens"] = self.max_tokens

        # Handle prompt tokens
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Apply extra arguments
        if self.extras:
            arguments.model_combine(self.extras)

        # Build prompt
        prefix = "".join(pre for pre in columns.get("prefix_column", []) if pre)
        text = "".join(txt for txt in columns.get("text_column", []) if txt)
        if prefix or text:
            prompt = prefix + text
            arguments.body["prompt"] = prompt
            input_metrics.add_text_metrics(prompt)

        return GenerationRequest(
            request_type="text_completions",
            arguments=arguments,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@PreprocessorRegistry.register("chat_completions")
class GenerativeChatCompletionsRequestFormatter(RequestFormatter):
    def __init__(
        self,
        model: str,
        extras: dict[str, Any] | GenerationRequestArguments | None = None,
        stream: bool = True,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.extras = (
            GenerationRequestArguments(**extras)
            if extras and isinstance(extras, dict)
            else extras
        )
        self.stream = stream
        self.max_completion_tokens = max_tokens or max_completion_tokens
        self.encode_image_kwargs = (
            encode_kwargs.get("image", {}) if encode_kwargs else {}
        )
        self.encode_video_kwargs = (
            encode_kwargs.get("video", {}) if encode_kwargs else {}
        )
        self.encode_audio_kwargs = (
            encode_kwargs.get("audio", {}) if encode_kwargs else {}
        )

    def __call__(  # noqa: C901, PLR0912, PLR0915
        self, columns: dict[str, list[Any]]
    ) -> GenerationRequest:
        """
        :param columns: A dict of GenerativeDatasetColumnType to Any
        """
        arguments = GenerationRequestArguments()
        arguments.body = {}  # The type checker works best with body assigned here
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            arguments.body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            arguments.body["stream"] = True
            arguments.body["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }

        # Handle output tokens
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens
            arguments.body.update(
                {
                    "max_completion_tokens": output_tokens,
                    "stop": None,
                    "ignore_eos": True,
                }
            )
        elif self.max_completion_tokens is not None:
            arguments.body["max_completion_tokens"] = self.max_completion_tokens

        # Handle prompt tokens
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Apply extra arguments
        if self.extras:
            arguments.model_combine(self.extras)

        # Build messages
        arguments.body["messages"] = []

        for prefix in columns.get("prefix_column", []):
            if not prefix:
                continue

            input_metrics.add_text_metrics(prefix)
            arguments.body["messages"].append({"role": "system", "content": prefix})

        for text in columns.get("text_column", []):
            if not text:
                continue

            input_metrics.add_text_metrics(text)

            arguments.body["messages"].append(
                {"role": "user", "content": [{"type": "text", "text": text}]}
            )

        for image in columns.get("image_column", []):
            if not image:
                continue

            image_dict = self.encode_image(image, **self.encode_image_kwargs)
            if (image_pixels := image_dict.get("image_pixels")) is not None:
                input_metrics.image_pixels = (
                    input_metrics.image_pixels or 0
                ) + image_pixels
            if (image_bytes := image_dict.get("image_bytes")) is not None:
                input_metrics.image_bytes = (
                    input_metrics.image_bytes or 0
                ) + image_bytes

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_dict.get("image")},
                        }
                    ],
                }
            )

        for video in columns.get("video_column", []):
            if not video:
                continue

            video_dict = self.encode_video(video, **self.encode_video_kwargs)
            if (video_frames := video_dict.get("video_frames")) is not None:
                input_metrics.video_frames = (
                    input_metrics.video_frames or 0
                ) + video_frames
            if (video_seconds := video_dict.get("video_seconds")) is not None:
                input_metrics.video_seconds = (
                    input_metrics.video_seconds or 0.0
                ) + video_seconds
            if (video_bytes := video_dict.get("video_bytes")) is not None:
                input_metrics.video_bytes = (
                    input_metrics.video_bytes or 0
                ) + video_bytes

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_dict.get("video")},
                        }
                    ],
                }
            )

        for audio in columns.get("audio_column", []):
            if not audio:
                continue

            audio_dict = self.encode_audio(
                audio, b64encode=True, **self.encode_audio_kwargs
            )
            if (audio_samples := audio_dict.get("audio_samples")) is not None:
                input_metrics.audio_samples = (
                    input_metrics.audio_samples or 0
                ) + audio_samples
            if (audio_seconds := audio_dict.get("audio_seconds")) is not None:
                input_metrics.audio_seconds = (
                    input_metrics.audio_seconds or 0.0
                ) + audio_seconds
            if (audio_bytes := audio_dict.get("audio_bytes")) is not None:
                input_metrics.audio_bytes = (
                    input_metrics.audio_bytes or 0
                ) + audio_bytes

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_dict.get("audio"),
                                "format": audio_dict.get("format"),
                            },
                        }
                    ],
                }
            )

        return GenerationRequest(
            request_type="chat_completions",
            arguments=arguments,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@PreprocessorRegistry.register("audio_transcriptions")
class GenerativeAudioTranscriptionRequestFormatter(RequestFormatter):
    def __init__(
        self,
        model: str,
        extras: dict[str, Any] | GenerationRequestArguments | None = None,
        stream: bool = True,
        encode_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.extras = (
            GenerationRequestArguments(**extras)
            if extras and isinstance(extras, dict)
            else extras
        )
        self.stream = stream
        self.encode_audio_kwargs = encode_kwargs or {}

    def __call__(  # noqa: C901
        self, columns: dict[str, list[Any]]
    ) -> GenerationRequest:
        arguments = GenerationRequestArguments(files={})
        arguments.body = {}  # The type checker works best with body assigned here
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            arguments.body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            arguments.body["stream"] = True
            # NOTE: File upload endpoints use flattened stream options
            arguments.body["stream_include_usage"] = True
            arguments.body["stream_continuous_usage_stats"] = True

        # Handle output tokens
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens

        # Handle prompt tokens (for audio duration tracking)
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Apply extra arguments
        if self.extras:
            arguments.model_combine(self.extras)

        # Build audio input
        audio_columns = columns.get("audio_column", [])
        if len(audio_columns) != 1:
            raise ValueError(
                f"GenerativeAudioTranscriptionRequestFormatter expects exactly "
                f"one audio column, but got {len(audio_columns)}."
            )

        audio_dict = self.encode_audio(
            audio_columns[0], b64encode=False, **self.encode_audio_kwargs
        )
        input_metrics.audio_samples = audio_dict.get("audio_samples")
        input_metrics.audio_seconds = audio_dict.get("audio_seconds")
        input_metrics.audio_bytes = audio_dict.get("audio_bytes")

        arguments.files = {
            "file": (
                audio_dict.get("file_name", "audio_input"),
                audio_dict.get("audio"),
                audio_dict.get("mimetype"),
            )
        }

        return GenerationRequest(
            request_type="audio_transcriptions",
            arguments=arguments,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@PreprocessorRegistry.register("audio_translations")
class GenerativeAudioTranslationRequestFormatter(
    GenerativeAudioTranscriptionRequestFormatter
):
    def __call__(self, columns: dict[str, list[Any]]) -> GenerationRequest:
        result = super().__call__(columns)
        result.request_type = "audio_translations"
        return result
