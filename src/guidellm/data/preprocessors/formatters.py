from __future__ import annotations

from typing import Any

from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.schemas import GenerativeDatasetColumnType
from guidellm.data.utils import (
    encode_audio_as_dict,
    encode_audio_as_file,
    encode_image,
    encode_video,
)
from guidellm.schemas import GenerationRequest, GenerationRequestArguments, UsageMetrics

__all__ = [
    "GenerativeAudioTranscriptionRequestFormatter",
    "GenerativeAudioTranslationRequestFormatter",
    "GenerativeChatCompletionsRequestFormatter",
    "GenerativeTextCompletionsRequestFormatter",
]


@PreprocessorRegistry.register("text_completions")
class GenerativeTextCompletionsRequestFormatter(DatasetPreprocessor):
    def __init__(
        self,
        model: str,
        extras: dict[str, Any] | GenerationRequestArguments | None = None,
        stream: bool = True,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
    ):
        self.model: str | None = model
        self.extras = (
            GenerationRequestArguments(**extras)
            if extras and isinstance(extras, dict)
            else extras
        )
        self.stream: bool = stream
        self.max_tokens: int | None = max_tokens or max_completion_tokens

    def __call__(
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> GenerationRequest:
        body: dict[str, Any] = {}
        arguments: GenerationRequestArguments = GenerationRequestArguments(body=body)
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            body["stream"] = True

        # Handle output tokens
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens
            body["max_tokens"] = output_tokens
            body["stop"] = None
            body["ignore_eos"] = True
        elif self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

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
            body["prompt"] = prefix + text

        return GenerationRequest(
            request_type="text_completions",
            arguments=arguments,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@PreprocessorRegistry.register("chat_completions")
class GenerativeChatCompletionsRequestFormatter(DatasetPreprocessor):
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

    def __call__(
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> GenerationRequest:
        body: dict[str, Any] = {}
        arguments = GenerationRequestArguments(body=body)
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            body.update({"stream": True, "stream_options": {"include_usage": True}})

        # Handle output tokens
        if output_tokens := sum(
            count for count in columns.get("output_tokens_count_column", []) if count
        ):
            output_metrics.text_tokens = output_tokens
            body.update(
                {
                    "max_completion_tokens": output_tokens,
                    "stop": None,
                    "ignore_eos": True,
                }
            )
        elif self.max_completion_tokens is not None:
            body["max_completion_tokens"] = self.max_completion_tokens

        # Handle prompt tokens
        if prompt_tokens := sum(
            count for count in columns.get("prompt_tokens_count_column", []) if count
        ):
            input_metrics.text_tokens = prompt_tokens

        # Apply extra arguments
        if self.extras:
            arguments.model_combine(self.extras)

        # Build messages
        body["messages"] = (
            [
                {"role": "system", "content": prefix}
                for prefix in columns.get("prefix_column", [])
                if prefix
            ]
            + [
                {"role": "user", "content": [{"type": "text", "text": text}]}
                for text in columns.get("text_column", [])
                if text
            ]
            + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": encode_image(
                                image, **self.encode_image_kwargs
                            ),
                        }
                    ],
                }
                for image in columns.get("image_column", [])
                if image
            ]
            + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": encode_video(
                                video, **self.encode_video_kwargs
                            ),
                        }
                    ],
                }
                for video in columns.get("video_column", [])
                if video
            ]
            + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": encode_audio_as_dict(
                                audio, **self.encode_audio_kwargs
                            ),
                        }
                    ],
                }
                for audio in columns.get("audio_column", [])
                if audio
            ]
        )

        return GenerationRequest(
            request_type="chat_completions",
            arguments=arguments,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@PreprocessorRegistry.register("audio_transcriptions")
class GenerativeAudioTranscriptionRequestFormatter(DatasetPreprocessor):
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
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> GenerationRequest:
        body: dict[str, Any] = {}
        arguments = GenerationRequestArguments(body=body, files={})
        input_metrics = UsageMetrics()
        output_metrics = UsageMetrics()

        # Add model
        if self.model is not None:
            body["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments.stream = True
            body.update({"stream": True, "stream_options": {"include_usage": True}})

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
        if audio := [aud for aud in columns.get("audio_column", []) if aud]:
            file_name, content, mime_type = encode_audio_as_file(
                audio[0], **self.encode_audio_kwargs
            )
            arguments.files = {"file": (file_name, content, mime_type)}
        else:
            raise ValueError("No audio column found for audio transcription request.")

        # Build prompt
        prefix = "".join(pre for pre in columns.get("prefix_column", []) if pre)
        text = "".join(txt for txt in columns.get("text_column", []) if txt)
        if prefix or text:
            body["prompt"] = prefix + text

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
    def __call__(
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> GenerationRequest:
        result = super().__call__(columns)
        result.request_type = "audio_translations"
        return result
