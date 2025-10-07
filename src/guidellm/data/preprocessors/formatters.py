from __future__ import annotations

from typing import Any, Literal

from guidellm.data.objects import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerativeDatasetColumnType,
)
from guidellm.data.preprocessors.preprocessor import (
    DatasetPreprocessor,
    PreprocessorRegistry,
)
from guidellm.data.utils import (
    encode_audio_as_dict,
    encode_audio_as_file,
    encode_image,
    encode_video,
)

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
        arguments = {"json_body": {}}
        stats = {}

        # Add model
        if self.model is not None:
            arguments["json_body"]["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments["json_body"].update(
                {"stream": True, "stream_options": {"include_usage": True}}
            )
            arguments["stream"] = True

        # Handle output tokens
        if output_tokens := columns.get("output_tokens_count_column", []):
            output_count = output_tokens[0]
            stats["output_tokens"] = output_count
            arguments["json_body"].update(
                {"max_tokens": output_count, "stop": None, "ignore_eos": True}
            )
        elif self.max_tokens is not None:
            arguments["json_body"]["max_tokens"] = self.max_tokens

        # Handle prompt tokens
        if prompt_tokens := columns.get("prompt_tokens_count_column", []):
            stats["prompt_tokens"] = prompt_tokens[0]

        # Apply extra arguments
        if self.extras:
            arguments = GenerationRequestArguments.model_combine_dict(
                arguments, self.extras
            )

        # Build prompt
        arguments["json_body"]["prompt"] = "".join(
            columns.get("prefix_column", []) + columns.get("text_column", [])
        )

        return GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(**arguments),
            stats=stats,
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
        arguments = {"json_body": {}}
        stats = {}

        # Add model
        if self.model is not None:
            arguments["json_body"]["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments["json_body"].update(
                {"stream": True, "stream_options": {"include_usage": True}}
            )
            arguments["stream"] = True

        # Handle output tokens
        if output_tokens := columns.pop("output_tokens_count_column", []):
            output_count = output_tokens[0]
            stats["output_tokens"] = output_count
            arguments["json_body"].update(
                {
                    "max_completion_tokens": output_count,
                    "stop": None,
                    "ignore_eos": True,
                }
            )
        elif self.max_completion_tokens is not None:
            arguments["json_body"]["max_completion_tokens"] = self.max_completion_tokens

        # Handle prompt tokens
        if prompt_tokens := columns.pop("prompt_tokens_count_column", []):
            stats["prompt_tokens"] = prompt_tokens[0]

        # Apply extra arguments
        if self.extras:
            arguments = GenerationRequestArguments.model_combine_dict(
                arguments, self.extras
            )

        # Build messages
        arguments["json_body"]["messages"] = (
            [
                {"role": "system", "content": prefix}
                for prefix in columns.pop("prefix_column", [])
            ]
            + [
                {"role": "user", "content": [{"type": "text", "text": text}]}
                for text in columns.pop("text_column", [])
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
                for image in columns.pop("image_column", [])
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
                for video in columns.pop("video_column", [])
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
                for audio in columns.pop("audio_column", [])
            ]
        )

        return GenerationRequest(
            request_type="chat_completions",
            arguments=GenerationRequestArguments(**arguments),
            stats=stats,
        )


@PreprocessorRegistry.register("audio_transcriptions")
class GenerativeAudioTranscriptionRequestFormatter(DatasetPreprocessor):
    def __init__(
        self,
        model: str,
        extra_args: dict[str, Any] | GenerationRequestArguments | None = None,
        stream: bool = True,
        encode_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.extra_args = extra_args
        self.stream = stream
        self.encode_audio_kwargs = encode_kwargs or {}

    def __call__(
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> GenerationRequest:
        arguments = {"json_body": {}}
        stats = {}

        # Add model
        if self.model is not None:
            arguments["json_body"]["model"] = self.model

        # Configure streaming
        if self.stream:
            arguments["json_body"].update(
                {"stream": True, "stream_options": {"include_usage": True}}
            )

        # Apply extra arguments
        if self.extra_args:
            arguments = GenerationRequestArguments.model_combine_dict(
                arguments, self.extra_args
            )

        # Handle stats tokens
        if output_tokens := columns.get("output_tokens_count_column", []):
            output_count = output_tokens[0]
            stats["output_tokens"] = output_count
        if prompt_tokens := columns.get("prompt_tokens_count_column", []):
            stats["prompt_tokens"] = prompt_tokens[0]

        # Build audio input
        if audio := columns.get("audio_column", []):
            arguments["files"]["file"] = encode_audio_as_file(
                audio[0], **self.encode_audio_kwargs
            )
        else:
            raise ValueError("No audio column found for audio transcription request.")

        # Build prompt
        if (prefix := columns.get("prefix_column", [])) or (
            text := columns.get("text_column", [])
        ):
            arguments["json_body"]["prompt"] = "".join(prefix) + "".join(text)

        return {
            "request": {
                "request_type": "audio_transcriptions",
                "arguments": arguments,
                "stats": stats,
            }
        }


@PreprocessorRegistry.register("audio_translations")
class GenerativeAudioTranslationRequestFormatter(
    GenerativeAudioTranscriptionRequestFormatter
):
    def __call__(
        self, columns: dict[GenerativeDatasetColumnType, list[Any]]
    ) -> dict[Literal["request"], dict[Literal["request_type"], Any]]:
        result = super().__call__(columns)
        result["request"]["request_type"] = "audio_translations"
        return result
