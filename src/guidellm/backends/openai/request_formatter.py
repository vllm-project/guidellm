"""
Handlers for converting data into given formats for requests.
"""

from typing import Protocol

from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
)
from guidellm.utils import RegistryMixin


class GenerationRequestFormatter(Protocol):
    """
    Protocol for handling generation request formatting and response parsing.

    Defines the interface for classes that format requests to various backend
    generation APIs and parse their responses into a standardized format.
    """

    def format(
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the generation request into the appropriate structure for
        the backend API.

        :param request: The generation request to format
        :param kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        ...


class GenerationRequestFormatterFactory(
    RegistryMixin[type[GenerationRequestFormatter]]
):
    """
    Factory for registering and creating response handlers by backend type.

    Registry-based system for associating handler classes with specific backend API
    types, enabling automatic selection of the appropriate handler for processing
    responses from different generation services.
    """

    @classmethod
    def create(
        cls,
        request_type: str,
    ) -> GenerationRequestFormatter:
        """
        Create a response handler class for the given request type.

        :param request_type: The type of generation request (e.g., "text_completions")
        :return: The corresponding instantiated GenerationResponseHandler
        :raises ValueError: When no handler is registered for the request type
        """
        handler_cls = cls.get_registered_object(request_type)
        if not handler_cls:
            raise ValueError(
                f"No response handler registered for type '{request_type}'."
            )

        return handler_cls()


@GenerationRequestFormatterFactory.register("text_completions")
class TextCompletionsResponseHandler(GenerationRequestFormatter):
    """
    Handler for formatting text completion requests and parsing their responses.

    Implements the GenerationRequestHandler protocol to provide request formatting
    and response parsing logic specific to text completion backend APIs.
    """

    def format(
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the text completion generation request into the appropriate structure.

        :param request: The generation request to format
        :return: The formatted request arguments
        """
        arguments: GenerationRequestArguments = GenerationRequestArguments()
        arguments.body = {}  # The type checker works better setting this field here

        # Add model
        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        # Configure streaming
        if kwargs.get("stream"):
            arguments.stream = True
            arguments.body["stream"] = True
            arguments.body["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }

        # Handle output tokens
        if data.output_metrics.text_tokens:
            arguments.body["max_tokens"] = data.output_metrics.text_tokens
            arguments.body["stop"] = None
            arguments.body["ignore_eos"] = True
        elif kwargs.get("max_tokens") is not None:
            arguments.body["max_tokens"] = kwargs["max_tokens"]

        # Apply extra arguments
        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        # Build prompt
        prefix = "".join(pre for pre in data.columns.get("prefix_column", []) if pre)
        text = "".join(txt for txt in data.columns.get("text_column", []) if txt)
        if prefix or text:
            prompt = prefix + text
            arguments.body["prompt"] = prompt

        return arguments


@GenerationRequestFormatterFactory.register("chat_completions")
class ChatCompletionsResponseHandler(GenerationRequestFormatter):
    """
    Handler for formatting chat completion requests and parsing their responses.

    Implements the GenerationRequestHandler protocol to provide request formatting
    and response parsing logic specific to chat completion backend APIs.
    """

    def format(  # noqa: C901, PLR0912, PLR0915
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the chat completion generation request into the appropriate structure.

        :param request: The generation request to format
        :return: The formatted request arguments
        """
        arguments = GenerationRequestArguments()
        arguments.body = {}  # The type checker works best with body assigned here

        # Add model
        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        # Configure streaming
        if kwargs.get("stream"):
            arguments.stream = True
            arguments.body["stream"] = True
            arguments.body["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }

        # Handle output tokens
        if data.output_metrics.text_tokens:
            arguments.body.update(
                {
                    "max_completion_tokens": data.output_metrics.text_tokens,
                    "stop": None,
                    "ignore_eos": True,
                }
            )
        elif kwargs.get("max_tokens") is not None:
            arguments.body["max_completion_tokens"] = kwargs["max_tokens"]

        # Apply extra arguments
        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        # Build messages
        arguments.body["messages"] = []

        for prefix in data.columns.get("prefix_column", []):
            if not prefix:
                continue

            arguments.body["messages"].append({"role": "system", "content": prefix})

        for text in data.columns.get("text_column", []):
            if not text:
                continue

            arguments.body["messages"].append(
                {"role": "user", "content": [{"type": "text", "text": text}]}
            )

        for image in data.columns.get("image_column", []):
            if not image:
                continue

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image.get("image")},
                        }
                    ],
                }
            )

        for video in data.columns.get("video_column", []):
            if not video:
                continue

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video.get("video")},
                        }
                    ],
                }
            )

        for audio in data.columns.get("audio_column", []):
            if not audio:
                continue

            arguments.body["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio.get("audio"),
                                "format": audio.get("format"),
                            },
                        }
                    ],
                }
            )

        return arguments


@GenerationRequestFormatterFactory.register(
    ["audio_transcriptions", "audio_translations"]
)
class AudioTranscriptionResponseHandler(GenerationRequestFormatter):
    """
    Handler for formatting audio transcription requests and parsing their responses.

    Implements the GenerationRequestHandler protocol to provide request formatting
    and response parsing logic specific to audio transcription backend APIs.
    """

    def format(
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:  # noqa: C901
        """
        Format the audio transcription generation request into the
        appropriate structure.

        :param request: The generation request to format
        :return: The formatted request arguments
        """
        arguments = GenerationRequestArguments(files={})
        arguments.body = {}

        # Add model
        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        # Configure streaming
        if kwargs.get("stream"):
            arguments.stream = True
            arguments.body["stream"] = True
            # NOTE: File upload endpoints use flattened stream options
            arguments.body["stream_include_usage"] = True
            arguments.body["stream_continuous_usage_stats"] = True

        # Apply extra arguments
        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        # Build audio input
        audio_columns = data.columns.get("audio_column", [])
        if len(audio_columns) != 1:
            raise ValueError(
                f"GenerativeAudioTranscriptionRequestFormatter expects exactly "
                f"one audio column, but got {len(audio_columns)}."
            )

        arguments.files = {
            "file": (
                audio_columns[0].get("file_name", "audio_input"),
                audio_columns[0].get("audio"),
                audio_columns[0].get("mimetype"),
            )
        }

        return arguments
