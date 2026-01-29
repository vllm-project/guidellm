"""
Response handlers for processing API responses from different generation backends.

Provides a pluggable system for handling responses from language model backends,
supporting both streaming and non-streaming responses. Each handler implements the
GenerationResponseHandler protocol to parse API responses, extract usage metrics,
and convert them into standardized GenerationResponse objects.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.utils import RegistryMixin, json

__all__ = [
    "AudioRequestHandler",
    "ChatCompletionsRequestHandler",
    "OpenAIRequestHandler",
    "OpenAIRequestHandlerFactory",
    "TextCompletionsRequestHandler",
]


class OpenAIRequestHandler(Protocol):
    """
    Protocol for handling OpenAI request endpoint

    Defines the interface to format the request for a given endpoint and to
    process both streaming and non-streaming responses from backend APIs,
    converting them into standardized GenerationResponse objects
    with consistent metrics extraction.
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
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        ...

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: Any,
    ) -> GenerationResponse:
        """
        Process a complete non-streaming API response.

        :param request: Original generation request
        :param response: Raw API response data from the backend
        :return: Standardized GenerationResponse with extracted metrics
        """
        ...

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from a streaming response.

        :param line: Raw line from the streaming response
        :return: 1 if content was updated, 0 if line was ignored, None if done
        """
        ...

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile accumulated streaming data into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with extracted metrics
        """
        ...


class OpenAIRequestHandlerFactory(RegistryMixin[type[OpenAIRequestHandler]]):
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
        handler_overrides: dict[str, type[OpenAIRequestHandler]] | None = None,
    ) -> OpenAIRequestHandler:
        """
        Create a response handler class for the given request type.

        :param request_type: The type of generation request (e.g., "text_completions")
        :param handler_overrides: Optional mapping of request types to handler classes
            to override the default registry by checking first and then falling back
            to the registered handlers.
        :return: The corresponding instantiated GenerationResponseHandler
        :raises ValueError: When no handler is registered for the request type
        """
        if handler_overrides and request_type in handler_overrides:
            return handler_overrides[request_type]()

        handler_cls = cls.get_registered_object(request_type)
        if not handler_cls:
            raise ValueError(
                f"No response handler registered for type '{request_type}'."
            )

        return handler_cls()


@OpenAIRequestHandlerFactory.register("/completions")
class TextCompletionsRequestHandler(OpenAIRequestHandler):
    """
    Response handler for OpenAI-style text completion endpoints.

    Processes responses from text completion APIs that return generated text in the
    'choices' array with 'text' fields. Handles both streaming and non-streaming
    responses, extracting usage metrics for input and output tokens.

    Example:
    ::
        handler = TextCompletionsResponseHandler()
        response = handler.compile_non_streaming(request, api_response)
    """

    def __init__(self):
        """
        Initialize the text completions response handler.

        Sets up internal state for accumulating streaming response data including
        text chunks and usage metrics.
        """
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, int | dict[str, int]] | None = None
        self.streaming_response_id: str | None = None

    def format(
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the text completion generation request into the appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
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

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: dict,
    ) -> GenerationResponse:
        """
        Process a complete text completion response.

        :param request: Original generation request
        :param response: Complete API response containing choices and usage data
        :return: Standardized GenerationResponse with extracted text and metrics
        """
        choices, usage = self.extract_choices_and_usage(response)
        choice = choices[0] if choices else {}
        text = choice.get("text", "")
        input_metrics, output_metrics = self.extract_metrics(usage, text)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),  # use vLLM ID if available
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from a text completion streaming response.

        Parses Server-Sent Events (SSE) formatted lines, extracting text content
        and usage metrics. Accumulates text chunks for final response compilation.

        :param line: Raw SSE line from the streaming response
        :return: 1 if text content was extracted, 0 if line ignored, None if done
        """
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        if "id" in data and self.streaming_response_id is None:
            self.streaming_response_id = data["id"]

        updated = False
        choices, usage = self.extract_choices_and_usage(data)
        choice = choices[0] if choices else {}

        if choices and (text := choice.get("text")):
            self.streaming_texts.append(text)
            updated = True

        if usage:
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile accumulated streaming text chunks into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated text and metrics
        """
        text = "".join(self.streaming_texts)
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage, text)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=self.streaming_response_id,  # use vLLM ID if available
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def extract_line_data(self, line: str) -> dict[str, Any] | None:
        """
        Extract JSON data from a streaming response line.

        :param line: Raw line from the streaming response
        :return: Parsed JSON data as dictionary, or None if line indicates completion
        """
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return {}

        line = line[len("data:") :].strip()

        return json.loads(line)

    def extract_choices_and_usage(
        self, response: dict
    ) -> tuple[list[dict], dict[str, int | dict[str, int]]]:
        """
        Extract choices and usage data from the API response.

        :param response: Complete API response containing choices and usage data
        :return: Tuple of choices list and usage dictionary
        """
        return response.get("choices", []), response.get("usage", {})

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None, text: str
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from API response usage data.

        :param usage: Usage data dictionary from API response
        :param text: Generated text for calculating word and character counts
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=len(text.split()) if text else 0,
                text_characters=len(text) if text else 0,
            )

        input_details: dict[str, int] = cast(
            "dict[str, int]", usage.get("prompt_tokens_details", {}) or {}
        )
        output_details: dict[str, int] = cast(
            "dict[str, int]", usage.get("completion_tokens_details", {}) or {}
        )
        usage_metrics: dict[str, int] = cast("dict[str, int]", usage)

        return UsageMetrics(
            text_tokens=(
                input_details.get("prompt_tokens")
                or usage_metrics.get("prompt_tokens")
                or 0
            ),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        ), UsageMetrics(
            text_tokens=(
                output_details.get("completion_tokens")
                or usage_metrics.get("completion_tokens")
                or 0
            ),
            text_words=len(text.split()) if text else 0,
            text_characters=len(text) if text else 0,
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )


@OpenAIRequestHandlerFactory.register("/chat/completions")
class ChatCompletionsRequestHandler(TextCompletionsRequestHandler):
    """
    Response handler for OpenAI-style chat completion endpoints.

    Extends TextCompletionsResponseHandler to handle chat completion responses where
    generated text is nested within message objects in the choices array. Processes
    both streaming and non-streaming chat completion responses.
    """

    def format(  # noqa: C901, PLR0912, PLR0915
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the chat completion generation request into the appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
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

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: dict,
    ) -> GenerationResponse:
        """
        Process a complete chat completion response.

        Extracts content from the message object within choices, handling the nested
        structure specific to chat completion endpoints.

        :param request: Original generation request
        :param response: Complete API response containing choices and usage data
        :return: Standardized GenerationResponse with extracted content and metrics
        """
        choices, usage = self.extract_choices_and_usage(response)
        choice: dict[str, dict] = choices[0] if choices else {}
        text = choice.get("message", {}).get("content", "")
        input_metrics, output_metrics = self.extract_metrics(usage, text)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),  # use vLLM ID if available
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from a chat completion streaming response.

        Handles the chat completion specific delta structure where content is nested
        within delta objects in the streaming response chunks.

        :param line: Raw SSE line from the streaming response
        :return: 1 if content was extracted, 0 if line ignored, None if done
        """
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        if "id" in data and self.streaming_response_id is None:
            self.streaming_response_id = data["id"]

        updated = False
        choices, usage = self.extract_choices_and_usage(data)
        choice: dict[str, dict] = choices[0] if choices else {}

        if choices and (content := choice.get("delta", {}).get("content")):
            self.streaming_texts.append(content)
            updated = True

        if usage:
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile accumulated streaming chat completion content into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated content and metrics
        """
        text = "".join(self.streaming_texts)
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage, text)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=self.streaming_response_id,  # use vLLM ID if available
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@OpenAIRequestHandlerFactory.register(["/audio/transcriptions", "/audio/translations"])
class AudioRequestHandler(ChatCompletionsRequestHandler):
    """
    Response handler for audio transcription and translation endpoints.

    Processes responses from audio processing APIs that convert speech to text,
    handling both transcription and translation services. Manages audio-specific
    usage metrics including audio tokens and processing duration.

    Example:
    ::
        handler = AudioResponseHandler()
        response = handler.compile_non_streaming(request, api_response)
    """

    def __init__(self):
        """
        Initialize the audio response handler.

        Sets up internal state for accumulating streaming response data including
        audio buffers, text chunks, and usage metrics.
        """
        self.streaming_buffer: bytearray = bytearray()
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, int | dict[str, int]] | None = None
        self.streaming_response_id: str | None = None

    def format(
        self,
        data: GenerationRequest,
        **kwargs,
    ) -> GenerationRequestArguments:  # noqa: C901
        """
        Format the audio transcription generation request into the
        appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
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

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None, text: str
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from audio API response usage data.

        Handles audio-specific metrics including processing duration and audio tokens
        in addition to standard text token counts.

        :param usage: Usage data dictionary from audio API response
        :param text: Generated text for calculating word and character counts
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=len(text.split()) if text else 0,
                text_characters=len(text) if text else 0,
            )

        usage_metrics: dict[str, int] = cast("dict[str, int]", usage)

        return UsageMetrics(
            audio_tokens=(usage_metrics.get("prompt_tokens") or 0),
        ), UsageMetrics(
            text_tokens=(usage_metrics.get("completion_tokens") or 0),
            text_words=len(text.split()) if text else 0,
            text_characters=len(text) if text else 0,
        )
