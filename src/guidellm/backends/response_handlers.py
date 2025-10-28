"""
Response handlers for processing API responses from different generation backends.

Provides a pluggable system for handling responses from language model backends,
supporting both streaming and non-streaming responses. Each handler implements the
GenerationResponseHandler protocol to parse API responses, extract usage metrics,
and convert them into standardized GenerationResponse objects.
"""

from __future__ import annotations

from typing import Any, Protocol

from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.utils import RegistryMixin, json

__all__ = [
    "AudioResponseHandler",
    "ChatCompletionsResponseHandler",
    "GenerationResponseHandler",
    "GenerationResponseHandlerFactory",
    "TextCompletionsResponseHandler",
]


class GenerationResponseHandler(Protocol):
    """
    Protocol for handling generation API responses.

    Defines the interface for processing both streaming and non-streaming responses
    from backend APIs, converting them into standardized GenerationResponse objects
    with consistent metrics extraction.
    """

    def compile_non_streaming(
        self, request: GenerationRequest, response: Any
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

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        """
        Compile accumulated streaming data into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with extracted metrics
        """
        ...


class GenerationResponseHandlerFactory(RegistryMixin[type[GenerationResponseHandler]]):
    """
    Factory for registering and creating response handlers by backend type.

    Registry-based system for associating handler classes with specific backend API
    types, enabling automatic selection of the appropriate handler for processing
    responses from different generation services.
    """


@GenerationResponseHandlerFactory.register("text_completions")
class TextCompletionsResponseHandler(GenerationResponseHandler):
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

    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
    ) -> GenerationResponse:
        """
        Process a complete text completion response.

        :param request: Original generation request
        :param response: Complete API response containing choices and usage data
        :return: Standardized GenerationResponse with extracted text and metrics
        """
        choices, usage = self.extract_choices_and_usage(response)
        input_metrics, output_metrics = self.extract_metrics(usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=choices[0].get("text", "") if choices else "",
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

        updated = False
        choices, usage = self.extract_choices_and_usage(data)

        if text := choices[0].get("text"):
            self.streaming_texts.append(text)
            updated = True

        if usage:
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        """
        Compile accumulated streaming text chunks into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated text and metrics
        """
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
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
        self, usage: dict[str, int | dict[str, int]] | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from API response usage data.

        :param usage: Usage data dictionary from API response
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if not usage:
            return UsageMetrics(), UsageMetrics()

        input_details: dict[str, int] = usage.get("prompt_tokens_details", {}) or {}
        output_details: dict[str, int] = (
            usage.get("completion_tokens_details", {}) or {}
        )

        return UsageMetrics(
            text_tokens=(
                input_details.get("prompt_tokens") or usage.get("prompt_tokens")
            ),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        ), UsageMetrics(
            text_tokens=(
                output_details.get("completion_tokens")
                or usage.get("completion_tokens")
            ),
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )


@GenerationResponseHandlerFactory.register("chat_completions")
class ChatCompletionsResponseHandler(TextCompletionsResponseHandler):
    """
    Response handler for OpenAI-style chat completion endpoints.

    Extends TextCompletionsResponseHandler to handle chat completion responses where
    generated text is nested within message objects in the choices array. Processes
    both streaming and non-streaming chat completion responses.
    """

    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
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
        input_metrics, output_metrics = self.extract_metrics(usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=(choices[0].get("message", {}).get("content", "") if choices else ""),
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

        updated = False
        choices, usage = self.extract_choices_and_usage(data)

        if choices and (content := choices[0].get("delta", {}).get("content")):
            self.streaming_texts.append(content)
            updated = True

        if usage:
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        """
        Compile accumulated streaming chat completion content into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated content and metrics
        """
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@GenerationResponseHandlerFactory.register(
    ["audio_transcriptions", "audio_translations"]
)
class AudioResponseHandler:
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

    def compile_non_streaming(
        self, request: GenerationRequest, response: dict
    ) -> GenerationResponse:
        """
        Process a complete audio transcription or translation response.

        Extracts transcribed or translated text and audio-specific usage metrics
        including processing duration and token counts for audio content.

        :param request: Original generation request
        :param response: Complete API response containing text and usage data
        :return: Standardized GenerationResponse with extracted text and metrics
        """
        usage: dict[str, int | dict[str, int]] = response.get("usage", {})
        input_details: dict[str, int] = usage.get("input_token_details", {}) or {}
        output_details: dict[str, int] = usage.get("output_token_details", {}) or {}
        text: str = response.get("text", "")

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text=text,
            input_metrics=UsageMetrics(
                text_tokens=input_details.get("text_tokens", usage.get("input_tokens")),
                audio_tokens=input_details.get(
                    "audio_tokens", usage.get("input_tokens")
                ),
                audio_seconds=input_details.get("seconds", usage.get("seconds")),
            ),
            output_metrics=UsageMetrics(
                text_tokens=output_details.get(
                    "text_tokens", usage.get("output_tokens")
                ),
            ),
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from an audio streaming response.

        Handles JSON-formatted streaming responses from audio processing endpoints,
        extracting text content and usage metrics as they become available.

        :param line: Raw JSON line from the streaming response
        :return: 1 if text content was extracted, 0 if line ignored, None if done
        """
        if line == "data: [DONE]":
            return None

        if not line or not (line := line.strip()) or not line.startswith("{"):
            return 0

        data: dict[str, Any] = json.loads(line)
        text: str
        usage: dict[str, int | dict[str, int]]
        updated = False

        if text := data.get("text"):
            self.streaming_texts.append(text)
            updated = True

        if usage := data.get("usage"):
            self.streaming_usage = usage

        return 1 if updated else 0

    def compile_streaming(self, request: GenerationRequest) -> GenerationResponse:
        """
        Compile accumulated streaming audio text into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated text and metrics
        """
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            text="".join(self.streaming_texts),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from audio API response usage data.

        Handles audio-specific metrics including processing duration and audio tokens
        in addition to standard text token counts.

        :param usage: Usage data dictionary from audio API response
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if not usage:
            return UsageMetrics(), UsageMetrics()

        input_details: dict[str, int] = usage.get("input_token_details", {}) or {}
        output_details: dict[str, int] = usage.get("output_token_details", {}) or {}

        return UsageMetrics(
            text_tokens=(input_details.get("text_tokens") or usage.get("input_tokens")),
            audio_tokens=(
                input_details.get("audio_tokens") or usage.get("audio_tokens")
            ),
            audio_seconds=(input_details.get("seconds") or usage.get("seconds")),
        ), UsageMetrics(
            text_tokens=output_details.get("text_tokens") or usage.get("output_tokens"),
        )
