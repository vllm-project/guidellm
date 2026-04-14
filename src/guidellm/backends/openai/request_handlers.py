"""
Request handlers for formatting requests and processing API responses from
different OpenAI endpoints.

Provides a pluggable system for handling format differences while supporting
both streaming and non-streaming responses. Each handler implements the
GenerationRequestHandler protocol to format json requests, parse API responses,
extract usage metrics, and convert results into standardized GenerationResponse.
"""

from __future__ import annotations

import base64
from typing import Any, Protocol, cast

from more_itertools import roundrobin

from guidellm.scheduler import HistoryT
from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.utils.imports import json
from guidellm.utils.registry import RegistryMixin

__all__ = [
    "AudioRequestHandler",
    "ChatCompletionsRequestHandler",
    "OpenAIRequestHandler",
    "OpenAIRequestHandlerFactory",
    "PoolingRequestHandler",
    "ResponsesRequestHandler",
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
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
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
    Factory for registering and creating OpenAI request handlers by request type.

    Registry-based system for associating handler classes with specific API
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
        Create a request handler class for the given request type.

        :param request_type: The type of generation request (e.g., "/chat/completions")
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


@OpenAIRequestHandlerFactory.register("/v1/completions")
class TextCompletionsRequestHandler(OpenAIRequestHandler):
    """
    Request handler for OpenAI-style legacy completion endpoints.

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

    def format(  # noqa: C901
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the text completion generation request into the appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        prev_requests: list[GenerationRequestArguments] = []
        if history:
            # NOTE: Does not include history to avoid infinite recursion
            prev_requests = [
                self.format(req, response=res, **kwargs) for req, res in history
            ]

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

        ## Build prompt ##
        prompts = []

        # Include previous requests
        for req in prev_requests:
            if req.body and "prompt" in req.body:
                prompts.append(req.body["prompt"])

        # Include prefix
        prompts.extend(data.columns.get("prefix_column", []))
        # Include text column
        prompts.extend(data.columns.get("text_column", []))

        # Include the response to the current prompt
        if response and response.text:
            prompts.append(response.text)

        if prompts:
            arguments.body["prompt"] = " ".join(prompts)

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
        self, usage: dict[str, int | dict[str, int]] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from API response usage data.

        :param usage: Usage data dictionary from API response
        :param text: Generated text for calculating word and character counts.
            None means text is not applicable (metrics will be None);
            empty string means text was applicable but empty (metrics will be 0).
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if text is None:
            # text not applicable (e.g. tool-call-only) — exclude from aggregation
            text_words = None
            text_chars = None
        else:
            text_words = len(text.split())
            text_chars = len(text)

        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=text_words,
                text_characters=text_chars,
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
            text_words=text_words,
            text_characters=text_chars,
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
        )


@OpenAIRequestHandlerFactory.register("/v1/chat/completions")
class ChatCompletionsRequestHandler(TextCompletionsRequestHandler):
    """
    Request handler for OpenAI-style chat completion endpoints.

    Extends TextCompletionsResponseHandler to handle chat completion requests where
    generated text is nested within message objects in the choices array. Processes
    both streaming and non-streaming chat completion responses, including tool call
    responses where the model outputs ``tool_calls`` instead of text content.
    """

    def __init__(self):
        super().__init__()
        self.streaming_tool_call_indices: set[int] = set()

    def _format_prompts(
        self, column_data: list[dict[str, Any]], column_type: str
    ) -> list[dict[str, Any]]:
        """
        Helper method to format different types of data columns
        into the appropriate structure for chat messages.
        """
        formatted_data = []
        for item in column_data:
            if column_type == "text_column":
                formatted_data.append({"type": "text", "text": item})
            elif column_type == "image_column":
                formatted_data.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item.get("image")},
                    }
                )
            elif column_type == "video_column":
                formatted_data.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": item.get("video")},
                    }
                )
            elif column_type == "audio_column":
                formatted_data.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(item.get("audio", b"")).decode(
                                "utf-8"
                            ),
                            "format": item.get("format"),
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported column type: {column_type}")

        return formatted_data

    def format(  # noqa: C901
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the chat completion generation request into the appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        prev_requests: list[GenerationRequestArguments] = []
        if history:
            # NOTE: Does not include history to avoid infinite recursion
            prev_requests = [
                self.format(req, response=res, **kwargs) for req, res in history
            ]

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

        # Include previous requests
        for req in prev_requests:
            if req.body and "messages" in req.body:
                arguments.body["messages"].extend(req.body["messages"])

        # Build the system prompt
        prefix = " ".join(data.columns.get("prefix_column", []))
        if prefix:
            arguments.body["messages"].append({"role": "system", "content": prefix})

        # Build each prompt then combine into a single user message
        prompts = [
            self._format_prompts(data.columns.get(col, []), col)
            for col in ("text_column", "image_column", "video_column", "audio_column")
        ]
        if prompts:
            # Interleave prompt types
            arguments.body["messages"].append(
                {"role": "user", "content": list(roundrobin(*prompts))}
            )

        # Add the response to the current prompt if available
        if response and response.text:
            arguments.body["messages"].append(
                {"role": "assistant", "content": response.text}
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
        message = choice.get("message", {})
        text = message.get("content")
        raw_tool_calls = message.get("tool_calls")
        if text is None and not raw_tool_calls:
            text = ""  # Edge case: null content and no tools
        input_metrics, output_metrics = self.extract_metrics(usage, text)
        if raw_tool_calls:
            output_metrics.tool_call_count = len(raw_tool_calls)
            if text is None:  # tool-only turn
                output_metrics.tool_call_tokens = output_metrics.text_tokens
            else:  # mixed content + tool call turn
                output_metrics.mixed_content_tool_tokens = output_metrics.text_tokens

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
        within delta objects in the streaming response chunks. Also accumulates
        ``tool_calls`` deltas when the model streams function call output.

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
        delta = choice.get("delta", {}) if choices else {}

        if content := delta.get("content"):
            self.streaming_texts.append(content)
            updated = True

        # tool_calls is an optional field for when the server is requesting a tool
        for tc_delta in delta.get("tool_calls", []):
            # Keep track of the index to properly count tool usage, since a tool call
            # can be split into multiple chunks when streaming.
            self.streaming_tool_call_indices.add(tc_delta["index"])
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
        text = "".join(self.streaming_texts) or None
        has_tool_calls = bool(self.streaming_tool_call_indices)
        if text is None and not has_tool_calls:
            text = ""
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage, text)
        if has_tool_calls:
            output_metrics.tool_call_count = len(self.streaming_tool_call_indices)
            if text is None:  # tool-only turn
                output_metrics.tool_call_tokens = output_metrics.text_tokens
            else:  # mixed content + tool call turn
                output_metrics.mixed_content_tool_tokens = output_metrics.text_tokens

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=self.streaming_response_id,  # use vLLM ID if available
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )


@OpenAIRequestHandlerFactory.register(
    ["/v1/audio/transcriptions", "/v1/audio/translations"]
)
class AudioRequestHandler(ChatCompletionsRequestHandler):
    """
    Request handler for audio transcription and translation endpoints.

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
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:  # noqa: C901
        """
        Format the audio transcription generation request into the
        appropriate structure.

        :param request: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        if history or response:
            raise ValueError("AudioRequestHandler does not support multiturn.")

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
        self, usage: dict[str, int | dict[str, int]] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics from audio API response usage data.

        Handles audio-specific metrics including processing duration and audio tokens
        in addition to standard text token counts.

        :param usage: Usage data dictionary from audio API response
        :param text: Generated text for calculating word and character counts.
            None means text is not applicable (metrics will be None);
            empty string means text was applicable but empty (metrics will be 0).
        :return: Tuple of input_metrics and output_metrics as UsageMetrics objects
        """
        if text is None:
            # text not applicable (e.g. tool-call-only) — exclude from aggregation
            text_words = None
            text_chars = None
        else:
            text_words = len(text.split())
            text_chars = len(text)

        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=text_words,
                text_characters=text_chars,
            )

        usage_metrics: dict[str, int] = cast("dict[str, int]", usage)

        return UsageMetrics(
            audio_tokens=(usage_metrics.get("prompt_tokens") or 0),
        ), UsageMetrics(
            text_tokens=(usage_metrics.get("completion_tokens") or 0),
            text_words=text_words,
            text_characters=text_chars,
        )


@OpenAIRequestHandlerFactory.register("/v1/responses")
class ResponsesRequestHandler(OpenAIRequestHandler):
    """
    Request handler for the OpenAI Responses API endpoint.

    Handles the /v1/responses format which uses `input` instead of `messages`,
    `instructions` for system prompts, and a different response/streaming shape
    than chat completions. Supports both streaming and non-streaming responses.
    """

    def __init__(self):
        self.streaming_texts: list[str] = []
        self.streaming_usage: dict[str, int | dict[str, int]] | None = None
        self.streaming_response_id: str | None = None
        self.streaming_tool_call_indices: set[int] = set()

    def _format_prompts(
        self, column_data: list, column_type: str
    ) -> list[dict[str, Any]]:
        formatted_data: list[dict[str, Any]] = []
        for item in column_data:
            if column_type == "text_column":
                formatted_data.append({"type": "input_text", "text": item})
            elif column_type == "image_column":
                formatted_data.append(
                    {
                        "type": "input_image",
                        "image_url": item.get("image"),
                    }
                )
            elif column_type == "audio_column":
                formatted_data.append(
                    {
                        "type": "input_file",
                        "file_data": base64.b64encode(item.get("audio", b"")).decode(
                            "utf-8"
                        ),
                    }
                )
        return formatted_data

    def _build_input_items(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None,
        prev_requests: list[GenerationRequestArguments],
    ) -> list[dict[str, Any]]:
        """Build the ``input`` array for the Responses API.

        The Responses API uses a flat ``input`` list of role-tagged message
        dicts (with nested content parts like ``input_text``, ``input_image``)
        instead of chat completions' ``messages`` array.
        """
        input_items: list[dict[str, Any]] = []

        for req in prev_requests:
            if req.body and "input" in req.body:
                prev_input = req.body["input"]
                if isinstance(prev_input, list):
                    input_items.extend(prev_input)

        prompts = [
            self._format_prompts(data.columns.get(col, []), col)
            for col in ("text_column", "image_column", "video_column", "audio_column")
        ]
        content_parts = list(roundrobin(*prompts))
        if content_parts:
            input_items.append({"role": "user", "content": content_parts})

        if response and response.text:
            input_items.append({"role": "assistant", "content": response.text})

        return input_items

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        prev_requests: list[GenerationRequestArguments] = []
        if history:
            prev_requests = [
                self.format(req, response=res, **kwargs) for req, res in history
            ]

        arguments = GenerationRequestArguments()
        arguments.body = {}

        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        if kwargs.get("stream"):
            arguments.stream = True
            arguments.body["stream"] = True
            # Unlike chat completions, we don't send stream_options here.
            # The Responses API's stream_options only controls obfuscation,
            # not usage reporting. vLLM always includes usage data in the
            # response.completed SSE event for this endpoint.
            # Unfortunately, this complicates getting accurate stats when canceled.

        if data.output_metrics.text_tokens:
            arguments.body["max_output_tokens"] = data.output_metrics.text_tokens
            # stop/ignore_eos are vLLM-specific sampling params that force
            # the model to generate exactly N tokens, matching the behavior
            # of the chat completions handler for controlled benchmarking.
            arguments.body["stop"] = None
            arguments.body["ignore_eos"] = True
        elif kwargs.get("max_tokens") is not None:
            arguments.body["max_output_tokens"] = kwargs["max_tokens"]

        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        prefix = " ".join(data.columns.get("prefix_column", []))
        if prefix:
            arguments.body["instructions"] = prefix

        arguments.body["input"] = self._build_input_items(data, response, prev_requests)

        return arguments

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: dict,
    ) -> GenerationResponse:
        text = self._extract_output_text(response)
        tool_call_count = sum(
            1
            for item in response.get("output", [])
            if item.get("type") == "function_call"
        )
        if text is None and not tool_call_count:
            text = ""
        usage = response.get("usage", {})
        input_metrics, output_metrics = self.extract_metrics(usage, text)
        if tool_call_count:
            output_metrics.tool_call_count = tool_call_count
            if text is None:  # tool-only turn
                output_metrics.tool_call_tokens = output_metrics.text_tokens
            else:  # mixed content + tool call turn
                output_metrics.mixed_content_tool_tokens = output_metrics.text_tokens

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        text = "".join(self.streaming_texts) or None
        has_tool_calls = bool(self.streaming_tool_call_indices)
        if text is None and not has_tool_calls:
            text = ""
        input_metrics, output_metrics = self.extract_metrics(self.streaming_usage, text)
        if has_tool_calls:
            output_metrics.tool_call_count = len(self.streaming_tool_call_indices)
            if text is None:  # tool-only turn
                output_metrics.tool_call_tokens = output_metrics.text_tokens
            else:  # mixed content + tool call turn
                output_metrics.mixed_content_tool_tokens = output_metrics.text_tokens

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=self.streaming_response_id,
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def extract_line_data(self, line: str) -> dict[str, Any] | None:
        """Parse a Responses API SSE line.

        The Responses API streams paired ``event: <type>`` and ``data: <json>``
        lines, unlike chat completions which only uses ``data:`` lines.  The
        event type is redundantly embedded in the JSON payload's ``type`` field,
        so ``event:`` lines are skipped, keeping only ``data:`` lines.
        """
        line = line.strip()

        if not line or not line.startswith("data:"):
            return {}

        if line == "data: [DONE]":
            return None

        return json.loads(line[len("data:") :].strip())

    def add_streaming_line(self, line: str) -> int | None:
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        event_type = data.get("type", "")

        if "id" in data and self.streaming_response_id is None:
            resp = data.get("response", {})
            if "id" in resp:
                self.streaming_response_id = resp["id"]

        if event_type == "response.output_text.delta":
            delta = data.get("delta", "")
            if delta:
                self.streaming_texts.append(delta)
                return 1
            return 0

        if (
            event_type == "response.output_item.added"
            and data.get("item", {}).get("type") == "function_call"
        ):
            self.streaming_tool_call_indices.add(data["output_index"])
            return 1

        if event_type in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            # All three are terminal SSE events. response.completed is the
            # normal case; response.failed and response.incomplete may be sent
            # by some providers instead. Each carries a final response object
            # with optional usage data. Returning None signals the streaming
            # loop in http.py to break out of the stream.
            resp = data.get("response") or {}
            usage = resp.get("usage")
            if usage:
                self.streaming_usage = usage
            if self.streaming_response_id is None and "id" in resp:
                self.streaming_response_id = resp["id"]
            return None

        return 0

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        # Responses API uses "input_tokens"/"output_tokens" in its usage
        # payload, unlike chat completions' "prompt_tokens"/"completion_tokens".
        if text is None:
            # text not applicable — exclude from aggregation
            text_words = None
            text_chars = None
        else:
            text_words = len(text.split())
            text_chars = len(text)

        if not usage:
            return UsageMetrics(), UsageMetrics(
                text_words=text_words,
                text_characters=text_chars,
            )

        usage_metrics: dict[str, int] = cast("dict[str, int]", usage)

        return UsageMetrics(
            text_tokens=usage_metrics.get("input_tokens", 0),
        ), UsageMetrics(
            text_tokens=usage_metrics.get("output_tokens", 0),
            text_words=text_words,
            text_characters=text_chars,
        )

    @staticmethod
    def _extract_output_text(response: dict) -> str | None:
        """Extract generated text from a Responses API response object.

        :returns: ``None`` when no message/output_text items exist (e.g. tool-call-
        only responses), so callers can distinguish "no text" from "empty text".
        """
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        return "".join(texts) if texts else None


@OpenAIRequestHandlerFactory.register("/pooling")
class PoolingRequestHandler(ChatCompletionsRequestHandler):
    """
    Request handler for vLLM pooling endpoints.

    Inherits from ChatCompletionsRequestHandler and overrides format() to handle
    pooling-specific request structure with nested data fields.
    """

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,  # noqa: ARG002
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> GenerationRequestArguments:
        """
        Format the pooling generation request into the appropriate structure.

        :param data: The generation request to format
        :param response: Optional previous response (unused for pooling)
        :param history: Optional request/response history (unused for pooling)
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        arguments = GenerationRequestArguments()
        arguments.body = {}

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

        # Apply extra arguments
        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        # Build pooling request body from text_column (which contains the dict)
        pooling_data = data.columns.get("pooling_column", [])
        if pooling_data and isinstance(pooling_data[0], dict):
            # Use the dict directly from text_column
            pooling_entry = pooling_data[0]
            if "data" in pooling_entry:
                arguments.body["data"] = pooling_entry["data"]
            if "priority" in pooling_entry:
                arguments.body["priority"] = pooling_entry["priority"]

        return arguments
