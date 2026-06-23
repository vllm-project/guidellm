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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, cast

from more_itertools import roundrobin

from guidellm.backends.openai.common import format_ws_error
from guidellm.scheduler import HistoryT
from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.schemas.tool_call import ToolCall, ToolCallFunction
from guidellm.settings import settings
from guidellm.utils.audio import pcm16_append_b64_chunks
from guidellm.utils.imports import json
from guidellm.utils.registry import RegistryMixin

WS_AUDIO_CHUNKS_BODY_KEY = "audio_chunks"

__all__ = [
    "WS_AUDIO_CHUNKS_BODY_KEY",
    "AudioRequestHandler",
    "ChatCompletionsRequestHandler",
    "EmbeddingsRequestHandler",
    "OpenAIRequestHandler",
    "OpenAIRequestHandlerFactory",
    "OpenAIWSRequestHandler",
    "OpenAIWSRequestHandlerFactory",
    "PoolingRequestHandler",
    "RealtimeTranscriptionWSRequestHandler",
    "ResponsesRequestHandler",
    "TextCompletionsRequestHandler",
    "ToolCall",
    "ToolCallFunction",
    "WSEventResult",
    "WSStreamingEventResult",
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

    @property
    def last_iteration_had_content(self) -> bool:
        """
        Whether the last chunk carried output (text/tool-call) tokens,
        not solely reasoning tokens.

        Used by the HTTP streaming loop to detect the first output token
        for TTFOT measurement.

        :return: True if the last chunk carried output tokens, not solely
            reasoning tokens.
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


def _check_streaming_error(data: Any) -> None:
    """Raise when a streaming SSE payload conveys a server-side error.

    Some OpenAI-compatible servers (e.g. vLLM's
    ``create_streaming_error_response``) return HTTP 200 and report failures
    through the stream body as ``data: {"error": {...}}`` followed by
    ``data: [DONE]``. Without this check the error payload is ignored, the
    request is recorded as a successful but empty generation, and timing
    metrics (TTFT, ITL) are reported as zero. Raising ``ValueError`` follows
    the same convention as other request-level failures so the worker marks
    the request as errored.

    :param data: Parsed JSON payload from a single streaming line.
    :raises ValueError: If the payload contains a non-empty ``error`` field.
    """
    if not isinstance(data, dict):
        return
    error = data.get("error")
    if not error:
        return
    if isinstance(error, dict):
        message = (
            error.get("message") or error.get("type") or error.get("code") or str(error)
        )
    else:
        message = str(error)
    raise ValueError(f"Streaming response returned an error: {message}")


class WSEventResult(Enum):
    """Classification of a processed WebSocket streaming event."""

    STREAM_END = auto()
    CONTENT = auto()
    REQUEST_ITERATION = auto()
    IGNORED = auto()


@dataclass(frozen=True)
class WSStreamingEventResult:
    """
    Result of processing one WebSocket JSON event frame.

    :param kind: How the backend should update timings and loop control.
    :param content_tokens: New content tokens when ``kind`` is ``CONTENT``.
    """

    kind: WSEventResult
    content_tokens: int = 0


class OpenAIWSRequestHandler(Protocol):
    """
    Protocol for WebSocket-based streaming request handlers.

    Defines the interface for handlers that interpret JSON event frames from a
    WebSocket connection, accumulate streaming state, and compile a final response.
    Mirrors the HTTP handler lifecycle (format -> stream -> compile) but uses
    structured event dicts instead of SSE text lines.
    """

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format and validate the generation request for the WebSocket endpoint.

        :param data: The generation request to format
        :param response: Optional previous response for multi-turn
        :param history: Optional conversation history
        :param kwargs: Additional keyword arguments (model, websocket_path, etc.)
        :return: The formatted request arguments with metadata
        """
        ...

    def add_streaming_event(self, event: dict[str, Any]) -> WSStreamingEventResult:
        """
        Process one JSON event frame from the WebSocket.

        :param event: Parsed JSON dict from a WebSocket text frame
        :return: Classified result for timing updates and loop control
        :raises RuntimeError: On server error events
        """
        ...

    @property
    def streaming_text(self) -> str:
        """Accumulated transcription text from processed events."""
        ...

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Assemble accumulated streaming state into a final GenerationResponse.

        Called after the event loop completes (normal or cancelled).

        :param request: Original generation request
        :param arguments: Request arguments from format()
        :return: Standardized GenerationResponse with extracted metrics
        """
        ...


class OpenAIWSRequestHandlerFactory(RegistryMixin["type[OpenAIWSRequestHandler]"]):
    """Factory for registering and creating WebSocket request handlers by path."""

    @classmethod
    def create(cls, request_type: str) -> OpenAIWSRequestHandler:
        """
        Create a WebSocket request handler for the given request path.

        :param request_type: The WebSocket path (e.g., "/v1/realtime")
        :return: Instantiated handler implementing OpenAIWSRequestHandler
        :raises ValueError: When no handler is registered for the path
        """
        handler_cls = cls.get_registered_object(request_type)
        if not handler_cls:
            raise ValueError(
                f"No WebSocket handler registered for type '{request_type}'."
            )
        return handler_cls()


def _apply_tool_call_metrics(
    output_metrics: UsageMetrics,
    tool_call_count: int,
    text: str | None,
) -> None:
    """Set tool-call-related fields on *output_metrics* in place.

    Shared by both chat completions and responses handlers so the logic for
    deciding between ``tool_call_tokens`` (tool-only turn) and
    ``mixed_content_tool_tokens`` (mixed turn) lives in one place.

    :param output_metrics: The mutable output metrics to update.
    :param tool_call_count: Number of tool calls in the response.
    :param text: The generated text, or ``None`` for tool-only turns.
    """
    if not tool_call_count:
        return
    output_metrics.tool_call_count = tool_call_count
    if text is None:  # tool-only turn
        output_metrics.tool_call_tokens = output_metrics.text_tokens
    else:  # mixed content + tool call turn
        output_metrics.mixed_content_tool_tokens = output_metrics.text_tokens


_DEFAULT_REASONING_TEMPLATE = "<think>{reasoning}</think>"


def _wrap_reasoning(reasoning_text: str | None, mode: bool | str) -> str | None:
    """Apply the configured reasoning format to reasoning text.

    :param reasoning_text: Raw reasoning text from the model response.
    :param mode: Wrapping mode — False disables, True uses the default
        ``<think>...</think>`` template, a string is used as a format
        template containing ``{reasoning}``.
    :return: Formatted reasoning string, or None when disabled or no text.
    """
    if not mode or not reasoning_text:
        return None
    template = _DEFAULT_REASONING_TEMPLATE if mode is True else mode
    return template.format(reasoning=reasoning_text)


def _compile_streaming_response(
    request: GenerationRequest,
    arguments: GenerationRequestArguments,
    streaming_texts: list[str],
    streaming_tool_calls: dict[int, ToolCall],
    streaming_usage: dict[str, int | dict[str, int]] | None,
    streaming_response_id: str | None,
    extract_metrics: Callable[
        [dict[str, int | dict[str, int]] | None, str | None],
        tuple[UsageMetrics, UsageMetrics],
    ],
    streaming_reasoning_texts: list[str] | None = None,
) -> GenerationResponse:
    """Compile accumulated streaming state into a final response.

    Shared by both chat completions and responses handlers so the logic for
    assembling text, tool calls, and metrics from streaming state lives in one
    place.

    :param request: Original generation request.
    :param arguments: The request arguments that were sent.
    :param streaming_texts: Text chunks accumulated during streaming.
    :param streaming_tool_calls: Tool calls keyed by stream index.
    :param streaming_usage: Usage dict from the final streaming event.
    :param streaming_response_id: Server-assigned response ID, if any.
    :param extract_metrics: Handler-specific metric extraction callable.
    :param streaming_reasoning_texts: Reasoning text chunks, if any.
    :return: Standardized GenerationResponse with extracted metrics.
    """
    text = "".join(streaming_texts) or None
    reasoning_text = (
        "".join(streaming_reasoning_texts) if streaming_reasoning_texts else None
    ) or None
    tool_calls: list[ToolCall] | None = (
        [streaming_tool_calls[i] for i in sorted(streaming_tool_calls)]
        if streaming_tool_calls
        else None
    )
    if text is None and not tool_calls:
        text = ""
    input_metrics, output_metrics = extract_metrics(streaming_usage, text)
    _apply_tool_call_metrics(output_metrics, len(tool_calls) if tool_calls else 0, text)

    return GenerationResponse(
        request_id=request.request_id,
        request_args=arguments.model_dump_json(),
        response_id=streaming_response_id,
        text=text,
        reasoning_text=reasoning_text,
        tool_calls=tool_calls,
        input_metrics=input_metrics,
        output_metrics=output_metrics,
    )


_FUNCTION_DETAIL_KEYS = ("name", "description", "parameters", "strict")


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

    @property
    def last_iteration_had_content(self) -> bool:
        """
        Text completions (``/v1/completions``) have no reasoning concept, so
        every token is content. ChatCompletionsRequestHandler and
        ResponsesRequestHandler override this with a tracked flag that starts
        ``False`` and only becomes ``True`` when a content or tool call delta
        arrives.

        :return: Always True for the text completions base class
        """
        return True

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

        data = json.loads(line)
        _check_streaming_error(data)
        return data

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
        # Full tool call payloads accumulated across streaming deltas,
        # keyed by the delta ``index`` field.  Needed for multi-turn tool
        # calling so the response carries the id/name/arguments of each call.
        self.streaming_tool_calls: dict[int, ToolCall] = {}
        self.streaming_reasoning_texts: list[str] = []
        self._last_iteration_had_content: bool = False

    @property
    def last_iteration_had_content(self) -> bool:
        """
        :return: True if the last chunk carried output (text/tool-call) tokens,
            not solely reasoning tokens.
        """
        return self._last_iteration_had_content

    @staticmethod
    def _ensure_tool_format(tool: dict[str, Any]) -> dict[str, Any]:
        """Normalise a single tool definition to Chat Completions format.

        Chat Completions expects
        ``{"type": "function", "function": {"name": ..., ...}}``.
        If the tool is already in that format it is returned as-is.  If it is in the
        flat Responses API format (top-level ``name``, no ``function`` key) the detail
        fields are wrapped into a nested ``function`` dict.

        :param tool: A single tool definition dict in either format.
        :return: The tool in Chat Completions format.
        """
        if "name" in tool and "function" not in tool:
            fn = {k: tool[k] for k in _FUNCTION_DETAIL_KEYS if k in tool}
            return {"type": tool.get("type", "function"), "function": fn}
        return tool

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

    @staticmethod
    def _build_tool_response_messages(
        tool_calls: list[ToolCall],
        tool_response_columns: list[Any],
    ) -> list[dict[str, Any]]:
        """Build synthetic ``role: "tool"`` messages for each tool call.

        Uses per-request tool response content from ``tool_response_columns``
        when available, falling back to
        :attr:`settings.default_synthetic_tool_response`.

        :param tool_calls: The tool call objects from the prior assistant response.
        :param tool_response_columns: Per-tool-call response content from the
            dataset, which may be ``str`` or ``bytes`` (orjson).
        :return: List of tool-role message dicts ready to append to messages.
        """
        messages: list[dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls):
            raw_content = (
                tool_response_columns[idx]
                if idx < len(tool_response_columns)
                else settings.default_synthetic_tool_response
            )
            # orjson.dumps returns bytes; ensure content is a string.
            content = (
                raw_content.decode("utf-8")
                if isinstance(raw_content, bytes)
                else raw_content
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )
        return messages

    @staticmethod
    def _apply_tool_call_overrides(
        body: dict[str, Any],
        data: GenerationRequest,
    ) -> None:
        """Inject tool definitions and constrain the request body for tool calling.

        Handles three concerns:

        1. Deserializes and injects tool definitions from dataset columns.
        2. Sets ``tool_choice`` to ``"required"`` or ``"none"`` depending on
           whether the current turn expects a tool call.
        3. Removes body keys that are incompatible with tool calling
           (``ignore_eos``, ``stop``, and token-limit keys on tool-call turns).

        :param body: The mutable request body dict being built.
        :param data: The current generation request.
        """
        tools_column = data.columns.get("tools_column", [])
        if tools_column:
            tools_value = tools_column[0]
            # JSON-serialized tool definitions (e.g. from synthetic data
            # generators that store tools as strings for HuggingFace
            # Features compatibility).  orjson produces bytes; stdlib
            # json produces str.
            if isinstance(tools_value, str | bytes):
                tools_value = json.loads(tools_value)
            if isinstance(tools_value, list):
                body["tools"] = [
                    ChatCompletionsRequestHandler._ensure_tool_format(t)
                    for t in tools_value
                ]
                body.setdefault("tool_choice", "required")

        if "tools" not in body:
            body.pop("tool_choice", None)
            return

        # Override tool_choice to "none" on turns that don't expect tool calls,
        # so the model produces a plain text response instead.
        if not data.expects_tool_call:
            body["tool_choice"] = "none"

        # Tool calling requires the model to stop naturally after producing
        # valid JSON; ignore_eos would force generation past that point and
        # break the server's constrained decoding grammar.
        # max_completion_tokens would truncate output mid-JSON and corrupt
        # the arguments sent in conversation history on follow-up turns.
        if data.expects_tool_call:
            body.pop("ignore_eos", None)
            body.pop("stop", None)
            body.pop("max_completion_tokens", None)
            body.pop("max_tokens", None)

    def format(  # noqa: C901, PLR0912, PLR0915
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the chat completion generation request into the appropriate structure.

        :param data: The generation request to format
        :param response: Optional prior response for multi-turn history
        :param history: Prior (request, response) pairs in the conversation
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
        user_content = list(roundrobin(*prompts))
        if user_content:
            arguments.body["messages"].append({"role": "user", "content": user_content})

        # Append the prior assistant response to the message history.
        # For tool call responses, include the assistant's tool_calls and
        # synthetic tool result messages so the model sees the full
        # multi-turn exchange.  For plain text responses, just add content.
        # When multiturn_reasoning is enabled, prepend wrapped reasoning
        # text to the assistant content so the model sees its own CoT.
        multiturn_reasoning = kwargs.get("multiturn_reasoning", False)
        wrapped_reasoning = _wrap_reasoning(
            response.reasoning_text if response else None, multiturn_reasoning
        )
        if response and response.tool_calls:
            assistant_content = response.text
            if wrapped_reasoning:
                assistant_content = wrapped_reasoning + (assistant_content or "")
            arguments.body["messages"].append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": [tc.model_dump() for tc in response.tool_calls],
                }
            )
            tool_response_columns = data.columns.get("tool_response_column", [])
            arguments.body["messages"].extend(
                self._build_tool_response_messages(
                    response.tool_calls, tool_response_columns
                )
            )
        elif response and (response.text or wrapped_reasoning):
            content = response.text or ""
            if wrapped_reasoning:
                content = wrapped_reasoning + content
            arguments.body["messages"].append({"role": "assistant", "content": content})

        # Inject tool definitions and apply tool-call-specific overrides.
        self._apply_tool_call_overrides(arguments.body, data)

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
        :param arguments: The request arguments that were sent
        :param response: Complete API response containing choices and usage data
        :return: Standardized GenerationResponse with extracted content and metrics
        """
        choices, usage = self.extract_choices_and_usage(response)
        choice: dict[str, dict] = choices[0] if choices else {}
        message = choice.get("message", {})
        text = message.get("content")
        reasoning_text = (
            message.get("reasoning") or message.get("reasoning_content") or None
        )
        raw_tool_calls = message.get("tool_calls")
        if text is None and not raw_tool_calls:
            text = ""  # Edge case: null content and no tools
        input_metrics, output_metrics = self.extract_metrics(usage, text)

        tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            tool_calls = [ToolCall.model_validate(tc) for tc in raw_tool_calls]
        _apply_tool_call_metrics(
            output_metrics, len(tool_calls) if tool_calls else 0, text
        )

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),  # use vLLM ID if available
            text=text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
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
        # Tracks whether this iteration produced user-visible content (not
        # reasoning-only). Used by the HTTP layer to set TTFOT timing.
        had_content = False
        choices, usage = self.extract_choices_and_usage(data)
        choice: dict[str, dict] = choices[0] if choices else {}
        delta = choice.get("delta", {}) if choices else {}

        # Reasoning tokens trigger TTFT (updated=True) but are not
        # considered "content" for the TTFOT metric.
        if reasoning := (delta.get("reasoning") or delta.get("reasoning_content")):
            self.streaming_reasoning_texts.append(reasoning)
            updated = True
        if content := delta.get("content"):
            self.streaming_texts.append(content)
            updated = True
            had_content = True

        # Accumulate streamed tool_calls deltas.  Each tool call may be split
        # across multiple chunks; we reassemble by ``index``.
        # ``tool_calls`` could be either missing or ``null`` in the delta
        # (some OpenAI-compatible servers emit this), handle both cases
        for tc_delta in delta.get("tool_calls") or []:
            self._accumulate_tool_call_delta(tc_delta)
            updated = True
            had_content = True

        if usage:
            self.streaming_usage = usage

        # Only update the flag when we processed a token-bearing iteration;
        # non-updating lines should not reset the flag.
        if updated:
            self._last_iteration_had_content = had_content
        return 1 if updated else 0

    def _accumulate_tool_call_delta(self, tc_delta: dict[str, Any]) -> None:
        """Merge a single streaming tool_call delta into accumulated state.

        Each tool call is split across multiple SSE chunks.  This method
        creates or updates the :class:`ToolCall` entry keyed by the
        delta's ``index`` field.

        :param tc_delta: A single element from the ``tool_calls`` array in a
            streaming chat completion delta.
        """
        idx = tc_delta["index"]

        if idx not in self.streaming_tool_calls:
            self.streaming_tool_calls[idx] = ToolCall(
                id=tc_delta.get("id", ""),
                type=tc_delta.get("type", "function"),
            )

        tc = self.streaming_tool_calls[idx]
        fn_delta = tc_delta.get("function", {})

        if fn_id := tc_delta.get("id"):
            tc.id = fn_id
        if fn_name := fn_delta.get("name"):
            tc.function.name += fn_name
        if fn_args := fn_delta.get("arguments"):
            tc.function.arguments += fn_args

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile accumulated streaming chat completion content into a final response.

        :param request: Original generation request
        :return: Standardized GenerationResponse with concatenated content and metrics
        """
        return _compile_streaming_response(
            request,
            arguments,
            self.streaming_texts,
            self.streaming_tool_calls,
            self.streaming_usage,
            self.streaming_response_id,
            self.extract_metrics,
            streaming_reasoning_texts=self.streaming_reasoning_texts,
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
        super().__init__()
        self.streaming_buffer: bytearray = bytearray()

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


@OpenAIWSRequestHandlerFactory.register("/v1/realtime")
class RealtimeTranscriptionWSRequestHandler(OpenAIWSRequestHandler):
    """
    WebSocket handler for realtime audio transcription (``/v1/realtime``).

    Concrete :class:`OpenAIWSRequestHandler` for vLLM realtime transcription:
    validates audio input in ``format()``, interprets transcription events in
    ``add_streaming_event()``, and assembles the final ``GenerationResponse``
    with audio metrics in ``compile_streaming()``.
    """

    def __init__(self) -> None:
        """
        Initialize streaming state for one realtime transcription request.

        Sets up audio metrics extraction and accumulators for transcription text
        and usage data from WebSocket event frames.
        """
        self._audio_metrics = AudioRequestHandler()
        self._streaming_texts: list[str] = []
        self._streaming_usage: dict[str, int | dict[str, int]] | None = None

    @property
    def streaming_text(self) -> str:
        """Accumulated transcription text from processed events."""
        return "".join(self._streaming_texts)

    @staticmethod
    def extract_single_audio(data: GenerationRequest) -> dict[str, Any]:
        """Return the single ``audio_column`` entry required for realtime streaming."""
        audio_columns = data.columns.get("audio_column", [])
        if len(audio_columns) != 1:
            raise ValueError(
                "Realtime WebSocket transcription expects exactly one audio_column "
                f"entry; got {len(audio_columns)}."
            )
        return audio_columns[0]

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs: Any,
    ) -> GenerationRequestArguments:
        """
        Validate the request and build metadata for ``request_args``.

        Validates the audio column once, PCM-encodes it into base64 chunks, and
        attaches those chunks to ``body`` for the WebSocket backend to send.

        :param data: Must contain exactly one ``audio_column`` entry.
        :param response: Not supported (raises if provided).
        :param history: Not supported (raises if provided).
        :param kwargs: Must include ``model`` and ``websocket_path``.
        :return: Arguments with wire-protocol metadata and ``audio_chunks`` in ``body``.
        """
        if history or response:
            raise ValueError(
                "RealtimeTranscriptionWSRequestHandler does not support multiturn."
            )
        audio_entry = self.extract_single_audio(data)
        model = kwargs.get("model")
        if model is None:
            raise ValueError("model is required for realtime WebSocket format()")
        websocket_path = kwargs.get("websocket_path")
        if websocket_path is None:
            raise ValueError(
                "websocket_path is required for realtime WebSocket format()"
            )
        chunk_samples = kwargs.get("chunk_samples", 3200)
        audio_chunks = pcm16_append_b64_chunks(
            audio_entry,
            chunk_samples=chunk_samples,
        )
        arguments = GenerationRequestArguments()
        arguments.body = {
            "model": model,
            "websocket_path": websocket_path,
            "chunk_samples": chunk_samples,
            WS_AUDIO_CHUNKS_BODY_KEY: audio_chunks,
        }
        return arguments

    def add_streaming_event(self, event: dict[str, Any]) -> WSStreamingEventResult:
        """
        Process one JSON event from the vLLM realtime WebSocket.

        :param event: Parsed JSON dict from a WebSocket text frame.
        :return: Classified streaming update for the generic WS backend loop.
        :raises RuntimeError: On ``error`` type events.
        """
        event_type = event.get("type")
        if event_type == "transcription.delta":
            delta = event.get("delta") or ""
            self._streaming_texts.append(delta)
            if delta:
                return WSStreamingEventResult(
                    kind=WSEventResult.CONTENT, content_tokens=1
                )
            return WSStreamingEventResult(kind=WSEventResult.REQUEST_ITERATION)
        if event_type == "transcription.done":
            self._streaming_usage = event.get("usage")
            final_text = event.get("text")
            # Server may send only ``text`` on done, replacing accumulated deltas.
            if final_text:
                self._streaming_texts = [final_text]
            return WSStreamingEventResult(kind=WSEventResult.STREAM_END)
        if event_type == "error":
            err = event.get("error")
            raise RuntimeError(format_ws_error(err))
        return WSStreamingEventResult(kind=WSEventResult.IGNORED)

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Assemble accumulated transcription text and usage into a response.

        :param request: Original generation request.
        :param arguments: Request arguments from format().
        :return: Final GenerationResponse with audio metrics.
        """
        full_text = self.streaming_text
        inp, outp = self._audio_metrics.extract_metrics(
            self._streaming_usage, full_text
        )
        body = dict(arguments.body or {})
        body.pop(WS_AUDIO_CHUNKS_BODY_KEY, None)
        request_args = arguments.model_copy(update={"body": body or None})
        return GenerationResponse(
            request_id=request.request_id,
            request_args=request_args.model_dump_json(),
            text=full_text,
            input_metrics=inp,
            output_metrics=outp,
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
        # Accumulated function_call items keyed by output_index, used to
        # reconstruct tool_calls on GenerationResponse after streaming.
        self.streaming_tool_calls: dict[int, ToolCall] = {}
        self.streaming_reasoning_texts: list[str] = []
        self._last_iteration_had_content: bool = False

    @property
    def last_iteration_had_content(self) -> bool:
        """
        :return: True if the last chunk carried output (text/tool-call) tokens,
            not solely reasoning tokens.
        """
        return self._last_iteration_had_content

    @staticmethod
    def _ensure_tool_format(tool: dict[str, Any]) -> dict[str, Any]:
        """Normalise a single tool definition to Responses API format.

        The Responses API expects ``{"type": "function", "name": ..., ...}`` with
        detail fields at the top level.  If the tool is already in that format it is
        returned as-is.  If it is in Chat Completions format (nested ``function``
        key, no top-level ``name``) the detail fields are flattened up.

        :param tool: A single tool definition dict in either format.
        :return: The tool in Responses API format.
        """
        if "function" in tool and "name" not in tool:
            fn = tool["function"]
            converted = {"type": tool.get("type", "function")}
            for key in _FUNCTION_DETAIL_KEYS:
                if key in fn:
                    converted[key] = fn[key]
            return converted
        return tool

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
        *,
        multiturn_reasoning: bool | str = False,
    ) -> list[dict[str, Any]]:
        """Build the ``input`` array for the Responses API.

        The Responses API uses a flat ``input`` list of role-tagged message
        dicts (with nested content parts like ``input_text``, ``input_image``)
        instead of chat completions' ``messages`` array.

        :param multiturn_reasoning: When truthy, prepend wrapped reasoning text
            to assistant content in the conversation history.
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

        # Replay the prior assistant response. For tool call responses,
        # include the function_call items and synthetic function_call_output
        # items so the model sees the full multi-turn exchange.
        wrapped_reasoning = _wrap_reasoning(
            response.reasoning_text if response else None, multiturn_reasoning
        )
        if response and response.tool_calls:
            for tc in response.tool_calls:
                input_items.append(self._tool_call_to_responses_item(tc))
            tool_response_columns = data.columns.get("tool_response_column", [])
            for tc_idx, tc in enumerate(response.tool_calls):
                raw_content = (
                    tool_response_columns[tc_idx]
                    if tc_idx < len(tool_response_columns)
                    else settings.default_synthetic_tool_response
                )
                output_content = (
                    raw_content.decode("utf-8")
                    if isinstance(raw_content, bytes)
                    else raw_content
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.id,
                        "output": output_content,
                    }
                )
        elif response and (response.text or wrapped_reasoning):
            content = response.text or ""
            if wrapped_reasoning:
                content = wrapped_reasoning + content
            input_items.append({"role": "assistant", "content": content})

        return input_items

    @staticmethod
    def _apply_tool_call_overrides(
        body: dict[str, Any],
        data: GenerationRequest,
    ) -> None:
        """Inject tool definitions and constrain the request body for tool calling.

        Handles three concerns:

        1. Deserializes and injects tool definitions from dataset columns,
           normalising to Responses API format when necessary.
        2. Sets ``tool_choice`` to ``"required"`` or ``"none"`` depending on
           whether the current turn expects a tool call.
        3. Removes body keys that are incompatible with tool calling
           (``ignore_eos``, ``stop``, and ``max_output_tokens`` on tool-call
           turns).

        :param body: The mutable request body dict being built.
        :param data: The current generation request.
        """
        tools_column = data.columns.get("tools_column", [])
        if tools_column:
            tools_value = tools_column[0]
            if isinstance(tools_value, str | bytes):
                tools_value = json.loads(tools_value)
            if isinstance(tools_value, list):
                body["tools"] = [
                    ResponsesRequestHandler._ensure_tool_format(t) for t in tools_value
                ]
                body.setdefault("tool_choice", "required")

        if "tools" not in body:
            body.pop("tool_choice", None)
            return

        if not data.expects_tool_call:
            body["tool_choice"] = "none"

        # Tool calling requires the model to stop naturally after producing
        # valid JSON; ignore_eos would force generation past that point.
        # max_output_tokens would truncate output mid-JSON and corrupt
        # the arguments sent in conversation history on follow-up turns.
        if data.expects_tool_call:
            body.pop("ignore_eos", None)
            body.pop("stop", None)
            body.pop("max_output_tokens", None)

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        use_server_history = kwargs.get("server_history") and history

        prev_requests: list[GenerationRequestArguments] = []
        if history and not use_server_history:
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

        arguments.body["input"] = self._build_input_items(
            data,
            response,
            prev_requests,
            multiturn_reasoning=kwargs.get("multiturn_reasoning", False),
        )

        # Server-side history: reference the previous response by ID and
        # include any tool outputs the server cannot know (tool execution
        # is client-side).  Only the immediate follow-up after a tool-call
        # response needs function_call_output items; subsequent turns have
        # no tool_calls on the last response so this branch is skipped.
        if use_server_history:
            self._apply_server_history(arguments.body, history)  # type: ignore[arg-type]

        self._apply_tool_call_overrides(arguments.body, data)

        return arguments

    @staticmethod
    def _extract_reasoning_text(response: dict) -> str | None:
        """Extract reasoning summary text from Responses API output items.

        Reasoning items have ``type: "reasoning"`` with a list of ``summary``
        objects, each containing a ``text`` field.

        :param response: Full Responses API response dict.
        :return: Concatenated reasoning text, or None if absent.
        """
        parts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "reasoning":
                continue
            for summary in item.get("summary", []):
                if txt := summary.get("text"):
                    parts.append(txt)
        return "".join(parts) or None

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: dict,
    ) -> GenerationResponse:
        text = self._extract_output_text(response)
        reasoning_text = self._extract_reasoning_text(response)
        raw_items = [
            item
            for item in response.get("output", [])
            if item.get("type") == "function_call"
        ]
        tool_calls: list[ToolCall] | None = (
            [self._responses_item_to_tool_call(item) for item in raw_items]
            if raw_items
            else None
        )
        if text is None and not tool_calls:
            text = ""
        usage = response.get("usage", {})
        input_metrics, output_metrics = self.extract_metrics(usage, text)
        _apply_tool_call_metrics(
            output_metrics, len(tool_calls) if tool_calls else 0, text
        )

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),
            text=text,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        return _compile_streaming_response(
            request,
            arguments,
            self.streaming_texts,
            self.streaming_tool_calls,
            self.streaming_usage,
            self.streaming_response_id,
            self.extract_metrics,
            streaming_reasoning_texts=self.streaming_reasoning_texts,
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

        data = json.loads(line[len("data:") :].strip())
        _check_streaming_error(data)
        return data

    @staticmethod
    def _responses_item_to_tool_call(item: dict[str, Any]) -> ToolCall:
        """Convert a Responses API ``function_call`` output item to a
        ``ToolCall``.

        The Responses API uses a flat structure (``call_id``, ``name``,
        ``arguments``) while ``ToolCall`` nests name/arguments
        inside a ``function`` sub-object.

        :param item: A Responses API ``function_call`` dict.
        :return: The equivalent ``ToolCall``.
        """
        return ToolCall(
            id=item.get("call_id", ""),
            type="function",
            function=ToolCallFunction(
                name=item.get("name", ""),
                arguments=item.get("arguments", ""),
            ),
        )

    @staticmethod
    def _tool_call_to_responses_item(tc: ToolCall) -> dict[str, Any]:
        """Convert a ``ToolCall`` back to a Responses API
        ``function_call`` input item for multi-turn replay.

        :param tc: The canonical tool call object.
        :return: A dict suitable for the Responses API ``input`` array.
        """
        return {
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }

    def _apply_server_history(
        self,
        body: dict[str, Any],
        history: HistoryT[GenerationRequest, GenerationResponse],
    ) -> None:
        """Apply server-side history fields to the request body.

        Sets ``previous_response_id`` from the last response in history and
        prepends ``function_call_output`` items when the last response
        contained tool calls (since tool execution is client-side and the
        server cannot infer those results).

        :param body: The mutable request body dict being built.
        :param history: The conversation history up to this point.
        """
        last_request, last_response = history[-1]
        if last_response and last_response.response_id:
            body["previous_response_id"] = last_response.response_id
        # Tool call responses need to be manually added from history because they are
        # stored on the prior request, rather than the current request.
        if last_response and last_response.tool_calls:
            body["input"] = (
                self._build_tool_outputs_from_history(last_request, last_response)
                + body["input"]
            )

    @staticmethod
    def _build_tool_outputs_from_history(
        prev_request: GenerationRequest,
        prev_response: GenerationResponse,
    ) -> list[dict[str, Any]]:
        """Build ``function_call_output`` items for server-history follow-ups.

        When using server-side history (``previous_response_id``), the server
        already stores the model's ``function_call`` outputs but cannot know
        what the tools returned (tool execution is client-side).  This method
        produces the minimal ``function_call_output`` items needed to complete
        the tool-calling cycle.

        :param prev_request: The request whose response triggered tool calls.
            Its ``tool_response_column`` supplies per-call response content.
        :param prev_response: The response containing tool calls.
        :return: List of ``function_call_output`` dicts for the ``input`` array.
        """
        tool_response_columns = prev_request.columns.get("tool_response_column", [])
        outputs: list[dict[str, Any]] = []
        for idx, tc in enumerate(prev_response.tool_calls):  # type: ignore[arg-type]
            raw_content = (
                tool_response_columns[idx]
                if idx < len(tool_response_columns)
                else settings.default_synthetic_tool_response
            )
            content = (
                raw_content.decode("utf-8")
                if isinstance(raw_content, bytes)
                else raw_content
            )
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tc.id,
                    "output": content,
                }
            )
        return outputs

    def _handle_streaming_function_call(
        self, event_type: str, data: dict[str, Any]
    ) -> int | None:
        """Process function_call-related streaming events.

        Accumulates ``ToolCall`` objects keyed by ``output_index``.
        The Responses API streams ``call_id`` / ``name`` / ``arguments`` at
        the top level; these are mapped into the canonical
        ``ToolCall`` shape used across all handlers.

        :returns: Token count delta, or ``None`` if unrecognized.
        """
        # First event for a new tool call: the server announces the
        # function_call output item with its call_id and name.  We
        # create a new ToolCall entry keyed by output_index so that
        # subsequent argument deltas can append to the right object.
        if (
            event_type == "response.output_item.added"
            and data.get("item", {}).get("type") == "function_call"
        ):
            idx = data["output_index"]
            item = data["item"]
            self.streaming_tool_calls[idx] = ToolCall(
                id=item.get("call_id", ""),
                type="function",
                function=ToolCallFunction(
                    name=item.get("name", ""),
                    arguments=item.get("arguments", ""),
                ),
            )
            return 1

        # Incremental argument chunk: append the JSON fragment to the
        # tool call that was created by the output_item.added event above.
        if event_type == "response.function_call_arguments.delta":
            idx = data.get("output_index", -1)
            if idx in self.streaming_tool_calls:
                self.streaming_tool_calls[idx].function.arguments += data.get(
                    "delta", ""
                )
            return 1

        # Final arguments payload: the server sends the complete argument
        # string once streaming is finished.  We overwrite whatever was
        # accumulated from deltas to guarantee consistency.
        if event_type == "response.function_call_arguments.done":
            idx = data.get("output_index", -1)
            if idx in self.streaming_tool_calls:
                self.streaming_tool_calls[idx].function.arguments = data.get(
                    "arguments",
                    self.streaming_tool_calls[idx].function.arguments,
                )
            return 1

        # Not a function-call event; let the caller handle it.
        return None

    def _handle_streaming_text_delta(
        self, event_type: str, data: dict[str, Any]
    ) -> int | None:
        """
        Handle reasoning and output text delta events, updating
        ``_last_iteration_had_content`` accordingly.

        :return: 0 or 1 if handled, None if not a text delta event.
        """
        if event_type == "response.reasoning_summary_text.delta":
            delta = data.get("delta", "")
            if delta:
                self.streaming_reasoning_texts.append(delta)
                self._last_iteration_had_content = False
                return 1
            return 0

        if event_type == "response.output_text.delta":
            delta = data.get("delta", "")
            if delta:
                self.streaming_texts.append(delta)
                self._last_iteration_had_content = True
                return 1
            return 0

        return None

    def add_streaming_line(self, line: str) -> int | None:
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        event_type = data.get("type", "")

        # Extract the response ID from the response.created event which
        # carries a nested "response" object containing the actual ID.
        if self.streaming_response_id is None:
            resp = data.get("response", {})
            if isinstance(resp, dict) and "id" in resp:
                self.streaming_response_id = resp["id"]

        text_result = self._handle_streaming_text_delta(event_type, data)
        if text_result is not None:
            return text_result

        # Function call deltas are always treated as content for TTFOT
        fc_result = self._handle_streaming_function_call(event_type, data)
        if fc_result is not None:
            self._last_iteration_had_content = True
            return fc_result

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
        # It also provides "input_tokens_details" and "output_tokens_details"
        # for multimodal breakdowns, mirroring chat completions'
        # "prompt_tokens_details"/"completion_tokens_details".
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
        input_details: dict[str, int] = cast(
            "dict[str, int]", usage.get("input_tokens_details", {}) or {}
        )
        output_details: dict[str, int] = cast(
            "dict[str, int]", usage.get("output_tokens_details", {}) or {}
        )

        return UsageMetrics(
            text_tokens=(
                input_details.get("text_tokens")
                or usage_metrics.get("input_tokens")
                or 0
            ),
            image_tokens=input_details.get("image_tokens"),
            video_tokens=input_details.get("video_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            audio_seconds=input_details.get("seconds"),
        ), UsageMetrics(
            text_tokens=(
                output_details.get("text_tokens")
                or usage_metrics.get("output_tokens")
                or 0
            ),
            text_words=text_words,
            text_characters=text_chars,
            image_tokens=output_details.get("image_tokens"),
            video_tokens=output_details.get("video_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            audio_seconds=output_details.get("seconds"),
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


@OpenAIRequestHandlerFactory.register("/v1/embeddings")
class EmbeddingsRequestHandler(OpenAIRequestHandler):
    """
    Request handler for OpenAI-style embeddings endpoints.

    Handles embeddings requests which do not support streaming and return
    embedding vectors instead of generated text. Processes input text into
    embeddings for performance benchmarking.
    """

    def format(
        self,
        data: GenerationRequest,
        response: GenerationResponse | None = None,  # noqa: ARG002
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> GenerationRequestArguments:
        """
        Format the embeddings generation request.

        :param data: The generation request to format
        :param response: Previous response (unused for embeddings)
        :param history: Request/response history (unused for embeddings)
        :param **kwargs: Additional keyword arguments (model, encoding_format, etc.)
        :return: The formatted request arguments
        """
        arguments = GenerationRequestArguments()
        arguments.body = {}
        arguments.stream = False  # Embeddings never stream

        # Add model
        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        # Build input from text columns
        input_texts = []
        for text in data.columns.get("text_column", []):
            if text:
                input_texts.append(text)

        # Use single string if only one text, otherwise list
        if len(input_texts) == 1:
            arguments.body["input"] = input_texts[0]
        else:
            arguments.body["input"] = input_texts

        # Apply extra arguments
        if kwargs.get("extras"):
            arguments.model_combine(kwargs["extras"])

        return arguments

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: Any,
    ) -> GenerationResponse:
        """
        Process a complete non-streaming embeddings API response.

        :param request: Original generation request
        :param arguments: Request arguments used
        :param response: Raw API response data
        :return: GenerationResponse with embeddings data
        """
        # Extract usage data
        usage = response.get("usage", {})

        # Build response (no text output for embeddings)
        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            text="",  # Embeddings don't generate text
            input_metrics=UsageMetrics(
                text_tokens=usage.get("prompt_tokens", 0),
            ),
            # output_metrics defaults to UsageMetrics() with all None values
        )

    def add_streaming_line(self, line: str) -> int | None:  # noqa: ARG002
        """
        Embeddings do not support streaming.

        :param line: Streaming line (unused)
        :return: None (not supported)
        :raises NotImplementedError: Embeddings never stream
        """
        raise NotImplementedError("Embeddings do not support streaming")

    def compile_streaming(  # noqa: ARG002
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Embeddings do not support streaming.

        :param request: Generation request (unused)
        :param arguments: Request arguments (unused)
        :return: Never returns
        :raises NotImplementedError: Embeddings never stream
        """
        raise NotImplementedError("Embeddings do not support streaming")
