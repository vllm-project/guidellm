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
import binascii
import io
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Protocol, cast

from more_itertools import roundrobin

from guidellm.backends.openai.common import format_ws_error
from guidellm.backends.openai.modality_parsers import (
    ModalityOutputParser,
    ModalityParserRegistry,
)
from guidellm.logger import logger
from guidellm.scheduler import HistoryT
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestTimings,
    ToolCall,
    ToolCallFunction,
    UsageMetrics,
)
from guidellm.settings import settings
from guidellm.utils.audio import (
    AudioContinuityStats,
    compute_audio_continuity,
    decode_audio_with_metadata,
    pcm16_append_b64_chunks,
)
from guidellm.utils.imports import json
from guidellm.utils.registry import RegistryMixin

WS_AUDIO_CHUNKS_BODY_KEY = "audio_chunks"
PCM_WAV_HEADER_SIZE = 44
WAV_FMT_CHUNK_SIZE = 16
PCM_BITS_PER_SAMPLE = 16
PCM16_BYTES_PER_SAMPLE = PCM_BITS_PER_SAMPLE // 8

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
    "SpeechRequestHandler",
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

    raw_response: ClassVar[bool] = False
    raw_streaming: bool = False

    def set_response_headers(self, headers: Mapping[str, str]) -> None:  # noqa: ARG002
        """
        Accept response headers for endpoint-specific processing.
        :param headers: HTTP response headers.
        :return: None.
        """
        return

    @property
    def last_iteration_had_audio(self) -> bool:
        """
        Return whether last iteration contained audio.
        :return: False, for non-audio output handlers.
        """
        return False

    def set_streaming_arrival_time(self, timestamp_s: float) -> None:  # noqa: ARG002
        """
        Accept the arrival timestamp of specific streaming.
        :param timestamp_s: Unix timestamp of the current stream item.
        :return: None.
        """
        return

    def mark_stream_complete(self) -> None:
        """
        Accept notification on successfully ended HTTP stream.
        :return: None.
        """
        return

    def update_response_timings(
        self, response: GenerationResponse, timings: RequestTimings
    ) -> None:
        """
        Get audio TTFP and RTF when response metrics are ready.
        Non-positive audio duration yields zero RTF.
        :param response: Compiled response.
        :param timings: HTTP request time measurements.
        :return: None.
        """
        if (
            timings.request_start is not None
            and timings.first_audio_iteration is not None
        ):
            response.output_metrics.audio_ttfp_ms = 1000 * (
                timings.first_audio_iteration - timings.request_start
            )
        duration = response.output_metrics.audio_seconds
        if (
            duration is not None
            and duration > 0
            and timings.request_start is not None
            and timings.request_end is not None
        ):
            response.output_metrics.audio_rtf = (
                timings.request_end - timings.request_start
            ) / duration
        if duration is not None and duration <= 0:
            response.output_metrics.audio_rtf = 0.0

    def format(
        self,
        data: GenerationRequest,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the generation request into the appropriate structure for
        the backend API.

        :param data: The generation request to format
        :param history: Prior (request, response) pairs in the conversation
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

    def add_streaming_chunk(self, chunk: bytes) -> int | None:
        """
        Process single raw response chunk.

        :param chunk: Raw line from the streaming response
        :return: 1 if chunk was added, 0 otherwise
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

    def post_validation(self, response: GenerationResponse) -> None:
        """Validate a compiled response before returning it.

        Default implementation is permissive (no-op). Handlers override
        this to reject responses that lack usable output for their
        endpoint type.

        :param response: The compiled generation response to validate.
        :raises ValueError: If the response is unusable.
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


def _apply_audio_metadata(item: dict[str, Any], metadata: Any) -> None:
    if not isinstance(metadata, dict):
        return

    audio_format = metadata.get("format")
    sample_rate = metadata.get("sample_rate_hz")
    frame_count = metadata.get("frame_count")
    channels = metadata.get("channels")
    if (
        not isinstance(audio_format, str)
        or not audio_format
        or type(sample_rate) is not int
        or sample_rate <= 0
        or type(frame_count) is not int
        or frame_count < 0
        or type(channels) is not int
        or channels <= 0
    ):
        return

    item.update(
        format=audio_format,
        sample_rate=sample_rate,
        channels=channels,
        audio_samples=frame_count,
        audio_seconds=frame_count / sample_rate,
    )


def _pcm16_layout(
    data: bytes,
    audio_format: str,
    sample_rate_hint: int | None,
    channels_hint: int | None,
) -> tuple[int, int, int, int] | None:
    """
    Returning tuple contains: (sample_rate, channels, header_bytes, frame_count).
    """
    sample_rate: int | None
    channels: int | None

    if audio_format == "wav":
        if (
            len(data) < PCM_WAV_HEADER_SIZE
            or data[:4] != b"RIFF"
            or data[8:12] != b"WAVE"
            or data[12:16] != b"fmt "
            or int.from_bytes(data[16:20], "little") != WAV_FMT_CHUNK_SIZE
            or int.from_bytes(data[20:22], "little") != 1
            or int.from_bytes(data[34:36], "little") != PCM_BITS_PER_SAMPLE
            or data[36:40] != b"data"
        ):
            return None
        sample_rate = int.from_bytes(data[24:28], "little")
        channels = int.from_bytes(data[22:24], "little")
        header_bytes = PCM_WAV_HEADER_SIZE
    elif audio_format in ("pcm", "pcm16"):
        sample_rate = sample_rate_hint
        channels = channels_hint
        header_bytes = 0
    else:
        return None

    if (
        type(sample_rate) is not int
        or sample_rate <= 0
        or type(channels) is not int
        or channels <= 0
    ):
        return None

    payload_bytes = len(data) - header_bytes
    bytes_per_frame = 2 * channels
    if payload_bytes < 0 or payload_bytes % bytes_per_frame:
        return None

    return sample_rate, channels, header_bytes, payload_bytes // bytes_per_frame


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

    def post_validation(self, response: GenerationResponse) -> None:
        """Validate a compiled response before returning it.

        Default implementation is permissive (no-op). Handlers override
        this to reject responses that lack usable output for their
        endpoint type.

        :param response: The compiled generation response to validate.
        :raises ValueError: If the response is unusable.
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


def _validate_text_response(response: GenerationResponse) -> None:
    """Reject a compiled response that has no usable output.

    A response is considered usable if it has non-empty text, tool calls,
    or output tokens. Used by text/chat completions and responses handlers.

    :param response: The compiled generation response to validate.
    :raises ValueError: If the response contains no usable output.
    """
    has_text = bool(response.text and response.text.strip())
    has_tool_calls = bool(response.tool_calls)
    output_tokens = response.output_metrics.total_tokens or 0
    if (
        not has_text
        and not has_tool_calls
        and not response.output_media
        and output_tokens <= 0
    ):
        raise ValueError(
            "[UNUSABLE_BACKEND_RESPONSE] backend resolved with empty response payload"
        )


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
    response_metrics: dict[str, Any] | None = None,
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
    :param response_metrics: Per-request metrics the backend reported, if any.
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
        response_metrics=response_metrics,
    )


_FUNCTION_DETAIL_KEYS = ("name", "description", "parameters", "strict")


def _constrain_tool_call_body(
    body: dict[str, Any],
    data: GenerationRequest,
    *token_limit_keys: str,
) -> None:
    """Set tool_choice and drop sampling keys that break constrained tool decoding.

    Tools must already be present in ``body``. An injection that also carries
    ``tools_column`` is a multi-step loop: keep ``tool_choice=required`` and
    drop the same sampling keys as ``client_tool_call``. ``token_limit_keys``
    are API-specific caps such as ``max_completion_tokens``.
    """
    if "tools" not in body:
        body.pop("tool_choice", None)
        return

    injection_requests_tools = data.turn_type == "tool_response_injection" and bool(
        data.columns.get("tools_column")
    )
    if data.turn_type == "standard" or (
        data.turn_type == "tool_response_injection" and not injection_requests_tools
    ):
        body["tool_choice"] = "none"

    if data.turn_type == "client_tool_call" or injection_requests_tools:
        body.pop("ignore_eos", None)
        body.pop("stop", None)
        for key in token_limit_keys:
            body.pop(key, None)


def _dataset_tool_choice(data: GenerationRequest) -> str:
    """Return ``tool_choice`` from ``tool_choice_column``, else ``required``.

    :param data: Current generation request.
    :return: ``required`` or ``auto``.
    :raises ValueError: If ``tool_choice_column`` is set to an unsupported value.
    """
    values = data.columns.get("tool_choice_column") or []
    if not values:
        return "required"
    choice = values[0]
    if choice not in ("required", "auto"):
        raise ValueError(
            f"Unsupported tool_choice_column value {choice!r}; "
            "expected 'required' or 'auto'."
        )
    return str(choice)


def _recorded_chat_messages(columns: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return OpenAI chat dicts from ``raw_messages_column`` when present.

    The column is always chat-completions format. Dataset cells store one
    value per column as a list: either ``[[{role, content}, ...]]`` or a
    flat list of dicts.

    :param columns: Request column mapping.
    :return: Message dicts, or ``None`` when the column is unset.
    """
    recorded = columns.get("raw_messages_column")
    if not recorded:
        return None
    first = recorded[0]
    if isinstance(first, list):
        return [item for item in first if isinstance(item, dict)]
    if isinstance(first, dict):
        return [item for item in recorded if isinstance(item, dict)]
    return None


def _require_request_payload(
    arguments: GenerationRequestArguments,
    *,
    endpoint: str,
    field: str,
    sources: str,
    columns: dict[str, Any],
) -> None:
    """Raise if the endpoint's required body field is missing or empty.

    Called after ``format()`` has merged extras, history, and the current
    turn so a payload supplied that way still counts as present.

    :param arguments: Formatted request arguments.
    :param endpoint: Request path shown in the error (e.g. ``/v1/completions``).
    :param field: Body key that must be non-empty (``prompt``, ``messages``,
        ``input``).
    :param sources: Human-readable column names that can fill ``field``.
    :param columns: Request columns, listed in the error when present and non-empty.
    :raises ValueError: If ``field`` is missing or empty.
    """
    body = arguments.body or {}
    if body.get(field):
        return
    nonempty = sorted(key for key, value in columns.items() if value)
    nonempty_names = ", ".join(nonempty) if nonempty else "none"
    raise ValueError(
        f"Cannot build {endpoint} {field}: missing {sources}. "
        f"Present, non-empty columns: {nonempty_names}."
    )


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
        self.response_metrics: dict[str, Any] | None = None

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
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the text completion generation request into the appropriate structure.

        :param data: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        :raises ValueError: If ``prompt`` cannot be built from columns, extras,
            or history.
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

        ## Build prompt ##
        prompts: list[str] = []

        # Include history: previous prompts and their responses
        if history:
            for req, res in history:
                prompts.extend(req.columns.get("prefix_column", []))
                prompts.extend(req.columns.get("text_column", []))
                if res and res.text:
                    prompts.append(res.text)

        # Include prefix
        prompts.extend(data.columns.get("prefix_column", []))
        # Include text column
        prompts.extend(data.columns.get("text_column", []))

        if prompts:
            arguments.body["prompt"] = " ".join(prompts)

        _require_request_payload(
            arguments,
            endpoint="/v1/completions",
            field="prompt",
            sources="text_column or prefix_column",
            columns=data.columns,
        )
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
            response_metrics=self.response_metrics,
        )

    def add_streaming_chunk(self, chunk: bytes) -> int | None:
        """
        Reject to process raw-byte requests for line-based completion handlers.

        :param chunk: Raw response bytes.
        :return: Zero for an empty chunk.
        :raises ValueError: If a non-empty raw chunk is received.
        """
        if not chunk:
            return 0
        raise ValueError(
            "[UNUSABLE_BACKEND_RESPONSE] raw-byte streaming is not supported "
            "by this request handler"
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
            response_metrics=self.response_metrics,
        )

    def post_validation(self, response: GenerationResponse) -> None:
        """Reject responses with no text, tool calls, or output tokens."""
        _validate_text_response(response)

    def extract_line_data(self, line: str) -> dict[str, Any] | None:
        """
        Extract JSON data from a streaming response line.

        :param line: Raw line from the streaming response
        :return: Parsed JSON data as dictionary, or None if line indicates completion
        """
        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return {}

        line = line[len("data:") :].strip()
        if line == "[DONE]":
            return None

        data = json.loads(line)
        _check_streaming_error(data)
        return data

    def extract_choices_and_usage(
        self, response: dict
    ) -> tuple[list[dict], dict[str, int | dict[str, int]]]:
        """
        Extract choices and usage data from the API response.

        A backend may report its own per-request metrics beside `usage`; vLLM does,
        under `metrics`. Those are recorded here rather than returned, so every handler
        picks them up without widening this signature.

        :param response: Complete API response containing choices and usage data
        :return: Tuple of choices list and usage dictionary
        """
        if isinstance(metrics := response.get("metrics"), dict):
            self.response_metrics = metrics
        return response.get("choices", []), response.get("usage", {})

    @staticmethod
    def _extract_input_metrics(
        input_details: dict[str, Any],
        usage_metrics: dict[str, int],
    ) -> UsageMetrics:
        """
        Extract input text and modality metrics (all that exist)

        :param input_details: Detailed input token usage from the backend.
        :param usage_metrics: Aggregate token usage counts.
        :return: Input usage metrics.
        """

        input_multimodal_with_value = input_details.get("multimodal_tokens")
        if not input_multimodal_with_value:
            input_multimodal: dict[str, Any] = {}
        elif isinstance(input_multimodal_with_value, dict):
            input_multimodal = input_multimodal_with_value
        else:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] "
                "prompt_tokens_details.multimodal_tokens must be an object"
            )

        input_image_tokens: int | None = input_multimodal.get(
            "image", input_details.get("image_tokens")
        )
        input_video_tokens: int | None = input_multimodal.get(
            "video", input_details.get("video_tokens")
        )
        input_audio_tokens: int | None = input_multimodal.get(
            "audio", input_details.get("audio_tokens")
        )

        input_multimodal_tokens = 0
        for count in (input_image_tokens, input_video_tokens, input_audio_tokens):
            input_multimodal_tokens += count if count is not None else 0

        input_text_tokens = input_details.get("prompt_tokens")
        if input_text_tokens is None:
            input_text_tokens = usage_metrics.get("prompt_tokens")
            if input_text_tokens is None:
                input_text_tokens = 0
            else:
                input_text_tokens -= input_multimodal_tokens
                if input_text_tokens < 0:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] input modality token counts "
                        "exceed the reported total prompt_tokens"
                    )

        return UsageMetrics(
            text_tokens=input_text_tokens,
            cached_tokens=input_details.get("cached_tokens"),
            image_tokens=input_image_tokens,
            video_tokens=input_video_tokens,
            audio_tokens=input_audio_tokens,
            audio_seconds=input_details.get("seconds"),
        )

    @staticmethod
    def _extract_output_metrics(
        output_details: dict[str, Any],
        usage_metrics: dict[str, int],
        text_words: int | None,
        text_chars: int | None,
    ) -> UsageMetrics:
        """
        Extract output text and modality metrics (all that available)

        :param output_details: Detailed output token usage from the backend.
        :param usage_metrics: Aggregate usage token counts.
        :param text_words: Word count, or None when text is not applicable.
        :param text_chars: Character count, or None when text is not applicable.
        :return: Output usage metrics.
        """

        output_multimodal_with_value = output_details.get("multimodal_tokens")
        if not output_multimodal_with_value:
            output_multimodal: dict[str, Any] = {}
        elif isinstance(output_multimodal_with_value, dict):
            output_multimodal = output_multimodal_with_value
        else:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] "
                "completion_tokens_details.multimodal_tokens must be an object"
            )

        output_image_tokens: int | None = output_multimodal.get(
            "image", output_details.get("image_tokens")
        )
        output_video_tokens: int | None = output_multimodal.get(
            "video", output_details.get("video_tokens")
        )
        output_audio_tokens: int | None = output_multimodal.get(
            "audio", output_details.get("audio_tokens")
        )

        output_multimodal_tokens = 0
        for count in (output_image_tokens, output_video_tokens, output_audio_tokens):
            output_multimodal_tokens += count if count is not None else 0

        output_text_tokens = output_details.get("completion_tokens")
        if output_text_tokens is None:
            output_text_tokens = usage_metrics.get("completion_tokens")
            if output_text_tokens is None:
                output_text_tokens = 0
            else:
                output_text_tokens -= output_multimodal_tokens
                if output_text_tokens < 0:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] output modality token counts "
                        "exceed the reported total completion_tokens"
                    )

        return UsageMetrics(
            text_tokens=output_text_tokens,
            text_words=text_words,
            text_characters=text_chars,
            image_tokens=output_image_tokens,
            video_tokens=output_video_tokens,
            audio_tokens=output_audio_tokens,
            audio_seconds=output_details.get("seconds"),
        )

    def extract_metrics(
        self, usage: dict[str, int | dict[str, int]] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract text and modality input and output usage metrics
        from API response usage data.

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

        input_details: dict[str, Any] = cast(
            "dict[str, Any]", usage.get("prompt_tokens_details", {}) or {}
        )
        output_details: dict[str, Any] = cast(
            "dict[str, Any]", usage.get("completion_tokens_details", {}) or {}
        )
        usage_metrics: dict[str, int] = cast("dict[str, int]", usage)

        return (
            self._extract_input_metrics(input_details, usage_metrics),
            self._extract_output_metrics(
                output_details, usage_metrics, text_words, text_chars
            ),
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
        self._last_iteration_had_audio = False
        self.streaming_media_packets: list[
            tuple[int, dict[str, Any], float | None]
        ] = []
        self._streaming_arrival_time: float | None = None
        self._last_audio_event_time: float | None = None
        self.streaming_audio_ids: dict[int, str] = {}
        self.streaming_audio_expirations: dict[int, int | float] = {}
        self.streaming_audio_transcripts: dict[int, list[str]] = {}
        self._output_sample_rate_hint: int | None = None
        self._output_channels_hint: int | None = None
        self._requested_audio_format = "wav"
        self._stream_complete = False

    def set_streaming_arrival_time(self, timestamp_s: float) -> None:
        """
        Update the arrival timestamp.
        :param timestamp_s: Unix timestamp of the current stream item.
        :return: None.
        """
        self._streaming_arrival_time = timestamp_s

    def mark_stream_complete(self) -> None:
        self._stream_complete = True

    def update_response_timings(
        self, response: GenerationResponse, timings: RequestTimings
    ) -> None:
        """
        Update the last audio event timings with the RTF stats
        (even if last audio event is empty)

        :param response: Compiled response to update.
        :param timings: HTTP request timing measurements.
        :return: None.
        """
        super().update_response_timings(response, timings)
        duration = response.output_metrics.audio_seconds
        last_audio_iteration = timings.last_audio_iteration
        if self._last_audio_event_time is not None and (
            last_audio_iteration is None
            or self._last_audio_event_time > last_audio_iteration
        ):
            last_audio_iteration = self._last_audio_event_time

        if (
            duration is not None
            and duration > 0
            and timings.request_start is not None
            and last_audio_iteration is not None
        ):
            response.output_metrics.audio_rtf = (
                last_audio_iteration - timings.request_start
            ) / duration

    def _append_streaming_media_item(
        self, choice_index: int, item: dict[str, Any]
    ) -> None:

        item["choice_index"] = choice_index
        if item.get("modality") == "audio":
            self._last_iteration_had_audio = True
        self.streaming_media_packets.append(
            (choice_index, item, self._streaming_arrival_time)
        )

    @property
    def last_iteration_had_content(self) -> bool:
        """
        :return: True if the last chunk carried output (text/tool-call) tokens,
            not solely reasoning tokens.
        """
        return self._last_iteration_had_content

    @property
    def last_iteration_had_audio(self) -> bool:
        """
        :return: Return True if current chat delta contained audio.
        """
        return self._last_iteration_had_audio

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
        self,
        column_data: list,
        column_type: str,
        content_extras: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Helper method to format different types of data columns
        into the appropriate structure for chat messages.
        """
        formatted_data = []
        for item in column_data:
            if column_type == "text_column":
                content = {"type": "text", "text": item}
                if content_extras:
                    content.update(content_extras)
                formatted_data.append(content)
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

    def _parse_output_media_from_content(  # noqa: C901
        self, content: Any
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Split a message's ``content`` into (text, output_media items), trying
        every registered modality parser on each list item.
        If content is a plain string, it returned unchanged.

        :param content: Raw message["content"] or delta["content"] value.
        :return: Tuple of (joined text or None, output_media items found).
        """
        if content is None:
            return None, []
        if isinstance(content, str):
            return content, []  # input is string
        if not isinstance(content, list):
            return None, []

        text_parts: list[str] = []
        media: list[dict[str, Any]] = []

        try:
            parser_names = ModalityParserRegistry.registered_names()
        except ValueError:
            parser_names = ()

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                if text := part.get("text"):
                    text_parts.append(text)
                continue
            """
            Try each registered modality parser; unregistered/unrecognized part
            types (e.g. a future modality with no parser landed yet) are
            silently skipped (maybe should raise?)
            """
            for parser_name in parser_names:
                parser_cls = ModalityParserRegistry.create(parser_name)
                if parser_cls is not None and (
                        item := parser_cls.parse_content_part(part)
                ) is not None:
                    media.append(item)
                    break

        return ("".join(text_parts) if text_parts else None), media

    def _augment_media_output_metrics(
        self, output_metrics: UsageMetrics, items: list[dict[str, Any]]
    ) -> None:
        """
        Add audio output_metrics measurements.

        :param output_metrics: Response metrics updated in place.
        :param items: Encoded output media packets.
        :return: None.
        :raises ValueError: If audio packet is malformed or undecodable.
        """
        audio_items = [item for item in items if item.get("modality") == "audio"]
        if not audio_items:
            return

        output_metrics.audio_bytes = sum(len(item["data"]) for item in audio_items)
        total_samples = 0
        total_seconds = 0.0
        complete_samples = True
        complete_seconds = True

        for audio_item in audio_items:
            audio_format, samples, seconds, rate, channels = (
                self._retrieve_audio_item_metrics(audio_item)
            )

            audio_item["format"] = audio_format
            audio_item["audio_samples"] = samples
            audio_item["audio_seconds"] = seconds
            audio_item["sample_rate"] = rate
            audio_item["channels"] = channels

            if samples is None:
                complete_samples = False
            else:
                total_samples += samples
            if seconds is None:
                complete_seconds = False
            else:
                total_seconds += seconds

        if output_metrics.audio_samples is None and complete_samples:
            output_metrics.audio_samples = total_samples
        if output_metrics.audio_seconds is None and complete_seconds:
            output_metrics.audio_seconds = total_seconds

    def _retrieve_audio_item_metrics(
        self,
        audio_item: dict[str, Any],
    ) -> tuple[str | None, int | None, float | None, int | None, int | None]:
        """
        Get metrics for one audio packet without modifying it.

        :param audio_item: audio packet with the supplied metadata.
        :return: Metadata format,sample count, duration, sample rate and channel count.
        :raises ValueError: If the packet is malformed or cannot be decoded.
        """
        audio_format = audio_item.get("format") or self._requested_audio_format
        samples = audio_item.get("audio_samples")
        seconds = audio_item.get("audio_seconds")
        rate = audio_item.get("sample_rate") or self._output_sample_rate_hint
        channels = audio_item.get("channels") or self._output_channels_hint

        if samples is None or seconds is None:
            layout = _pcm16_layout(audio_item["data"], audio_format, rate, channels)
            if layout is not None:
                rate, channels, _, frames = layout
                samples = frames
                seconds = frames / rate
            elif audio_format in ("pcm", "pcm16"):
                audio_len = len(audio_item["data"])
                pcm_len = 2 * (channels or 1)
                if (audio_len - int(audio_len / pcm_len) * pcm_len) > 0:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] incomplete PCM16 frame"
                    )
            else:
                try:
                    decoded, rate, seconds = decode_audio_with_metadata(
                        audio_item["data"]
                    )
                except Exception as exc:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] chat audio decoding failed"
                    ) from exc
                samples = int(decoded.shape[-1])
                channels = int(decoded.shape[0]) if decoded.ndim > 1 else 1

        if samples == 0 or seconds == 0:
            raise ValueError("[UNUSABLE_BACKEND_RESPONSE] empty chat audio packet")

        return audio_format, samples, seconds, rate, channels

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
            # The OpenAI spec allows the server to do multiple tool calls.
            # If the quantity of calls exceeds those in the dataset, inject the
            # default synthetic tool response.
            raw_content = (
                tool_response_columns[idx]
                if idx < len(tool_response_columns)
                else settings.default_synthetic_tool_response
            )
            # The project JSON utils can return bytes or string; ensure string.
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
        2. Sets ``tool_choice`` from ``tool_choice_column`` when present,
           otherwise ``"required"``, or ``"none"`` on turns that must not
           emit tool calls.
        3. Removes body keys that are incompatible with tool calling
           (``ignore_eos``, ``stop``, and token-limit keys on tool-call turns).

        :param body: The mutable request body dict being built.
        :param data: The current generation request.
        """
        tools_column = data.columns.get("tools_column", [])
        if tools_column:
            tools_value = tools_column[0]
            # JSON-serialized tool definitions (e.g. from synthetic data
            # generators that store tools as strings for HuggingFace Features
            # compatibility). The project JSON utils can return bytes or str.
            if isinstance(tools_value, str | bytes):
                tools_value = json.loads(tools_value)
            if isinstance(tools_value, list):
                body["tools"] = [
                    ChatCompletionsRequestHandler._ensure_tool_format(t)
                    for t in tools_value
                ]
                body.setdefault("tool_choice", _dataset_tool_choice(data))

        _constrain_tool_call_body(body, data, "max_completion_tokens")

    def _append_request_content_messages(
        self,
        messages: list[dict[str, Any]],
        req: GenerationRequest,
        **kwargs,
    ) -> None:
        """Append current-turn content: recorded chat dicts or text/multimodal.

        When ``raw_messages_column`` is set, those dicts are extended as-is so
        multi-turn replay can send user, assistant, and tool messages without
        wrapping them as a single user string.

        :param messages: Message list to extend in place.
        :param req: Request whose columns supply the current turn.
        :param kwargs: Forwarded format kwargs (``extras`` for multimodal).
        """
        recorded = _recorded_chat_messages(req.columns)
        if recorded is not None:
            messages.extend(recorded)
            return

        prefix = " ".join(req.columns.get("prefix_column", []))
        if prefix:
            messages.append({"role": "system", "content": prefix})

        extras = kwargs.get("extras")
        content_extras = extras.content if extras is not None else None
        prompts = [
            self._format_prompts(
                req.columns.get(col, []),
                col,
                content_extras,
            )
            for col in (
                "text_column",
                "image_column",
                "video_column",
                "audio_column",
            )
        ]
        user_content = list(roundrobin(*prompts))
        if user_content:
            messages.append({"role": "user", "content": user_content})

    def _build_history_messages(
        self,
        history: HistoryT[GenerationRequest, GenerationResponse],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Build the messages array from completed conversation turns.

        :param history: Completed (request, response) pairs.
        :param kwargs: Forwarded config (``multiturn_reasoning``, etc.).
        :return: Flat list of message dicts ready to extend into the body.
        """
        messages: list[dict[str, Any]] = []
        for idx, (req, res) in enumerate(history):
            # Passes in the prior response so that past data can be included,
            # like for tool calling.
            prior_response = history[idx - 1][1] if idx > 0 else None
            messages.extend(
                self._build_turn_messages(req, res, prior_response, **kwargs)
            )
        return messages

    def _audio_history_content(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert generated audio packets into replayable content parts.

        :param items: Audio packets belonging to one assistant choice.
        :return: Input-audio content parts preserving packet order.
        :raises ValueError: If raw PCM lacks the layout needed for replay.
        """
        columns: list[dict[str, Any]] = []
        for item in items:
            data = item["data"]
            audio_format = item.get("format") or "wav"
            if audio_format in ("pcm", "pcm16"):
                rate = item.get("sample_rate")
                channels = item.get("channels")
                if (
                    type(rate) is not int
                    or rate <= 0
                    or type(channels) is not int
                    or channels <= 0
                ):
                    raise ValueError(
                        "Cannot replay PCM audio without sample rate and channels"
                    )
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as writer:
                    writer.setnchannels(channels)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                    writer.writeframes(data)
                data = buffer.getvalue()
                audio_format = "wav"
            columns.append({"audio": data, "format": audio_format})

        return self._format_prompts(columns, "audio_column")

    def _build_turn_messages(  # noqa: C901
        self,
        req: GenerationRequest,
        res: GenerationResponse | None,
        prior_response: GenerationResponse | None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Build messages for a single history turn.

        Dispatches on ``req.turn_type``:

        * ``"client_tool_call"``: user content + assistant tool_calls (no tool
          response messages — those come from the following injection turn).
        * ``"tool_response_injection"``: tool-role messages built from
          ``prior_response.tool_calls`` IDs + this turn's
          ``tool_response_column``, then the assistant text response.
        * ``"standard"``: user content + assistant text response.

        :param req: The request for this history turn.
        :param res: The response the server gave for this turn.
        :param prior_response: The response from the immediately preceding
            history turn (used by injection turns for ``tool_call_id``s).
        :param kwargs: Forwarded config (``multiturn_reasoning``, etc.).
        :return: List of message dicts for this turn.
        """
        messages: list[dict[str, Any]] = []
        multiturn_reasoning = kwargs.get("multiturn_reasoning", False)

        if req.turn_type == "tool_response_injection":
            # Injection turn: tool-role messages then assistant text.
            if prior_response and prior_response.tool_calls:
                tool_response_columns = req.columns.get("tool_response_column", [])
                messages.extend(
                    self._build_tool_response_messages(
                        prior_response.tool_calls, tool_response_columns
                    )
                )
            if res is not None and res.text is not None:
                wrapped = _wrap_reasoning(res.reasoning_text, multiturn_reasoning)
                content = res.text
                if wrapped:
                    content = wrapped + content
                messages.append({"role": "assistant", "content": content})
        else:
            # Standard or tool_call turn: recorded messages, or system + user content.
            self._append_request_content_messages(messages, req, **kwargs)

            # Assistant response for history replay.
            wrapped = _wrap_reasoning(
                res.reasoning_text if res else None, multiturn_reasoning
            )
            if res is not None:
                if res.tool_calls:
                    # Tool-call turn: assistant with tool_calls only.
                    # Tool response messages come from the injection turn.
                    assistant_content = res.text
                    if wrapped:
                        assistant_content = wrapped + (assistant_content or "")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": [tc.model_dump() for tc in res.tool_calls],
                        }
                    )
                elif res.text is not None or wrapped:
                    content = res.text or ""
                    if wrapped:
                        content = wrapped + content
                    messages.append({"role": "assistant", "content": content})

        if res is not None and not res.text:
            self._append_audio_content(messages, res)

        return messages

    def _append_audio_content(
        self,
        messages: list[dict[str, Any]],
        response: GenerationResponse,
    ) -> None:
        """
        Append generated audio to the assistant message.
        """
        audio_items = [
            item
            for item in response.output_media or []
            if item.get("modality") == "audio"
        ]
        if not audio_items:
            return

        choice_index = audio_items[0].get("choice_index", 0)
        audio_items = [
            item for item in audio_items if item.get("choice_index", 0) == choice_index
        ]

        if messages and messages[-1]["role"] == "assistant":
            assistant = messages[-1]
        else:
            assistant = {"role": "assistant", "content": None}
            messages.append(assistant)

        transcripts: list[str] = []
        for item in audio_items:
            transcript = item.get("transcript")
            if isinstance(transcript, str) and transcript:
                transcripts.append(transcript)

        if transcripts:
            assistant["content"] = (assistant.get("content") or "") + "".join(
                transcripts
            )
            return

        content = self._audio_history_content(audio_items)
        existing_content = assistant.get("content")
        if existing_content:
            content.insert(0, {"type": "text", "text": existing_content})
        assistant["content"] = content

    def format(  # noqa: C901, PLR0912, PLR0915
        self,
        data: GenerationRequest,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format the chat completion generation request into the appropriate structure.

        :param data: The generation request to format
        :param history: Prior (request, response) pairs in the conversation
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        :raises ValueError: If ``messages`` cannot be built from columns, extras,
            history, or a tool-injection turn.
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

        # Build messages from history
        arguments.body["messages"] = (
            self._build_history_messages(history, **kwargs) if history else []
        )

        # Build the current turn's messages
        if data.turn_type == "tool_response_injection":
            # Injection turn: send tool results back to the server.
            # tool_call_ids come from the last history entry's response.
            prior_response = history[-1][1] if history else None
            if prior_response and prior_response.tool_calls:
                tool_response_columns = data.columns.get("tool_response_column", [])
                arguments.body["messages"].extend(
                    self._build_tool_response_messages(
                        prior_response.tool_calls, tool_response_columns
                    )
                )
        else:
            # Standard or tool_call turn: recorded messages, or system + user content.
            self._append_request_content_messages(
                arguments.body["messages"], data, **kwargs
            )

        # Inject tool definitions and apply tool-call-specific overrides.
        self._apply_tool_call_overrides(arguments.body, data)

        self._output_sample_rate_hint = kwargs.get("audio_output_sample_rate")
        self._output_channels_hint = kwargs.get("audio_output_channels")
        self._requested_audio_format = "wav"
        if isinstance(arguments.body, dict):
            audio_settings = arguments.body.get("audio")
            if isinstance(audio_settings, dict):
                audio_format = audio_settings.get("format")
                if isinstance(audio_format, str):
                    self._requested_audio_format = audio_format

        _require_request_payload(
            arguments,
            endpoint="/v1/chat/completions",
            field="messages",
            sources=(
                "raw_messages_column, text_column, prefix_column, or media columns"
            ),
            columns=data.columns,
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
        :param arguments: The request arguments that were sent
        :param response: Complete API response containing choices and usage data
        :return: Standardized GenerationResponse with extracted content and metrics
        """
        choices, usage = self.extract_choices_and_usage(response)

        """
        Fix: do NOT assume choices[0] carries both text and audio. Single
        ["text","audio"] modalities request returns text and audio as
        SEPARATE choices. Scan every choice; take the first non-null text/
        reasoning/tool_calls and merge in every choice's parsed audio via
        the registry. Zero-behavior-change for today's single-choice,
        text-only case -- the loop body just executes once.
        """
        text: str | None = None
        reasoning_text: str | None = None
        raw_tool_calls: list[dict] | None = None
        output_media: list[dict[str, Any]] = []

        for choice in choices:
            message = choice.get("message", {})
            content_text, content_media = self._parse_output_media_from_content(
                message.get("content")
            )
            if text is None:
                text = content_text
            output_media.extend(content_media)
            if reasoning_text is None:
                reasoning_text = (
                    message.get("reasoning") or message.get("reasoning_content") or None
                )
            if raw_tool_calls is None:
                raw_tool_calls = message.get("tool_calls")

            audio_parser = ModalityParserRegistry.create("audio")
            if (
                audio_parser is not None
                and (
                    item := audio_parser.parse_dedicated_field(
                        message.get("audio")
                    )
                )
                is not None
            ):
                _apply_audio_metadata(item, choice.get("audio_metadata"))
                item["choice_index"] = choice.get("index", 0)
                output_media.append(item)

        if text is None and not raw_tool_calls and not output_media:
            text = ""  # Edge case: null content, no tools, no media
        input_metrics, output_metrics = self.extract_metrics(usage, text)

        if output_media:
            self._augment_media_output_metrics(output_metrics, output_media)

        tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            tool_calls = [ToolCall.model_validate(tc) for tc in raw_tool_calls]
        _apply_tool_call_metrics(
            output_metrics, len(tool_calls) if tool_calls else 0, text
        )

        # if audio_only case - verify that text metrics reflect this
        audio_only = bool(output_media) and not (text or reasoning_text or tool_calls)
        if audio_only:
            text = None
            output_metrics.text_tokens = None
            output_metrics.text_words = None
            output_metrics.text_characters = None

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),  # use vLLM ID if available
            text=text,
            use_request_output_metrics=not audio_only,
            reasoning_text=reasoning_text,
            tool_calls=tool_calls,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            response_metrics=self.response_metrics,
            output_media=output_media or None,
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from a chat completion streaming response.

        Handles the chat completion specific delta structure where content is nested
        within delta objects in the streaming response chunks. Also accumulates
        ``tool_calls`` deltas when the model streams function call output and marks
         audio event timestamp for rtf even if no audio payload was present.

        :param line: Raw SSE line from the streaming response
        :return: 1 if content was extracted, 0 if line ignored, None if done
        """

        self._last_iteration_had_audio = False
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        if "id" in data and self.streaming_response_id is None:
            self.streaming_response_id = data["id"]

        # Tracks whether this iteration produced user-visible content (not
        # reasoning-only). Used by the HTTP layer to set TTFOT timing.
        updated = False
        had_content = False
        choices, usage = self.extract_choices_and_usage(data)
        if choices and data.get("modality") == "audio":
            self._last_audio_event_time = self._streaming_arrival_time

        for choice in choices:
            audio_parser = ModalityParserRegistry.create("audio")
            choice_updated, choice_had_content = self._process_streaming_choice(
                data, choice, audio_parser
            )
            updated |= choice_updated
            had_content |= choice_had_content

        if usage:
            self.streaming_usage = usage

        # Only update the flag when we processed a token-bearing iteration;
        # non-updating lines should not reset the flag.
        if updated:
            self._last_iteration_had_content = had_content
        return 1 if updated else 0

    def _process_streaming_choice(
        self,
        data: dict[str, Any],
        choice: dict[str, Any],
        audio_parser: ModalityOutputParser | None,
    ) -> tuple[bool, bool]:
        """
        Process one choice delta and return its update and content flags.
        """
        updated = False
        had_content = False
        index = choice.get("index", 0)
        delta = choice.get("delta") or {}

        # Reasoning tokens trigger TTFT (updated=True) but are not
        # considered "content" for the TTFOT metric.
        if reasoning := (delta.get("reasoning") or delta.get("reasoning_content")):
            self.streaming_reasoning_texts.append(reasoning)
            updated = True

        content_updated, content_had_content = self._process_streaming_choice_content(
            data, choice, index, delta, audio_parser
        )
        updated |= content_updated
        had_content |= content_had_content

        # Accumulate streamed tool_calls deltas.  Each tool call may be split
        # across multiple chunks; we reassemble by ``index``.
        # ``tool_calls`` could be either missing or ``null`` in the delta
        # (some OpenAI-compatible servers emit this), handle both cases
        for tc_delta in delta.get("tool_calls") or []:
            self._accumulate_tool_call_delta(tc_delta)
            updated = True
            had_content = True

        return updated, had_content

    def _process_streaming_choice_content(
        self,
        data: dict[str, Any],
        choice: dict[str, Any],
        index: int,
        delta: dict[str, Any],
        audio_parser: ModalityOutputParser | None,
    ) -> tuple[bool, bool]:
        """
        Process text and audio content for one choice.
        """

        content = delta.get("content")
        content_was_audio = False
        if (
            data.get("modality") == "audio"
            and content
            and isinstance(content, str)
            and audio_parser is not None
        ):
            audio_item = audio_parser.parse_dedicated_field({"data": content})
            if audio_item is not None:
                _apply_audio_metadata(audio_item, choice.get("audio_metadata"))
                self._append_streaming_media_item(index, audio_item)
                content_was_audio = True

        content_updated = False
        choice_had_content = False
        if not content_was_audio:
            content_text, content_media = self._parse_output_media_from_content(content)
            if content_text:
                self.streaming_texts.append(content_text)
                content_updated = True
                choice_had_content = True

            for media_item in content_media:
                if media_item.get("modality") == "audio":
                    _apply_audio_metadata(media_item, choice.get("audio_metadata"))
                self._append_streaming_media_item(index, media_item)

            if audio_parser is not None:
                audio_data = delta.get("audio")
                if isinstance(audio_data, dict):
                    audio_id = audio_data.get("id")
                    if isinstance(audio_id, str) and audio_id:
                        self.streaming_audio_ids[index] = audio_id

                    transcript = audio_data.get("transcript")
                    if isinstance(transcript, str) and transcript:
                        self.streaming_audio_transcripts.setdefault(
                            index, []
                        ).append(transcript)

                    expiration_time = audio_data.get("expires_at")
                    if isinstance(expiration_time, (int, float)) and not isinstance(
                            expiration_time, bool
                    ):
                        self.streaming_audio_expirations[index] = expiration_time

                    # Transcript-only and metadata-only updates are valid stream
                    # events, but they are not valid payloads for the audio parser.
                    if not audio_data.get("data"):
                        audio_data = None

                audio_item = audio_parser.parse_dedicated_field(audio_data)
                if audio_item is not None:
                    _apply_audio_metadata(audio_item, choice.get("audio_metadata"))
                    self._append_streaming_media_item(index, audio_item)

        return content_updated, choice_had_content

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

    def _add_streaming_audio_metadata(
            self, output_media: list[dict[str, Any]]
    ) -> list[str]:
        """
        Attach streamed audio metadata (IDs, expiration times, transcripts)
        to the packets.

        :param output_media: Compiled output media items.
        :return: Transcripts for choices that had no audio packet.
        """
        audio_items_by_choice: dict[int, list[dict[str, Any]]] = {}
        for item in output_media:
            if item.get("modality") == "audio":
                choice_index = item.get("choice_index", 0)
                audio_items_by_choice.setdefault(choice_index, []).append(item)

        for choice_index, audio_items in audio_items_by_choice.items():
            audio_id = self.streaming_audio_ids.get(choice_index)
            expiration_time = self.streaming_audio_expirations.get(choice_index)

            for item in audio_items:
                if audio_id is not None and not item.get("id"):
                    item["id"] = audio_id
                if expiration_time is not None and (
                        audio_id is None or item.get("id") == audio_id
                ):
                    item["expires_at"] = expiration_time

            transcript = "".join(
                self.streaming_audio_transcripts.get(choice_index, [])
            )
            if transcript:
                # The response-level transcript spans the audio packets; store
                # it once so replay does not duplicate it.
                for item in audio_items:
                    item.pop("transcript", None)
                audio_items[0]["transcript"] = transcript

        return [
            transcript
            for choice_index, parts in self.streaming_audio_transcripts.items()
            if choice_index not in audio_items_by_choice
               and (transcript := "".join(parts))
        ]

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile accumulated streaming chat completion content into a final response.

        :param request: Original generation request
        :param arguments: Additional arguments for complete request definition
        :return: Standardized GenerationResponse with concatenated content and metrics
        """
        response = _compile_streaming_response(
            request,
            arguments,
            self.streaming_texts,
            self.streaming_tool_calls,
            self.streaming_usage,
            self.streaming_response_id,
            self.extract_metrics,
            streaming_reasoning_texts=self.streaming_reasoning_texts,
            response_metrics=self.response_metrics,
        )

        # Each upstream audio packet is independently encoded. Preserve
        # packet order and its own valid container instead of joining WAVs.
        output_media: list[dict[str, Any]] = [
            item for _, item, _ in self.streaming_media_packets
        ]
        unattached_transcripts = self._add_streaming_audio_metadata(output_media)
        if unattached_transcripts and not response.text:
            transcript_text = "".join(unattached_transcripts)
            response.text = transcript_text
            _, response.output_metrics = self.extract_metrics(
                self.streaming_usage, transcript_text
            )

        if output_media:
            response.output_media = output_media
            self._augment_media_output_metrics(response.output_metrics, output_media)
            if not (response.text or response.reasoning_text or response.tool_calls):
                response.text = None
                response.use_request_output_metrics = False
                response.output_metrics.text_tokens = None
                response.output_metrics.text_words = None
                response.output_metrics.text_characters = None

        if self._stream_complete:
            by_choice: dict[int, list[tuple[dict[str, Any], float | None]]] = {}
            for index, item, arrival in self.streaming_media_packets:
                if item.get("modality") == "audio":
                    by_choice.setdefault(index, []).append((item, arrival))

            continuity: list[AudioContinuityStats] = []
            for packets in by_choice.values():
                first_packet = packets[0][0]
                sample_rate = first_packet.get("sample_rate")
                channels = first_packet.get("channels")
                if (
                    type(sample_rate) is not int
                    or sample_rate <= 0
                    or type(channels) is not int
                    or channels <= 0
                    or any(
                        arrival is None
                        or item.get("sample_rate") != sample_rate
                        or item.get("channels") != channels
                        or type(item.get("audio_samples")) is not int
                        or item["audio_samples"] <= 0
                        for item, arrival in packets
                    )
                ):
                    continuity = []
                    break

                arrival_times: list[float] = [
                    cast("float", time_s) for _, time_s in packets
                ]
                chunks_bytes: list[int] = [
                    item["audio_samples"] * channels * 2 for item, _ in packets
                ]
                continuity.append(
                    compute_audio_continuity(
                        chunk_arrival_times_s=arrival_times,
                        chunk_bytes=chunks_bytes,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                )

            if continuity:
                response.output_metrics.audio_underrun_seconds = max(
                    result.max_underrun_seconds for result in continuity
                )
                response.output_metrics.audio_continuity_ok = int(
                    all(result.is_continuous for result in continuity)
                )
        return response


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
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:  # noqa: C901
        """
        Format the audio transcription generation request into the
        appropriate structure.

        :param data: The generation request to format
        :param **kwargs: Additional keyword arguments for request formatting
        :return: The formatted request arguments
        """
        if history:
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
                audio_columns[0].get("file_name") or "audio_input",
                audio_columns[0].get("audio"),
                audio_columns[0].get("mimetype"),
            )
        }

        return arguments

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: dict,
    ) -> GenerationResponse:
        """
        Process a complete audio transcription or translation response.

        :param request: Original generation request
        :param arguments: The request arguments that were sent
        :param response: Complete API response containing top-level text and usage
        :return: Standardized GenerationResponse with extracted text and metrics
        """
        text = response.get("text")
        input_metrics, output_metrics = self.extract_metrics(
            response.get("usage"), text
        )

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            response_id=response.get("id"),
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            response_metrics=self.response_metrics,
        )

    def post_validation(self, response: GenerationResponse) -> None:
        """Reject audio responses that omit the transcription text field.

        An empty string is a valid transcription for silent or VAD-filtered audio.

        :param response: The compiled audio response to validate.
        :raises ValueError: If the response has no transcription text field.
        """
        if response.text is None:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] backend resolved with empty response "
                "payload"
            )

    @staticmethod
    def _align_audio_usage_for_chat_completions(  # noqa: C901
        audio_usage: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Align Audio/Realtime usage fields with the Chat Completions usage metrics.

        :param audio_usage: Usage data using Audio / Realtime keys.
        :return: A copy with Chat Completions keys added where needed.
        :raises ValueError: If keys aliases are conflicting or usage details are
                invalid.
        """
        chat_usage = dict(audio_usage)

        for audio_key, chat_key in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            audio_value = audio_usage.get(audio_key)
            chat_value = audio_usage.get(chat_key)
            # check that values are not both exist and contradict
            if (
                audio_value is not None
                and chat_value is not None
                and audio_value != chat_value
            ):
                raise ValueError(
                    "[UNUSABLE_BACKEND_RESPONSE] conflicting usage fields "
                    f"'{audio_key}' and '{chat_key}'"
                )

            if chat_value is None and audio_value is not None:
                chat_usage[chat_key] = audio_value

        for audio_key, chat_key, chat_text_key in (
            ("input_token_details", "prompt_tokens_details", "prompt_tokens"),
            ("output_token_details", "completion_tokens_details", "completion_tokens"),
        ):
            audio_details = audio_usage.get(audio_key)
            chat_details = audio_usage.get(chat_key)

            if audio_details is not None and not isinstance(audio_details, dict):
                raise ValueError(
                    f"[UNUSABLE_BACKEND_RESPONSE] invalid usage details '{audio_key}'"
                )
            if chat_details is not None and not isinstance(chat_details, dict):
                raise ValueError(
                    f"[UNUSABLE_BACKEND_RESPONSE] invalid usage details '{chat_key}'"
                )
            if audio_details is None:
                continue

            aligned_details = dict(chat_details or {})
            for key, value in audio_details.items():
                chat_detail_key = chat_text_key if key == "text_tokens" else key
                chat_value = aligned_details.get(chat_detail_key)

                if value is not None and chat_value is not None and value != chat_value:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] conflicting usage fields "
                        f"'{audio_key}.{key}' and '{chat_key}.{chat_detail_key}'"
                    )

                if chat_value is None and value is not None:
                    aligned_details[chat_detail_key] = value

            chat_usage[chat_key] = aligned_details

        return chat_usage

    def extract_metrics(
        self, usage: dict[str, Any] | None, text: str | None
    ) -> tuple[UsageMetrics, UsageMetrics]:
        """
        Extract input and output usage metrics using the parent's metrics extractor.
        Handle input audio duration and text/audio token counts without double-counting.
        Input text/audio counts remain unknown if they cannot be determined from the
        reported usage.

        :param usage: Prompt/completion token usage or duration-only audio usage
        :param text: Generated text, or None when text is not applicable
        :return: Separate input text/audio metrics and output text metrics
        :raises ValueError: If input token counts are negative or exceed the total.
        """
        chat_usage = (
            self._align_audio_usage_for_chat_completions(usage) if usage else usage
        )
        input_metrics, output_metrics = super().extract_metrics(chat_usage, text)
        if not usage or chat_usage is None:
            return input_metrics, output_metrics

        input_details = chat_usage.get("prompt_tokens_details") or {}
        output_details = chat_usage.get("completion_tokens_details") or {}
        input_total_tokens = chat_usage.get("prompt_tokens")

        # The parent's text fallback cannot classify an unsplit ASR input total.
        if input_details.get("prompt_tokens") is None and (
            input_total_tokens is None or input_metrics.audio_tokens is None
        ):
            input_metrics.text_tokens = None

        if input_metrics.audio_tokens is None and input_total_tokens is not None:
            if input_metrics.text_tokens is not None:
                input_metrics.audio_tokens = (
                    input_total_tokens - input_metrics.text_tokens
                )
            elif input_total_tokens == 0:
                input_metrics.text_tokens = 0
                input_metrics.audio_tokens = 0

        input_counts = (
            input_metrics.text_tokens,
            input_metrics.audio_tokens,
        )
        if any(count is not None and count < 0 for count in input_counts) or (
            input_total_tokens is not None
            and sum(count for count in input_counts if count is not None)
            > input_total_tokens
        ):
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] input token counts are negative "
                "or exceed the reported prompt_tokens total"
            )

        if chat_usage.get("seconds") is not None:
            input_metrics.audio_seconds = chat_usage["seconds"]

        # Duration-only usage must not inherit the parent's missing-count zero.
        if (
            output_details.get("completion_tokens") is None
            and chat_usage.get("completion_tokens") is None
        ):
            output_metrics.text_tokens = None

        return input_metrics, output_metrics


@OpenAIRequestHandlerFactory.register("/v1/audio/speech")
class SpeechRequestHandler(OpenAIRequestHandler):
    """
    Request handler for OpenAI / vLLM-Omni text-to-speech (TTS), for
    /v1/audio/speech OpenAI endpoint.
    Response shape depends on streaming mode:
      - Non-streaming (default, stream=false and no stream_format): raw binary audio;
        usage from response headers (x-vllm-omni-*),  via set_response_headers().
      - SSE streaming (with stream=true, the vLLM-Omni streaming default):
        speech.audio.delta/.done/.error text events -- reuses the existing
        line-oriented streaming loop with the underneath audio payload.
      - Raw byte streaming (stream_format="audio"): raw PCM/WAV bytes

      Response modes:
      - Non-streaming (stream=false and no stream_format): complete binary audio.
      - SSE (stream=true or stream_format="sse"): speech.audio.* events.
      - Raw audio (stream_format="audio"): streamed PCM or WAV bytes.
    """

    raw_response: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.raw_streaming = False
        self._response_headers: dict[str, str] = {}
        self._last_iteration_had_audio = False
        self._output_sample_rate_hint: int | None = None
        self._output_channels_hint: int | None = None
        self._response_sample_rate: int | None = None
        self._response_channels: int | None = None
        self._requested_audio_format = "wav"
        self._streaming_arrival_time: float | None = None
        self._stream_complete = False
        self._streaming_response = False
        self._streamed_audio_header = b""
        self._streamed_audio_bytes_seen = 0
        self.streaming_audio_chunks: list[bytes] = []
        self.streaming_audio_arrivals: list[float | None] = []
        self.streaming_audio_format: str | None = None
        self.streaming_usage: dict[str, Any] | None = None

    @property
    def last_iteration_had_content(self) -> bool:
        """
        :return: False because audio has a separate flag.
        """
        return False

    @property
    def last_iteration_had_audio(self) -> bool:
        """
        Return whether the last chunk contained audio.
        :return: True if at least one additional PCM chunk arrived.
        """
        return self._last_iteration_had_audio

    def set_response_headers(self, headers: Mapping[str, str]) -> None:
        """
        Store usage headers and server's PCM layout.
        :param headers: HTTP response headers.
        :return: None.
        """
        self._response_headers = {key.lower(): value for key, value in headers.items()}
        self._update_audio_layout(
            self._response_headers.get("x-audio-sample-rate"),
            self._response_headers.get("x-audio-channels"),
        )

    def _update_audio_layout(self, sample_rate: Any, channels: Any) -> None:
        """
        Merge server layout into the active layout.

        :param sample_rate: Sample rate from raw header or SSE event (or None).
        :param channels: Number of channels from raw header or SSE event (or None).
        :return: None.
        :raises ValueError: If invalid value or changes inside the response.
        """
        values: list[int | None] = []
        for name, raw, current in (
            ("sample_rate", sample_rate, self._response_sample_rate),
            ("channels", channels, self._response_channels),
        ):
            if raw is None:
                values.append(current)
                continue
            value = int(raw) if isinstance(raw, str) and raw.isdigit() else raw
            if type(value) is not int or value <= 0:
                raise ValueError(f"[UNUSABLE_BACKEND_RESPONSE] invalid audio {name}")
            if current is not None and value != current:
                raise ValueError(f"[UNUSABLE_BACKEND_RESPONSE] audio {name} changed")
            values.append(value)

        self._response_sample_rate, self._response_channels = values

    def set_streaming_arrival_time(self, timestamp_s: float) -> None:
        """
        Update the arrival timestamp.
        :param timestamp_s: Unix timestamp of the current stream item.
        :return: None.
        """
        self._streaming_arrival_time = timestamp_s

    def mark_stream_complete(self) -> None:
        """
        Record successful EOF for a raw audio stream.
        SSE completion is recorded only by speech.audio.done.
        :return: None.
        """
        if self.raw_streaming:
            self._stream_complete = True

    def format(
        self,
        data: GenerationRequest,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        """
        Format text synthesis and optional voice-cloning inputs.

        :param data: Request columns and requested generation limits.
        :param history: Unsupported prior conversation turns.
        :param kwargs: Model, streaming options, layout hints, and extras.
        :return: Speech request arguments.
        :raises ValueError: If history or reference columns are invalid.
        """
        if history:
            raise ValueError("SpeechRequestHandler does not support multiturn.")

        arguments = GenerationRequestArguments()
        arguments.body = {}

        self._output_sample_rate_hint = kwargs.get("audio_output_sample_rate")
        self._output_channels_hint = kwargs.get("audio_output_channels")
        if kwargs.get("model") is not None:
            arguments.body["model"] = kwargs["model"]

        text_parts = [
            *data.columns.get("prefix_column", []),
            *data.columns.get("text_column", []),
        ]
        if text_parts:
            arguments.body["input"] = " ".join(text_parts)

        # Voice cloning
        self._format_reference_audio(arguments, data)

        max_tokens = data.output_metrics.text_tokens
        if max_tokens is None:
            max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            arguments.body["max_new_tokens"] = max_tokens

        ref_text_columns = data.columns.get("ref_text_column", [])
        if ref_text_columns:
            if len(ref_text_columns) != 1 or not isinstance(ref_text_columns[0], str):
                raise ValueError(
                    "ref_text_column must contain one reference transcript"
                )
            arguments.body["ref_text"] = ref_text_columns[0]

        if kwargs.get("extras"):
            arguments.model_combine(
                kwargs["extras"]
            )  # task_type, voice, language, etc.

        self._configure_speech_streaming(arguments, kwargs)
        return arguments

    @staticmethod
    def _format_reference_audio(
        arguments: GenerationRequestArguments,
        data: GenerationRequest,
    ) -> None:
        """
        Add voice-cloning reference audio from the audio_column.
        :param arguments: Request arguments to be filled.
        :param data: Request columns containing reference audio.
        :return: None.
        :raises ValueError: If a reference item has no audio bytes.
        """
        # Voice cloning: reuse audio_column as the reference-audio input.
        ref_audio_columns = data.columns.get("audio_column", [])
        references: list[str] = []
        for ref_item in ref_audio_columns:
            audio = ref_item.get("audio")
            if not isinstance(audio, bytes) or not audio:
                raise ValueError("Reference audio must contain non-empty bytes")
            encoded = base64.b64encode(audio).decode("utf-8")
            fmt = ref_item.get("format") or "wav"
            references.append(f"data:audio/{fmt};base64,{encoded}")

        if references:
            body = arguments.body
            if body is None:
                body = {}
                arguments.body = body
            body["ref_audio"] = references[0] if len(references) == 1 else references

    def _configure_speech_streaming(
        self,
        arguments: GenerationRequestArguments,
        kwargs: dict[str, Any],
    ) -> None:
        """
        Set streaming options and the audio format.
        Streaming mode selection:
            stream_format="audio" -> raw bytes streaming;
            stream=True (and no stream_format or "sse" stream_format) -> SSE

        :param arguments: Request arguments being built.
        :param kwargs: Additional speech request options.
        :return: None.
        """
        body = arguments.body or {}
        self._requested_audio_format = body.get("response_format") or "wav"
        stream_format = body.get("stream_format")
        self.raw_streaming = stream_format == "audio"

        if stream_format in ("audio", "sse"):
            arguments.stream = True
            if stream_format == "sse":
                body["stream"] = True
        elif "stream" in body:
            arguments.stream = bool(body["stream"])
        elif arguments.stream is True or (
            arguments.stream is None and kwargs.get("stream")
        ):
            arguments.stream = True
            body["stream"] = True

    @staticmethod
    def extract_line_data(line: str) -> dict[str, Any] | None:
        if not line or not (line := line.strip()) or not line.startswith("data:"):
            return {}
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return None
        data = json.loads(payload)
        if data.get("type") != "speech.audio.error":
            _check_streaming_error(data)
        return data

    def _add_chunk(self, chunk: bytes, audio_format: str) -> None:
        """
        Append an audio bytes' chunk and update the payload.
        :param chunk: Non-empty encoded audio chunk.
        :param audio_format: Format of the accumulated response.
        :return: None.
        """
        previous_bytes = self._streamed_audio_bytes_seen
        self.streaming_audio_chunks.append(chunk)
        self.streaming_audio_arrivals.append(self._streaming_arrival_time)
        self._streamed_audio_bytes_seen += len(chunk)

        # Keep encoded chunks, but don't interpret them as pcm for TTFP.
        if audio_format not in ("pcm", "pcm16", "wav"):
            self._last_iteration_had_audio = False
            return

        header_bytes = PCM_WAV_HEADER_SIZE if audio_format == "wav" else 0
        channels = self._response_channels or self._output_channels_hint
        if audio_format == "wav":
            self._streamed_audio_header = (self._streamed_audio_header + chunk)[
                :PCM_WAV_HEADER_SIZE
            ]
            if len(self._streamed_audio_header) < PCM_WAV_HEADER_SIZE:
                self._last_iteration_had_audio = False
                return

            wav_layout = _pcm16_layout(
                self._streamed_audio_header,
                audio_format,
                None,
                None,
            )
            if wav_layout is None:
                # Do not count container/header bytes as playable audio.
                self._last_iteration_had_audio = False
                return

            channels = wav_layout[1]

        if type(channels) is not int or channels <= 0:
            self._last_iteration_had_audio = False
            return

        bytes_per_frame = PCM16_BYTES_PER_SAMPLE * channels
        previous_frames = max(0, previous_bytes - header_bytes) // bytes_per_frame
        current_frames = (
            max(0, self._streamed_audio_bytes_seen - header_bytes) // bytes_per_frame
        )
        self._last_iteration_had_audio = current_frames > previous_frames

    def _populate_audio_duration(
        self,
        output_metrics: UsageMetrics,
        data: bytes,
        audio_format: str,
        arguments: GenerationRequestArguments,
        complete: bool = True,
    ) -> tuple[int, int, int, int] | None:
        """
        Populate audio sample and duration metrics when the layout is known.

        Headerless PCM remains with unknown layout-dependent metrics when
        no server metadata or configured layout hints are available. Encoded
        containers are decoded after having the complete response.

        :param output_metrics: Metrics updated in place.
        :param data: Accumulated encoded audio or PCM16 bytes.
        :param audio_format: Response audio format.
        :param arguments: Request arguments containing optional layout hints.
        :param complete: Whether all response bytes have been received.
        :return: PCM16 layout, or None for other or unknown layouts.
        :raises ValueError: If completed audio is malformed or cannot be decoded.
        """
        if not data:
            return None

        body = arguments.body or {}
        sample_rate = (
            self._response_sample_rate
            or self._output_sample_rate_hint
            or body.get("sample_rate")
        )
        channels = self._response_channels or self._output_channels_hint
        layout = _pcm16_layout(data, audio_format, sample_rate, channels)
        if layout is not None:
            rate, _, _, frames = layout
            output_metrics.audio_samples = frames
            output_metrics.audio_seconds = frames / rate
            return layout

        if not complete:
            return None

        if audio_format in ("pcm", "pcm16"):
            if len(data) % PCM16_BYTES_PER_SAMPLE:
                raise ValueError("[UNUSABLE_BACKEND_RESPONSE] incomplete PCM16 sample")

            if type(channels) is int and channels > 0:
                bytes_per_frame = PCM16_BYTES_PER_SAMPLE * channels
                if len(data) % bytes_per_frame:
                    raise ValueError(
                        "[UNUSABLE_BACKEND_RESPONSE] incomplete PCM16 frame"
                    )

            logger.warning(
                "PCM speech output has no usable sample-rate/channel layout; "
                "preserving audio bytes and leaving sample count and duration "
                "unknown. Configure audio_output_sample_rate and "
                "audio_output_channels to calculate them."
            )
            return None

        try:
            samples, _, duration = decode_audio_with_metadata(data)
        except Exception as exc:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] audio decoding failed"
            ) from exc

        output_metrics.audio_samples = int(samples.shape[-1])
        output_metrics.audio_seconds = duration
        return None

    def compile_non_streaming(
        self,
        request: GenerationRequest,
        arguments: GenerationRequestArguments,
        response: bytes,
    ) -> GenerationResponse:
        """
        Process a raw-bytes TTS response. Usage comes from
        x-vllm-omni-{input,output,total,input-text,input-audio}-tokens
        response headers (set via set_response_headers() before this call)
        when present; falls back to None (not zero) when talking to a
        server that doesn't set them, per design principle #1.
        """
        audio_format = (arguments.body or {}).get("response_format") or "wav"

        def _hdr_int(name: str) -> int | None:
            raw = self._response_headers.get(name)
            return int(raw) if raw is not None and raw.isdigit() else None

        input_metrics = UsageMetrics(
            text_tokens=_hdr_int("x-vllm-omni-input-text-tokens"),
            audio_tokens=_hdr_int("x-vllm-omni-input-audio-tokens"),
        )
        output_metrics = UsageMetrics(
            audio_tokens=_hdr_int("x-vllm-omni-output-tokens"),
            audio_bytes=len(response),
        )

        self._populate_audio_duration(output_metrics, response, audio_format, arguments)
        if audio_format not in ("pcm", "pcm16") and (
            output_metrics.audio_seconds is None
        ):
            raise ValueError("[UNUSABLE_BACKEND_RESPONSE] audio decoding has failed.")

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            text=None,
            use_request_output_metrics=False,
            output_media=[
                {"modality": "audio", "data": response, "format": audio_format}
            ],
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def add_streaming_line(self, line: str) -> int | None:
        """
        Parse a single speech.audio.* SSE line. Reused unmodified line-oriented
        loop -- the payload is text (base64 JSON), so this is not
        the raw-byte streaming path (that's add_streaming_chunk(), below).
        """

        self._last_iteration_had_audio = False
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        event_type = data.get("type")
        if event_type == "speech.audio.delta":
            self._update_audio_layout(data.get("sample_rate"), data.get("channels"))
            encoded = data.get("audio")
            if not encoded:
                return 0
            try:
                chunk = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError, TypeError) as exc:
                raise ValueError(
                    f"[UNUSABLE_BACKEND_RESPONSE] invalid speech audio delta: {exc}"
                ) from exc
            if not chunk:
                return 0
            self.streaming_audio_format = (
                data.get("response_format") or self.streaming_audio_format
            )
            self._add_chunk(
                chunk,
                self.streaming_audio_format or self._requested_audio_format,
            )
            return 0

        if event_type == "speech.audio.done":
            self.streaming_usage = data.get("usage")
            self._stream_complete = True
            return None
        if event_type == "speech.audio.error":
            raise ValueError(
                f"Streaming speech response returned an error: "
                f"{data.get('error') or {}}"
            )
        return 0

    def add_streaming_chunk(self, chunk: bytes) -> int | None:
        """
        Accumulate a raw speech chunk.
        :param chunk: Audio bytes received from the HTTP stream.
        :return: Zero because audio chunks are not counted as text-token.
        """
        if not chunk:
            return 0
        self._add_chunk(chunk, self._requested_audio_format)
        return 0

    def compile_streaming(
        self, request: GenerationRequest, arguments: GenerationRequestArguments
    ) -> GenerationResponse:
        """
        Compile the accumulated speech response.

        :param request: Original generation request.
        :param arguments: Original generation request.
        :return: Audio response and available usage measurements.
        """

        self._streaming_response = True
        audio_bytes = b"".join(self.streaming_audio_chunks)
        # speech.audio.done's usage uses input_tokens/output_tokens naming,
        # NOT prompt_tokens/completion_tokens -- deliberately not reusing
        # TextCompletionsRequestHandler.extract_metrics() here, since that
        # would silently record zero usage against the wrong keys.

        audio_format = (
            self.streaming_audio_format
            or (arguments.body or {}).get("response_format")
            or "wav"
        )
        usage = self.streaming_usage or {}
        details = usage.get("input_token_details") or {}
        input_metrics = UsageMetrics(
            text_tokens=details.get("text_tokens"),
            audio_tokens=details.get("audio_tokens"),
        )
        output_metrics = UsageMetrics(
            audio_tokens=usage.get("output_tokens"),
            audio_bytes=len(audio_bytes),
        )
        layout = self._populate_audio_duration(
            output_metrics,
            audio_bytes,
            audio_format,
            arguments,
            complete=self._stream_complete,
        )

        if self._stream_complete and layout is not None:
            sample_rate, channels, header_bytes, _ = layout
            header_remaining = header_bytes
            times: list[float] = []
            sizes: list[int] = []
            complete_timing = True
            for chunk, arrival in zip(
                self.streaming_audio_chunks,
                self.streaming_audio_arrivals,
                strict=True,
            ):
                skipped = min(len(chunk), header_remaining)
                header_remaining -= skipped
                pcm_bytes = len(chunk) - skipped
                if pcm_bytes:
                    if arrival is None:
                        complete_timing = False
                        break
                    times.append(arrival)
                    sizes.append(pcm_bytes)

            if complete_timing and sizes:
                continuity = compute_audio_continuity(
                    times,
                    sizes,
                    sample_rate=sample_rate,
                    channels=channels,
                )
                output_metrics.audio_underrun_seconds = continuity.max_underrun_seconds
                output_metrics.audio_continuity_ok = int(continuity.is_continuous)

        return GenerationResponse(
            request_id=request.request_id,
            request_args=arguments.model_dump_json(),
            text=None,
            use_request_output_metrics=False,
            output_media=[
                {
                    "modality": "audio",
                    "data": audio_bytes,
                    "format": audio_format,
                }
            ],
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )

    def post_validation(self, response: GenerationResponse) -> None:
        """
        Reject responses with no generated audio bytes.

        :param response: Compiled speech response.
        :return: None.
        :raises ValueError: If completion or usable audio is missing.
        """
        if self._streaming_response and not self._stream_complete:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] speech stream ended "
                "before speech.audio.done"
            )

        if (
            not response.output_media
            or not response.output_media[0].get("data")
            or response.output_metrics.audio_samples == 0
        ):
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] backend resolved with empty audio payload"
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
        self,
        column_data: list,
        column_type: str,
        content_extras: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        formatted_data: list[dict[str, Any]] = []
        for item in column_data:
            if column_type == "text_column":
                content = {"type": "input_text", "text": item}
                if content_extras:
                    content.update(content_extras)
                formatted_data.append(content)
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

    def _build_history_input_items(
        self,
        history: HistoryT[GenerationRequest, GenerationResponse],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Build the ``input`` array from completed conversation turns.

        Iterates through history with neighbor access so that injection
        turns can pull ``tool_call_id``s from the preceding turn's response.

        :param history: Completed (request, response) pairs.
        :param kwargs: Forwarded config (``multiturn_reasoning``, etc.).
        :return: Flat list of input item dicts.
        """
        items: list[dict[str, Any]] = []
        for idx, (req, res) in enumerate(history):
            prior_response = history[idx - 1][1] if idx > 0 else None
            items.extend(
                self._build_turn_input_items(req, res, prior_response, **kwargs)
            )
        return items

    def _build_turn_input_items(  # noqa: C901
        self,
        req: GenerationRequest,
        res: GenerationResponse | None,
        prior_response: GenerationResponse | None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Build input items for a single history turn.

        Dispatches on ``req.turn_type`` analogously to the chat completions
        handler's ``_build_turn_messages``.

        :param req: The request for this history turn.
        :param res: The response the server gave for this turn.
        :param prior_response: Response from the preceding history turn
            (used by injection turns for ``call_id``s).
        :param kwargs: Forwarded config (``multiturn_reasoning``, etc.).
        :return: List of input item dicts for this turn.
        """
        items: list[dict[str, Any]] = []
        multiturn_reasoning = kwargs.get("multiturn_reasoning", False)

        if req.turn_type == "tool_response_injection":
            # Injection turn: function_call_output items then assistant text.
            if prior_response and prior_response.tool_calls:
                items.extend(
                    self._build_function_call_outputs(
                        prior_response.tool_calls,
                        req.columns.get("tool_response_column", []),
                    )
                )
            if res is not None and res.text is not None:
                wrapped = _wrap_reasoning(res.reasoning_text, multiturn_reasoning)
                content = res.text
                if wrapped:
                    content = wrapped + content
                items.append({"role": "assistant", "content": content})
        else:
            self._append_request_content_input_items(items, req, **kwargs)

            wrapped = _wrap_reasoning(
                res.reasoning_text if res else None, multiturn_reasoning
            )
            if res is not None:
                if res.tool_calls:
                    # Tool-call turn: function_call items only.
                    # function_call_output items come from the injection turn.
                    for tc in res.tool_calls:
                        items.append(self._tool_call_to_responses_item(tc))
                elif res.text is not None or wrapped:
                    content = res.text or ""
                    if wrapped:
                        content = wrapped + content
                    items.append({"role": "assistant", "content": content})

        return items

    @staticmethod
    def _build_function_call_outputs(
        tool_calls: list[ToolCall],
        tool_response_columns: list[Any],
    ) -> list[dict[str, Any]]:
        """Build ``function_call_output`` items from tool calls and response data.

        :param tool_calls: Tool call objects supplying ``call_id``s.
        :param tool_response_columns: Per-call response content from the
            dataset, falling back to the default synthetic response.
        :return: List of ``function_call_output`` dicts.
        """
        outputs: list[dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls):
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

    @staticmethod
    def _apply_tool_call_overrides(
        body: dict[str, Any],
        data: GenerationRequest,
    ) -> None:
        """Inject tool definitions and constrain the request body for tool calling.

        Handles three concerns:

        1. Deserializes and injects tool definitions from dataset columns,
           normalising to Responses API format when necessary.
        2. Sets ``tool_choice`` from ``tool_choice_column`` when present,
           otherwise ``"required"``, or ``"none"`` on turns that must not
           emit tool calls.
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
                body.setdefault("tool_choice", _dataset_tool_choice(data))

        _constrain_tool_call_body(body, data, "max_output_tokens")

    def _append_request_content_input_items(
        self,
        items: list[dict[str, Any]],
        req: GenerationRequest,
        **kwargs,
    ) -> None:
        """Append current-turn content as Responses ``input`` items.

        ``text_column`` / media wrap as a user message. ``raw_messages_column``
        is ignored here; a missing ``input`` after format raises.

        :param items: Input item list to extend in place.
        :param req: Request whose columns supply the current turn.
        :param kwargs: Forwarded format kwargs (``extras`` for multimodal).
        """
        extras = kwargs.get("extras")
        content_extras = extras.content if extras is not None else None
        prompts = [
            self._format_prompts(
                req.columns.get(col, []),
                col,
                content_extras,
            )
            for col in (
                "text_column",
                "image_column",
                "video_column",
                "audio_column",
            )
        ]
        content_parts = list(roundrobin(*prompts))
        if content_parts:
            items.append({"role": "user", "content": content_parts})

    def format(  # noqa: C901
        self,
        data: GenerationRequest,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,
        **kwargs,
    ) -> GenerationRequestArguments:
        use_server_history = kwargs.get("server_history") and history

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

        # Build input items from history
        input_items: list[dict[str, Any]] = []
        if history and not use_server_history:
            input_items = self._build_history_input_items(history, **kwargs)

        # Build the current turn's input items.
        # When server_history is active, _apply_server_history handles
        # function_call_output injection, so skip it here.
        if data.turn_type == "tool_response_injection" and not use_server_history:
            prior_response = history[-1][1] if history else None
            if prior_response and prior_response.tool_calls:
                input_items.extend(
                    self._build_function_call_outputs(
                        prior_response.tool_calls,
                        data.columns.get("tool_response_column", []),
                    )
                )
        elif data.turn_type != "tool_response_injection":
            self._append_request_content_input_items(input_items, data, **kwargs)

        arguments.body["input"] = input_items

        # Server-side history: reference the previous response by ID and
        # include any tool outputs the server cannot know (tool execution
        # is client-side).  Only the immediate follow-up after a tool-call
        # response needs function_call_output items; subsequent turns have
        # no tool_calls on the last response so this branch is skipped.
        if use_server_history:
            self._apply_server_history(arguments.body, history, data)  # type: ignore[arg-type]

        self._apply_tool_call_overrides(arguments.body, data)

        _require_request_payload(
            arguments,
            endpoint="/v1/responses",
            field="input",
            sources="text_column or media columns",
            columns=data.columns,
        )
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

    def post_validation(self, response: GenerationResponse) -> None:
        """Reject responses with no text, tool calls, or output tokens."""
        _validate_text_response(response)

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

        line = line[len("data:") :].strip()
        if line == "[DONE]":
            return None

        data = json.loads(line)
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
        current_request: GenerationRequest,
    ) -> None:
        """Apply server-side history fields to the request body.

        Sets ``previous_response_id`` from the last response in history.
        For injection turns, prepends ``function_call_output`` items using
        the injection turn's ``tool_response_column`` and the preceding
        response's tool call IDs.

        :param body: The mutable request body dict being built.
        :param history: The conversation history up to this point.
        :param current_request: The current request being formatted.
        """
        _, last_response = history[-1]
        if last_response and last_response.response_id:
            body["previous_response_id"] = last_response.response_id
        # For injection turns the tool_response_column lives on the
        # current request and the tool_call IDs come from the last
        # history entry's response.
        if (
            current_request.turn_type == "tool_response_injection"
            and last_response
            and last_response.tool_calls
        ):
            body["input"] = (
                self._build_function_call_outputs(
                    last_response.tool_calls,
                    current_request.columns.get("tool_response_column", []),
                )
                + body["input"]
            )

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
            if event_type == "response.failed":
                _check_streaming_error(resp)
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
            cached_tokens=input_details.get("cached_tokens"),
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

    def post_validation(self, response: GenerationResponse) -> None:  # noqa: ARG002
        """Pooling responses produce non-text output; skip validation."""

    def format(
        self,
        data: GenerationRequest,
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> GenerationRequestArguments:
        """
        Format the pooling generation request into the appropriate structure.

        :param data: The generation request to format
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
        history: HistoryT[GenerationRequest, GenerationResponse] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> GenerationRequestArguments:
        """
        Format the embeddings generation request.

        :param data: The generation request to format
        :param history: Request/response history (unused for embeddings)
        :param **kwargs: Additional keyword arguments (model, encoding_format, etc.)
        :return: The formatted request arguments
        :raises ValueError: If ``input`` cannot be built from ``text_column`` or extras.
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

        _require_request_payload(
            arguments,
            endpoint="/v1/embeddings",
            field="input",
            sources="text_column",
            columns=data.columns,
        )
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
