"""Optional OpenTelemetry tracing support for GuideLLM benchmark execution."""

from __future__ import annotations

import os
import threading
from typing import Any

try:
    from opentelemetry import context as otel_context
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    OTEL_API_AVAILABLE = True
except ImportError:
    OTEL_API_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcOTLPSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpOTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    OTEL_SDK_AVAILABLE = True
except ImportError:
    OTEL_SDK_AVAILABLE = False

__all__ = [
    "TraceSpan",
    "activate_trace_context",
    "capture_trace_context",
    "deactivate_trace_context",
    "inject_trace_headers",
    "shutdown_tracing",
    "start_span",
]

_configure_lock = threading.Lock()


class _TracingState:
    """Mutable process-local configuration state."""

    configured_pid: int | None = None
    tracer_provider: Any = None


_state = _TracingState()


def _configure_from_environment() -> None:
    """Configure an OTLP exporter when standard OTEL environment variables enable it."""
    if (
        not OTEL_API_AVAILABLE
        or not OTEL_SDK_AVAILABLE
        or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return
    if os.environ.get("OTEL_SDK_DISABLED", "false").lower() == "true":
        return

    process_id = os.getpid()
    if _state.configured_pid == process_id:
        return

    with _configure_lock:
        if _state.configured_pid == process_id:
            return

        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        exporter: SpanExporter
        if protocol == "http/protobuf":
            exporter = HttpOTLPSpanExporter()
        elif protocol == "grpc":
            exporter = GrpcOTLPSpanExporter()
        else:
            raise ValueError(
                "OTEL_EXPORTER_OTLP_PROTOCOL must be 'grpc' or 'http/protobuf', "
                f"got {protocol!r}"
            )

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))

        _state.tracer_provider = provider
        _state.configured_pid = process_id


class TraceSpan:
    """Small lifecycle wrapper that keeps OpenTelemetry optional at runtime."""

    def __init__(self, span: Any = None, token: Any = None):
        """
        Initialize a trace span wrapper.

        :param span: OpenTelemetry span, or None when tracing is unavailable
        :param token: Context attachment token paired with the span
        """
        self._span = span
        self._token = token
        self._ended = False

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """
        Add non-null attributes to the span.

        :param attributes: Attribute names and values to record
        """
        if self._span is None:
            return
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(key, value)

    def end(self, error: BaseException | None = None) -> None:
        """
        Record an optional failure, detach the context, and finish the span.

        :param error: Exception associated with the operation, if any
        """
        if self._span is None or self._ended:
            return
        if error is not None:
            self._span.record_exception(error)
            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            self._span.set_attribute("error.type", type(error).__name__)
        otel_context.detach(self._token)
        self._span.end()
        self._ended = True


def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    client: bool = False,
) -> TraceSpan:
    """
    Start and activate a span when the OpenTelemetry API is installed.

    :param name: Span name
    :param attributes: Initial span attributes
    :param client: Whether the span represents a client operation
    :return: Lifecycle wrapper; always safe to use when tracing is unavailable
    """
    if not OTEL_API_AVAILABLE:
        return TraceSpan()

    _configure_from_environment()
    tracer = (
        _state.tracer_provider.get_tracer("guidellm")
        if _state.tracer_provider is not None
        else trace.get_tracer("guidellm")
    )
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT if client else SpanKind.INTERNAL,
        attributes={
            key: value for key, value in (attributes or {}).items() if value is not None
        },
    )
    token = otel_context.attach(trace.set_span_in_context(span))
    return TraceSpan(span, token)


def inject_trace_headers(
    headers: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """
    Inject the active W3C trace context into a copy of outbound headers.

    :param headers: Existing outbound headers
    :return: Headers with trace context, or the original empty value when unavailable
    """
    if not OTEL_API_AVAILABLE:
        return headers

    _configure_from_environment()
    carrier = dict(headers or {})
    propagate.inject(carrier)
    return carrier or None


def capture_trace_context() -> dict[str, str]:
    """
    Serialize the active trace context for transfer to a worker process.

    :return: Pickleable W3C trace-context carrier
    """
    if not OTEL_API_AVAILABLE:
        return {}

    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def activate_trace_context(carrier: dict[str, str] | None) -> Any:
    """
    Activate trace context received from another process.

    :param carrier: W3C trace-context carrier from the parent process
    :return: Context token to pass to :func:`deactivate_trace_context`
    """
    if not OTEL_API_AVAILABLE or not carrier:
        return None

    return otel_context.attach(propagate.extract(carrier))


def deactivate_trace_context(token: Any) -> None:
    """
    Detach a previously activated cross-process trace context.

    :param token: Token returned by :func:`activate_trace_context`
    """
    if OTEL_API_AVAILABLE and token is not None:
        otel_context.detach(token)


def shutdown_tracing() -> None:
    """Flush and shut down the OTLP provider owned by the current process."""
    if _state.tracer_provider is None or _state.configured_pid != os.getpid():
        return

    _state.tracer_provider.shutdown()
    _state.tracer_provider = None
    _state.configured_pid = None
