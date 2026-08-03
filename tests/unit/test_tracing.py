"""Tests for optional OpenTelemetry tracing helpers."""

from __future__ import annotations

import pickle
from multiprocessing import get_context
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from guidellm import tracing
from guidellm.tracing import (
    activate_trace_context,
    capture_trace_context,
    deactivate_trace_context,
    inject_trace_headers,
    shutdown_tracing,
    start_span,
)


def _read_trace_context_in_worker(carrier: dict[str, str], output_queue: Any) -> None:
    """Activate a carrier in a spawned process and return its span context."""
    context_token = activate_trace_context(carrier)
    try:
        span_context = trace.get_current_span().get_span_context()
        output_queue.put(
            (span_context.trace_id, span_context.span_id, span_context.is_remote)
        )
    finally:
        deactivate_trace_context(context_token)


@pytest.fixture(scope="module")
def span_exporter() -> InMemorySpanExporter:
    """Install an in-memory trace provider for this module."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.mark.sanity
def test_span_hierarchy_survives_process_context_transfer(
    span_exporter: InMemorySpanExporter,
) -> None:
    """Preserve run hierarchy through a pickleable carrier. ## WRITTEN BY AI ##"""
    run_span = start_span("guidellm.run", {"guidellm.run.id": "run-1"})
    benchmark_span = start_span(
        "guidellm.benchmark",
        {"guidellm.benchmark.id": "benchmark-1"},
    )
    carrier = pickle.loads(  # noqa: S301 - trusted local round-trip
        pickle.dumps(capture_trace_context())
    )
    benchmark_span.end()
    run_span.end()

    context_token = activate_trace_context(carrier)
    request_span = start_span(
        "gen_ai.request",
        {"guidellm.request.id": "request-1"},
        client=True,
    )
    headers = inject_trace_headers({"authorization": "secret"})
    request_span.end()
    deactivate_trace_context(context_token)

    assert headers is not None
    assert headers["authorization"] == "secret"
    assert headers["traceparent"].startswith("00-")
    spans = {span.name: span for span in span_exporter.get_finished_spans()[-3:]}
    run = spans["guidellm.run"]
    benchmark = spans["guidellm.benchmark"]
    request = spans["gen_ai.request"]
    assert benchmark.parent is not None
    assert benchmark.parent.span_id == run.context.span_id
    assert request.parent is not None
    assert request.parent.trace_id == run.context.trace_id
    assert request.parent.span_id == benchmark.context.span_id
    assert request.attributes["guidellm.request.id"] == "request-1"


@pytest.mark.regression
def test_span_records_failures(span_exporter: InMemorySpanExporter) -> None:
    """Mark failed operations with error details. ## WRITTEN BY AI ##"""
    request_span = start_span("gen_ai.request")
    request_span.end(ValueError("bad response"))

    span = span_exporter.get_finished_spans()[-1]
    assert span.status.is_ok is False
    assert span.attributes["error.type"] == "ValueError"
    assert span.events[-1].name == "exception"


@pytest.mark.regression
def test_context_reaches_spawned_worker_process(
    span_exporter: InMemorySpanExporter,
) -> None:
    """Activate benchmark context under the spawn start method. ## WRITTEN BY AI ##"""
    _ = span_exporter
    benchmark_span = start_span("guidellm.benchmark")
    expected = trace.get_current_span().get_span_context()
    carrier = capture_trace_context()
    mp_context = get_context("spawn")
    output_queue = mp_context.Queue()
    process = mp_context.Process(
        target=_read_trace_context_in_worker,
        args=(carrier, output_queue),
    )

    try:
        process.start()
        trace_id, parent_span_id, is_remote = output_queue.get(timeout=10)
    finally:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        output_queue.close()
        output_queue.join_thread()
        benchmark_span.end()

    assert process.exitcode == 0
    assert trace_id == expected.trace_id
    assert parent_span_id == expected.span_id
    assert is_remote is True


@pytest.mark.regression
def test_automatic_provider_is_recreated_for_each_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reuse an automatically configured forked provider. ## WRITTEN BY AI ##"""
    providers = [MagicMock(), MagicMock()]
    provider_factory = MagicMock(side_effect=providers)
    process_ids = iter([100, 100, 200])
    monkeypatch.setattr(tracing._state, "configured_pid", None)
    monkeypatch.setattr(tracing._state, "tracer_provider", None)
    monkeypatch.setattr(tracing, "TracerProvider", provider_factory)
    monkeypatch.setattr(tracing, "GrpcOTLPSpanExporter", MagicMock())
    monkeypatch.setattr(tracing, "BatchSpanProcessor", MagicMock())
    monkeypatch.setattr(tracing.os, "getpid", lambda: next(process_ids))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    tracing._configure_from_environment()
    first_provider = tracing._state.tracer_provider
    tracing._configure_from_environment()
    tracing._configure_from_environment()

    assert provider_factory.call_count == 2
    assert first_provider is providers[0]
    assert tracing._state.tracer_provider is providers[1]


@pytest.mark.regression
def test_shutdown_flushes_only_current_process_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flush the worker provider without touching inherited state.

    ## WRITTEN BY AI ##
    """
    current_provider = MagicMock()
    inherited_provider = MagicMock()
    monkeypatch.setattr(tracing.os, "getpid", lambda: 200)
    monkeypatch.setattr(tracing._state, "configured_pid", 200)
    monkeypatch.setattr(tracing._state, "tracer_provider", current_provider)

    shutdown_tracing()

    current_provider.shutdown.assert_called_once_with()
    assert tracing._state.tracer_provider is None

    monkeypatch.setattr(tracing._state, "configured_pid", 100)
    monkeypatch.setattr(tracing._state, "tracer_provider", inherited_provider)
    shutdown_tracing()
    inherited_provider.shutdown.assert_not_called()
