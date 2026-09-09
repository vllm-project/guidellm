"""Regression tests for non-streaming OpenAI tool-call validation ordering."""

from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import Mock, patch

import pytest
from pytest_httpx import HTTPXMock

from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.backends.openai.request_handlers import (
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
)
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)
from guidellm.schemas.backends import OpenAIHTTPBackendArgs
from tests.unit.testing_utils import async_timeout


def _make_backend(**kwargs) -> OpenAIHTTPBackend:
    """Create an OpenAIHTTPBackend from keyword arguments via BackendArgs."""
    return OpenAIHTTPBackend(OpenAIHTTPBackendArgs(**kwargs))


@pytest.fixture
def mock_request_handler():
    """Provide a mocked OpenAI request handler and its factory patch."""
    mock_handler = Mock(spec=OpenAIRequestHandler)
    mock_handler.format.return_value = GenerationRequestArguments(
        body={"model": "test-model"}
    )
    patch_context = patch.object(
        OpenAIRequestHandlerFactory,
        "create",
        return_value=mock_handler,
    )
    return mock_handler, patch_context


@pytest.mark.regression
@pytest.mark.asyncio
@async_timeout(10.0)
@pytest.mark.parametrize(
    ("behavior", "expected_error", "expected_yields"),
    [
        ("error_stop", ValueError, 0),
        ("ignore_stop", asyncio.CancelledError, 1),
    ],
)
async def test_non_streaming_missing_tool_call_stop_behavior(
    httpx_mock: HTTPXMock,
    mock_request_handler,
    behavior: Literal["ignore_stop", "error_stop"],
    expected_error: type[BaseException],
    expected_yields: int,
):
    """Missing tool calls match streaming stop behavior before termination.

    ## WRITTEN BY AI ##
    """
    httpx_mock.add_response(
        url="http://test/v1/chat/completions",
        json={"choices": [{"message": {"content": "no tool call"}}]},
    )
    backend = _make_backend(
        target="http://test",
        model="test-model",
        stream=False,
        validate_backend=False,
        request_format="/v1/chat/completions",
        tool_call_missing_behavior=behavior,
    )
    await backend.process_startup()
    request = GenerationRequest(
        columns={"text_column": ["call the tool"]},
        turn_type="client_tool_call",
    )
    request_info = RequestInfo(
        request_id="test-id",
        status="pending",
        scheduler_node_id=1,
        scheduler_process_id=1,
        scheduler_start_time=123.0,
        timings=RequestTimings(),
    )
    mock_handler, handler_patch = mock_request_handler
    mock_handler.compile_non_streaming.return_value = GenerationResponse(
        request_id="test-id",
        request_args="test args",
        text="no tool call",
    )
    yielded = []

    async def consume_response():
        async for item in backend.resolve(request, request_info):
            yielded.append(item)

    with handler_patch, pytest.raises(expected_error, match="tool call"):
        await consume_response()

    assert len(yielded) == expected_yields
