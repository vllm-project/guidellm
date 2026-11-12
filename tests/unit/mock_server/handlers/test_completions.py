from __future__ import annotations

import json

import pytest

from guidellm.backends.openai.request_handlers import TextCompletionsRequestHandler
from guidellm.mock_server.handlers.completions import CompletionsHandler
from guidellm.mock_server.models import CompletionsRequest
from guidellm.schemas.mock_server import MockServerConfig


class _RecordingStream:
    """Stands in for the Sanic stream the handler writes server-sent events to."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    async def write(self, data: str) -> None:
        self.lines.extend(line for line in data.split("\n") if line)


async def _stream_completion(max_tokens: int) -> list[str]:
    handler = CompletionsHandler(MockServerConfig(ttft_ms=0.0, itl_ms=0.0))
    request = CompletionsRequest(
        model="test-model",
        prompt="Hello",
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    stream_response = await handler._handle_stream(request)
    recorder = _RecordingStream()
    await stream_response.streaming_fn(recorder)
    return recorder.lines


class TestCompletionsHandlerStreaming:
    @pytest.mark.smoke
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_stream_chunks_use_choice_index_zero(self):
        """
        Every streamed chunk reports ``choices[0].index == 0``. The index names the
        choice, not the token position, and the mock server only ever returns one
        choice.

        ## WRITTEN BY AI ##
        """
        lines = await _stream_completion(max_tokens=4)

        chunks = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        indexes = [choice["index"] for chunk in chunks for choice in chunk["choices"]]

        assert len(indexes) == 5  # four token chunks and the finish chunk
        assert indexes == [0] * len(indexes)

    @pytest.mark.smoke
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_stream_is_accepted_by_the_completions_request_handler(self):
        """
        The benchmark client's ``/v1/completions`` handler consumes the mock
        server's multi-token stream without rejecting it as multiple choices, and
        sees one content iteration per streamed token.

        ## WRITTEN BY AI ##
        """
        lines = await _stream_completion(max_tokens=4)

        client_handler = TextCompletionsRequestHandler()
        iterations = [client_handler.add_streaming_line(line) for line in lines]

        assert iterations[-1] is None  # [DONE]
        assert sum(count for count in iterations if count) == 4
        assert client_handler.streaming_usage is not None
        assert client_handler.streaming_usage["completion_tokens"] == 4
