"""
Unit tests for OpenAI request handlers.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.backends.openai.request_handlers import (
    AudioRequestHandler,
    ChatCompletionsRequestHandler,
    EmbeddingsRequestHandler,
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
    OpenAIWSRequestHandlerFactory,
    PoolingRequestHandler,
    RealtimeTranscriptionWSRequestHandler,
    ResponsesRequestHandler,
    TextCompletionsRequestHandler,
    WSEventResult,
)
from guidellm.schemas import GenerationRequest, GenerationResponse, UsageMetrics
from guidellm.utils.registry import RegistryMixin


@pytest.fixture
def generation_request():
    """Create a basic GenerationRequest for testing."""
    return GenerationRequest(
        columns={"text_column": ["test prompt"]},
    )


class TestOpenAIRequestHandler:
    """Test cases for OpenAIRequestHandler protocol.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test OpenAIRequestHandler is a Protocol with correct methods.

        ### WRITTEN BY AI ###
        """
        # Verify it's a Protocol by checking its methods
        assert hasattr(OpenAIRequestHandler, "format")
        assert hasattr(OpenAIRequestHandler, "compile_non_streaming")
        assert hasattr(OpenAIRequestHandler, "add_streaming_line")
        assert hasattr(OpenAIRequestHandler, "compile_streaming")


class TestOpenAIRequestHandlerFactory:
    """Test cases for OpenAIRequestHandlerFactory.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test that OpenAIRequestHandlerFactory has correct inheritance.

        ### WRITTEN BY AI ###
        """
        assert issubclass(OpenAIRequestHandlerFactory, RegistryMixin)
        assert hasattr(OpenAIRequestHandlerFactory, "register")
        assert hasattr(OpenAIRequestHandlerFactory, "create")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("request_type", "handler_overrides", "expected_class"),
        [
            ("/v1/completions", None, TextCompletionsRequestHandler),
            ("/v1/chat/completions", None, ChatCompletionsRequestHandler),
            ("/v1/responses", None, ResponsesRequestHandler),
            ("/v1/audio/transcriptions", None, AudioRequestHandler),
            ("/v1/audio/translations", None, AudioRequestHandler),
            ("/pooling", None, PoolingRequestHandler),
            ("/v1/embeddings", None, EmbeddingsRequestHandler),
            (
                "/v1/completions",
                {"/v1/completions": ChatCompletionsRequestHandler},
                ChatCompletionsRequestHandler,
            ),
        ],
        ids=[
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/pooling",
            "/v1/embeddings",
            "override_text_completions",
        ],
    )
    def test_create(self, request_type, handler_overrides, expected_class):
        """Test create method with various request types and overrides.

        ### WRITTEN BY AI ###
        """
        handler = OpenAIRequestHandlerFactory.create(request_type, handler_overrides)
        assert isinstance(handler, expected_class)

    @pytest.mark.sanity
    def test_create_invalid_request_type(self):
        """Test create method with invalid request type.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError, match="No response handler registered"):
            OpenAIRequestHandlerFactory.create("invalid_type")


class TestRealtimeTranscriptionWSRequestHandler:
    """Realtime WebSocket path handler (``/v1/realtime``)."""

    @pytest.fixture(autouse=True)
    def _patch_pcm16_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Avoid torchcodec decode when format() PCM-encodes audio.

        ## WRITTEN BY AI ##
        """
        monkeypatch.setattr(
            "guidellm.backends.openai.request_handlers.pcm16_append_b64_chunks",
            lambda *a, **k: ["YWFhYQ=="],
        )

    def test_extract_single_audio_requires_one_column(self) -> None:
        handler = RealtimeTranscriptionWSRequestHandler()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": []},
        )
        with pytest.raises(ValueError, match="exactly one audio_column"):
            handler.extract_single_audio(req)

    def test_format_builds_body(self) -> None:
        """format() validates audio and attaches PCM chunks to body.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        args = handler.format(
            req,
            model="m1",
            websocket_path="/v1/realtime",
            chunk_samples=1600,
        )
        assert args.body == {
            "model": "m1",
            "websocket_path": "/v1/realtime",
            "chunk_samples": 1600,
            "audio_chunks": ["YWFhYQ=="],
        }

    def test_ws_factory_create(self) -> None:
        """OpenAIWSRequestHandlerFactory returns RealtimeTranscriptionWSRequestHandler.

        ## WRITTEN BY AI ##
        """
        handler = OpenAIWSRequestHandlerFactory.create("/v1/realtime")
        assert isinstance(handler, RealtimeTranscriptionWSRequestHandler)

    def test_http_factory_rejects_realtime_path(self) -> None:
        """Realtime path is registered on the WS factory only.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="No response handler registered"):
            OpenAIRequestHandlerFactory.create("/v1/realtime")

    def test_ws_factory_rejects_unknown_path(self) -> None:
        """Unknown WS paths raise ValueError from the WS factory.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="No WebSocket handler registered"):
            OpenAIWSRequestHandlerFactory.create("/v1/unknown")

    def test_add_streaming_event_delta_returns_content(self) -> None:
        """transcription.delta with content returns CONTENT.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        result = handler.add_streaming_event(
            {"type": "transcription.delta", "delta": "hi"}
        )
        assert result.kind is WSEventResult.CONTENT
        assert result.content_tokens == 1
        assert handler.streaming_text == "hi"

    def test_add_streaming_event_empty_delta_returns_request_iteration(self) -> None:
        """Empty transcription.delta returns REQUEST_ITERATION.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        result = handler.add_streaming_event(
            {"type": "transcription.delta", "delta": ""}
        )
        assert result.kind is WSEventResult.REQUEST_ITERATION
        assert result.content_tokens == 0
        assert handler.streaming_text == ""

    def test_add_streaming_event_done_returns_stream_end(self) -> None:
        """transcription.done returns STREAM_END and stores usage.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        handler.add_streaming_event({"type": "transcription.delta", "delta": "a"})
        result = handler.add_streaming_event(
            {
                "type": "transcription.done",
                "text": "hello",
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )
        assert result.kind is WSEventResult.STREAM_END
        assert handler._streaming_usage == {
            "prompt_tokens": 5,
            "completion_tokens": 1,
        }
        assert handler.streaming_text == "hello"

    def test_add_streaming_event_done_without_text_joins_deltas(self) -> None:
        """transcription.done without text keeps accumulated deltas.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        handler.add_streaming_event({"type": "transcription.delta", "delta": "hel"})
        handler.add_streaming_event({"type": "transcription.delta", "delta": "lo"})
        handler.add_streaming_event(
            {
                "type": "transcription.done",
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )
        assert handler.streaming_text == "hello"

    def test_add_streaming_event_done_only_sets_text(self) -> None:
        """Single transcription.done event sets text without prior deltas.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        handler.add_streaming_event(
            {
                "type": "transcription.done",
                "text": "only",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )
        assert handler.streaming_text == "only"

    def test_add_streaming_event_error_raises(self) -> None:
        """error events raise RuntimeError.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        with pytest.raises(RuntimeError, match="401: auth failed"):
            handler.add_streaming_event(
                {"type": "error", "error": {"message": "auth failed", "code": "401"}}
            )

    @pytest.mark.parametrize(
        ("error_payload", "expected"),
        [
            (None, "WebSocket error"),
            ("", "WebSocket error"),
            ("plain failure", "plain failure"),
            ({"code": "500"}, "500"),
        ],
        ids=["none", "empty", "plain_string", "code_only"],
    )
    def test_add_streaming_event_error_formats_payload(
        self, error_payload: object, expected: str
    ) -> None:
        """format_ws_error branches are surfaced through error events.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        with pytest.raises(RuntimeError, match=expected):
            handler.add_streaming_event({"type": "error", "error": error_payload})

    def test_add_streaming_event_unknown_returns_ignored(self) -> None:
        """Unrecognized event types return IGNORED.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        result = handler.add_streaming_event({"type": "noise.event"})
        assert result.kind is WSEventResult.IGNORED
        assert result.content_tokens == 0
        assert handler.streaming_text == ""
        assert handler._streaming_usage is None

    def test_compile_streaming_partial_without_done(self) -> None:
        """compile_streaming works with deltas only (cancel/partial path).

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        args = handler.format(
            req,
            model="m1",
            websocket_path="/v1/realtime",
            chunk_samples=1600,
        )
        handler.add_streaming_event({"type": "transcription.delta", "delta": "partial"})
        resp = handler.compile_streaming(req, args)
        assert resp.text == "partial"
        assert resp.request_id == "r1"
        assert handler._streaming_usage is None
        assert resp.input_metrics.audio_tokens is None
        assert resp.output_metrics.text_tokens is None

    def test_compile_streaming_builds_response(self) -> None:
        """compile_streaming assembles text and metrics.

        ## WRITTEN BY AI ##
        """
        handler = RealtimeTranscriptionWSRequestHandler()
        req = GenerationRequest(
            request_id="r1",
            columns={"audio_column": [{"audio": b"x"}]},
        )
        args = handler.format(
            req,
            model="m1",
            websocket_path="/v1/realtime",
            chunk_samples=1600,
        )
        handler.add_streaming_event({"type": "transcription.delta", "delta": "hi"})
        handler.add_streaming_event(
            {
                "type": "transcription.done",
                "text": "hi",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        )
        resp = handler.compile_streaming(req, args)
        assert resp.text == "hi"
        assert resp.request_id == "r1"
        assert resp.input_metrics.audio_tokens == 5
        assert resp.output_metrics.text_tokens == 1
        assert '"model":"m1"' in resp.request_args
        assert "audio_chunks" not in resp.request_args


class TestTextCompletionsRequestHandler:
    """Test cases for TextCompletionsRequestHandler.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of TextCompletionsRequestHandler.

        ### WRITTEN BY AI ###
        """
        return TextCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test TextCompletionsRequestHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = TextCompletionsRequestHandler()
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "extract_line_data")
        assert hasattr(handler, "extract_choices_and_usage")
        assert hasattr(handler, "extract_metrics")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test TextCompletionsRequestHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, TextCompletionsRequestHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    # Request formatting tests
    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, model="test-model")

        assert result.body["model"] == "test-model"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with streaming enabled.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, stream=True)

        assert result.stream is True
        assert result.body["stream"] is True
        assert "stream_options" in result.body
        assert result.body["stream_options"]["include_usage"] is True
        assert result.body["stream_options"]["continuous_usage_stats"] is True

    @pytest.mark.sanity
    def test_format_output_tokens(self, valid_instances):
        """Test format method with output_tokens handling.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            output_metrics=UsageMetrics(text_tokens=100),
        )

        result = instance.format(data)

        assert result.body["max_tokens"] == 100
        assert result.body["stop"] is None
        assert result.body["ignore_eos"] is True

    @pytest.mark.sanity
    def test_format_max_tokens_kwarg(self, valid_instances):
        """Test format method with max_tokens keyword argument.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, max_tokens=50)

        assert result.body["max_tokens"] == 50

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()
        extras = {"body": {"temperature": 0.7, "top_p": 0.9}}

        result = instance.format(data, extras=extras)

        assert result.body.get("temperature") == 0.7
        assert result.body.get("top_p") == 0.9

    @pytest.mark.sanity
    def test_format_prefix_and_text(self, valid_instances):
        """Test format method with prefix and text columns.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "prefix_column": ["prefix1", "prefix2"],
                "text_column": ["Hello", "world"],
            },
        )

        result = instance.format(data)

        assert result.body["prompt"] == "prefix1 prefix2 Hello world"

    @pytest.mark.sanity
    def test_format_ignore_eos(self, valid_instances):
        """Test format method sets ignore_eos when output tokens specified.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            output_metrics=UsageMetrics(text_tokens=100),
        )

        result = instance.format(data)

        assert result.body["ignore_eos"] is True

    # Response handling tests
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "response",
            "expected_text",
            "expected_input_tokens",
            "expected_output_tokens",
        ),
        [
            (
                {
                    "choices": [{"text": "Hello, world!"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
                "Hello, world!",
                5,
                3,
            ),
            (
                {
                    "choices": [{"text": "Test response"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"prompt_tokens": 10},
                        "completion_tokens_details": {"completion_tokens": 5},
                    },
                },
                "Test response",
                10,
                5,
            ),
            ({"choices": [{"text": ""}], "usage": {}}, "", None, None),
            ({"choices": [], "usage": {}}, "", None, None),
            ({}, "", None, None),
        ],
    )
    def test_non_streaming(
        self,
        valid_instances,
        generation_request,
        response,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """Test compile_non_streaming method.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == expected_text
        assert result.input_metrics.text_tokens == expected_input_tokens
        assert result.output_metrics.text_tokens == expected_output_tokens
        assert result.output_metrics.text_words == len(expected_text.split())
        assert result.output_metrics.text_characters == len(expected_text)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "lines",
            "expected_text",
            "expected_input_tokens",
            "expected_output_tokens",
        ),
        [
            (
                [
                    'data: {"choices": [{"text": "Hello"}], "usage": {}}',
                    "",
                    'data: {"choices": [{"text": ", "}], "usage": {}}',
                    (
                        'data: {"choices": [{"text": "world!"}], '
                        '"usage": {"prompt_tokens": 5, "completion_tokens": 3}}'
                    ),
                    "data: [DONE]",
                ],
                "Hello, world!",
                5,
                3,
            ),
            (
                [
                    'data: {"choices": [{"text": "Test"}], "usage": {}}',
                    "",
                    "data: [DONE]",
                ],
                "Test",
                None,
                None,
            ),
            (["", "data: [DONE]"], "", None, None),
        ],
    )
    def test_streaming(
        self,
        valid_instances,
        generation_request,
        lines,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """Test streaming with add_streaming_line and compile_streaming.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
            elif result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == expected_text
        assert response.input_metrics.text_tokens == expected_input_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens
        assert response.output_metrics.text_words == len(expected_text.split())
        assert response.output_metrics.text_characters == len(expected_text)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("line", "expected_output"),
        [
            ('data: {"choices": [{"text": "Test"}]}', {"choices": [{"text": "Test"}]}),
            ("data: [DONE]", None),
            ("", {}),
            ("invalid line", {}),
            ('data: {"test": "value"}', {"test": "value"}),
        ],
    )
    def test_extract_line_data(self, valid_instances, line, expected_output):
        """Test extract_line_data method.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        result = instance.extract_line_data(line)
        assert result == expected_output

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "line",
        [
            'data: {"error": {"message": "TimeoutError", "type": '
            '"GatewayTimeout", "code": 504}}',
            'data: {"error": "boom"}',
        ],
    )
    def test_extract_line_data_streaming_error(self, valid_instances, line):
        """Streaming SSE error payloads must surface as request failures.

        Some OpenAI-compatible servers return HTTP 200 and report errors via
        the stream body (e.g. vLLM's ``create_streaming_error_response``)
        rather than an HTTP status, which would otherwise be recorded as a
        successful empty generation.
        """
        instance = valid_instances
        with pytest.raises(ValueError, match="Streaming response returned an error"):
            instance.extract_line_data(line)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("response", "expected_choices", "expected_usage"),
        [
            (
                {"choices": [{"text": "Hello"}], "usage": {"prompt_tokens": 5}},
                [{"text": "Hello"}],
                {"prompt_tokens": 5},
            ),
            (
                {"choices": [], "usage": {}},
                [],
                {},
            ),
            (
                {},
                [],
                {},
            ),
        ],
    )
    def test_extract_choices_and_usage(
        self, valid_instances, response, expected_choices, expected_usage
    ):
        """Test extract_choices_and_usage method.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        choices, usage = instance.extract_choices_and_usage(response)
        assert choices == expected_choices
        assert usage == expected_usage

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("usage", "text", "expected_input_tokens", "expected_output_tokens"),
        [
            (
                {"prompt_tokens": 5, "completion_tokens": 3},
                "Hello world",
                5,
                3,
            ),
            (
                {
                    "prompt_tokens_details": {"prompt_tokens": 10, "image_tokens": 2},
                    "completion_tokens_details": {"completion_tokens": 5},
                },
                "Test response",
                10,
                5,
            ),
            (
                None,
                "Hello world",
                None,
                None,
            ),
            (
                {},
                "",
                None,
                None,
            ),
        ],
    )
    def test_extract_metrics(
        self,
        valid_instances,
        usage,
        text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """Test extract_metrics method.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        input_metrics, output_metrics = instance.extract_metrics(usage, text)

        assert input_metrics.text_tokens == expected_input_tokens
        assert output_metrics.text_tokens == expected_output_tokens
        assert output_metrics.text_words == (len(text.split()) if text else 0)
        assert output_metrics.text_characters == len(text)


class TestChatCompletionsRequestHandler:
    """Test cases for ChatCompletionsRequestHandler.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of ChatCompletionsRequestHandler.

        ### WRITTEN BY AI ###
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ChatCompletionsRequestHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = ChatCompletionsRequestHandler()
        assert issubclass(ChatCompletionsRequestHandler, TextCompletionsRequestHandler)
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ChatCompletionsRequestHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, ChatCompletionsRequestHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    # Request formatting tests
    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)
        assert "messages" in result.body
        assert isinstance(result.body["messages"], list)

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, model="gpt-4")

        assert result.body["model"] == "gpt-4"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with streaming enabled.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, stream=True)

        assert result.stream is True
        assert result.body["stream"] is True
        assert "stream_options" in result.body
        assert result.body["stream_options"]["include_usage"] is True
        assert result.body["stream_options"]["continuous_usage_stats"] is True

    @pytest.mark.sanity
    def test_format_output_tokens(self, valid_instances):
        """Test format method with max_completion_tokens.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            output_metrics=UsageMetrics(text_tokens=100),
        )

        result = instance.format(data)

        assert result.body["max_completion_tokens"] == 100
        assert result.body["stop"] is None
        assert result.body["ignore_eos"] is True

    @pytest.mark.sanity
    def test_format_max_tokens_kwarg(self, valid_instances):
        """Test format method with max_tokens keyword argument.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, max_tokens=50)

        assert result.body["max_completion_tokens"] == 50

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()
        extras = {"body": {"temperature": 0.5, "top_k": 40}}

        result = instance.format(data, extras=extras)

        assert result.body.get("temperature") == 0.5
        assert result.body.get("top_k") == 40

    @pytest.mark.sanity
    def test_format_messages_text(self, valid_instances):
        """Test format method with text messages.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello", "How are you?"]},
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "user"
        assert len(result.body["messages"][0]["content"]) == 2
        assert result.body["messages"][0]["content"][0]["type"] == "text"
        assert result.body["messages"][0]["content"][0]["text"] == "Hello"
        assert result.body["messages"][0]["content"][1]["type"] == "text"
        assert result.body["messages"][0]["content"][1]["text"] == "How are you?"

    @pytest.mark.sanity
    def test_format_messages_prefix(self, valid_instances):
        """Test format method with prefix as system message.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"prefix_column": ["You are a helpful assistant."]},
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "system"
        assert result.body["messages"][0]["content"] == "You are a helpful assistant."

    @pytest.mark.sanity
    def test_format_messages_image(self, valid_instances):
        """Test format method with image_url content.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "image_column": [
                    {"image": "https://example.com/image.jpg"},
                ]
            },
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "user"
        assert result.body["messages"][0]["content"][0]["type"] == "image_url"
        assert (
            result.body["messages"][0]["content"][0]["image_url"]["url"]
            == "https://example.com/image.jpg"
        )

    @pytest.mark.sanity
    def test_format_messages_video(self, valid_instances):
        """Test format method with video_url content.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "video_column": [
                    {"video": "https://example.com/video.mp4"},
                ]
            },
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "user"
        assert result.body["messages"][0]["content"][0]["type"] == "video_url"
        assert (
            result.body["messages"][0]["content"][0]["video_url"]["url"]
            == "https://example.com/video.mp4"
        )

    @pytest.mark.sanity
    def test_format_messages_audio(self, valid_instances):
        """Test format method with input_audio content.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {"audio": b"base64data", "format": "wav"},
                ]
            },
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "user"
        assert result.body["messages"][0]["content"][0]["type"] == "input_audio"
        assert (
            result.body["messages"][0]["content"][0]["input_audio"]["data"]
            == "YmFzZTY0ZGF0YQ=="
        )
        assert (
            result.body["messages"][0]["content"][0]["input_audio"]["format"] == "wav"
        )

    @pytest.mark.regression
    def test_format_multimodal(self, valid_instances):
        """Test format method combining multiple modalities.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "prefix_column": ["You are a helpful assistant."],
                "text_column": ["Describe this image"],
                "image_column": [{"image": "https://example.com/image.jpg"}],
            },
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 2
        # System message from prefix
        assert result.body["messages"][0]["role"] == "system"
        assert result.body["messages"][0]["content"] == "You are a helpful assistant."
        # User message with interleaved text and image content
        assert result.body["messages"][1]["role"] == "user"
        assert len(result.body["messages"][1]["content"]) == 2
        # roundrobin interleaves: text first, then image
        assert result.body["messages"][1]["content"][0]["type"] == "text"
        assert result.body["messages"][1]["content"][0]["text"] == "Describe this image"
        assert result.body["messages"][1]["content"][1]["type"] == "image_url"
        assert (
            result.body["messages"][1]["content"][1]["image_url"]["url"]
            == "https://example.com/image.jpg"
        )

    # Response handling tests
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "response",
            "expected_text",
            "expected_input_tokens",
            "expected_output_tokens",
        ),
        [
            (
                {
                    "choices": [{"message": {"content": "Hello, world!"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
                "Hello, world!",
                5,
                3,
            ),
            (
                {
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                    },
                },
                "Test response",
                10,
                5,
            ),
            (
                {"choices": [{"message": {"content": ""}}], "usage": {}},
                "",
                None,
                None,
            ),
            (
                {"choices": [], "usage": {}},
                "",
                None,
                None,
            ),
        ],
    )
    def test_non_streaming(
        self,
        valid_instances,
        generation_request,
        response,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """Test compile_non_streaming method for chat completions.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == expected_text
        assert result.input_metrics.text_tokens == expected_input_tokens
        assert result.output_metrics.text_tokens == expected_output_tokens

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("lines", "expected_text", "expected_input_tokens", "expected_output_tokens"),
        [
            (
                [
                    'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {}}',
                    "",
                    'data: {"choices": [{"delta": {"content": ", "}}], "usage": {}}',
                    (
                        'data: {"choices": [{"delta": {"content": "world!"}}], '
                        '"usage": {"prompt_tokens": 5, "completion_tokens": 3}}'
                    ),
                    "data: [DONE]",
                ],
                "Hello, world!",
                5,
                3,
            ),
            (
                [
                    'data: {"choices": [{"delta": {"content": "Test"}}], "usage": {}}',
                    "",
                    "data: [DONE]",
                ],
                "Test",
                None,
                None,
            ),
            (
                ["", "data: [DONE]"],
                "",
                None,
                None,
            ),
        ],
    )
    def test_streaming(
        self,
        valid_instances,
        generation_request,
        lines,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """Test streaming pathway for chat completions.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
            elif result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == expected_text
        assert response.input_metrics.text_tokens == expected_input_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens

    @pytest.mark.sanity
    def test_streaming_reasoning_tokens(self, valid_instances, generation_request):
        """Test that reasoning tokens are properly detected for TTFT measurement.

        Reasoning-capable models (e.g., DeepSeek-R1, o1) emit delta.reasoning
        before delta.content. This test verifies that the first reasoning token
        triggers the updated flag, ensuring accurate TTFT measurement.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            # First chunk has reasoning token
            (
                'data: {"id": "chatcmpl-123", "choices": '
                '[{"index": 0, "delta": {"reasoning": "Okay"}}], "usage": {}}'
            ),
            # More reasoning tokens
            'data: {"choices": [{"delta": {"reasoning": ", let me"}}], "usage": {}}',
            'data: {"choices": [{"delta": {"reasoning": " think..."}}], "usage": {}}',
            # Finally content tokens
            'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {}}',
            (
                'data: {"choices": [{"delta": {"content": " world!"}}], '
                '"usage": {"prompt_tokens": 5, "completion_tokens": 10}}'
            ),
            "data: [DONE]",
        ]

        updated_count = 0
        first_update_on_line = None
        for idx, line in enumerate(lines):
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
                if first_update_on_line is None:
                    first_update_on_line = idx
            elif result is None:
                break

        # Verify that the first update happened on the first reasoning token (line 0)
        assert first_update_on_line == 0, (
            f"Expected first token detection on line 0 (reasoning token), "
            f"but got {first_update_on_line}"
        )

        # Verify all chunks with content were counted (5 lines with tokens)
        assert updated_count == 5

        response = instance.compile_streaming(generation_request, arguments)
        # Reasoning tokens should NOT appear in response.text; only content does
        assert "Okay" not in response.text
        assert "let me think..." not in response.text
        assert response.text == "Hello world!"
        assert response.input_metrics.text_tokens == 5
        assert response.output_metrics.text_tokens == 10

    @pytest.mark.sanity
    def test_streaming_both_reasoning_and_content_in_same_chunk(
        self, valid_instances, generation_request
    ):
        """Test handling chunks with both reasoning and content fields.

        Edge case: verify that if a chunk contains both delta.reasoning
        and delta.content, reasoning triggers the update flag but only
        content is captured in the response text.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            # Chunk with both reasoning and content (edge case)
            (
                'data: {"choices": [{"delta": '
                '{"reasoning": "Let me think...", "content": "Answer: "}}], '
                '"usage": {}}'
            ),
            'data: {"choices": [{"delta": {"content": "42"}}], "usage": {}}',
            "data: [DONE]",
        ]

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break
            if result > 0:
                updated_count += 1

        # First chunk has both reasoning and content (counts as 1 iteration)
        # Second chunk has content only (counts as 1 iteration)
        assert updated_count == 2

        response = instance.compile_streaming(generation_request, arguments)
        # Reasoning text should NOT appear; only content is captured
        assert "Let me think..." not in response.text
        assert response.text == "Answer: 42"

    @pytest.mark.sanity
    def test_streaming_captures_reasoning_text(
        self, valid_instances, generation_request
    ):
        """
        Verify reasoning text is accumulated on response.reasoning_text
        during streaming, separate from response.text.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            'data: {"choices": [{"delta": {"reasoning": "Step 1. "}}], "usage": {}}',
            'data: {"choices": [{"delta": {"reasoning": "Step 2."}}], "usage": {}}',
            'data: {"choices": [{"delta": {"content": "Final answer"}}], "usage": {}}',
            "data: [DONE]",
        ]
        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == "Final answer"
        assert response.reasoning_text == "Step 1. Step 2."

    @pytest.mark.sanity
    def test_non_streaming_captures_reasoning_text(
        self, valid_instances, generation_request
    ):
        """
        Verify compile_non_streaming extracts reasoning from the message.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        api_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning": "Let me think step by step.",
                        "content": "The answer is 42.",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
        }
        response = instance.compile_non_streaming(
            generation_request, arguments, api_response
        )
        assert response.text == "The answer is 42."
        assert response.reasoning_text == "Let me think step by step."

    @pytest.mark.sanity
    def test_format_excludes_reasoning_from_history_by_default(self, valid_instances):
        """
        By default (multiturn_reasoning=False), reasoning_text
        should not appear in the assistant message content.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="4",
            reasoning_text="Let me add 2 and 2.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
        )

        # Find the assistant message in the history
        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "4"
        assert "Let me add" not in assistant_msgs[0]["content"]

    @pytest.mark.sanity
    def test_format_includes_reasoning_in_history_when_enabled(self, valid_instances):
        """
        When multiturn_reasoning=True, reasoning_text should
        be wrapped in <think> tags and prepended to the assistant message content.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="4",
            reasoning_text="Let me add 2 and 2.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=True,
        )

        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "<think>Let me add 2 and 2.</think>4"

    @pytest.mark.sanity
    def test_format_includes_reasoning_only_response_in_history(self, valid_instances):
        """
        When multiturn_reasoning=True and the prior response has
        reasoning_text but no regular text content, an assistant message
        should still be injected containing the reasoning.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="",
            reasoning_text="Let me think about this carefully.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=True,
        )

        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert (
            assistant_msgs[0]["content"]
            == "<think>Let me think about this carefully.</think>"
        )

    @pytest.mark.sanity
    def test_format_drops_reasoning_only_response_from_history_by_default(
        self, valid_instances
    ):
        """
        When multiturn_reasoning=False (default) and the prior
        response has reasoning_text but empty text content, an
        assistant message with empty content should still be included
        to preserve alternating user/assistant structure.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="",
            reasoning_text="Let me think about this carefully.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
        )

        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == ""

    @pytest.mark.sanity
    def test_format_preserves_empty_assistant_in_multi_turn_history(
        self, valid_instances
    ):
        """
        When a prior turn produced an empty text response (e.g. reasoning
        model exhausted token budget), the assistant message must still appear
        in the history to prevent consecutive user messages.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        turn1_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        turn1_response = GenerationResponse(
            request_id="t1",
            request_args=None,
            text="Hi there!",
        )
        turn2_request = GenerationRequest(
            columns={"text_column": ["Tell me more"]},
        )
        turn2_response = GenerationResponse(
            request_id="t2",
            request_args=None,
            text="",
            reasoning_text="Let me think about this...",
        )

        current_request = GenerationRequest(
            columns={"text_column": ["Continue"]},
        )
        result = instance.format(
            current_request,
            history=[
                (turn1_request, turn1_response),
                (turn2_request, turn2_response),
            ],
        )

        messages = result.body["messages"]
        roles = [m["role"] for m in messages]

        # Verify no consecutive user messages
        for j in range(len(roles) - 1):
            assert not (roles[j] == "user" and roles[j + 1] == "user"), (
                f"Consecutive user messages at indices {j} and {j + 1}: {roles}"
            )

        # Verify the empty assistant response is present
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["content"] == "Hi there!"
        assert assistant_msgs[1]["content"] == ""

    @pytest.mark.sanity
    def test_format_includes_reasoning_with_custom_template(self, valid_instances):
        """
        When multiturn_reasoning is a custom format string, reasoning_text
        should be wrapped using that template.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="4",
            reasoning_text="Let me add 2 and 2.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        template = "Here is my thought process:{reasoning}Here is my response:"
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=template,
        )

        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        expected = "Here is my thought process:Let me add 2 and 2.Here is my response:4"
        assert assistant_msgs[0]["content"] == expected

    @pytest.mark.sanity
    def test_format_includes_reasoning_raw_template(self, valid_instances):
        """
        When multiturn_reasoning='{reasoning}', reasoning_text should be
        prepended with no extra delimiters (raw mode).

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="4",
            reasoning_text="Let me add 2 and 2.",
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning="{reasoning}",
        )

        assistant_msgs = [
            m for m in result.body["messages"] if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Let me add 2 and 2.4"

    @pytest.mark.sanity
    def test_last_iteration_had_content_reasoning_then_content(
        self, valid_instances, generation_request
    ):
        """
        Verify last_iteration_had_content tracks content vs reasoning deltas.

        Reasoning-only iterations set the flag to False; content sets it True.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        instance.format(generation_request)

        # Initial state before any streaming
        assert instance.last_iteration_had_content is False

        # Reasoning-only delta
        instance.add_streaming_line(
            'data: {"choices": [{"delta": {"reasoning": "thinking..."}}], "usage": {}}'
        )
        assert instance.last_iteration_had_content is False

        # Content delta
        instance.add_streaming_line(
            'data: {"choices": [{"delta": {"content": "Hello"}}], "usage": {}}'
        )
        assert instance.last_iteration_had_content is True

    @pytest.mark.sanity
    def test_last_iteration_had_content_tool_call(
        self, valid_instances, generation_request
    ):
        """
        Verify tool call deltas set last_iteration_had_content to True.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        instance.format(generation_request)

        tc_line = (
            'data: {"choices": [{"delta": {"tool_calls": '
            '[{"index": 0, "id": "call_1", "type": "function", '
            '"function": {"name": "foo", "arguments": ""}}]}}], "usage": {}}'
        )
        instance.add_streaming_line(tc_line)
        assert instance.last_iteration_had_content is True

    # Tool call response handling tests

    @pytest.mark.sanity
    def test_non_streaming_tool_calls(self, valid_instances, generation_request):
        """
        Test compile_non_streaming extracts tool_calls when content is null.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA"}',
                },
            }
        ]
        response = {
            "id": "chatcmpl-xyz",
            "choices": [
                {
                    "message": {"content": None, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text is None
        assert result.input_metrics.text_tokens == 10
        assert result.output_metrics.text_tokens == 15
        assert result.output_metrics.text_words is None
        assert result.output_metrics.text_characters is None
        assert result.output_metrics.tool_call_tokens == 15
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_non_streaming_tool_calls_content_preferred(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming with both content and tool_calls. Text comes from
        content; tool_call_count is set, tool_call_tokens is None, and
        mixed_content_tool_tokens equals the completion total because the API does
        not split completion_tokens between natural language text and tool JSON.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "choices": [
                {
                    "message": {
                        "content": "I will call the function.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fn",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == "I will call the function."
        assert result.output_metrics.tool_call_tokens is None
        assert result.output_metrics.mixed_content_tool_tokens == 8
        assert result.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_non_streaming_multiple_tool_calls(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming with multiple parallel tool calls.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "SF"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": '{"timezone": "PST"}',
                },
            },
        ]
        response = {
            "choices": [
                {
                    "message": {"content": None, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 20},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text is None
        assert result.output_metrics.text_tokens == 20
        assert result.output_metrics.text_words is None
        assert result.output_metrics.text_characters is None
        assert result.output_metrics.tool_call_tokens == 20
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count == 2

    @pytest.mark.sanity
    def test_non_streaming_no_tool_calls_unchanged(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming with normal text response is unchanged.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == "Hello!"
        assert result.output_metrics.tool_call_tokens is None
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count is None

    @pytest.mark.sanity
    def test_streaming_tool_calls(self, valid_instances, generation_request):
        """
        Test streaming accumulates tool_calls deltas and sets tool call metrics.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            (
                'data: {"id": "chatcmpl-1", "choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "id": "call_abc", "type": "function", '
                '"function": {"name": "get_weather", "arguments": ""}}]}}], '
                '"usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "function": {"arguments": "{\\"loc"}}]}}], '
                '"usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "function": {"arguments": "ation\\": \\"SF\\"}"}}]}}], '
                '"usage": {"prompt_tokens": 10, "completion_tokens": 12}}'
            ),
            "data: [DONE]",
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text is None
        assert response.input_metrics.text_tokens == 10
        assert response.output_metrics.text_tokens == 12
        assert response.output_metrics.text_words is None
        assert response.output_metrics.text_characters is None
        assert response.output_metrics.tool_call_tokens == 12
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_streaming_multiple_tool_calls(self, valid_instances, generation_request):
        """
        Test streaming with multiple parallel tool calls on different indices.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            (
                'data: {"id": "chatcmpl-2", "choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "id": "call_1", "type": "function", '
                '"function": {"name": "fn_a", "arguments": ""}}]}}], "usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 1, "id": "call_2", "type": "function", '
                '"function": {"name": "fn_b", "arguments": ""}}]}}], "usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "function": {"arguments": "{\\"x\\": 1}"}}]}}], '
                '"usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 1, "function": {"arguments": "{\\"y\\": 2}"}}]}}], '
                '"usage": {"prompt_tokens": 8, "completion_tokens": 18}}'
            ),
            "data: [DONE]",
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text is None
        assert response.output_metrics.text_words is None
        assert response.output_metrics.text_characters is None
        assert response.output_metrics.tool_call_tokens == 18
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count == 2

    @pytest.mark.sanity
    def test_streaming_text_preferred_over_tool_calls(
        self, valid_instances, generation_request
    ):
        """
        Test streaming when both content and tool_calls deltas appear: final text
        is concatenated content; tool_call_count is set, tool_call_tokens is None,
        and mixed_content_tool_tokens equals the completion total because the API
        does not split completion_tokens between natural language text and tool JSON.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            (
                'data: {"id": "chatcmpl-3", "choices": [{"delta": '
                '{"content": "Some text"}}], "usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"tool_calls": '
                '[{"index": 0, "id": "call_x", "type": "function", '
                '"function": {"name": "fn", "arguments": "{}"}}]}}], '
                '"usage": {"prompt_tokens": 2, "completion_tokens": 8}}'
            ),
            "data: [DONE]",
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text == "Some text"
        assert response.output_metrics.tool_call_tokens is None
        assert response.output_metrics.mixed_content_tool_tokens == 8
        assert response.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_streaming_no_tool_calls_unchanged(
        self, valid_instances, generation_request
    ):
        """
        Test that normal text streaming is unaffected by tool call support.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            (
                'data: {"id": "chatcmpl-4", '
                '"choices": [{"delta": {"content": "Hi"}}], '
                '"usage": {}}'
            ),
            (
                'data: {"choices": [{"delta": {"content": " there"}}], '
                '"usage": {"prompt_tokens": 3, "completion_tokens": 2}}'
            ),
            "data: [DONE]",
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text == "Hi there"
        assert response.input_metrics.text_tokens == 3
        assert response.output_metrics.text_tokens == 2
        assert response.output_metrics.tool_call_tokens is None
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count is None

    @pytest.mark.smoke
    def test_initialization_has_streaming_tool_calls(self, valid_instances):
        """
        Test ChatCompletionsRequestHandler initializes streaming_tool_calls.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        assert hasattr(instance, "streaming_tool_calls")
        assert instance.streaming_tool_calls == {}

    @pytest.mark.sanity
    def test_format_strips_tool_choice_without_tools(self, valid_instances):
        """
        Test that tool_choice from extras is stripped when no tools are present
        in the request body.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["test prompt"]},
            turn_type="standard",
        )

        result = instance.format(data, extras={"body": {"tool_choice": "required"}})

        assert "tool_choice" not in result.body
        assert "tools" not in result.body


class TestAudioRequestHandler:
    """Test cases for AudioRequestHandler.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of AudioRequestHandler.

        ### WRITTEN BY AI ###
        """
        return AudioRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test AudioRequestHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = AudioRequestHandler()
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "extract_metrics")
        assert hasattr(handler, "streaming_buffer")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test AudioRequestHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, AudioRequestHandler)
        assert isinstance(instance.streaming_buffer, bytearray)
        assert len(instance.streaming_buffer) == 0
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    # Request formatting tests
    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal audio data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        result = instance.format(data)

        assert result.body is not None
        assert result.files is not None
        assert "file" in result.files

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        result = instance.format(data, model="whisper-1")

        assert result.body["model"] == "whisper-1"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with flattened stream options.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        result = instance.format(data, stream=True)

        assert result.stream is True
        assert result.body["stream"] is True
        # Audio endpoints use flattened stream options
        assert result.body["stream_include_usage"] is True
        assert result.body["stream_continuous_usage_stats"] is True

    @pytest.mark.sanity
    def test_format_file_upload(self, valid_instances):
        """Test format method creates correct file tuple.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        audio_data = b"fake_audio_bytes"
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": audio_data,
                        "file_name": "recording.wav",
                        "mimetype": "audio/wav",
                    }
                ]
            },
        )

        result = instance.format(data)

        assert "file" in result.files
        file_tuple = result.files["file"]
        assert file_tuple[0] == "recording.wav"
        assert file_tuple[1] == audio_data
        assert file_tuple[2] == "audio/wav"

    @pytest.mark.sanity
    def test_format_missing_audio(self, valid_instances):
        """Test format method raises error when no audio column provided.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={},
        )

        with pytest.raises(ValueError, match="expects exactly one audio column"):
            instance.format(data)

    @pytest.mark.sanity
    def test_format_multiple_audio(self, valid_instances):
        """Test format method raises error with multiple audio columns.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio1",
                        "file_name": "test1.mp3",
                        "mimetype": "audio/mpeg",
                    },
                    {
                        "audio": b"audio2",
                        "file_name": "test2.mp3",
                        "mimetype": "audio/mpeg",
                    },
                ]
            },
        )

        with pytest.raises(ValueError, match="expects exactly one audio column"):
            instance.format(data)

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )
        extras = {"body": {"language": "en", "temperature": 0.0}}

        result = instance.format(data, extras=extras)

        assert result.body.get("language") == "en"
        assert result.body.get("temperature") == 0.0

    # Response handling tests
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "usage",
            "text",
            "expected_audio_tokens",
            "expected_output_tokens",
        ),
        [
            (
                {"prompt_tokens": 538, "total_tokens": 982, "completion_tokens": 444},
                "Hello world",
                538,
                444,
            ),
            (None, "Hello", None, None),
            ({}, "", None, None),
        ],
    )
    def test_extract_metrics(
        self,
        valid_instances,
        usage,
        text,
        expected_audio_tokens,
        expected_output_tokens,
    ):
        """Test extract_metrics method for audio.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        input_metrics, output_metrics = instance.extract_metrics(usage, text)

        assert input_metrics.audio_tokens == expected_audio_tokens
        assert output_metrics.text_tokens == expected_output_tokens
        assert output_metrics.text_words == (len(text.split()) if text else 0)
        assert output_metrics.text_characters == len(text)

    @pytest.mark.smoke
    def test_audio_blocks_multiturn_with_history(self, valid_instances):
        """Test audio handler blocks multiturn with non-empty history.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        # Create a history with one turn
        history = [
            (
                GenerationRequest(columns={"audio_column": [{"audio": b"prev"}]}),
                None,
            )
        ]

        with pytest.raises(ValueError, match="does not support multiturn"):
            instance.format(data, history=history)

    @pytest.mark.smoke
    def test_audio_blocks_multiturn_with_history_and_response(self, valid_instances):
        """Test audio handler blocks multiturn with non-None history response.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        prev_request = GenerationRequest(columns={"audio_column": [{"audio": b"x"}]})
        prev_response = GenerationResponse(
            request_id="test", request_args=None, text="test"
        )

        with pytest.raises(ValueError, match="does not support multiturn"):
            instance.format(data, history=[(prev_request, prev_response)])

    @pytest.mark.sanity
    def test_audio_allows_single_turn(self, valid_instances):
        """Test audio handler allows single turn requests.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "audio_column": [
                    {
                        "audio": b"audio_data",
                        "file_name": "test.mp3",
                        "mimetype": "audio/mpeg",
                    }
                ]
            },
        )

        # Should succeed without history or response
        result = instance.format(data, history=None, response=None)

        assert result is not None
        assert result.files is not None
        assert "file" in result.files


class TestTextCompletionsRequestHandlerMultiturn:
    """Test cases for TextCompletionsRequestHandler multiturn support.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of TextCompletionsRequestHandler.

        ### WRITTEN BY AI ###
        """
        return TextCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_text_format_with_empty_history(self, valid_instances):
        """Test format with empty history.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )

        result = instance.format(data, history=[])

        assert result.body["prompt"] == "Hello"

    @pytest.mark.sanity
    def test_text_format_with_single_turn_history(self, valid_instances):
        """Test format with single turn history builds cumulative prompt.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Previous turn
        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="4"
        )

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        # Should include previous request and response
        assert "What is 2+2?" in result.body["prompt"]
        assert "4" in result.body["prompt"]
        assert "What is 3+3?" in result.body["prompt"]

    @pytest.mark.sanity
    def test_text_format_with_multi_turn_history(self, valid_instances):
        """Test format with multiple turns in history.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Build history with 3 turns
        history = [
            (
                GenerationRequest(columns={"text_column": ["Turn 1"]}),
                GenerationResponse(
                    request_id="r1", request_args=None, text="Response 1"
                ),
            ),
            (
                GenerationRequest(columns={"text_column": ["Turn 2"]}),
                GenerationResponse(
                    request_id="r2", request_args=None, text="Response 2"
                ),
            ),
            (
                GenerationRequest(columns={"text_column": ["Turn 3"]}),
                GenerationResponse(
                    request_id="r3", request_args=None, text="Response 3"
                ),
            ),
        ]

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["Turn 4"]},
        )

        result = instance.format(data, history=history)

        # Should include all turns in order
        prompt = result.body["prompt"]
        assert "Turn 1" in prompt
        assert "Response 1" in prompt
        assert "Turn 2" in prompt
        assert "Response 2" in prompt
        assert "Turn 3" in prompt
        assert "Response 3" in prompt
        assert "Turn 4" in prompt

    @pytest.mark.regression
    def test_text_format_prevents_infinite_recursion(self, valid_instances):
        """Test format doesn't pass history to recursive calls.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Create history
        prev_request = GenerationRequest(
            columns={"text_column": ["Previous"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="Response"
        )
        history = [(prev_request, prev_response)]

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["Current"]},
        )

        # This should not cause infinite recursion
        result = instance.format(data, history=history)

        # Verify it succeeded
        assert result.body["prompt"] is not None

    @pytest.mark.sanity
    def test_text_format_with_response_in_history(self, valid_instances):
        """Test format uses response content in cumulative prompt.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Previous turn with response
        prev_request = GenerationRequest(
            columns={"text_column": ["Question"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="The answer is 42"
        )

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["Follow up"]},
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        prompt = result.body["prompt"]
        # Should include the response text from history
        assert "The answer is 42" in prompt


class TestChatCompletionsRequestHandlerMultiturn:
    """Test cases for ChatCompletionsRequestHandler multiturn support.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of ChatCompletionsRequestHandler.

        ### WRITTEN BY AI ###
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_chat_format_with_empty_history(self, valid_instances):
        """Test format with empty history creates single user message.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )

        result = instance.format(data, history=[])

        messages = result.body["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.sanity
    def test_chat_format_with_single_turn_history(self, valid_instances):
        """Test format with single turn creates user/assistant/user sequence.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Previous turn
        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="The answer is 4"
        )

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        messages = result.body["messages"]
        # Should have: user (prev) + assistant (prev response) + user (current)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "The answer is 4"
        assert messages[2]["role"] == "user"

    @pytest.mark.sanity
    def test_chat_format_with_multi_turn_history(self, valid_instances):
        """Test format with multiple turns alternates user/assistant.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Build history with 3 turns
        history = [
            (
                GenerationRequest(columns={"text_column": ["Question 1"]}),
                GenerationResponse(request_id="r1", request_args=None, text="Answer 1"),
            ),
            (
                GenerationRequest(columns={"text_column": ["Question 2"]}),
                GenerationResponse(request_id="r2", request_args=None, text="Answer 2"),
            ),
            (
                GenerationRequest(columns={"text_column": ["Question 3"]}),
                GenerationResponse(request_id="r3", request_args=None, text="Answer 3"),
            ),
        ]

        # Current turn
        data = GenerationRequest(
            columns={"text_column": ["Question 4"]},
        )

        result = instance.format(data, history=history)

        messages = result.body["messages"]
        # Should have: 3 * (user + assistant) + current user = 7 messages
        assert len(messages) == 7

        # Check alternating pattern
        for i in range(0, 6, 2):
            assert messages[i]["role"] == "user"
            assert messages[i + 1]["role"] == "assistant"
        assert messages[6]["role"] == "user"

        # Check content
        assert messages[1]["content"] == "Answer 1"
        assert messages[3]["content"] == "Answer 2"
        assert messages[5]["content"] == "Answer 3"

    @pytest.mark.sanity
    def test_chat_format_with_system_prefix_and_history(self, valid_instances):
        """Test format with prefix (system) and history maintains order.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Previous turn
        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="Hi there!"
        )

        # Current turn with system prefix
        data = GenerationRequest(
            columns={
                "prefix_column": ["You are a helpful assistant."],
                "text_column": ["How are you?"],
            },
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        messages = result.body["messages"]
        # Should have: user (prev) + assistant (prev) + system + user (current)
        # (History is added first, then system prefix, then current user)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["role"] == "system"
        assert messages[2]["content"] == "You are a helpful assistant."
        assert messages[3]["role"] == "user"

    @pytest.mark.regression
    def test_chat_format_multimodal_with_history(self, valid_instances):
        """Test format with multimodal content (images) and history.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        # Previous turn with image
        prev_request = GenerationRequest(
            columns={
                "text_column": ["Describe this"],
                "image_column": [{"image": "https://example.com/img1.jpg"}],
            },
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="It's a cat"
        )

        # Current turn with different image
        data = GenerationRequest(
            columns={
                "text_column": ["And this one?"],
                "image_column": [{"image": "https://example.com/img2.jpg"}],
            },
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        messages = result.body["messages"]
        # Should have: user (prev with image) + assistant + user (current with image)
        assert len(messages) == 3

        # Check first user message has both text and image
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert any(item["type"] == "text" for item in messages[0]["content"])
        assert any(item["type"] == "image_url" for item in messages[0]["content"])

        # Check assistant message
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "It's a cat"

        # Check second user message has both text and image
        assert messages[2]["role"] == "user"
        assert isinstance(messages[2]["content"], list)
        assert any(item["type"] == "text" for item in messages[2]["content"])
        assert any(item["type"] == "image_url" for item in messages[2]["content"])


class TestResponsesRequestHandler:
    """Test cases for ResponsesRequestHandler.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def valid_instances(self):
        """
        Create instance of ResponsesRequestHandler.

        ## WRITTEN BY AI ##
        """
        return ResponsesRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """
        Test ResponsesRequestHandler class signatures.

        ## WRITTEN BY AI ##
        """
        handler = ResponsesRequestHandler()
        assert OpenAIRequestHandler in ResponsesRequestHandler.__mro__
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "extract_metrics")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """
        Test ResponsesRequestHandler initialization.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        assert isinstance(instance, ResponsesRequestHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    @pytest.mark.smoke
    def test_factory_registration(self):
        """
        Test that the handler is registered in the factory for /v1/responses.

        ## WRITTEN BY AI ##
        """
        handler = OpenAIRequestHandlerFactory.create("/v1/responses")
        assert isinstance(handler, ResponsesRequestHandler)

    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """
        Test format method with minimal data.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)
        assert "input" in result.body

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """
        Test format method with model parameter.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, model="gpt-4o")

        assert result.body["model"] == "gpt-4o"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """
        Test format method with streaming enabled (no stream_options).

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, stream=True)

        assert result.stream is True
        assert result.body["stream"] is True
        assert "stream_options" not in result.body

    @pytest.mark.sanity
    def test_format_output_tokens(self, valid_instances):
        """
        Test format method uses max_output_tokens with stop/ignore_eos for parity.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            output_metrics=UsageMetrics(text_tokens=100),
        )

        result = instance.format(data)

        assert result.body["max_output_tokens"] == 100
        assert result.body["stop"] is None
        assert result.body["ignore_eos"] is True
        assert "max_completion_tokens" not in result.body
        assert "max_tokens" not in result.body

    @pytest.mark.sanity
    def test_format_max_tokens_kwarg(self, valid_instances):
        """
        Test format method with max_tokens keyword maps to max_output_tokens
        without stop/ignore_eos.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, max_tokens=50)

        assert result.body["max_output_tokens"] == 50
        assert "stop" not in result.body
        assert "ignore_eos" not in result.body

    @pytest.mark.sanity
    def test_format_instructions_from_prefix(self, valid_instances):
        """
        Test format method maps prefix_column to instructions field.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"prefix_column": ["You are a helpful assistant."]},
        )

        result = instance.format(data)

        assert result.body["instructions"] == "You are a helpful assistant."

    @pytest.mark.sanity
    def test_format_input_items_text(self, valid_instances):
        """
        Test format method creates input items with input_text type.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello", "How are you?"]},
        )

        result = instance.format(data)

        input_items = result.body["input"]
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"
        content = input_items[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "input_text"
        assert content[0]["text"] == "Hello"
        assert content[1]["type"] == "input_text"
        assert content[1]["text"] == "How are you?"

    @pytest.mark.sanity
    def test_format_input_items_image(self, valid_instances):
        """
        Test format method creates input items with input_image type.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "image_column": [{"image": "https://example.com/img.jpg"}],
            },
        )

        result = instance.format(data)

        input_items = result.body["input"]
        assert len(input_items) == 1
        content = input_items[0]["content"]
        assert content[0]["type"] == "input_image"
        assert content[0]["image_url"] == "https://example.com/img.jpg"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "response",
            "expected_text",
            "expected_input_tokens",
            "expected_output_tokens",
        ),
        [
            (
                {
                    "id": "resp_123",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Hello, world!"}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
                "Hello, world!",
                5,
                3,
            ),
            (
                {
                    "id": "resp_456",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Part 1"},
                                {"type": "output_text", "text": " Part 2"},
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 8},
                },
                "Part 1 Part 2",
                10,
                8,
            ),
            (
                {"id": "resp_789", "output": [], "usage": {}},
                "",
                None,
                None,
            ),
            (
                {"output": []},
                "",
                None,
                None,
            ),
        ],
    )
    def test_non_streaming(
        self,
        valid_instances,
        generation_request,
        response,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """
        Test compile_non_streaming method for Responses API format.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == expected_text
        assert result.input_metrics.text_tokens == expected_input_tokens
        assert result.output_metrics.text_tokens == expected_output_tokens

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("lines", "expected_text", "expected_input_tokens", "expected_output_tokens"),
        [
            (
                [
                    "event: response.created",
                    (
                        "data: {"
                        '"type":"response.created",'
                        '"response":{"id":"resp_1"},'
                        '"sequence_number":0}'
                    ),
                    "",
                    "event: response.output_text.delta",
                    (
                        "data: {"
                        '"type":"response.output_text.delta",'
                        '"delta":"Hello",'
                        '"sequence_number":4}'
                    ),
                    "",
                    "event: response.output_text.delta",
                    (
                        "data: {"
                        '"type":"response.output_text.delta",'
                        '"delta":", world!",'
                        '"sequence_number":5}'
                    ),
                    "",
                    "event: response.completed",
                    (
                        "data: {"
                        '"type":"response.completed",'
                        '"response":{"id":"resp_1",'
                        '"usage":{"input_tokens":5,'
                        '"output_tokens":3}},'
                        '"sequence_number":8}'
                    ),
                    "data: [DONE]",
                ],
                "Hello, world!",
                5,
                3,
            ),
            (
                [
                    "event: response.output_text.delta",
                    (
                        "data: {"
                        '"type":"response.output_text.delta",'
                        '"delta":"Test",'
                        '"sequence_number":4}'
                    ),
                    "",
                    "event: response.completed",
                    (
                        "data: {"
                        '"type":"response.completed",'
                        '"response":{"id":"resp_2","usage":{}},'
                        '"sequence_number":6}'
                    ),
                    "data: [DONE]",
                ],
                "Test",
                None,
                None,
            ),
            (
                [
                    "event: response.created",
                    (
                        "data: {"
                        '"type":"response.created",'
                        '"response":{"id":"resp_3"},'
                        '"sequence_number":0}'
                    ),
                    "",
                    "event: response.completed",
                    (
                        "data: {"
                        '"type":"response.completed",'
                        '"response":{"id":"resp_3","usage":{}},'
                        '"sequence_number":2}'
                    ),
                    "data: [DONE]",
                ],
                "",
                None,
                None,
            ),
        ],
    )
    def test_streaming(
        self,
        valid_instances,
        generation_request,
        lines,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """
        Test streaming with Responses API SSE event format.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == expected_text
        assert response.input_metrics.text_tokens == expected_input_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("usage", "text", "expected_input_tokens", "expected_output_tokens"),
        [
            (
                {"input_tokens": 5, "output_tokens": 3},
                "Hello world",
                5,
                3,
            ),
            (
                {"input_tokens": 0, "output_tokens": 0},
                "",
                0,
                0,
            ),
            (
                None,
                "Hello world",
                None,
                None,
            ),
            (
                {},
                "",
                None,
                None,
            ),
        ],
    )
    def test_extract_metrics(
        self,
        valid_instances,
        usage,
        text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """
        Test extract_metrics maps input_tokens/output_tokens correctly.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        input_metrics, output_metrics = instance.extract_metrics(usage, text)

        assert input_metrics.text_tokens == expected_input_tokens
        assert output_metrics.text_tokens == expected_output_tokens
        assert output_metrics.text_words == (len(text.split()) if text else 0)
        assert output_metrics.text_characters == len(text)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("line", "expected_output"),
        [
            (
                'data: {"type":"response.output_text.delta","delta":"Hi"}',
                {"type": "response.output_text.delta", "delta": "Hi"},
            ),
            ("data: [DONE]", None),
            ("", {}),
            ("event: response.created", {}),
            ("event: response.output_text.delta", {}),
            ("  event: response.completed  ", {}),
            ("invalid line", {}),
            ('data: {"test": "value"}', {"test": "value"}),
        ],
    )
    def test_extract_line_data(self, valid_instances, line, expected_output):
        """
        Test extract_line_data handles Responses API SSE format.

        Explicitly skips event: lines and parses data: JSON lines.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        result = instance.extract_line_data(line)
        assert result == expected_output

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("lines", "expected_text", "expected_input_tokens", "expected_output_tokens"),
        [
            (
                [
                    "event: response.output_text.delta",
                    (
                        "data: {"
                        '"type":"response.output_text.delta",'
                        '"delta":"Hello",'
                        '"sequence_number":4}'
                    ),
                    "",
                    "event: response.failed",
                    (
                        "data: {"
                        '"type":"response.failed",'
                        '"response":{"id":"resp_err",'
                        '"usage":{"input_tokens":5,"output_tokens":1}},'
                        '"sequence_number":6}'
                    ),
                ],
                "Hello",
                5,
                1,
            ),
            (
                [
                    "event: response.output_text.delta",
                    (
                        "data: {"
                        '"type":"response.output_text.delta",'
                        '"delta":"Partial",'
                        '"sequence_number":4}'
                    ),
                    "",
                    "event: response.incomplete",
                    (
                        "data: {"
                        '"type":"response.incomplete",'
                        '"response":{"id":"resp_inc",'
                        '"usage":{"input_tokens":10,"output_tokens":2}},'
                        '"sequence_number":6}'
                    ),
                ],
                "Partial",
                10,
                2,
            ),
            (
                [
                    "event: response.failed",
                    (
                        "data: {"
                        '"type":"response.failed",'
                        '"response":{"id":"resp_fail_no_usage"},'
                        '"sequence_number":1}'
                    ),
                ],
                "",
                None,
                None,
            ),
        ],
    )
    def test_streaming_terminal_events(
        self,
        valid_instances,
        generation_request,
        lines,
        expected_text,
        expected_input_tokens,
        expected_output_tokens,
    ):
        """
        Test that response.failed and response.incomplete terminate the stream.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == expected_text
        assert response.input_metrics.text_tokens == expected_input_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens

    @pytest.mark.sanity
    def test_streaming_reasoning_triggers_ttft_not_content(
        self, valid_instances, generation_request
    ):
        """
        Verify Responses API reasoning events trigger TTFT (return 1)
        but set last_iteration_had_content to False, matching Chat
        Completions parity for TTFOT measurement.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        instance.format(generation_request)

        assert instance.last_iteration_had_content is False

        # Reasoning summary delta -- should return 1 (TTFT) but not content
        result = instance.add_streaming_line(
            'data: {"type": "response.reasoning_summary_text.delta", '
            '"delta": "Let me think..."}'
        )
        assert result == 1
        assert instance.last_iteration_had_content is False

        # Output text delta -- now content flag flips to True
        result = instance.add_streaming_line(
            'data: {"type": "response.output_text.delta", "delta": "Hello"}'
        )
        assert result == 1
        assert instance.last_iteration_had_content is True

    @pytest.mark.sanity
    def test_streaming_reasoning_only_leaves_content_false(
        self, valid_instances, generation_request
    ):
        """
        If a Responses API stream emits only reasoning before completing,
        last_iteration_had_content stays False.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        instance.format(generation_request)

        instance.add_streaming_line(
            'data: {"type": "response.reasoning_summary_text.delta", '
            '"delta": "thinking..."}'
        )
        instance.add_streaming_line(
            'data: {"type": "response.reasoning_summary_text.delta", '
            '"delta": " more thinking"}'
        )
        assert instance.last_iteration_had_content is False

        # Stream completes
        instance.add_streaming_line(
            'data: {"type": "response.completed", "response": '
            '{"id": "resp_1", "usage": {"input_tokens": 5, "output_tokens": 0}}}'
        )
        assert instance.last_iteration_had_content is False

    @pytest.mark.sanity
    def test_streaming_captures_reasoning_text(
        self, valid_instances, generation_request
    ):
        """
        Verify Responses API streaming accumulates reasoning_text on the
        compiled response.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            'data: {"type": "response.reasoning_summary_text.delta", '
            '"delta": "First, "}',
            'data: {"type": "response.reasoning_summary_text.delta", '
            '"delta": "consider..."}',
            'data: {"type": "response.output_text.delta", "delta": "Answer"}',
            'data: {"type": "response.completed", "response": '
            '{"id": "resp_1", "usage": {"input_tokens": 5, "output_tokens": 3}}}',
        ]
        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)
        assert response.text == "Answer"
        assert response.reasoning_text == "First, consider..."

    @pytest.mark.sanity
    def test_non_streaming_captures_reasoning_text(
        self, valid_instances, generation_request
    ):
        """
        Verify compile_non_streaming extracts reasoning from Responses API
        output items with type 'reasoning'.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        api_response = {
            "id": "resp_1",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"text": "Step 1. "}, {"text": "Step 2."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Final answer"}],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = instance.compile_non_streaming(
            generation_request, arguments, api_response
        )
        assert response.text == "Final answer"
        assert response.reasoning_text == "Step 1. Step 2."

    @pytest.mark.sanity
    def test_format_excludes_reasoning_from_history_by_default(self, valid_instances):
        """
        By default, reasoning_text should not appear in the assistant
        input item content.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="World",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        result = instance.format(data, history=[(prev_request, prev_response)])

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        assert assistant_items[0]["content"] == "World"

    @pytest.mark.sanity
    def test_format_includes_reasoning_in_history_when_enabled(self, valid_instances):
        """
        When multiturn_reasoning=True, reasoning_text should be
        wrapped in <think> tags and prepended to the assistant input item content.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="World",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=True,
        )

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        assert (
            assistant_items[0]["content"] == "<think>Think about greeting.</think>World"
        )

    @pytest.mark.sanity
    def test_format_includes_reasoning_only_response_in_history(self, valid_instances):
        """
        When multiturn_reasoning=True and the prior response has
        reasoning_text but no regular text content, an assistant input item
        should still be injected containing the reasoning.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=True,
        )

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        assert assistant_items[0]["content"] == "<think>Think about greeting.</think>"

    @pytest.mark.sanity
    def test_format_drops_reasoning_only_response_from_history_by_default(
        self, valid_instances
    ):
        """
        When multiturn_reasoning=False (default) and the prior
        response has reasoning_text but empty text content, an
        assistant input item with empty content should still be
        included to preserve alternating user/assistant structure.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
        )

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        assert assistant_items[0]["content"] == ""

    @pytest.mark.sanity
    def test_format_preserves_empty_assistant_in_multi_turn_history(
        self, valid_instances
    ):
        """
        When a prior turn produced an empty text response (e.g. reasoning
        model exhausted token budget), the assistant input item must still
        appear in the history to prevent consecutive user items.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        turn1_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        turn1_response = GenerationResponse(
            request_id="t1",
            request_args=None,
            text="Hi there!",
        )
        turn2_request = GenerationRequest(
            columns={"text_column": ["Tell me more"]},
        )
        turn2_response = GenerationResponse(
            request_id="t2",
            request_args=None,
            text="",
            reasoning_text="Let me think about this...",
        )

        current_request = GenerationRequest(
            columns={"text_column": ["Continue"]},
        )
        result = instance.format(
            current_request,
            history=[
                (turn1_request, turn1_response),
                (turn2_request, turn2_response),
            ],
        )

        items = result.body["input"]
        roles = [item["role"] for item in items if "role" in item]

        # Verify no consecutive user items
        for j in range(len(roles) - 1):
            assert not (roles[j] == "user" and roles[j + 1] == "user"), (
                f"Consecutive user items at indices {j} and {j + 1}: {roles}"
            )

        # Verify the empty assistant response is present
        assistant_items = [item for item in items if item.get("role") == "assistant"]
        assert len(assistant_items) == 2
        assert assistant_items[0]["content"] == "Hi there!"
        assert assistant_items[1]["content"] == ""

    @pytest.mark.sanity
    def test_format_includes_reasoning_with_custom_template(self, valid_instances):
        """
        When multiturn_reasoning is a custom format string, reasoning_text
        should be wrapped using that template in the Responses API input.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="World",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        template = "Here is my thought process:{reasoning}Here is my response:"
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning=template,
        )

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        expected = (
            "Here is my thought process:Think about greeting.Here is my response:World"
        )
        assert assistant_items[0]["content"] == expected

    @pytest.mark.sanity
    def test_format_includes_reasoning_raw_template(self, valid_instances):
        """
        When multiturn_reasoning='{reasoning}', reasoning_text should be
        prepended with no extra delimiters (raw mode) in the Responses API input.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Hello"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="World",
            reasoning_text="Think about greeting.",
        )
        data = GenerationRequest(columns={"text_column": ["Follow up"]})
        result = instance.format(
            data,
            history=[(prev_request, prev_response)],
            multiturn_reasoning="{reasoning}",
        )

        assistant_items = [
            item for item in result.body["input"] if item.get("role") == "assistant"
        ]
        assert len(assistant_items) == 1
        assert assistant_items[0]["content"] == "Think about greeting.World"

    @pytest.mark.sanity
    def test_format_with_history(self, valid_instances):
        """
        Test format builds input with conversation history.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="4"
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )

        result = instance.format(data, history=[(prev_request, prev_response)])

        input_items = result.body["input"]
        assert len(input_items) == 3
        assert input_items[0]["role"] == "user"
        assert input_items[1]["role"] == "assistant"
        assert input_items[1]["content"] == "4"
        assert input_items[2]["role"] == "user"

    @pytest.mark.sanity
    def test_format_with_server_history(self, valid_instances):
        """
        Test format uses previous_response_id instead of replaying history
        when server_history is enabled.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev", request_args=None, text="4", response_id="resp_abc123"
        )

        data = GenerationRequest(
            columns={"text_column": ["What is 3+3?"]},
        )

        result = instance.format(
            data, history=[(prev_request, prev_response)], server_history=True
        )

        assert result.body["previous_response_id"] == "resp_abc123"
        input_items = result.body["input"]
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"

    @pytest.mark.sanity
    def test_format_with_server_history_first_turn(self, valid_instances):
        """
        Test format does not set previous_response_id on the first turn
        (no history) even when server_history is enabled.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        data = GenerationRequest(
            columns={"text_column": ["Hello!"]},
        )

        result = instance.format(data, server_history=True)

        assert "previous_response_id" not in result.body
        input_items = result.body["input"]
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"

    @pytest.mark.sanity
    def test_format_with_server_history_tool_calls(self, valid_instances):
        """
        Test format includes function_call_output items when server_history
        is enabled and the current request is a tool_response_injection turn.
        The injection turn's tool_response_column supplies the output content
        and the preceding response's tool_calls supply the call_ids.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Call get_weather for SF"]},
            turn_type="client_tool_call",
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text=None,
            response_id="resp_tool_001",
            tool_calls=[
                ToolCall(
                    id="call_xyz",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"location": "SF"}',
                    ),
                )
            ],
        )

        data = GenerationRequest(
            turn_type="tool_response_injection",
        )

        result = instance.format(
            data, history=[(prev_request, prev_response)], server_history=True
        )

        assert result.body["previous_response_id"] == "resp_tool_001"
        input_items = result.body["input"]

        # function_call_output should be present
        fco_items = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        assert fco_items[0]["call_id"] == "call_xyz"
        assert fco_items[0]["output"] == '{"status": "ok"}'

        # function_call items must NOT be present (server already has them)
        fc_items = [i for i in input_items if i.get("type") == "function_call"]
        assert len(fc_items) == 0

        # No user message on injection turns
        user_items = [i for i in input_items if i.get("role") == "user"]
        assert len(user_items) == 0

    @pytest.mark.sanity
    def test_format_with_server_history_tool_calls_custom_response(
        self, valid_instances
    ):
        """
        Test format sources tool response content from tool_response_column
        on the injection turn when using server_history with tool calls.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["Call get_weather"]},
            turn_type="client_tool_call",
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text=None,
            response_id="resp_tool_002",
            tool_calls=[
                ToolCall(
                    id="call_custom",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"location": "NYC"}',
                    ),
                )
            ],
        )

        data = GenerationRequest(
            columns={"tool_response_column": ['{"temp": 72, "unit": "F"}']},
            turn_type="tool_response_injection",
        )

        result = instance.format(
            data, history=[(prev_request, prev_response)], server_history=True
        )

        input_items = result.body["input"]
        fco_items = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        assert fco_items[0]["call_id"] == "call_custom"
        assert fco_items[0]["output"] == '{"temp": 72, "unit": "F"}'

    @pytest.mark.sanity
    def test_format_with_server_history_no_tool_calls(self, valid_instances):
        """
        Test format with server_history does NOT include function_call_output
        items when the previous response was plain text (regression check).

        ## WRITTEN BY AI ##
        """
        instance = valid_instances

        prev_request = GenerationRequest(
            columns={"text_column": ["What is 2+2?"]},
        )
        prev_response = GenerationResponse(
            request_id="prev",
            request_args=None,
            text="4",
            response_id="resp_plain",
        )

        data = GenerationRequest(
            columns={"text_column": ["And 3+3?"]},
        )

        result = instance.format(
            data, history=[(prev_request, prev_response)], server_history=True
        )

        assert result.body["previous_response_id"] == "resp_plain"
        input_items = result.body["input"]

        # No function_call_output items should be present
        fco_items = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco_items) == 0

        # Only the user message
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"

    # Tool call response handling tests

    @pytest.mark.sanity
    def test_non_streaming_tool_calls(self, valid_instances, generation_request):
        """
        Test compile_non_streaming extracts function_call items when no text present.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "id": "resp_tc1",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "SF"}',
                    "call_id": "call_abc123",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text is None
        assert result.input_metrics.text_tokens == 10
        assert result.output_metrics.text_tokens == 15
        assert result.output_metrics.text_words is None
        assert result.output_metrics.text_characters is None
        assert result.output_metrics.tool_call_tokens == 15
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count == 1
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "SF"}'

    @pytest.mark.sanity
    def test_non_streaming_tool_calls_content_preferred(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming with both message and function_call output items.
        Text comes from message content; tool_call_count is set, tool_call_tokens is
        None, and mixed_content_tool_tokens equals the completion total because the
        API does not split completion_tokens between natural language text and tool
        JSON.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "id": "resp_tc2",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "I will call the function."}
                    ],
                },
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "SF"}',
                    "call_id": "call_1",
                },
            ],
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == "I will call the function."
        assert result.output_metrics.tool_call_tokens is None
        assert result.output_metrics.mixed_content_tool_tokens == 8
        assert result.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_non_streaming_multiple_tool_calls(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming counts multiple function_call items.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "id": "resp_tc3",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "SF"}',
                    "call_id": "call_1",
                },
                {
                    "type": "function_call",
                    "name": "get_time",
                    "arguments": '{"timezone": "PST"}',
                    "call_id": "call_2",
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 20},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text is None
        assert result.output_metrics.text_tokens == 20
        assert result.output_metrics.text_words is None
        assert result.output_metrics.text_characters is None
        assert result.output_metrics.tool_call_tokens == 20
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count == 2
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].id == "call_2"
        assert result.tool_calls[1].function.name == "get_time"
        assert result.tool_calls[1].function.arguments == '{"timezone": "PST"}'

    @pytest.mark.sanity
    def test_non_streaming_no_tool_calls_unchanged(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming leaves tool_call fields None for normal text.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        response = {
            "id": "resp_tc4",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }

        result = instance.compile_non_streaming(generation_request, arguments, response)

        assert result.text == "Hello!"
        assert result.output_metrics.tool_call_tokens is None
        assert result.output_metrics.mixed_content_tool_tokens is None
        assert result.output_metrics.tool_call_count is None

    @pytest.mark.sanity
    def test_streaming_tool_calls(self, valid_instances, generation_request):
        """
        Test streaming accumulates function_call output items and sets metrics.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            "event: response.output_item.added",
            (
                'data: {"type":"response.output_item.added",'
                '"output_index":0,'
                '"item":{"type":"function_call","name":"get_weather",'
                '"call_id":"call_abc","arguments":""}}'
            ),
            "",
            "event: response.function_call_arguments.delta",
            (
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":0,"delta":"{\\"loc\\""}'
            ),
            "",
            "event: response.function_call_arguments.done",
            (
                'data: {"type":"response.function_call_arguments.done",'
                '"output_index":0,"arguments":"{\\"loc\\":\\"SF\\"}"}'
            ),
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_s1",'
                '"usage":{"input_tokens":10,"output_tokens":12}}}'
            ),
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text is None
        assert response.output_metrics.text_tokens == 12
        assert response.output_metrics.text_words is None
        assert response.output_metrics.text_characters is None
        assert response.output_metrics.tool_call_tokens == 12
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count == 1
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"loc":"SF"}'

    @pytest.mark.sanity
    def test_streaming_multiple_tool_calls(self, valid_instances, generation_request):
        """
        Test streaming with multiple parallel function calls on different indices.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            "event: response.output_item.added",
            (
                'data: {"type":"response.output_item.added",'
                '"output_index":0,'
                '"item":{"type":"function_call","name":"fn_a",'
                '"call_id":"call_1","arguments":""}}'
            ),
            "",
            "event: response.output_item.added",
            (
                'data: {"type":"response.output_item.added",'
                '"output_index":1,'
                '"item":{"type":"function_call","name":"fn_b",'
                '"call_id":"call_2","arguments":""}}'
            ),
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_s2",'
                '"usage":{"input_tokens":8,"output_tokens":18}}}'
            ),
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text is None
        assert response.output_metrics.text_words is None
        assert response.output_metrics.text_characters is None
        assert response.output_metrics.tool_call_tokens == 18
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count == 2

    @pytest.mark.sanity
    def test_streaming_text_preferred_over_tool_calls(
        self, valid_instances, generation_request
    ):
        """
        Test streaming when both text deltas and function_call items appear: final
        text is concatenated content; tool_call_count is set, tool_call_tokens is
        None, and mixed_content_tool_tokens equals the completion total because the
        API does not split completion_tokens between natural language text and tool
        JSON.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            "event: response.output_text.delta",
            ('data: {"type":"response.output_text.delta","delta":"Some text"}'),
            "",
            "event: response.output_item.added",
            (
                'data: {"type":"response.output_item.added",'
                '"output_index":1,'
                '"item":{"type":"function_call","name":"fn",'
                '"call_id":"call_x","arguments":"{}"}}'
            ),
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_s3",'
                '"usage":{"input_tokens":5,"output_tokens":4}}}'
            ),
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text == "Some text"
        assert response.output_metrics.tool_call_tokens is None
        assert response.output_metrics.mixed_content_tool_tokens == 4
        assert response.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_streaming_no_tool_calls_unchanged(
        self, valid_instances, generation_request
    ):
        """
        Test normal streaming text response has no tool_call metrics.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        arguments = instance.format(generation_request)

        lines = [
            "event: response.output_text.delta",
            ('data: {"type":"response.output_text.delta","delta":"Hi there"}'),
            "",
            "event: response.completed",
            (
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_s4",'
                '"usage":{"input_tokens":3,"output_tokens":2}}}'
            ),
        ]

        for line in lines:
            result = instance.add_streaming_line(line)
            if result is None:
                break

        response = instance.compile_streaming(generation_request, arguments)

        assert response.text == "Hi there"
        assert response.input_metrics.text_tokens == 3
        assert response.output_metrics.text_tokens == 2
        assert response.output_metrics.tool_call_tokens is None
        assert response.output_metrics.mixed_content_tool_tokens is None
        assert response.output_metrics.tool_call_count is None

    @pytest.mark.smoke
    def test_initialization_has_streaming_tool_calls(self, valid_instances):
        """
        Test ResponsesRequestHandler initializes streaming_tool_calls.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        assert hasattr(instance, "streaming_tool_calls")
        assert instance.streaming_tool_calls == {}

    @pytest.mark.sanity
    def test_format_tool_call_overrides(self, valid_instances):
        """
        Test _apply_tool_call_overrides injects tools, sets tool_choice, and
        removes incompatible keys on tool-call turns.  Tools provided in Chat
        Completions format are normalised to flat Responses API format.

        ## WRITTEN BY AI ##
        """
        import json as stdlib_json

        instance = valid_instances
        chat_tools = [
            {"type": "function", "function": {"name": "fn", "parameters": {}}}
        ]
        expected_tools = [{"type": "function", "name": "fn", "parameters": {}}]
        data = GenerationRequest(
            columns={"tools_column": [stdlib_json.dumps(chat_tools)]},
            turn_type="client_tool_call",
        )

        result = instance.format(data)

        assert result.body["tools"] == expected_tools
        assert result.body["tool_choice"] == "required"
        assert "ignore_eos" not in result.body
        assert "stop" not in result.body
        assert "max_output_tokens" not in result.body

    @pytest.mark.sanity
    def test_format_tool_choice_none_on_non_tool_turn(self, valid_instances):
        """
        Test that non-tool turns set tool_choice to 'none' when tools are
        present from extras.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
        data = GenerationRequest(turn_type="standard")

        result = instance.format(data, extras={"body": {"tools": tools}})

        assert result.body["tools"] == tools
        assert result.body["tool_choice"] == "none"

    @pytest.mark.sanity
    def test_format_strips_tool_choice_without_tools(self, valid_instances):
        """
        Test that tool_choice from extras is stripped when no tools are present
        in the request body.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["test prompt"]},
            turn_type="standard",
        )

        result = instance.format(data, extras={"body": {"tool_choice": "required"}})

        assert "tool_choice" not in result.body
        assert "tools" not in result.body

    @pytest.mark.sanity
    def test_history_replays_tool_calls_and_injection(self, valid_instances):
        """
        Test history replay produces function_call items from a tool_call
        turn and function_call_output items from the injection turn.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        instance = valid_instances
        tool_call_req = GenerationRequest(
            columns={"text_column": ["call the tool"]},
            turn_type="client_tool_call",
        )
        tool_call_resp = GenerationResponse(
            request_id="req-1",
            request_args="{}",
            tool_calls=[
                ToolCall(
                    id="call_abc",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"loc": "SF"}',
                    ),
                )
            ],
        )
        injection_req = GenerationRequest(
            columns={},
            turn_type="tool_response_injection",
        )
        injection_resp = GenerationResponse(
            request_id="req-2",
            request_args="{}",
            text="The weather is sunny.",
        )

        current = GenerationRequest(
            columns={"text_column": ["follow-up question"]},
        )
        result = instance.format(
            current,
            history=[
                (tool_call_req, tool_call_resp),
                (injection_req, injection_resp),
            ],
        )

        input_items = result.body["input"]
        fc_items = [i for i in input_items if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["call_id"] == "call_abc"
        assert fc_items[0]["name"] == "get_weather"

        fco_items = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        assert fco_items[0]["call_id"] == "call_abc"
        assert fco_items[0]["output"] == '{"status": "ok"}'


class TestEnsureChatCompletionsTool:
    """Tests for ChatCompletionsRequestHandler._ensure_tool_format normalizer.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_passthrough_native_format(self):
        """
        Chat Completions-format tools are returned unchanged.

        ## WRITTEN BY AI ##
        """
        tool = {
            "type": "function",
            "function": {"name": "fn", "description": "d", "parameters": {}},
        }
        assert ChatCompletionsRequestHandler._ensure_tool_format(tool) == tool

    @pytest.mark.smoke
    def test_converts_responses_format(self):
        """
        Responses API-format tools are wrapped into the nested ``function`` structure.

        ## WRITTEN BY AI ##
        """
        responses_tool = {
            "type": "function",
            "name": "fn",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": True,
        }
        expected = {
            "type": "function",
            "function": {
                "name": "fn",
                "description": "d",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
        result = ChatCompletionsRequestHandler._ensure_tool_format(responses_tool)
        assert result == expected

    @pytest.mark.sanity
    def test_partial_fields(self):
        """
        Only fields present in the source tool are carried over.

        ## WRITTEN BY AI ##
        """
        responses_tool = {"type": "function", "name": "fn"}
        result = ChatCompletionsRequestHandler._ensure_tool_format(responses_tool)
        assert result == {"type": "function", "function": {"name": "fn"}}


class TestEnsureResponsesTool:
    """Tests for ResponsesRequestHandler._ensure_tool_format normalizer.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_passthrough_native_format(self):
        """
        Responses API-format tools are returned unchanged.

        ## WRITTEN BY AI ##
        """
        tool = {
            "type": "function",
            "name": "fn",
            "description": "d",
            "parameters": {},
        }
        assert ResponsesRequestHandler._ensure_tool_format(tool) == tool

    @pytest.mark.smoke
    def test_converts_chat_completions_format(self):
        """
        Chat Completions-format tools are flattened to top-level keys.

        ## WRITTEN BY AI ##
        """
        chat_tool = {
            "type": "function",
            "function": {
                "name": "fn",
                "description": "d",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
        expected = {
            "type": "function",
            "name": "fn",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": True,
        }
        assert ResponsesRequestHandler._ensure_tool_format(chat_tool) == expected

    @pytest.mark.sanity
    def test_partial_fields(self):
        """
        Only fields present in the source function dict are carried over.

        ## WRITTEN BY AI ##
        """
        chat_tool = {"type": "function", "function": {"name": "fn"}}
        result = ResponsesRequestHandler._ensure_tool_format(chat_tool)
        assert result == {"type": "function", "name": "fn"}


class TestPoolingRequestHandler:
    """Test cases for PoolingRequestHandler.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of PoolingRequestHandler.

        ### WRITTEN BY AI ###
        """
        return PoolingRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test PoolingRequestHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = PoolingRequestHandler()
        assert issubclass(PoolingRequestHandler, ChatCompletionsRequestHandler)
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test PoolingRequestHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, PoolingRequestHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    # Request formatting tests
    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(
            data, model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        )

        assert (
            result.body["model"]
            == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        )

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with streaming enabled.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, stream=True)

        assert result.stream is True
        assert result.body["stream"] is True
        assert "stream_options" in result.body
        assert result.body["stream_options"]["include_usage"] is True
        assert result.body["stream_options"]["continuous_usage_stats"] is True

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()
        extras = {"body": {"temperature": 0.5, "top_k": 40}}

        result = instance.format(data, extras=extras)

        assert result.body.get("temperature") == 0.5
        assert result.body.get("top_k") == 40

    @pytest.mark.sanity
    def test_format_pooling_data(self, valid_instances):
        """Test format method with pooling column data from real dataset.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "pooling_column": [
                    {
                        "data": {
                            "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
                            "data_format": "url",
                            "out_data_format": "b64_json",
                            "indices": [1, 2, 3, 8, 11, 12],
                        }
                    }
                ]
            },
        )

        result = instance.format(data)

        assert "data" in result.body
        assert (
            result.body["data"]["data"]
            == "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif"
        )
        assert result.body["data"]["data_format"] == "url"
        assert result.body["data"]["out_data_format"] == "b64_json"
        assert result.body["data"]["indices"] == [1, 2, 3, 8, 11, 12]

    @pytest.mark.sanity
    def test_format_pooling_data_with_priority(self, valid_instances):
        """Test format method with pooling data and priority field.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "pooling_column": [
                    {
                        "data": {
                            "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
                            "data_format": "url",
                            "out_data_format": "b64_json",
                            "indices": [1, 2, 3, 8, 11, 12],
                        },
                        "priority": "high",
                    }
                ]
            },
        )

        result = instance.format(data)

        assert "data" in result.body
        assert (
            result.body["data"]["data"]
            == "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif"
        )
        assert result.body["data"]["data_format"] == "url"
        assert result.body["data"]["out_data_format"] == "b64_json"
        assert result.body["data"]["indices"] == [1, 2, 3, 8, 11, 12]
        assert result.body["priority"] == "high"

    @pytest.mark.sanity
    def test_format_pooling_with_model_and_streaming(self, valid_instances):
        """Test format method with pooling data, model, and streaming.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={
                "pooling_column": [
                    {
                        "data": {
                            "data": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
                            "data_format": "url",
                            "out_data_format": "b64_json",
                            "indices": [1, 2, 3, 8, 11, 12],
                        }
                    }
                ]
            },
        )

        result = instance.format(
            data,
            model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
            stream=True,
        )

        assert (
            result.body["model"]
            == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        )
        assert result.stream is True
        assert result.body["stream"] is True
        assert "data" in result.body
        assert (
            result.body["data"]["data"]
            == "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif"
        )
        assert result.body["data"]["data_format"] == "url"
        assert result.body["data"]["out_data_format"] == "b64_json"
        assert result.body["data"]["indices"] == [1, 2, 3, 8, 11, 12]


class TestEmbeddingsRequestHandler:
    """Test cases for EmbeddingsRequestHandler.

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of EmbeddingsRequestHandler.

        ### WRITTEN BY AI ###
        """
        return EmbeddingsRequestHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test EmbeddingsRequestHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = EmbeddingsRequestHandler()
        assert hasattr(handler, "format")
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test EmbeddingsRequestHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, EmbeddingsRequestHandler)

    # Request formatting tests
    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)
        assert result.stream is False  # Embeddings never stream

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()

        result = instance.format(data, model="BAAI/bge-small-en-v1.5")

        assert result.body["model"] == "BAAI/bge-small-en-v1.5"
        assert result.stream is False

    @pytest.mark.sanity
    def test_format_single_text(self, valid_instances):
        """Test format method with single text input.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello world"]},
        )

        result = instance.format(data)

        assert result.body["input"] == "Hello world"
        assert isinstance(result.body["input"], str)

    @pytest.mark.sanity
    def test_format_multiple_texts(self, valid_instances):
        """Test format method with multiple text inputs.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            columns={"text_column": ["Hello", "How are you?"]},
        )

        result = instance.format(data)

        assert result.body["input"] == ["Hello", "How are you?"]
        assert isinstance(result.body["input"], list)

    @pytest.mark.sanity
    def test_format_with_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest()
        extras = {"body": {"user": "test-user"}}

        result = instance.format(data, extras=extras)

        assert result.body.get("user") == "test-user"

    @pytest.mark.sanity
    def test_compile_non_streaming(self, valid_instances):
        """Test compile_non_streaming method.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        request = GenerationRequest()
        arguments = instance.format(request, model="test-model")
        response_data = {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        result = instance.compile_non_streaming(request, arguments, response_data)

        assert isinstance(result, GenerationResponse)
        assert result.request_id == request.request_id
        assert result.text == ""  # No text output for embeddings
        assert result.input_metrics.text_tokens == 10
        assert result.output_metrics.text_tokens is None

    @pytest.mark.sanity
    def test_compile_non_streaming_no_usage(self, valid_instances):
        """Test compile_non_streaming with missing usage data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        request = GenerationRequest()
        arguments = instance.format(request)
        response_data = {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        }

        result = instance.compile_non_streaming(request, arguments, response_data)

        assert result.input_metrics.text_tokens == 0
        assert result.output_metrics.text_tokens is None

    @pytest.mark.sanity
    def test_add_streaming_line_raises(self, valid_instances):
        """Test that add_streaming_line raises NotImplementedError.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances

        with pytest.raises(
            NotImplementedError, match="Embeddings do not support streaming"
        ):
            instance.add_streaming_line("data: test")

    @pytest.mark.sanity
    def test_compile_streaming_raises(self, valid_instances):
        """Test that compile_streaming raises NotImplementedError.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        request = GenerationRequest()
        arguments = instance.format(request)

        with pytest.raises(
            NotImplementedError, match="Embeddings do not support streaming"
        ):
            instance.compile_streaming(request, arguments)


class TestChatCompletionsToolChoiceOverride:
    """Verify tool_choice is overridden to 'none' on non-tool-call turns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def handler(self):
        """
        ## WRITTEN BY AI ##
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_tool_choice_none_when_expects_false(self, handler):
        """When turn_type='standard' and tools come from dataset, tool_choice='none'.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="standard",
        )
        extras = {"body": {"tool_choice": "required"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "none"

    @pytest.mark.smoke
    def test_tool_choice_preserved_when_expects_true(self, handler):
        """When turn_type='client_tool_call', the configured tool_choice is kept.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        extras = {"body": {"tool_choice": "required"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "required"

    @pytest.mark.sanity
    def test_auto_tool_choice_preserved_when_expects_true(self, handler):
        """When turn_type='client_tool_call' with auto mode, tool_choice stays 'auto'.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        extras = {"body": {"tool_choice": "auto"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "auto"

    @pytest.mark.sanity
    def test_no_override_without_tools(self, handler):
        """Without tools in body, no tool_choice override happens.

        ## WRITTEN BY AI ##
        """
        data = GenerationRequest(
            columns={"text_column": ["test"]},
            turn_type="standard",
        )
        result = handler.format(data)

        assert "tool_choice" not in result.body

    @pytest.mark.sanity
    def test_per_request_tools_deserialized_from_json(self, handler):
        """JSON-serialized tools from synthetic data are deserialized.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "get_data"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        result = handler.format(data)

        assert result.body["tools"] == tools

    @pytest.mark.smoke
    def test_no_token_limits_on_tool_call_turn(self, handler):
        """On tool-call turns, ignore_eos, stop, max_completion_tokens, and
        max_tokens are all absent.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        result = handler.format(data)

        assert "ignore_eos" not in result.body
        assert "stop" not in result.body
        assert "max_completion_tokens" not in result.body
        assert "max_tokens" not in result.body

    @pytest.mark.sanity
    def test_finalizer_to_format_no_token_limits_on_tool_call_turn(self, handler):
        """The finalizer moves output_metrics to the injection turn so the
        handler applies no token limits to the tool call turn and correct
        limits to the injection turn.

        ## WRITTEN BY AI ##
        """
        import json

        from guidellm.data.finalizers.generative import (
            GenerativeRequestFinalizer,
            GenerativeRequestFinalizerArgs,
        )

        finalizer = GenerativeRequestFinalizer(GenerativeRequestFinalizerArgs())
        items = [
            {
                "text_column": ["call the tool"],
                "tools_column": [
                    json.dumps([{"type": "function", "function": {"name": "fn"}}])
                ],
                "output_tokens_count_column": [100],
            },
        ]
        requests = finalizer(items)

        assert len(requests) == 2
        tool_call_req, injection_req = requests

        # Tool call turn: no token limits
        tc_result = handler.format(tool_call_req)
        assert "max_completion_tokens" not in tc_result.body
        assert "max_tokens" not in tc_result.body
        assert "ignore_eos" not in tc_result.body
        assert "stop" not in tc_result.body

        # Injection turn: token limits applied from output_metrics
        inj_result = handler.format(injection_req)
        assert inj_result.body["max_completion_tokens"] == 100
        assert inj_result.body["ignore_eos"] is True
        assert inj_result.body["stop"] is None

    @pytest.mark.smoke
    def test_max_completion_tokens_kept_on_plain_text_turn(self, handler):
        """On the final plain-text turn, max_completion_tokens is preserved.

        ## WRITTEN BY AI ##
        """
        import json

        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="standard",
            output_metrics=UsageMetrics(text_tokens=100),
        )
        result = handler.format(data)

        assert result.body["max_completion_tokens"] == 100


class TestChatCompletionsToolResponseColumn:
    """Verify chat completions handler uses tool_response_column from the
    injection turn, with tool_call_ids sourced from the preceding response.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def handler(self):
        """
        ## WRITTEN BY AI ##
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_uses_tool_response_from_injection_column(self, handler):
        """Tool response content from injection turn's tool_response_column
        is used in history replay.

        ## WRITTEN BY AI ##
        """
        import json
        from unittest.mock import MagicMock

        from guidellm.schemas.tool_call import (
            ToolCall,
            ToolCallFunction,
        )

        tools = [{"type": "function", "function": {"name": "fn"}}]
        tool_call_request = GenerationRequest(
            columns={
                "text_column": ["call the tool"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        tool_call_response = MagicMock(spec=GenerationResponse)
        tool_call_response.tool_calls = [
            ToolCall(id="call_1", function=ToolCallFunction(name="fn"))
        ]
        tool_call_response.text = None
        tool_call_response.reasoning_text = None

        injection_request = GenerationRequest(
            columns={"tool_response_column": ['{"result": "custom data"}']},
            turn_type="tool_response_injection",
        )
        injection_response = MagicMock(spec=GenerationResponse)
        injection_response.tool_calls = None
        injection_response.text = "The tool returned custom data."
        injection_response.reasoning_text = None

        current_request = GenerationRequest(
            columns={"text_column": ["now respond"]},
            turn_type="standard",
        )

        result = handler.format(
            current_request,
            history=[
                (tool_call_request, tool_call_response),
                (injection_request, injection_response),
            ],
        )

        tool_messages = [m for m in result.body["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == '{"result": "custom data"}'
        assert tool_messages[0]["tool_call_id"] == "call_1"

    @pytest.mark.sanity
    def test_falls_back_to_default_without_column(self, handler):
        """Without tool_response_column, the default placeholder is used.

        ## WRITTEN BY AI ##
        """
        import json
        from unittest.mock import MagicMock

        from guidellm.schemas.tool_call import (
            ToolCall,
            ToolCallFunction,
        )
        from guidellm.settings import settings

        tools = [{"type": "function", "function": {"name": "fn"}}]
        tool_call_request = GenerationRequest(
            columns={
                "text_column": ["call the tool"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        tool_call_response = MagicMock(spec=GenerationResponse)
        tool_call_response.tool_calls = [
            ToolCall(id="call_1", function=ToolCallFunction(name="fn"))
        ]
        tool_call_response.text = None
        tool_call_response.reasoning_text = None

        injection_request = GenerationRequest(
            columns={},
            turn_type="tool_response_injection",
        )
        injection_response = MagicMock(spec=GenerationResponse)
        injection_response.tool_calls = None
        injection_response.text = "Ok."
        injection_response.reasoning_text = None

        current_request = GenerationRequest(
            columns={"text_column": ["now respond"]},
            turn_type="standard",
        )

        result = handler.format(
            current_request,
            history=[
                (tool_call_request, tool_call_response),
                (injection_request, injection_response),
            ],
        )

        tool_messages = [m for m in result.body["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == settings.default_synthetic_tool_response

    @pytest.mark.sanity
    def test_bytes_tool_response_decoded(self, handler):
        """Tool response content stored as bytes (from orjson) is decoded to str.

        ## WRITTEN BY AI ##
        """
        import json
        from unittest.mock import MagicMock

        from guidellm.schemas.tool_call import (
            ToolCall,
            ToolCallFunction,
        )

        tools = [{"type": "function", "function": {"name": "fn"}}]
        tool_call_request = GenerationRequest(
            columns={
                "text_column": ["call the tool"],
                "tools_column": [json.dumps(tools)],
            },
            turn_type="client_tool_call",
        )
        tool_call_response = MagicMock(spec=GenerationResponse)
        tool_call_response.tool_calls = [
            ToolCall(id="call_1", function=ToolCallFunction(name="fn"))
        ]
        tool_call_response.text = None
        tool_call_response.reasoning_text = None

        injection_request = GenerationRequest(
            columns={"tool_response_column": [b'{"result": "bytes data"}']},
            turn_type="tool_response_injection",
        )
        injection_response = MagicMock(spec=GenerationResponse)
        injection_response.tool_calls = None
        injection_response.text = "Done."
        injection_response.reasoning_text = None

        current_request = GenerationRequest(
            columns={"text_column": ["now respond"]},
            turn_type="standard",
        )

        result = handler.format(
            current_request,
            history=[
                (tool_call_request, tool_call_response),
                (injection_request, injection_response),
            ],
        )

        tool_messages = [m for m in result.body["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == '{"result": "bytes data"}'
        assert isinstance(tool_messages[0]["content"], str)


class TestChatCompletionsInjectionTurnFormat:
    """Verify ChatCompletionsRequestHandler formats injection turns correctly
    as the current turn and in history replay.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def handler(self):
        """
        ## WRITTEN BY AI ##
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_injection_turn_as_current_turn(self, handler):
        """Injection turn as current turn: tool response messages, no user content.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        tool_req = GenerationRequest(
            columns={"text_column": ["ask"]},
            turn_type="client_tool_call",
        )
        tool_resp = GenerationResponse(
            request_id="r1",
            request_args="{}",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="get_data", arguments="{}"),
                )
            ],
        )

        injection = GenerationRequest(
            columns={"tool_response_column": ['{"data": 42}']},
            turn_type="tool_response_injection",
        )

        result = handler.format(injection, history=[(tool_req, tool_resp)])

        messages = result.body["messages"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(user_msgs) == 1  # from history
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == '{"data": 42}'

    @pytest.mark.smoke
    def test_consecutive_tool_turns_in_history(self, handler):
        """Consecutive tool turns each produce their own injection turn pair.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        tc1_req = GenerationRequest(
            columns={"text_column": ["q1"]}, turn_type="client_tool_call"
        )
        tc1_resp = GenerationResponse(
            request_id="r1",
            request_args="{}",
            tool_calls=[
                ToolCall(id="c1", function=ToolCallFunction(name="fn1", arguments="{}"))
            ],
        )
        inj1_req = GenerationRequest(
            columns={"tool_response_column": ['{"r": 1}']},
            turn_type="tool_response_injection",
        )
        inj1_resp = GenerationResponse(
            request_id="r2", request_args="{}", text="Result 1."
        )
        tc2_req = GenerationRequest(
            columns={"text_column": ["q2"]}, turn_type="client_tool_call"
        )
        tc2_resp = GenerationResponse(
            request_id="r3",
            request_args="{}",
            tool_calls=[
                ToolCall(id="c2", function=ToolCallFunction(name="fn2", arguments="{}"))
            ],
        )
        inj2_req = GenerationRequest(
            columns={"tool_response_column": ['{"r": 2}']},
            turn_type="tool_response_injection",
        )
        inj2_resp = GenerationResponse(
            request_id="r4", request_args="{}", text="Result 2."
        )

        current = GenerationRequest(
            columns={"text_column": ["final question"]},
        )
        result = handler.format(
            current,
            history=[
                (tc1_req, tc1_resp),
                (inj1_req, inj1_resp),
                (tc2_req, tc2_resp),
                (inj2_req, inj2_resp),
            ],
        )

        messages = result.body["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert tool_msgs[0]["content"] == '{"r": 1}'
        assert tool_msgs[1]["tool_call_id"] == "c2"
        assert tool_msgs[1]["content"] == '{"r": 2}'

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 4  # 2 tool_call + 2 injection text

    @pytest.mark.sanity
    def test_injection_with_no_prior_tool_calls(self, handler):
        """Injection turn with no prior tool_calls produces no tool messages.

        ## WRITTEN BY AI ##
        """
        prev_req = GenerationRequest(
            columns={"text_column": ["normal"]},
        )
        prev_resp = GenerationResponse(
            request_id="r1", request_args="{}", text="response"
        )

        injection = GenerationRequest(
            columns={},
            turn_type="tool_response_injection",
        )

        result = handler.format(injection, history=[(prev_req, prev_resp)])

        messages = result.body["messages"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 0


class TestResponsesInjectionTurnFormat:
    """Verify ResponsesRequestHandler formats injection turns correctly.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def handler(self):
        """
        ## WRITTEN BY AI ##
        """
        return ResponsesRequestHandler()

    @pytest.mark.smoke
    def test_injection_turn_as_current_turn(self, handler):
        """Injection turn sends function_call_output items, no user content.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        tool_req = GenerationRequest(
            columns={"text_column": ["ask"]},
            turn_type="client_tool_call",
        )
        tool_resp = GenerationResponse(
            request_id="r1",
            request_args="{}",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="get_data", arguments="{}"),
                )
            ],
        )

        injection = GenerationRequest(
            columns={"tool_response_column": ['{"data": 42}']},
            turn_type="tool_response_injection",
        )

        result = handler.format(injection, history=[(tool_req, tool_resp)])

        input_items = result.body["input"]
        fco = [i for i in input_items if i.get("type") == "function_call_output"]
        user = [i for i in input_items if i.get("role") == "user"]
        assert len(fco) == 1
        assert fco[0]["call_id"] == "call_1"
        assert fco[0]["output"] == '{"data": 42}'
        assert len(user) == 1  # from history

    @pytest.mark.sanity
    def test_server_history_injection_turn(self, handler):
        """Server-side history: injection turn sets previous_response_id
        and includes function_call_output items.

        ## WRITTEN BY AI ##
        """
        from guidellm.schemas.tool_call import ToolCall, ToolCallFunction

        tool_req = GenerationRequest(
            columns={"text_column": ["ask"]},
            turn_type="client_tool_call",
        )
        tool_resp = GenerationResponse(
            request_id="r1",
            request_args="{}",
            response_id="resp_001",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolCallFunction(name="fn", arguments="{}"),
                )
            ],
        )

        injection = GenerationRequest(
            columns={"tool_response_column": ['{"ok": true}']},
            turn_type="tool_response_injection",
        )

        result = handler.format(
            injection, history=[(tool_req, tool_resp)], server_history=True
        )

        assert result.body["previous_response_id"] == "resp_001"
        input_items = result.body["input"]
        fco = [i for i in input_items if i.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert fco[0]["call_id"] == "call_1"
        assert fco[0]["output"] == '{"ok": true}'
