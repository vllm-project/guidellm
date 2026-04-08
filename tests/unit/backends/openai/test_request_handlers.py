"""
Unit tests for OpenAI request handlers.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.backends.openai.request_handlers import (
    AudioRequestHandler,
    ChatCompletionsRequestHandler,
    OpenAIRequestHandler,
    OpenAIRequestHandlerFactory,
    PoolingRequestHandler,
    ResponsesRequestHandler,
    TextCompletionsRequestHandler,
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

        assert len(result.body["messages"]) == 2
        assert result.body["messages"][0]["role"] == "system"
        assert result.body["messages"][0]["content"] == "You are a helpful assistant."
        assert result.body["messages"][1]["role"] == "user"
        assert result.body["messages"][1]["content"] == []

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
        assert result.output_metrics.tool_call_count == 1

    @pytest.mark.sanity
    def test_non_streaming_tool_calls_content_preferred(
        self, valid_instances, generation_request
    ):
        """
        Test compile_non_streaming prefers content over tool_calls when both present.

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
        assert result.output_metrics.tool_call_count is None

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
        assert response.output_metrics.tool_call_count == 2

    @pytest.mark.sanity
    def test_streaming_text_preferred_over_tool_calls(
        self, valid_instances, generation_request
    ):
        """
        Test that streaming text content is used when both text and tool_calls
        are present (text takes precedence).

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
                '"function": {"name": "fn", "arguments": "{}"}}]}}], "usage": {}}'
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
        assert response.output_metrics.tool_call_count is None

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
        assert response.output_metrics.tool_call_count is None

    @pytest.mark.smoke
    def test_initialization_has_streaming_tool_call_indices(self, valid_instances):
        """
        Test ChatCompletionsRequestHandler initializes streaming_tool_call_indices.

        ## WRITTEN BY AI ##
        """
        instance = valid_instances
        assert hasattr(instance, "streaming_tool_call_indices")
        assert instance.streaming_tool_call_indices == set()


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
    def test_audio_blocks_multiturn_with_response(self, valid_instances):
        """Test audio handler blocks multiturn with non-None response.

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

        # Mock response

        response = GenerationResponse(request_id="test", request_args=None, text="test")

        with pytest.raises(ValueError, match="does not support multiturn"):
            instance.format(data, response=response)

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

        result = instance.format(
            data, response=prev_response, history=[(prev_request, prev_response)]
        )

        prompt = result.body["prompt"]
        # Should include the response text
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
