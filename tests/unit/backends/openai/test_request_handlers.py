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
    TextCompletionsRequestHandler,
)
from guidellm.schemas import GenerationRequest, UsageMetrics
from guidellm.utils import RegistryMixin


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
            ("/v1/audio/transcriptions", None, AudioRequestHandler),
            ("/v1/audio/translations", None, AudioRequestHandler),
            (
                "/v1/completions",
                {"/v1/completions": ChatCompletionsRequestHandler},
                ChatCompletionsRequestHandler,
            ),
        ],
        ids=[
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
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
                "prefix_column": ["System: ", "Context: "],
                "text_column": ["Hello ", "world"],
            },
        )

        result = instance.format(data)

        assert result.body["prompt"] == "System: Context: Hello world"

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

        assert len(result.body["messages"]) == 2
        assert result.body["messages"][0]["role"] == "user"
        assert result.body["messages"][0]["content"][0]["type"] == "text"
        assert result.body["messages"][0]["content"][0]["text"] == "Hello"
        assert result.body["messages"][1]["role"] == "user"
        assert result.body["messages"][1]["content"][0]["type"] == "text"
        assert result.body["messages"][1]["content"][0]["text"] == "How are you?"

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
                    {"audio": "base64data", "format": "wav"},
                ]
            },
        )

        result = instance.format(data)

        assert len(result.body["messages"]) == 1
        assert result.body["messages"][0]["role"] == "user"
        assert result.body["messages"][0]["content"][0]["type"] == "input_audio"
        assert (
            result.body["messages"][0]["content"][0]["input_audio"]["data"]
            == "base64data"
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

        assert len(result.body["messages"]) == 3
        # System message from prefix
        assert result.body["messages"][0]["role"] == "system"
        assert result.body["messages"][0]["content"] == "You are a helpful assistant."
        # Text message
        assert result.body["messages"][1]["role"] == "user"
        assert result.body["messages"][1]["content"][0]["type"] == "text"
        # Image message
        assert result.body["messages"][2]["role"] == "user"
        assert result.body["messages"][2]["content"][0]["type"] == "image_url"

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
