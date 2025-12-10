from __future__ import annotations

import pytest

from guidellm.backends import (
    AudioResponseHandler,
    ChatCompletionsResponseHandler,
    GenerationResponseHandler,
    GenerationResponseHandlerFactory,
    TextCompletionsResponseHandler,
)
from guidellm.schemas import GenerationRequest, GenerationRequestArguments
from guidellm.utils import RegistryMixin


@pytest.fixture
def generation_request():
    """Create a basic GenerationRequest for testing."""
    return GenerationRequest(
        request_type="text_completions",
        arguments=GenerationRequestArguments(
            method="POST",
            url="http://test.com",
            body={"prompt": "Test prompt"},
        ),
    )


class TestGenerationResponseHandler:
    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationResponseHandler is a Protocol with correct methods."""

        # Verify it's a Protocol by checking its methods
        assert hasattr(GenerationResponseHandler, "compile_non_streaming")
        assert hasattr(GenerationResponseHandler, "add_streaming_line")
        assert hasattr(GenerationResponseHandler, "compile_streaming")


class TestGenerationResponseHandlerFactory:
    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test that GenerationResponseHandlerFactory has correct inheritance."""
        assert issubclass(GenerationResponseHandlerFactory, RegistryMixin)
        assert hasattr(GenerationResponseHandlerFactory, "register")
        assert hasattr(GenerationResponseHandlerFactory, "create")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("request_type", "handler_overrides", "expected_class"),
        [
            ("text_completions", None, TextCompletionsResponseHandler),
            ("chat_completions", None, ChatCompletionsResponseHandler),
            ("audio_transcriptions", None, AudioResponseHandler),
            ("audio_translations", None, AudioResponseHandler),
            (
                "text_completions",
                {"text_completions": ChatCompletionsResponseHandler},
                ChatCompletionsResponseHandler,
            ),
        ],
        ids=[
            "text_completions",
            "chat_completions",
            "audio_transcriptions",
            "audio_translations",
            "override_text_completions",
        ],
    )
    def test_create(
        self,
        request_type,
        handler_overrides,
        expected_class,
    ):
        """Test create method with various request types and overrides."""
        handler = GenerationResponseHandlerFactory.create(
            request_type, handler_overrides
        )
        assert isinstance(handler, expected_class)

    @pytest.mark.sanity
    def test_create_invalid_request_type(self):
        """Test create method with invalid request type."""
        with pytest.raises(ValueError, match="No response handler registered"):
            GenerationResponseHandlerFactory.create("invalid_type")


class TestTextCompletionsResponseHandler:
    @pytest.fixture(
        params=[{}],
        ids=["default"],
    )
    def valid_instances(self, request):
        """Create instance of TextCompletionsResponseHandler."""
        return TextCompletionsResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test TextCompletionsResponseHandler class signatures."""
        handler = TextCompletionsResponseHandler()
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
        """Test TextCompletionsResponseHandler initialization."""
        instance = valid_instances
        assert isinstance(instance, TextCompletionsResponseHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

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
        """Test compile_non_streaming method."""
        instance: TextCompletionsResponseHandler = valid_instances

        result = instance.compile_non_streaming(generation_request, response)

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
        """Test streaming with add_streaming_line and compile_streaming."""
        instance: TextCompletionsResponseHandler = valid_instances

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
            elif result is None:
                break

        response = instance.compile_streaming(generation_request)
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
        """Test extract_line_data method."""
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
        """Test extract_choices_and_usage method."""
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
        """Test extract_metrics method."""
        instance = valid_instances
        input_metrics, output_metrics = instance.extract_metrics(usage, text)

        assert input_metrics.text_tokens == expected_input_tokens
        assert output_metrics.text_tokens == expected_output_tokens
        assert output_metrics.text_words == (len(text.split()) if text else 0)
        assert output_metrics.text_characters == len(text)


class TestChatCompletionsResponseHandler:
    @pytest.fixture(
        params=[{}],
        ids=["default"],
    )
    def valid_instances(self, request):
        """Create instance of ChatCompletionsResponseHandler."""
        return ChatCompletionsResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ChatCompletionsResponseHandler class signatures."""
        handler = ChatCompletionsResponseHandler()
        assert issubclass(
            ChatCompletionsResponseHandler, TextCompletionsResponseHandler
        )
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ChatCompletionsResponseHandler initialization."""
        instance: ChatCompletionsResponseHandler = valid_instances
        assert isinstance(instance, ChatCompletionsResponseHandler)
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

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
        """Test compile_non_streaming method for chat completions."""
        instance: ChatCompletionsResponseHandler = valid_instances

        result = instance.compile_non_streaming(generation_request, response)

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
        """Test streaming pathway for chat completions."""
        instance = ChatCompletionsResponseHandler()

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
            elif result is None:
                break

        response = instance.compile_streaming(generation_request)
        assert response.text == expected_text
        assert response.input_metrics.text_tokens == expected_input_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens


class TestAudioResponseHandler:
    @pytest.fixture(
        params=[{}],
        ids=["default"],
    )
    def valid_instances(self, request):
        """Create instance of AudioResponseHandler."""
        return AudioResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test AudioResponseHandler class signatures."""
        handler = AudioResponseHandler()
        assert hasattr(handler, "compile_non_streaming")
        assert hasattr(handler, "add_streaming_line")
        assert hasattr(handler, "compile_streaming")
        assert hasattr(handler, "extract_metrics")
        assert hasattr(handler, "streaming_buffer")
        assert hasattr(handler, "streaming_texts")
        assert hasattr(handler, "streaming_usage")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test AudioResponseHandler initialization."""
        instance = valid_instances
        assert isinstance(instance, AudioResponseHandler)
        assert isinstance(instance.streaming_buffer, bytearray)
        assert len(instance.streaming_buffer) == 0
        assert instance.streaming_texts == []
        assert instance.streaming_usage is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "response",
            "expected_text",
            "expected_audio_tokens",
            "expected_output_tokens",
            "expected_seconds",
        ),
        [
            (
                {
                    "text": "Hello, world!",
                    "usage": {"input_tokens": 1000, "output_tokens": 3},
                },
                "Hello, world!",
                1000,
                3,
                0,
            ),
            (
                {
                    "text": "Test transcription",
                    "usage": {
                        "audio_tokens": 500,
                        "output_tokens": 5,
                        "seconds": 10,
                    },
                },
                "Test transcription",
                500,
                5,
                10,
            ),
            (
                {"text": "", "usage": {}},
                "",
                None,
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
        expected_audio_tokens,
        expected_output_tokens,
        expected_seconds,
    ):
        """Test compile_non_streaming method for audio."""
        instance: AudioResponseHandler = valid_instances

        result = instance.compile_non_streaming(generation_request, response)

        assert result.text == expected_text
        assert result.input_metrics.audio_tokens == expected_audio_tokens
        assert result.output_metrics.text_tokens == expected_output_tokens
        assert result.output_metrics.text_words == len(expected_text.split())
        assert result.output_metrics.text_characters == len(expected_text)
        assert result.input_metrics.audio_seconds == expected_seconds

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "lines",
            "expected_text",
            "expected_audio_tokens",
            "expected_output_tokens",
            "expected_seconds",
        ),
        [
            (
                [
                    '{"text": "Hello", "usage": {}}',
                    (
                        '{"text": " world!", '
                        '"usage": {"audio_tokens": 1000, '
                        '"output_tokens": 3, "seconds": 5}}'
                    ),
                    "data: [DONE]",
                ],
                "Hello world!",
                1000,
                3,
                5,
            ),
            (
                [
                    '{"text": "Test", "usage": {}}',
                    "data: [DONE]",
                ],
                "Test",
                None,
                None,
                None,
            ),
            (
                ["data: [DONE]"],
                "",
                None,
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
        expected_audio_tokens,
        expected_output_tokens,
        expected_seconds,
    ):
        """Test streaming pathway for audio."""
        instance: AudioResponseHandler = valid_instances

        updated_count = 0
        for line in lines:
            result = instance.add_streaming_line(line)
            if result == 1:
                updated_count += 1
            elif result is None:
                break

        response = instance.compile_streaming(generation_request)
        assert response.text == expected_text
        assert response.input_metrics.audio_tokens == expected_audio_tokens
        assert response.output_metrics.text_tokens == expected_output_tokens
        assert response.input_metrics.audio_seconds == expected_seconds

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "usage",
            "text",
            "expected_audio_tokens",
            "expected_output_tokens",
            "expected_seconds",
        ),
        [
            (
                {"input_tokens": 1000, "output_tokens": 5},
                "Hello world",
                1000,
                5,
                0,
            ),
            (
                {
                    "audio_tokens": 500,
                    "output_tokens": 3,
                    "seconds": 10,
                },
                "Test",
                500,
                3,
                10,
            ),
            (
                {
                    "input_token_details": {
                        "audio_tokens": 800,
                        "text_tokens": 5,
                        "seconds": 10,
                    },
                    "output_token_details": {"text_tokens": 10},
                },
                "Hello world test",
                800,
                10,
                10,
            ),
            (None, "Hello", None, None, None),
            ({}, "", None, None, None),
        ],
    )
    def test_extract_metrics(
        self,
        valid_instances,
        usage,
        text,
        expected_audio_tokens,
        expected_output_tokens,
        expected_seconds,
    ):
        """Test extract_metrics method for audio."""
        instance: AudioResponseHandler = valid_instances
        input_metrics, output_metrics = instance.extract_metrics(usage, text)

        assert input_metrics.audio_tokens == expected_audio_tokens
        assert input_metrics.audio_seconds == expected_seconds
        assert output_metrics.text_tokens == expected_output_tokens
        assert output_metrics.text_words == (len(text.split()) if text else 0)
        assert output_metrics.text_characters == len(text)
