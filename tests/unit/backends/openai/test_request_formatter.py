"""
Unit tests for OpenAI request formatters.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest

from guidellm.backends.openai.request_formatter import (
    AudioTranscriptionResponseHandler,
    ChatCompletionsResponseHandler,
    GenerationRequestFormatter,
    GenerationRequestFormatterFactory,
    TextCompletionsResponseHandler,
)
from guidellm.schemas import GenerationRequest, UsageMetrics
from guidellm.utils import RegistryMixin


class TestGenerationRequestFormatter:
    """Test cases for GenerationRequestFormatter protocol.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationRequestFormatter is a Protocol with correct methods.

        ### WRITTEN BY AI ###
        """
        # Verify it's a Protocol by checking its methods
        assert hasattr(GenerationRequestFormatter, "format")


class TestGenerationRequestFormatterFactory:
    """Test cases for GenerationRequestFormatterFactory.

    ### WRITTEN BY AI ###
    """

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test that GenerationRequestFormatterFactory has correct inheritance.

        ### WRITTEN BY AI ###
        """
        assert issubclass(GenerationRequestFormatterFactory, RegistryMixin)
        assert hasattr(GenerationRequestFormatterFactory, "register")
        assert hasattr(GenerationRequestFormatterFactory, "create")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("request_type", "expected_class"),
        [
            ("text_completions", TextCompletionsResponseHandler),
            ("chat_completions", ChatCompletionsResponseHandler),
            ("audio_transcriptions", AudioTranscriptionResponseHandler),
            ("audio_translations", AudioTranscriptionResponseHandler),
        ],
        ids=[
            "text_completions",
            "chat_completions",
            "audio_transcriptions",
            "audio_translations",
        ],
    )
    def test_create(self, request_type, expected_class):
        """Test create method with various request types.

        ### WRITTEN BY AI ###
        """
        formatter = GenerationRequestFormatterFactory.create(request_type)
        assert isinstance(formatter, expected_class)

    @pytest.mark.sanity
    def test_create_invalid_request_type(self):
        """Test create method with invalid request type.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(ValueError, match="No response handler registered"):
            GenerationRequestFormatterFactory.create("invalid_type")


class TestTextCompletionsRequestFormatter:
    """Test cases for TextCompletionsResponseHandler (request formatter).

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of TextCompletionsResponseHandler.

        ### WRITTEN BY AI ###
        """
        return TextCompletionsResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test TextCompletionsResponseHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = TextCompletionsResponseHandler()
        assert hasattr(handler, "format")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test TextCompletionsResponseHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, TextCompletionsResponseHandler)

    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="text_completions")

        result = instance.format(data)

        assert result.body is not None
        assert isinstance(result.body, dict)

    @pytest.mark.sanity
    def test_format_with_model(self, valid_instances):
        """Test format method with model parameter.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="text_completions")

        result = instance.format(data, model="test-model")

        assert result.body["model"] == "test-model"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with streaming enabled.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="text_completions")

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
            request_type="text_completions",
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
        data = GenerationRequest(request_type="text_completions")

        result = instance.format(data, max_tokens=50)

        assert result.body["max_tokens"] == 50

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="text_completions")
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
            request_type="text_completions",
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
            request_type="text_completions",
            output_metrics=UsageMetrics(text_tokens=100),
        )

        result = instance.format(data)

        assert result.body["ignore_eos"] is True


class TestChatCompletionsRequestFormatter:
    """Test cases for ChatCompletionsResponseHandler (request formatter).

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of ChatCompletionsResponseHandler.

        ### WRITTEN BY AI ###
        """
        return ChatCompletionsResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ChatCompletionsResponseHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = ChatCompletionsResponseHandler()
        assert hasattr(handler, "format")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ChatCompletionsResponseHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, ChatCompletionsResponseHandler)

    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="chat_completions")

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
        data = GenerationRequest(request_type="chat_completions")

        result = instance.format(data, model="gpt-4")

        assert result.body["model"] == "gpt-4"

    @pytest.mark.sanity
    def test_format_streaming(self, valid_instances):
        """Test format method with streaming enabled.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="chat_completions")

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
            request_type="chat_completions",
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
        data = GenerationRequest(request_type="chat_completions")

        result = instance.format(data, max_tokens=50)

        assert result.body["max_completion_tokens"] == 50

    @pytest.mark.sanity
    def test_format_extras(self, valid_instances):
        """Test format method with extra parameters.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(request_type="chat_completions")
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
            request_type="chat_completions",
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
            request_type="chat_completions",
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
            request_type="chat_completions",
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
            request_type="chat_completions",
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
            request_type="chat_completions",
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
            request_type="chat_completions",
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


class TestAudioTranscriptionRequestFormatter:
    """Test cases for AudioTranscriptionResponseHandler (request formatter).

    ### WRITTEN BY AI ###
    """

    @pytest.fixture
    def valid_instances(self):
        """Create instance of AudioTranscriptionResponseHandler.

        ### WRITTEN BY AI ###
        """
        return AudioTranscriptionResponseHandler()

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test AudioTranscriptionResponseHandler class signatures.

        ### WRITTEN BY AI ###
        """
        handler = AudioTranscriptionResponseHandler()
        assert hasattr(handler, "format")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test AudioTranscriptionResponseHandler initialization.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        assert isinstance(instance, AudioTranscriptionResponseHandler)

    @pytest.mark.smoke
    def test_format_minimal(self, valid_instances):
        """Test format method with minimal audio data.

        ### WRITTEN BY AI ###
        """
        instance = valid_instances
        data = GenerationRequest(
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
            request_type="audio_transcriptions",
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
