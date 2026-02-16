"""
Unit tests for VLLMResponseHandler.

Tests compilation of OpenAI-style response dicts (choices+usage and text+usage)
and streaming SSE line parsing without request type.
"""

from __future__ import annotations

import pytest

from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.schemas import GenerationRequest, GenerationResponse


@pytest.fixture
def request_without_arguments():
    """GenerationRequest with no arguments (columns only)."""
    return GenerationRequest(columns={"text_column": ["test"]})


@pytest.fixture
def handler():
    """Fresh VLLMResponseHandler instance."""
    return VLLMResponseHandler()


class TestVLLMResponseHandler:
    """
    Test cases for VLLMResponseHandler.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_has_required_methods(self):
        """
        Handler exposes compile_non_streaming, add_streaming_line, compile_streaming.
        ## WRITTEN BY AI ##
        """
        assert hasattr(VLLMResponseHandler, "compile_non_streaming")
        assert hasattr(VLLMResponseHandler, "add_streaming_line")
        assert hasattr(VLLMResponseHandler, "compile_streaming")

    @pytest.mark.smoke
    def test_compile_non_streaming_choices_usage(
        self, handler, request_without_arguments
    ):
        """
        compile_non_streaming with choices+usage (text/chat) shape.
        ## WRITTEN BY AI ##
        """
        response_dict = {
            "id": "gen-123",
            "choices": [{"text": "Hello world"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
        out = handler.compile_non_streaming(request_without_arguments, response_dict)
        assert isinstance(out, GenerationResponse)
        assert out.request_id == request_without_arguments.request_id
        assert out.response_id == "gen-123"
        assert out.text == "Hello world"
        assert out.input_metrics.text_tokens == 5
        assert out.output_metrics.text_tokens == 2
        assert out.output_metrics.text_words == 2
        assert out.output_metrics.text_characters == 11

    @pytest.mark.smoke
    def test_compile_non_streaming_choices_message(
        self, handler, request_without_arguments
    ):
        """
        compile_non_streaming with choices[].message (chat) shape.
        ## WRITTEN BY AI ##
        """
        response_dict = {
            "choices": [{"message": {"content": "Hi there", "role": "assistant"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        out = handler.compile_non_streaming(request_without_arguments, response_dict)
        assert out.text == "Hi there"
        assert out.input_metrics.text_tokens == 1
        assert out.output_metrics.text_tokens == 2

    @pytest.mark.smoke
    def test_compile_non_streaming_text_usage(self, handler, request_without_arguments):
        """
        compile_non_streaming with text+usage (audio) shape.
        ## WRITTEN BY AI ##
        """
        response_dict = {
            "text": "Transcribed text",
            "usage": {"prompt_tokens": 0, "completion_tokens": 3},
        }
        out = handler.compile_non_streaming(request_without_arguments, response_dict)
        assert out.text == "Transcribed text"
        assert out.output_metrics.text_tokens == 3

    @pytest.mark.smoke
    def test_compile_non_streaming_empty_usage(
        self, handler, request_without_arguments
    ):
        """
        compile_non_streaming with no usage still fills text and word/char counts.
        ## WRITTEN BY AI ##
        """
        response_dict = {"choices": [{"text": "Hi"}]}
        out = handler.compile_non_streaming(request_without_arguments, response_dict)
        assert out.text == "Hi"
        assert out.output_metrics.text_words == 1
        assert out.output_metrics.text_characters == 2

    @pytest.mark.smoke
    def test_add_streaming_line_choices_text(self, handler):
        """
        add_streaming_line parses choices[].text delta.
        ## WRITTEN BY AI ##
        """
        line = 'data: {"choices": [{"text": "ab"}]}'
        assert handler.add_streaming_line(line) == 1
        assert handler.add_streaming_line('data: {"choices": [{"text": "c"}]}') == 1
        assert handler.streaming_texts == ["ab", "c"]

    @pytest.mark.smoke
    def test_add_streaming_line_choices_delta(self, handler):
        """
        add_streaming_line parses choices[].delta.content (chat) delta.
        ## WRITTEN BY AI ##
        """
        handler.add_streaming_line('data: {"choices": [{"delta": {"content": "x"}}]}')
        handler.add_streaming_line('data: {"choices": [{"delta": {"content": "y"}}]}')
        assert handler.streaming_texts == ["x", "y"]

    @pytest.mark.smoke
    def test_add_streaming_line_text(self, handler):
        """
        add_streaming_line parses audio-style text delta.
        ## WRITTEN BY AI ##
        """
        handler.add_streaming_line('data: {"text": "hello"}')
        assert handler.add_streaming_line('data: {"text": " world"}') == 1
        assert handler.streaming_texts == ["hello", " world"]

    @pytest.mark.smoke
    def test_add_streaming_line_done_returns_none(self, handler):
        """
        add_streaming_line returns None for data: [DONE].
        ## WRITTEN BY AI ##
        """
        assert handler.add_streaming_line("data: [DONE]") is None

    @pytest.mark.smoke
    def test_add_streaming_line_empty_returns_zero(self, handler):
        """
        add_streaming_line returns 0 for line with no content.
        ## WRITTEN BY AI ##
        """
        assert handler.add_streaming_line("data: {}") == 0

    @pytest.mark.smoke
    def test_add_streaming_line_invalid_json_returns_zero(self, handler):
        """
        add_streaming_line with invalid JSON in data returns 0; no crash.
        ## WRITTEN BY AI ##
        """
        result = handler.add_streaming_line("data: { invalid }")
        assert result == 0
        assert handler.streaming_texts == []

    @pytest.mark.smoke
    def test_compile_streaming_uses_accumulated_text(
        self, handler, request_without_arguments
    ):
        """
        compile_streaming builds response from accumulated chunks.
        ## WRITTEN BY AI ##
        """
        handler.add_streaming_line('data: {"choices": [{"text": "Hello"}]}')
        handler.add_streaming_line('data: {"choices": [{"text": " "}]}')
        handler.add_streaming_line('data: {"choices": [{"text": "world"}]}')
        out = handler.compile_streaming(request_without_arguments)
        assert out.text == "Hello world"
        assert out.request_id == request_without_arguments.request_id

    @pytest.mark.smoke
    def test_compile_streaming_stores_response_id(
        self, handler, request_without_arguments
    ):
        """
        compile_streaming uses response_id from streaming chunks.
        ## WRITTEN BY AI ##
        """
        handler.add_streaming_line(
            'data: {"id": "stream-1", "choices": [{"text": "x"}]}'
        )
        out = handler.compile_streaming(request_without_arguments)
        assert out.response_id == "stream-1"
        assert out.text == "x"

    @pytest.mark.smoke
    def test_request_args_from_arguments_if_present(self, request_without_arguments):
        """
        When request has arguments, request_args is serialized in response.
        ## WRITTEN BY AI ##
        """
        from unittest.mock import Mock

        # Request has no 'arguments'; handler uses getattr(request, "arguments", None).
        # Mock request with arguments.model_dump_json (e.g. from alternate pipelines).
        request_with_arguments = Mock(spec=GenerationRequest)
        request_with_arguments.request_id = request_without_arguments.request_id
        request_with_arguments.arguments = Mock()
        request_with_arguments.arguments.model_dump_json.return_value = (
            '{"body":{"prompt":"hi","max_tokens":10}}'
        )

        response_dict = {"choices": [{"text": "hey"}], "usage": {}}
        handler = VLLMResponseHandler()
        out = handler.compile_non_streaming(request_with_arguments, response_dict)
        assert out.request_args is not None
        assert "prompt" in out.request_args
        assert "max_tokens" in out.request_args
