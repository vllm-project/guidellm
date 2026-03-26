"""
Unit tests for VLLMResponseHandler.

Tests build_response: text, usage metrics, response_id, and edge cases.
"""

from __future__ import annotations

import pytest

from guidellm.backends.vllm_python.vllm_response import VLLMResponseHandler
from guidellm.schemas import GenerationRequest, GenerationResponse


@pytest.fixture
def request_fixture():
    """GenerationRequest for testing."""
    return GenerationRequest(columns={"text_column": ["test"]})


class TestVLLMResponseHandler:
    """
    Test cases for VLLMResponseHandler.build_response.
    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_build_response_with_usage(self, request_fixture):
        """
        build_response with text and usage populates metrics correctly.
        ## WRITTEN BY AI ##
        """
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        out = VLLMResponseHandler.build_response(
            request_fixture, "Hello world", usage, response_id="gen-123"
        )
        assert isinstance(out, GenerationResponse)
        assert out.request_id == request_fixture.request_id
        assert out.response_id == "gen-123"
        assert out.text == "Hello world"
        assert out.input_metrics.text_tokens == 5
        assert out.output_metrics.text_tokens == 2
        assert out.output_metrics.text_words == 2
        assert out.output_metrics.text_characters == 11

    @pytest.mark.sanity
    def test_build_response_without_usage(self, request_fixture):
        """
        build_response with None usage still fills word/char counts.
        ## WRITTEN BY AI ##
        """
        out = VLLMResponseHandler.build_response(request_fixture, "Hi", None)
        assert out.text == "Hi"
        assert out.output_metrics.text_words == 1
        assert out.output_metrics.text_characters == 2
        assert out.input_metrics.text_tokens is None

    @pytest.mark.sanity
    def test_build_response_empty_text(self, request_fixture):
        """
        build_response with empty text sets text to None.
        ## WRITTEN BY AI ##
        """
        out = VLLMResponseHandler.build_response(request_fixture, "", None)
        assert out.text is None
        assert out.output_metrics.text_words == 0
        assert out.output_metrics.text_characters == 0

    @pytest.mark.sanity
    def test_build_response_no_response_id(self, request_fixture):
        """
        build_response with no response_id defaults to None.
        ## WRITTEN BY AI ##
        """
        out = VLLMResponseHandler.build_response(request_fixture, "hi", None)
        assert out.response_id is None

    @pytest.mark.sanity
    def test_build_response_with_detailed_usage(self, request_fixture):
        """
        build_response with prompt_tokens_details and completion_tokens_details.
        ## WRITTEN BY AI ##
        """
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {
                "image_tokens": 3,
            },
            "completion_tokens_details": {
                "audio_tokens": 1,
            },
        }
        out = VLLMResponseHandler.build_response(request_fixture, "result", usage)
        assert out.input_metrics.text_tokens == 10
        assert out.input_metrics.image_tokens == 3
        assert out.output_metrics.text_tokens == 5
        assert out.output_metrics.audio_tokens == 1

    @pytest.mark.sanity
    def test_build_response_request_args_is_none(self, request_fixture):
        """
        build_response always sets request_args to None.
        ## WRITTEN BY AI ##
        """
        out = VLLMResponseHandler.build_response(request_fixture, "hi", None)
        assert out.request_args is None
