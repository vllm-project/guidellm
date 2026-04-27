"""
Unit tests for GenerationResponse.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    StandardBaseModel,
    UsageMetrics,
)


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    @pytest.fixture(
        params=[
            {
                "request_id": "test-123",
                "request_args": "model=gpt-3.5-turbo",
            },
            {
                "request_id": "test-456",
                "request_args": "model=gpt-4",
                "text": "Generated text",
            },
            {
                "request_id": "test-789",
                "request_args": None,
                "text": "Another response",
                "input_metrics": UsageMetrics(text_tokens=50),
                "output_metrics": UsageMetrics(text_tokens=25),
            },
        ],
        ids=["minimal", "with_text", "with_metrics"],
    )
    def valid_instances(self, request):
        """Fixture providing valid GenerationResponse instances."""
        constructor_args = request.param
        instance = GenerationResponse(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationResponse inheritance and type relationships."""
        assert issubclass(GenerationResponse, StandardBaseModel)
        assert hasattr(GenerationResponse, "model_dump")
        assert hasattr(GenerationResponse, "model_validate")

        # Check all expected fields and properties are defined
        fields = GenerationResponse.model_fields
        expected_fields = [
            "request_id",
            "request_args",
            "text",
            "input_metrics",
            "output_metrics",
        ]
        for field in expected_fields:
            assert field in fields

        # Check methods exist
        assert hasattr(GenerationResponse, "compile_stats")

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[GenerationResponse, dict[str, str]]
    ):
        """Test GenerationResponse initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationResponse)
        assert instance.request_id == constructor_args["request_id"]
        assert instance.request_args == constructor_args["request_args"]

        # Check defaults for optional fields
        if "text" not in constructor_args:
            assert instance.text is None
        else:
            assert instance.text == constructor_args["text"]

        # Check metrics (either provided or default)
        assert hasattr(instance, "input_metrics")
        assert hasattr(instance, "output_metrics")
        assert isinstance(instance.input_metrics, UsageMetrics)
        assert isinstance(instance.output_metrics, UsageMetrics)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("request_id", None),
            ("request_id", 123),
            ("request_id", {}),
            ("request_args", 123),
            ("request_args", []),
            ("text", 123),
            ("text", []),
            ("input_metrics", "invalid"),
            ("output_metrics", "invalid"),
        ],
    )
    def test_invalid_initialization_values(self, field: str, value: Any):
        """Test GenerationResponse with invalid field values."""
        data = {"request_id": "test-id", "request_args": "test_args"}
        data[field] = value
        with pytest.raises(ValidationError):
            GenerationResponse(**data)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationResponse initialization without required fields."""
        with pytest.raises(ValidationError):
            GenerationResponse()

        with pytest.raises(ValidationError):
            GenerationResponse(request_id="test")

        with pytest.raises(ValidationError):
            GenerationResponse(request_args="test")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("prefer_response", "expected_tokens"),
        [
            (True, 75),
            (False, 60),
        ],
    )
    def test_compile_stats(self, prefer_response: bool, expected_tokens: int):
        """Test compile_stats method functionality."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text="Generated response",
            input_metrics=UsageMetrics(text_tokens=50),
            output_metrics=UsageMetrics(text_tokens=25),
        )

        request = GenerationRequest(
            request_id="test-123",
            input_metrics=UsageMetrics(text_tokens=40),
            output_metrics=UsageMetrics(text_tokens=20),
        )

        request_info = RequestInfo(request_id="test-123", status="completed")

        stats = response.compile_stats(request, request_info, prefer_response)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.output == "Generated response"

        total_tokens = (
            stats.input_metrics.text_tokens + stats.output_metrics.text_tokens
        )
        assert total_tokens == expected_tokens

    @pytest.mark.smoke
    def test_compile_stats_with_defaults(self):
        """Test compile_stats with default metrics."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text="Generated response",
        )

        request = GenerationRequest(
            request_id="test-123",
        )

        request_info = RequestInfo(request_id="test-123", status="completed")

        stats = response.compile_stats(request, request_info)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.output == "Generated response"

    @pytest.mark.smoke
    def test_compile_stats_failed_request(self):
        """Test compile_stats with failed request status."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
            text=None,
            output_metrics=UsageMetrics(text_tokens=25),
        )

        request = GenerationRequest(
            request_id="test-123",
            output_metrics=UsageMetrics(text_tokens=20),
        )

        request_info = RequestInfo(request_id="test-123", status="errored")

        stats = response.compile_stats(request, request_info)
        assert isinstance(stats, GenerativeRequestStats)
        assert stats.request_id == "test-123"
        assert stats.info.status == "errored"

    @pytest.mark.smoke
    def test_compile_stats_persists_tool_calls(self):
        """
        compile_stats carries tool_calls from the response to the stats.

        ## WRITTEN BY AI ##
        """
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        response = GenerationResponse(
            request_id="test-tc",
            request_args="test_args",
            text=None,
            tool_calls=tool_calls,
        )

        request = GenerationRequest(request_id="test-tc")
        info = RequestInfo(request_id="test-tc", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.tool_calls == tool_calls

    @pytest.mark.smoke
    def test_compile_stats_tool_calls_none_when_absent(self):
        """
        compile_stats leaves tool_calls as None for plain-text responses.

        ## WRITTEN BY AI ##
        """
        response = GenerationResponse(
            request_id="test-plain",
            request_args="test_args",
            text="Hello world",
        )

        request = GenerationRequest(request_id="test-plain")
        info = RequestInfo(request_id="test-plain", status="completed")

        stats = response.compile_stats(request, info)
        assert stats.tool_calls is None

    @pytest.mark.sanity
    def test_compile_stats_mismatched_request_id(self):
        """Test compile_stats with mismatched request IDs."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
        )

        request = GenerationRequest(
            request_id="test-456",
        )

        request_info = RequestInfo(request_id="test-123")

        with pytest.raises(ValueError, match="Mismatched request IDs"):
            response.compile_stats(request, request_info)

    @pytest.mark.sanity
    def test_compile_stats_mismatched_info_id(self):
        """Test compile_stats with mismatched info request ID."""
        response = GenerationResponse(
            request_id="test-123",
            request_args="test_args",
        )

        request = GenerationRequest(
            request_id="test-123",
        )

        request_info = RequestInfo(request_id="test-456")

        with pytest.raises(ValueError, match="Mismatched request IDs"):
            response.compile_stats(request, request_info)

    @pytest.mark.sanity
    def test_marshalling(
        self, valid_instances: tuple[GenerationResponse, dict[str, str]]
    ):
        """Test GenerationResponse serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["request_id"] == constructor_args["request_id"]
        assert data_dict["request_args"] == constructor_args["request_args"]

        # Test reconstruction
        reconstructed = GenerationResponse.model_validate(data_dict)
        assert reconstructed.request_id == instance.request_id
        assert reconstructed.request_args == instance.request_args
        assert reconstructed.text == instance.text
        assert (
            reconstructed.input_metrics.model_dump()
            == instance.input_metrics.model_dump()
        )
        assert (
            reconstructed.output_metrics.model_dump()
            == instance.output_metrics.model_dump()
        )


class TestGenerativeRequestStatsToolCalls:
    """
    Tests for tool_calls field on GenerativeRequestStats.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_tool_calls_round_trips_through_serialization(self):
        """
        tool_calls survives model_dump / model_validate.

        ## WRITTEN BY AI ##
        """
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fn", "arguments": "{}"},
            }
        ]
        timings = RequestTimings(resolve_start=1.0, resolve_end=2.0)
        stats = GenerativeRequestStats(
            request_id="rt-1",
            info=RequestInfo(
                request_id="rt-1", status="completed", timings=timings
            ),
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(),
            tool_calls=tool_calls,
        )
        data = stats.model_dump()
        assert data["tool_calls"] == tool_calls

        restored = GenerativeRequestStats.model_validate(data)
        assert restored.tool_calls == tool_calls

    @pytest.mark.smoke
    def test_tool_calls_defaults_to_none(self):
        """
        tool_calls is None when not provided.

        ## WRITTEN BY AI ##
        """
        stats = GenerativeRequestStats(
            request_id="rt-2",
            info=RequestInfo(request_id="rt-2", status="completed"),
            input_metrics=UsageMetrics(),
            output_metrics=UsageMetrics(),
        )
        assert stats.tool_calls is None
