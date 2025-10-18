"""
Unit tests for GenerationRequest, GenerationResponse, RequestTimings.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    RequestTimings,
)
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.utils import StandardBaseModel


class TestGenerationRequest:
    """Test cases for GenerationRequest model."""

    @pytest.fixture(
        params=[
            {
                "request_type": "text_completions",
                "arguments": GenerationRequestArguments(),
            },
            {
                "request_type": "chat_completions",
                "arguments": GenerationRequestArguments(body={"temperature": 0.7}),
            },
            {
                "request_id": "custom-id",
                "request_type": "text_completions",
                "arguments": GenerationRequestArguments(body={"prompt": "test"}),
            },
        ]
    )
    def valid_instances(self, request):
        """Fixture providing valid GenerationRequest instances."""
        constructor_args = request.param
        instance = GenerationRequest(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationRequest inheritance and type relationships."""
        assert issubclass(GenerationRequest, StandardBaseModel)
        assert hasattr(GenerationRequest, "model_dump")
        assert hasattr(GenerationRequest, "model_validate")

        # Check all expected fields are defined
        fields = GenerationRequest.model_fields
        expected_fields = [
            "request_id",
            "request_type",
            "arguments",
            "input_metrics",
            "output_metrics",
        ]
        for field in expected_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerationRequest initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationRequest)
        assert instance.arguments == constructor_args["arguments"]

        # Check defaults
        expected_request_type = constructor_args.get("request_type", "text_completions")
        assert instance.request_type == expected_request_type

        if "request_id" in constructor_args:
            assert instance.request_id == constructor_args["request_id"]
        else:
            assert isinstance(instance.request_id, str)
            # Should be valid UUID
            uuid.UUID(instance.request_id)

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerationRequest with invalid field values."""
        # Invalid request_type (not a string)
        with pytest.raises(ValidationError):
            GenerationRequest(request_type=123, arguments=GenerationRequestArguments())

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationRequest initialization without required field."""
        with pytest.raises(ValidationError):
            GenerationRequest()  # Missing required 'request_type' field

    @pytest.mark.smoke
    def test_auto_id_generation(self):
        """Test that request_id is auto-generated if not provided."""
        request1 = GenerationRequest(
            request_type="text_completions", arguments=GenerationRequestArguments()
        )
        request2 = GenerationRequest(
            request_type="text_completions", arguments=GenerationRequestArguments()
        )

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0
        assert len(request2.request_id) > 0

        # Should be valid UUIDs
        uuid.UUID(request1.request_id)
        uuid.UUID(request2.request_id)

    @pytest.mark.regression
    def test_content_types(self):
        """Test GenerationRequest with different argument types."""
        # Basic arguments
        request1 = GenerationRequest(
            request_type="text_completions", arguments=GenerationRequestArguments()
        )
        assert isinstance(request1.arguments, GenerationRequestArguments)

        # Arguments with body
        request2 = GenerationRequest(
            request_type="chat_completions",
            arguments=GenerationRequestArguments(body={"prompt": "test"}),
        )
        assert request2.arguments.body == {"prompt": "test"}

        # Arguments with headers
        request3 = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(
                headers={"Authorization": "Bearer token"}
            ),
        )
        assert request3.arguments.headers == {"Authorization": "Bearer token"}

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test GenerationRequest serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert "arguments" in data_dict

        # Test reconstruction
        reconstructed = GenerationRequest.model_validate(data_dict)
        assert reconstructed.arguments == instance.arguments
        assert reconstructed.request_type == instance.request_type
        assert reconstructed.request_id == instance.request_id


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
        ]
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
    def test_initialization(self, valid_instances):
        """Test GenerationResponse initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationResponse)
        assert instance.request_id == constructor_args["request_id"]
        assert instance.request_args == constructor_args["request_args"]

        # Check defaults for optional fields
        if "text" not in constructor_args:
            assert instance.text is None

        # Check default metrics
        assert hasattr(instance, "input_metrics")
        assert hasattr(instance, "output_metrics")

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerationResponse with invalid field values."""
        # Invalid iterations type
        with pytest.raises(ValidationError):
            GenerationResponse(request_id="test", request_args={}, iterations="not_int")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationResponse initialization without required fields."""
        with pytest.raises(ValidationError):
            GenerationResponse()  # Missing required fields

        with pytest.raises(ValidationError):
            GenerationResponse(request_id="test")  # Missing request_args

    @pytest.mark.smoke
    def test_compile_stats_method(self):
        """Test compile_stats method functionality."""
        from guidellm.schemas.request import GenerationRequestArguments

        response = GenerationResponse(
            request_id="test-123", request_args="test_args", text="Generated response"
        )

        request = GenerationRequest(
            request_id="test-123",
            request_type="text_completions",
            arguments=GenerationRequestArguments(),
        )

        request_info = RequestInfo(request_id="test-123")

        # Test that compile_stats works
        stats = response.compile_stats(request, request_info)
        assert stats is not None
        assert hasattr(stats, "request_id")
        assert stats.request_id == "test-123"

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
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
        if hasattr(instance, "text"):
            assert reconstructed.text == instance.text


class TestRequestTimings:
    """Test cases for RequestTimings model."""

    @pytest.fixture(
        params=[
            {},
            {"first_iteration": 1234567890.0},
            {"last_iteration": 1234567895.0},
            {
                "first_iteration": 1234567890.0,
                "last_iteration": 1234567895.0,
            },
        ]
    )
    def valid_instances(self, request):
        """Fixture providing valid RequestTimings instances."""
        constructor_args = request.param
        instance = RequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test RequestTimings inheritance and type relationships."""
        assert issubclass(RequestTimings, RequestTimings)
        assert hasattr(RequestTimings, "model_dump")
        assert hasattr(RequestTimings, "model_validate")

        # Check inherited fields from RequestTimings
        fields = RequestTimings.model_fields
        expected_inherited_fields = ["request_start", "request_end"]
        for field in expected_inherited_fields:
            assert field in fields

        # Check own fields
        expected_own_fields = ["first_iteration", "last_iteration"]
        for field in expected_own_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test RequestTimings initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, RequestTimings)
        assert isinstance(instance, RequestTimings)

        # Check field values
        expected_first = constructor_args.get("first_iteration")
        expected_last = constructor_args.get("last_iteration")
        assert instance.first_iteration == expected_first
        assert instance.last_iteration == expected_last

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test RequestTimings with invalid field values."""
        # Invalid timestamp type
        with pytest.raises(ValidationError):
            RequestTimings(first_iteration="not_float")

        with pytest.raises(ValidationError):
            RequestTimings(last_iteration="not_float")

    @pytest.mark.smoke
    def test_optional_fields(self):
        """Test that all timing fields are optional."""
        # Should be able to create with no fields
        timings1 = RequestTimings()
        assert timings1.first_iteration is None
        assert timings1.last_iteration is None

        # Should be able to create with only one field
        timings2 = RequestTimings(first_iteration=123.0)
        assert timings2.first_iteration == 123.0
        assert timings2.last_iteration is None

        timings3 = RequestTimings(last_iteration=456.0)
        assert timings3.first_iteration is None
        assert timings3.last_iteration == 456.0

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test RequestTimings serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Test reconstruction
        reconstructed = RequestTimings.model_validate(data_dict)
        assert reconstructed.first_iteration == instance.first_iteration
        assert reconstructed.last_iteration == instance.last_iteration
        assert reconstructed.request_start == instance.request_start
        assert reconstructed.request_end == instance.request_end
