"""
Unit tests for GenerationRequest, GenerationResponse, GenerationRequestTimings.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import MeasuredRequestTimings
from guidellm.utils import StandardBaseModel


class TestGenerationRequest:
    """Test cases for GenerationRequest model."""

    @pytest.fixture(
        params=[
            {"content": "test content"},
            {
                "content": ["message1", "message2"],
                "request_type": "chat_completions",
                "params": {"temperature": 0.7},
            },
            {
                "request_id": "custom-id",
                "content": {"role": "user", "content": "test"},
                "stats": {"prompt_tokens": 50},
                "constraints": {"output_tokens": 100},
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
            "content",
            "params",
            "stats",
            "constraints",
        ]
        for field in expected_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerationRequest initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationRequest)
        assert instance.content == constructor_args["content"]

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
        # Invalid request_type
        with pytest.raises(ValidationError):
            GenerationRequest(content="test", request_type="invalid_type")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationRequest initialization without required field."""
        with pytest.raises(ValidationError):
            GenerationRequest()  # Missing required 'content' field

    @pytest.mark.smoke
    def test_auto_id_generation(self):
        """Test that request_id is auto-generated if not provided."""
        request1 = GenerationRequest(content="test1")
        request2 = GenerationRequest(content="test2")

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0
        assert len(request2.request_id) > 0

        # Should be valid UUIDs
        uuid.UUID(request1.request_id)
        uuid.UUID(request2.request_id)

    @pytest.mark.regression
    def test_content_types(self):
        """Test GenerationRequest with different content types."""
        # String content
        request1 = GenerationRequest(content="string content")
        assert request1.content == "string content"

        # List content
        request2 = GenerationRequest(content=["item1", "item2"])
        assert request2.content == ["item1", "item2"]

        # Dict content
        dict_content = {"role": "user", "content": "test"}
        request3 = GenerationRequest(content=dict_content)
        assert request3.content == dict_content

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test GenerationRequest serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert data_dict["content"] == constructor_args["content"]

        # Test reconstruction
        reconstructed = GenerationRequest.model_validate(data_dict)
        assert reconstructed.content == instance.content
        assert reconstructed.request_type == instance.request_type
        assert reconstructed.request_id == instance.request_id


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    @pytest.fixture(
        params=[
            {
                "request_id": "test-123",
                "request_args": {"model": "gpt-3.5-turbo"},
            },
            {
                "request_id": "test-456",
                "request_args": {"model": "gpt-4"},
                "value": "Generated text",
                "delta": "new text",
                "iterations": 5,
                "request_prompt_tokens": 50,
                "request_output_tokens": 100,
                "response_prompt_tokens": 55,
                "response_output_tokens": 95,
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
            "value",
            "delta",
            "iterations",
            "request_prompt_tokens",
            "request_output_tokens",
            "response_prompt_tokens",
            "response_output_tokens",
        ]
        for field in expected_fields:
            assert field in fields

        # Check properties exist
        assert hasattr(GenerationResponse, "prompt_tokens")
        assert hasattr(GenerationResponse, "output_tokens")
        assert hasattr(GenerationResponse, "total_tokens")
        assert hasattr(GenerationResponse, "preferred_prompt_tokens")
        assert hasattr(GenerationResponse, "preferred_output_tokens")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerationResponse initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationResponse)
        assert instance.request_id == constructor_args["request_id"]
        assert instance.request_args == constructor_args["request_args"]

        # Check defaults for optional fields
        if "value" not in constructor_args:
            assert instance.value is None
        if "delta" not in constructor_args:
            assert instance.delta is None
        if "iterations" not in constructor_args:
            assert instance.iterations == 0

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
    def test_prompt_tokens_property(self):
        """Test prompt_tokens property logic."""
        # When both are available, prefers response_prompt_tokens
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            response_prompt_tokens=55,
        )
        assert response1.prompt_tokens == 55

        # When only request_prompt_tokens is available
        response2 = GenerationResponse(
            request_id="test", request_args={}, request_prompt_tokens=50
        )
        assert response2.prompt_tokens == 50

        # When only response_prompt_tokens is available
        response3 = GenerationResponse(
            request_id="test", request_args={}, response_prompt_tokens=55
        )
        assert response3.prompt_tokens == 55

        # When neither is available
        response4 = GenerationResponse(request_id="test", request_args={})
        assert response4.prompt_tokens is None

    @pytest.mark.smoke
    def test_output_tokens_property(self):
        """Test output_tokens property logic."""
        # When both are available, prefers response_output_tokens
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            request_output_tokens=100,
            response_output_tokens=95,
        )
        assert response1.output_tokens == 95

        # When only request_output_tokens is available
        response2 = GenerationResponse(
            request_id="test", request_args={}, request_output_tokens=100
        )
        assert response2.output_tokens == 100

        # When only response_output_tokens is available
        response3 = GenerationResponse(
            request_id="test", request_args={}, response_output_tokens=95
        )
        assert response3.output_tokens == 95

        # When neither is available
        response4 = GenerationResponse(request_id="test", request_args={})
        assert response4.output_tokens is None

    @pytest.mark.smoke
    def test_total_tokens_property(self):
        """Test total_tokens property calculation."""
        # When both prompt and output tokens are available
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            response_prompt_tokens=50,
            response_output_tokens=100,
        )
        assert response1.total_tokens == 150

        # When one is missing
        response2 = GenerationResponse(
            request_id="test", request_args={}, response_prompt_tokens=50
        )
        assert response2.total_tokens is None

        # When both are missing
        response3 = GenerationResponse(request_id="test", request_args={})
        assert response3.total_tokens is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("preferred_source", "expected_prompt", "expected_output"),
        [
            ("request", 50, 100),
            ("response", 55, 95),
        ],
    )
    def test_preferred_token_methods(
        self, preferred_source, expected_prompt, expected_output
    ):
        """Test preferred_*_tokens methods."""
        response = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            request_output_tokens=100,
            response_prompt_tokens=55,
            response_output_tokens=95,
        )

        assert response.preferred_prompt_tokens(preferred_source) == expected_prompt
        assert response.preferred_output_tokens(preferred_source) == expected_output

    @pytest.mark.regression
    def test_preferred_tokens_fallback(self):
        """Test preferred_*_tokens methods with fallback logic."""
        # Only response tokens available
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            response_prompt_tokens=55,
            response_output_tokens=95,
        )

        assert response1.preferred_prompt_tokens("request") == 55  # Falls back
        assert response1.preferred_output_tokens("request") == 95  # Falls back

        # Only request tokens available
        response2 = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            request_output_tokens=100,
        )

        assert response2.preferred_prompt_tokens("response") == 50  # Falls back
        assert response2.preferred_output_tokens("response") == 100  # Falls back

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
        assert reconstructed.value == instance.value
        assert reconstructed.iterations == instance.iterations


class TestGenerationRequestTimings:
    """Test cases for GenerationRequestTimings model."""

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
        """Fixture providing valid GenerationRequestTimings instances."""
        constructor_args = request.param
        instance = GenerationRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationRequestTimings inheritance and type relationships."""
        assert issubclass(GenerationRequestTimings, MeasuredRequestTimings)
        assert issubclass(GenerationRequestTimings, StandardBaseModel)
        assert hasattr(GenerationRequestTimings, "model_dump")
        assert hasattr(GenerationRequestTimings, "model_validate")

        # Check inherited fields from MeasuredRequestTimings
        fields = GenerationRequestTimings.model_fields
        expected_inherited_fields = ["request_start", "request_end"]
        for field in expected_inherited_fields:
            assert field in fields

        # Check own fields
        expected_own_fields = ["first_iteration", "last_iteration"]
        for field in expected_own_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerationRequestTimings initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationRequestTimings)
        assert isinstance(instance, MeasuredRequestTimings)

        # Check field values
        expected_first = constructor_args.get("first_iteration")
        expected_last = constructor_args.get("last_iteration")
        assert instance.first_iteration == expected_first
        assert instance.last_iteration == expected_last

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerationRequestTimings with invalid field values."""
        # Invalid timestamp type
        with pytest.raises(ValidationError):
            GenerationRequestTimings(first_iteration="not_float")

        with pytest.raises(ValidationError):
            GenerationRequestTimings(last_iteration="not_float")

    @pytest.mark.smoke
    def test_optional_fields(self):
        """Test that all timing fields are optional."""
        # Should be able to create with no fields
        timings1 = GenerationRequestTimings()
        assert timings1.first_iteration is None
        assert timings1.last_iteration is None

        # Should be able to create with only one field
        timings2 = GenerationRequestTimings(first_iteration=123.0)
        assert timings2.first_iteration == 123.0
        assert timings2.last_iteration is None

        timings3 = GenerationRequestTimings(last_iteration=456.0)
        assert timings3.first_iteration is None
        assert timings3.last_iteration == 456.0

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test GenerationRequestTimings serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Test reconstruction
        reconstructed = GenerationRequestTimings.model_validate(data_dict)
        assert reconstructed.first_iteration == instance.first_iteration
        assert reconstructed.last_iteration == instance.last_iteration
        assert reconstructed.request_start == instance.request_start
        assert reconstructed.request_end == instance.request_end
