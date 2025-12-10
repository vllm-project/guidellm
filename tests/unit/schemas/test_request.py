"""
Unit tests for GenerationRequest, GenerationRequestArguments, and UsageMetrics.
"""

from __future__ import annotations

import typing
import uuid

import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    StandardBaseDict,
    StandardBaseModel,
    UsageMetrics,
)
from guidellm.schemas.request import GenerativeRequestType


@pytest.mark.smoke
def test_generative_request_type():
    """Test that GenerativeRequestType is defined correctly."""
    assert hasattr(typing, "get_args")
    args = typing.get_args(GenerativeRequestType)
    assert len(args) == 4
    assert "text_completions" in args
    assert "chat_completions" in args
    assert "audio_transcriptions" in args
    assert "audio_translations" in args


class TestGenerationRequestArguments:
    """Test cases for GenerationRequestArguments model."""

    @pytest.fixture(
        params=[
            {},
            {"method": "POST", "body": {"prompt": "test"}},
            {
                "method": "GET",
                "headers": {"Authorization": "Bearer token"},
                "params": {"key": "value"},
            },
            {
                "method": "POST",
                "stream": True,
                "headers": {"Content-Type": "application/json"},
                "params": {"limit": 10},
                "body": {"prompt": "hello"},
                "files": {"file": "data.txt"},
            },
        ],
        ids=["empty", "method_body", "method_headers_params", "all_fields"],
    )
    def valid_instances(self, request):
        """Fixture providing valid GenerationRequestArguments instances."""
        constructor_args = request.param
        instance = GenerationRequestArguments(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test GenerationRequestArguments inheritance and type relationships."""
        assert issubclass(GenerationRequestArguments, StandardBaseDict)
        assert hasattr(GenerationRequestArguments, "model_dump")
        assert hasattr(GenerationRequestArguments, "model_validate")
        assert hasattr(GenerationRequestArguments, "model_combine")

        # Check fields
        fields = GenerationRequestArguments.model_fields
        expected_fields = ["method", "stream", "headers", "params", "body", "files"]
        for field in expected_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test GenerationRequestArguments initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, GenerationRequestArguments)

        # Check field values
        for key, expected_value in constructor_args.items():
            assert getattr(instance, key) == expected_value

        # Check defaults for fields not provided
        for field in ["method", "stream", "headers", "params", "body", "files"]:
            if field not in constructor_args:
                assert getattr(instance, field) is None

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerationRequestArguments with invalid field values."""
        # Invalid method type
        with pytest.raises(ValidationError):
            GenerationRequestArguments(method=123)

        # Invalid stream type
        with pytest.raises(ValidationError):
            GenerationRequestArguments(stream="not_bool")

        # Invalid headers type
        with pytest.raises(ValidationError):
            GenerationRequestArguments(headers="not_dict")

        # Invalid params type
        with pytest.raises(ValidationError):
            GenerationRequestArguments(params="not_dict")

        # Invalid body type
        with pytest.raises(ValidationError):
            GenerationRequestArguments(body="not_dict")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationRequestArguments initialization without any fields."""
        # Should succeed since all fields are optional
        instance = GenerationRequestArguments()
        assert isinstance(instance, GenerationRequestArguments)
        assert instance.method is None
        assert instance.stream is None
        assert instance.headers is None
        assert instance.params is None
        assert instance.body is None
        assert instance.files is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        (
            "base_kwargs",
            "additional_kwargs",
            "expected_method",
            "expected_headers",
            "expected_body",
        ),
        [
            (
                {
                    "method": "POST",
                    "headers": {"Auth": "token1"},
                    "body": {"key1": "val1"},
                },
                {
                    "method": "GET",
                    "headers": {"Type": "json"},
                    "body": {"key2": "val2"},
                },
                "GET",
                {"Auth": "token1", "Type": "json"},
                {"key1": "val1", "key2": "val2"},
            ),
            (
                {"method": "POST"},
                {"stream": True, "headers": {"Auth": "token"}},
                "POST",
                {"Auth": "token"},
                None,
            ),
            (
                {"params": {"page": 1}, "files": {"file1": "data"}},
                {"params": {"limit": 10}, "files": {"file2": "more"}},
                None,
                None,
                None,
            ),
        ],
        ids=["overwrite_and_merge", "partial_merge", "params_and_files"],
    )
    def test_model_combine(
        self,
        base_kwargs,
        additional_kwargs,
        expected_method,
        expected_headers,
        expected_body,
    ):
        """Test GenerationRequestArguments.model_combine method."""
        base_args = GenerationRequestArguments(**base_kwargs)
        additional_args = GenerationRequestArguments(**additional_kwargs)

        # Combine args
        result = base_args.model_combine(additional_args)

        # Check method and stream (overwrite behavior)
        if expected_method is not None:
            assert result.method == expected_method
        elif "method" in base_kwargs:
            assert result.method == base_kwargs["method"]

        if "stream" in additional_kwargs:
            assert result.stream == additional_kwargs["stream"]
        elif "stream" in base_kwargs:
            assert result.stream == base_kwargs.get("stream")

        # Check headers (merge behavior)
        if expected_headers is not None:
            assert result.headers == expected_headers

        # Check body (merge behavior)
        if expected_body is not None:
            assert result.body == expected_body

        # Check params merge
        if "params" in base_kwargs and "params" in additional_kwargs:
            expected_params = {**base_kwargs["params"], **additional_kwargs["params"]}
            assert result.params == expected_params

        # Check files merge
        if "files" in base_kwargs and "files" in additional_kwargs:
            expected_files = {**base_kwargs["files"], **additional_kwargs["files"]}
            assert result.files == expected_files

    @pytest.mark.smoke
    def test_model_combine_with_dict(self):
        """Test GenerationRequestArguments.model_combine with dict input."""
        base_args = GenerationRequestArguments(
            method="POST",
            headers={"Authorization": "Bearer token1"},
        )

        additional_dict = {
            "method": "GET",
            "headers": {"Content-Type": "application/json"},
        }

        # Combine with dict
        result = base_args.model_combine(additional_dict)

        # Method should be overwritten
        assert result.method == "GET"

        # Headers should be merged
        assert result.headers == {
            "Authorization": "Bearer token1",
            "Content-Type": "application/json",
        }

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test GenerationRequestArguments serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Test reconstruction
        reconstructed = GenerationRequestArguments.model_validate(data_dict)
        for key, expected_value in constructor_args.items():
            assert getattr(reconstructed, key) == expected_value


class TestUsageMetrics:
    """Test cases for UsageMetrics model."""

    @pytest.fixture(
        params=[
            {},
            {"text_tokens": 100, "text_words": 50},
            {"image_tokens": 200, "image_count": 5, "image_pixels": 1024},
            {"audio_tokens": 150, "audio_seconds": 30.5},
            {
                "video_tokens": 75,
                "video_frames": 300,
                "video_seconds": 10.0,
                "video_bytes": 5000000,
            },
            {
                "text_tokens": 100,
                "image_tokens": 50,
                "video_tokens": 25,
                "audio_tokens": 25,
            },
        ],
        ids=[
            "empty",
            "text_metrics",
            "image_metrics",
            "audio_metrics",
            "video_metrics",
            "all_token_types",
        ],
    )
    def valid_instances(self, request):
        """Fixture providing valid UsageMetrics instances."""
        constructor_args = request.param
        instance = UsageMetrics(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test UsageMetrics inheritance and type relationships."""
        assert issubclass(UsageMetrics, StandardBaseDict)
        assert hasattr(UsageMetrics, "model_dump")
        assert hasattr(UsageMetrics, "model_validate")
        assert hasattr(UsageMetrics, "add_text_metrics")

        # Check fields
        fields = UsageMetrics.model_fields
        expected_fields = [
            "text_tokens",
            "text_words",
            "text_characters",
            "image_tokens",
            "image_count",
            "image_pixels",
            "image_bytes",
            "video_tokens",
            "video_frames",
            "video_seconds",
            "video_bytes",
            "audio_tokens",
            "audio_samples",
            "audio_seconds",
            "audio_bytes",
        ]
        for field in expected_fields:
            assert field in fields

        # Check computed property
        assert hasattr(UsageMetrics, "total_tokens")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test UsageMetrics initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, UsageMetrics)

        # Check field values
        for key, expected_value in constructor_args.items():
            assert getattr(instance, key) == expected_value

        # Check defaults for fields not provided
        all_fields = [
            "text_tokens",
            "text_words",
            "text_characters",
            "image_tokens",
            "image_count",
            "image_pixels",
            "image_bytes",
            "video_tokens",
            "video_frames",
            "video_seconds",
            "video_bytes",
            "audio_tokens",
            "audio_samples",
            "audio_seconds",
            "audio_bytes",
        ]
        for field in all_fields:
            if field not in constructor_args:
                assert getattr(instance, field) is None

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test UsageMetrics with invalid field values."""
        # Invalid token count type
        with pytest.raises(ValidationError):
            UsageMetrics(text_tokens="not_int")

        # Invalid seconds type
        with pytest.raises(ValidationError):
            UsageMetrics(audio_seconds="not_float")

        # Invalid image count type
        with pytest.raises(ValidationError):
            UsageMetrics(image_count=1.5)

        # Invalid video frames type
        with pytest.raises(ValidationError):
            UsageMetrics(video_frames="invalid")

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test UsageMetrics initialization without any fields."""
        # Should succeed since all fields are optional
        metrics = UsageMetrics()
        assert isinstance(metrics, UsageMetrics)
        assert metrics.text_tokens is None
        assert metrics.image_tokens is None
        assert metrics.video_tokens is None
        assert metrics.audio_tokens is None

    @pytest.mark.smoke
    def test_optional_fields(self):
        """Test that all usage metric fields are optional."""
        # Should be able to create with no fields
        metrics = UsageMetrics()
        assert metrics.text_tokens is None
        assert metrics.image_tokens is None
        assert metrics.audio_tokens is None
        assert metrics.video_tokens is None

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("metrics_kwargs", "expected_total"),
        [
            ({}, None),
            ({"text_tokens": 100}, 100),
            ({"text_tokens": 100, "image_tokens": 50}, 150),
            (
                {
                    "text_tokens": 100,
                    "image_tokens": 50,
                    "video_tokens": 25,
                    "audio_tokens": 25,
                },
                200,
            ),
            ({"image_tokens": 50, "video_tokens": 25}, 75),
        ],
        ids=[
            "no_tokens",
            "text_only",
            "text_and_image",
            "all_modalities",
            "image_and_video",
        ],
    )
    def test_total_tokens_property(self, metrics_kwargs, expected_total):
        """Test UsageMetrics.total_tokens computed property."""
        metrics = UsageMetrics(**metrics_kwargs)
        assert metrics.total_tokens == expected_total

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("initial_chars", "initial_words", "text", "expected_chars", "expected_words"),
        [
            (None, None, "Hello world", 11, 2),
            (0, 0, "Hello world", 11, 2),
            (11, 2, "Test message", 23, 4),
            (0, 0, "One", 3, 1),
            (10, 5, "", 10, 5),
        ],
        ids=[
            "first_text",
            "from_zero",
            "accumulate",
            "single_word",
            "empty_string",
        ],
    )
    def test_add_text_metrics(
        self,
        initial_chars,
        initial_words,
        text,
        expected_chars,
        expected_words,
    ):
        """Test UsageMetrics.add_text_metrics method."""
        metrics = UsageMetrics(
            text_characters=initial_chars,
            text_words=initial_words,
        )
        metrics.add_text_metrics(text)
        assert metrics.text_characters == expected_chars
        assert metrics.text_words == expected_words

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test UsageMetrics serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Test reconstruction
        reconstructed = UsageMetrics.model_validate(data_dict)
        for key, expected_value in constructor_args.items():
            assert getattr(reconstructed, key) == expected_value


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
            {
                "request_type": "audio_transcriptions",
                "arguments": GenerationRequestArguments(
                    method="POST",
                    files={"file": "audio.mp3"},
                ),
                "input_metrics": UsageMetrics(audio_seconds=30.0),
                "output_metrics": UsageMetrics(text_tokens=100),
            },
        ],
        ids=[
            "minimal",
            "with_body",
            "custom_id",
            "with_metrics",
        ],
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

        # Check request_type
        expected_request_type = constructor_args.get("request_type", "text_completions")
        assert instance.request_type == expected_request_type

        # Check request_id
        if "request_id" in constructor_args:
            assert instance.request_id == constructor_args["request_id"]
        else:
            assert isinstance(instance.request_id, str)
            # Should be valid UUID
            uuid.UUID(instance.request_id)

        # Check metrics defaults
        if "input_metrics" in constructor_args:
            assert instance.input_metrics == constructor_args["input_metrics"]
        else:
            assert isinstance(instance.input_metrics, UsageMetrics)

        if "output_metrics" in constructor_args:
            assert instance.output_metrics == constructor_args["output_metrics"]
        else:
            assert isinstance(instance.output_metrics, UsageMetrics)

    @pytest.mark.sanity
    def test_invalid_initialization_values(self):
        """Test GenerationRequest with invalid field values."""
        # Invalid request_type (not a string)
        with pytest.raises(ValidationError):
            GenerationRequest(
                request_type=123,
                arguments=GenerationRequestArguments(),
            )

        # Invalid arguments type
        with pytest.raises(ValidationError):
            GenerationRequest(
                request_type="text_completions",
                arguments="not_a_dict",
            )

        # Invalid input_metrics type
        with pytest.raises(ValidationError):
            GenerationRequest(
                request_type="text_completions",
                arguments=GenerationRequestArguments(),
                input_metrics="invalid",
            )

        # Invalid output_metrics type
        with pytest.raises(ValidationError):
            GenerationRequest(
                request_type="text_completions",
                arguments=GenerationRequestArguments(),
                output_metrics=123,
            )

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test GenerationRequest initialization without required field."""
        # Missing required 'request_type' field
        with pytest.raises(ValidationError):
            GenerationRequest()

        # Missing required 'arguments' field
        with pytest.raises(ValidationError):
            GenerationRequest(request_type="text_completions")

    @pytest.mark.smoke
    def test_auto_id_generation(self):
        """Test that request_id is auto-generated if not provided."""
        request1 = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(),
        )
        request2 = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(),
        )

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0
        assert len(request2.request_id) > 0

        # Should be valid UUIDs
        uuid.UUID(request1.request_id)
        uuid.UUID(request2.request_id)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("request_type", "expected_type"),
        [
            ("text_completions", "text_completions"),
            ("chat_completions", "chat_completions"),
            ("audio_transcriptions", "audio_transcriptions"),
            ("audio_translations", "audio_translations"),
            ("custom_type", "custom_type"),
        ],
        ids=[
            "text_completions",
            "chat_completions",
            "audio_transcriptions",
            "audio_translations",
            "custom_type",
        ],
    )
    def test_request_types(self, request_type, expected_type):
        """Test GenerationRequest with different request types."""
        request = GenerationRequest(
            request_type=request_type,
            arguments=GenerationRequestArguments(),
        )
        assert request.request_type == expected_type

    @pytest.mark.regression
    def test_content_types(self):
        """Test GenerationRequest with different argument types."""
        # Basic arguments
        request1 = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(),
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

    @pytest.mark.smoke
    def test_metrics_defaults(self):
        """Test that input_metrics and output_metrics are initialized."""
        request = GenerationRequest(
            request_type="text_completions",
            arguments=GenerationRequestArguments(),
        )

        assert isinstance(request.input_metrics, UsageMetrics)
        assert isinstance(request.output_metrics, UsageMetrics)
        assert request.input_metrics.total_tokens is None
        assert request.output_metrics.total_tokens is None

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test GenerationRequest serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)
        assert "arguments" in data_dict
        assert "input_metrics" in data_dict
        assert "output_metrics" in data_dict

        # Test reconstruction
        reconstructed = GenerationRequest.model_validate(data_dict)
        assert reconstructed.arguments == instance.arguments
        assert reconstructed.request_type == instance.request_type
        assert reconstructed.request_id == instance.request_id
        assert reconstructed.input_metrics.model_dump() == (
            instance.input_metrics.model_dump()
        )
        assert reconstructed.output_metrics.model_dump() == (
            instance.output_metrics.model_dump()
        )
