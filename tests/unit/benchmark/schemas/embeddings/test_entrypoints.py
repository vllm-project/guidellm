from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.schemas.embeddings.entrypoints import BenchmarkEmbeddingsArgs


class TestBenchmarkEmbeddingsArgs:
    """Tests for BenchmarkEmbeddingsArgs schema."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface and key properties."""
        fields = BenchmarkEmbeddingsArgs.model_fields

        # Standard benchmark args
        for field_name in (
            "target",
            "model",
            "backend",
            "profile",
            "data",
            "outputs",
        ):
            assert field_name in fields

        # Embeddings-specific args
        for field_name in ("encoding_format",):
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test initialization with minimal required fields."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )

        assert args.target == "http://localhost:8000"
        assert args.encoding_format == "float"  # Default is "float"

    @pytest.mark.sanity
    def test_initialization_with_encoding_format(self):
        """Test initialization with encoding format."""
        # Float encoding
        args_float = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            encoding_format="float",
        )
        assert args_float.encoding_format == "float"

        # Base64 encoding
        args_base64 = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            encoding_format="base64",
        )
        assert args_base64.encoding_format == "base64"

    @pytest.mark.sanity
    def test_initialization_all_fields(self):
        """Test initialization with all embeddings-specific fields."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-embedding-model",
            backend="openai_http",
            profile="sweep",
            data=["embeddings_data.json"],
            outputs=["json"],
            encoding_format="float",
        )

        # Standard fields
        assert args.target == "http://localhost:8000"
        assert args.model == "test-embedding-model"
        assert args.backend == "openai_http"
        assert args.profile == "sweep"
        assert args.data == ["embeddings_data.json"]
        assert args.outputs == ["json"]

        # Embeddings-specific fields
        assert args.encoding_format == "float"

    @pytest.mark.sanity
    def test_invalid_initialization_missing_target(self):
        """Missing target should fail validation."""
        with pytest.raises(ValidationError):
            BenchmarkEmbeddingsArgs()  # type: ignore[call-arg]

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("target", None),
            ("target", 123),
            ("model", 123),
            ("encoding_format", 123),
        ],
    )
    def test_invalid_initialization_values(self, field_name: str, bad_value):
        """Type mismatches should raise."""
        base = {"target": "http://localhost:8000"}
        base[field_name] = bad_value
        with pytest.raises(ValidationError):
            BenchmarkEmbeddingsArgs(**base)  # type: ignore[arg-type]

    @pytest.mark.smoke
    def test_marshalling(self):
        """Test model_dump / model_validate round-trip."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            data=["test_data.json"],
            encoding_format="float",
        )

        dumped = args.model_dump()
        rebuilt = BenchmarkEmbeddingsArgs.model_validate(dumped)

        assert rebuilt.target == args.target
        assert rebuilt.model == args.model
        assert rebuilt.encoding_format == args.encoding_format

    @pytest.mark.sanity
    def test_optional_fields(self):
        """Test that embeddings-specific fields are optional."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )

        # All embeddings-specific fields should have defaults
        assert args.encoding_format == "float"  # Default is "float", not None

    @pytest.mark.sanity
    def test_encoding_format_optional(self):
        """Test encoding format has default value."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )
        # Default is "float"
        assert args.encoding_format == "float"

    @pytest.mark.regression
    def test_standard_benchmark_args_inherited(self):
        """Test that standard BenchmarkArgs fields are inherited."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            backend="openai_http",
            profile="sweep",
            data=["data.json"],
            outputs=["json"],
        )

        # These are inherited from BenchmarkArgs
        assert hasattr(args, "target")
        assert hasattr(args, "model")
        assert hasattr(args, "backend")
        assert hasattr(args, "profile")
        assert hasattr(args, "data")
        assert hasattr(args, "outputs")

        # Verify values
        assert args.target == "http://localhost:8000"
        assert args.model == "test-model"
        assert args.backend == "openai_http"
        assert args.profile == "sweep"
        assert args.data == ["data.json"]
        assert args.outputs == ["json"]
