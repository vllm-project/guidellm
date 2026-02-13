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
        for field_name in (
            "enable_quality_validation",
            "baseline_model",
            "quality_tolerance",
            "enable_mteb",
            "mteb_tasks",
            "encoding_format",
        ):
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test initialization with minimal required fields."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )

        assert args.target == "http://localhost:8000"
        assert args.enable_quality_validation is False
        assert args.baseline_model is None
        assert args.quality_tolerance == 1e-2
        assert args.enable_mteb is False
        assert args.mteb_tasks is None
        assert args.encoding_format == "float"  # Default is "float"

    @pytest.mark.sanity
    def test_initialization_with_quality_validation(self):
        """Test initialization with quality validation enabled."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            enable_quality_validation=True,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            quality_tolerance=5e-4,
        )

        assert args.enable_quality_validation is True
        assert args.baseline_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert args.quality_tolerance == 5e-4

    @pytest.mark.sanity
    def test_initialization_with_mteb(self):
        """Test initialization with MTEB enabled."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            enable_mteb=True,
            mteb_tasks=["STS12", "STS13", "STSBenchmark"],
        )

        assert args.enable_mteb is True
        assert args.mteb_tasks == ["STS12", "STS13", "STSBenchmark"]

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
            outputs=["json", "csv", "html"],
            enable_quality_validation=True,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            quality_tolerance=1e-3,
            enable_mteb=True,
            mteb_tasks=["STS12", "STS13"],
            encoding_format="float",
        )

        # Standard fields
        assert args.target == "http://localhost:8000"
        assert args.model == "test-embedding-model"
        assert args.backend == "openai_http"
        assert args.profile == "sweep"
        assert args.data == ["embeddings_data.json"]
        assert args.outputs == ["json", "csv", "html"]

        # Embeddings-specific fields
        assert args.enable_quality_validation is True
        assert args.baseline_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert args.quality_tolerance == 1e-3
        assert args.enable_mteb is True
        assert args.mteb_tasks == ["STS12", "STS13"]
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
            ("enable_quality_validation", "not_a_bool"),
            ("quality_tolerance", "not_a_float"),
            ("enable_mteb", "not_a_bool"),
            ("mteb_tasks", "not_a_list"),
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
            data=["test_data.json"],  # Need at least one data item
            enable_quality_validation=True,
            baseline_model="sentence-transformers/all-MiniLM-L6-v2",
            quality_tolerance=1e-3,
        )

        dumped = args.model_dump()
        rebuilt = BenchmarkEmbeddingsArgs.model_validate(dumped)

        assert rebuilt.target == args.target
        assert rebuilt.model == args.model
        assert rebuilt.enable_quality_validation == args.enable_quality_validation
        assert rebuilt.baseline_model == args.baseline_model
        assert rebuilt.quality_tolerance == args.quality_tolerance

    @pytest.mark.regression
    def test_quality_tolerance_default_value(self):
        """Test default quality tolerance matches vLLM pattern (1e-2)."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )
        assert args.quality_tolerance == 1e-2

    @pytest.mark.regression
    def test_mteb_tasks_default_none(self):
        """Test MTEB tasks default to None (will use DEFAULT_MTEB_TASKS in validator)."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            enable_mteb=True,
        )
        # mteb_tasks should be None by default
        # The validator will set DEFAULT_MTEB_TASKS if None
        assert args.mteb_tasks is None or isinstance(args.mteb_tasks, list)

    @pytest.mark.sanity
    def test_optional_fields(self):
        """Test that embeddings-specific fields are optional."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
        )

        # All embeddings-specific fields should have defaults
        assert args.enable_quality_validation is False
        assert args.baseline_model is None
        assert args.quality_tolerance == 1e-2
        assert args.enable_mteb is False
        assert args.mteb_tasks is None
        assert args.encoding_format == "float"  # Default is "float", not None

    @pytest.mark.regression
    def test_quality_validation_without_baseline_model(self):
        """Test quality validation can be enabled without explicit baseline model."""
        # Should be valid - baseline model can be determined later or use default
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            enable_quality_validation=True,
        )

        assert args.enable_quality_validation is True
        assert args.baseline_model is None

    @pytest.mark.regression
    def test_mteb_tasks_as_list(self):
        """Test MTEB tasks can be specified as a list."""
        tasks = ["STS12", "STS13", "STS14", "STS15", "STSBenchmark"]
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            enable_mteb=True,
            mteb_tasks=tasks,
        )

        assert args.mteb_tasks == tasks
        assert len(args.mteb_tasks) == 5

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
            outputs=["json", "csv"],
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
        assert args.outputs == ["json", "csv"]
