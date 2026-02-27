from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.embeddings.accumulator import (
    EmbeddingsBenchmarkAccumulator,
)


class TestEmbeddingsBenchmarkAccumulator:
    """Tests for EmbeddingsBenchmarkAccumulator."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface and key properties."""
        # Check that class has expected attributes (will be set during init
        # with config)
        assert hasattr(EmbeddingsBenchmarkAccumulator, "model_fields")
        assert (
            "encoding_format_breakdown" in EmbeddingsBenchmarkAccumulator.model_fields
        )

    @pytest.mark.smoke
    def test_initialization(self):
        """Test accumulator has proper default fields."""
        # EmbeddingsBenchmarkAccumulator requires a BenchmarkConfig for full
        # instantiation but we can test that the class has expected fields
        fields = EmbeddingsBenchmarkAccumulator.model_fields

        assert "encoding_format_breakdown" in fields
        assert "timings" in fields
        assert "scheduler_metrics" in fields
        assert "metrics" in fields
        assert "requests" in fields

    @pytest.mark.sanity
    def test_encoding_format_breakdown_field(self):
        """Test that encoding_format_breakdown field exists and is a dict."""
        # Test that the field schema is correct
        fields = EmbeddingsBenchmarkAccumulator.model_fields
        assert "encoding_format_breakdown" in fields

        # Field should be a dict type
        field_info = fields["encoding_format_breakdown"]
        assert field_info.annotation == dict[str, int]

    @pytest.mark.regression
    def test_accumulator_field_defaults(self):
        """Test that accumulator fields have proper default factories."""
        fields = EmbeddingsBenchmarkAccumulator.model_fields

        # Check fields with default factories
        assert "timings" in fields
        assert "scheduler_metrics" in fields
        assert "metrics" in fields
        assert "requests" in fields

        # Check that encoding_format_breakdown has dict factory
        assert fields["encoding_format_breakdown"].default_factory is not None

    @pytest.mark.regression
    def test_type_literal(self):
        """Test that type_ field is correctly set."""
        fields = EmbeddingsBenchmarkAccumulator.model_fields
        assert "type_" in fields

        # Check the default value
        assert fields["type_"].default == "embeddings_benchmark_accumulator"
