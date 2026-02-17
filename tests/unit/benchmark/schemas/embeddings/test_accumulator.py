from __future__ import annotations

import pytest

from guidellm.benchmark.schemas.embeddings.accumulator import (
    EmbeddingsBenchmarkAccumulator,
    EmbeddingsQualityMetricsAccumulator,
)


class TestEmbeddingsQualityMetricsAccumulator:
    """Tests for EmbeddingsQualityMetricsAccumulator."""

    @pytest.mark.smoke
    def test_initialization(self):
        """Test accumulator initialization."""
        accumulator = EmbeddingsQualityMetricsAccumulator()
        assert accumulator.cosine_similarities == []

    @pytest.mark.sanity
    def test_add_cosine_similarity(self):
        """Test adding cosine similarity values."""
        accumulator = EmbeddingsQualityMetricsAccumulator()

        # Add some cosine similarity values
        accumulator.cosine_similarities.append(0.98)
        accumulator.cosine_similarities.append(0.97)
        accumulator.cosine_similarities.append(0.99)

        assert len(accumulator.cosine_similarities) == 3
        assert accumulator.cosine_similarities[0] == 0.98
        assert accumulator.cosine_similarities[1] == 0.97
        assert accumulator.cosine_similarities[2] == 0.99

    @pytest.mark.sanity
    def test_multiple_instances_independent(self):
        """Test that multiple accumulator instances are independent."""
        acc1 = EmbeddingsQualityMetricsAccumulator()
        acc2 = EmbeddingsQualityMetricsAccumulator()

        acc1.cosine_similarities.append(0.95)
        acc2.cosine_similarities.append(0.99)

        assert len(acc1.cosine_similarities) == 1
        assert len(acc2.cosine_similarities) == 1
        assert acc1.cosine_similarities[0] != acc2.cosine_similarities[0]


class TestEmbeddingsBenchmarkAccumulator:
    """Tests for EmbeddingsBenchmarkAccumulator."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface and key properties."""
        # Check that class has expected attributes (will be set during init
        # with config)
        assert hasattr(EmbeddingsBenchmarkAccumulator, "model_fields")
        assert "quality" in EmbeddingsBenchmarkAccumulator.model_fields
        assert (
            "encoding_format_breakdown"
            in EmbeddingsBenchmarkAccumulator.model_fields
        )

    @pytest.mark.smoke
    def test_initialization(self):
        """Test accumulator has proper default fields."""
        # EmbeddingsBenchmarkAccumulator requires a BenchmarkConfig for full
        # instantiation but we can test that the class has expected fields
        fields = EmbeddingsBenchmarkAccumulator.model_fields

        assert "quality_enabled" in fields
        assert "quality" in fields
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

    @pytest.mark.sanity
    def test_quality_metrics_accumulator_field(self):
        """Test that quality field exists and has correct type."""
        fields = EmbeddingsBenchmarkAccumulator.model_fields
        assert "quality" in fields
        assert "quality_enabled" in fields

        # Field should be optional EmbeddingsQualityMetricsAccumulator
        field_info = fields["quality"]
        # Check field is optional (can be None)
        assert field_info.is_required() is False

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
