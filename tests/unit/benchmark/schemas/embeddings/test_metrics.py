from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.schemas.embeddings.metrics import (
    EmbeddingsMetrics,
    EmbeddingsQualityMetrics,
)
from guidellm.schemas import (
    DistributionSummary,
    Percentiles,
    StatusBreakdown,
    StatusDistributionSummary,
)


def create_percentiles(p50=0.5) -> Percentiles:
    """Helper to create Percentiles with all required fields."""
    return Percentiles(
        p001=p50 * 0.5,
        p01=p50 * 0.6,
        p05=p50 * 0.7,
        p10=p50 * 0.8,
        p25=p50 * 0.9,
        p50=p50,
        p75=p50 * 1.05,
        p90=p50 * 1.1,
        p95=p50 * 1.15,
        p99=p50 * 1.2,
        p999=p50 * 1.25,
    )


def create_distribution_summary(
    mean=0.5, median=0.5, mode=0.5, variance=0.01, std_dev=0.1,
    min_val=0.1, max_val=1.0, count=100, total_sum=50.0
) -> DistributionSummary:
    """Helper to create DistributionSummary with all required fields."""
    return DistributionSummary(
        mean=mean,
        median=median,
        mode=mode,
        variance=variance,
        std_dev=std_dev,
        min=min_val,
        max=max_val,
        count=count,
        total_sum=total_sum,
        percentiles=create_percentiles(median),
    )


class TestEmbeddingsQualityMetrics:
    """Tests for EmbeddingsQualityMetrics schema."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface and key properties."""
        fields = EmbeddingsQualityMetrics.model_fields
        for field_name in (
            "baseline_cosine_similarity",
            "self_consistency_score",
            "mteb_main_score",
            "mteb_task_scores",
        ):
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test initialization with minimal required fields."""
        metrics = EmbeddingsQualityMetrics()
        assert metrics.baseline_cosine_similarity is None
        assert metrics.self_consistency_score is None
        assert metrics.mteb_main_score is None
        assert metrics.mteb_task_scores is None

    @pytest.mark.sanity
    def test_initialization_with_cosine_similarity(self):
        """Test initialization with baseline cosine similarity."""
        dist = create_distribution_summary(
            mean=0.98,
            median=0.985,
            mode=0.985,
            variance=0.0001,
            std_dev=0.01,
            min_val=0.95,
            max_val=0.99,
            count=100,
            total_sum=98.0,
        )
        status_dist = StatusDistributionSummary(
            successful=dist,
            errored=None,
            incomplete=None,
            total=None,
        )

        metrics = EmbeddingsQualityMetrics(
            baseline_cosine_similarity=status_dist
        )
        assert metrics.baseline_cosine_similarity is not None
        assert metrics.baseline_cosine_similarity.successful.mean == 0.98

    @pytest.mark.sanity
    def test_initialization_with_mteb_scores(self):
        """Test initialization with MTEB scores."""
        metrics = EmbeddingsQualityMetrics(
            mteb_main_score=75.5,
            mteb_task_scores={
                "STS12": 72.3,
                "STS13": 78.1,
                "STSBenchmark": 80.9,
            },
        )
        assert metrics.mteb_main_score == 75.5
        assert metrics.mteb_task_scores is not None
        assert len(metrics.mteb_task_scores) == 3
        assert metrics.mteb_task_scores["STS12"] == 72.3

    @pytest.mark.sanity
    def test_initialization_all_fields(self):
        """Test initialization with all fields populated."""
        cos_dist = create_distribution_summary(
            mean=0.98,
            median=0.985,
            mode=0.985,
            variance=0.0001,
            std_dev=0.01,
            min_val=0.95,
            max_val=0.99,
            count=100,
            total_sum=98.0,
        )
        cons_dist = create_distribution_summary(
            mean=0.995,
            median=0.997,
            mode=0.997,
            variance=0.000025,
            std_dev=0.005,
            min_val=0.98,
            max_val=0.999,
            count=100,
            total_sum=99.5,
        )

        metrics = EmbeddingsQualityMetrics(
            baseline_cosine_similarity=StatusDistributionSummary(
                successful=cos_dist, errored=None, incomplete=None, total=None
            ),
            self_consistency_score=StatusDistributionSummary(
                successful=cons_dist, errored=None, incomplete=None, total=None
            ),
            mteb_main_score=75.5,
            mteb_task_scores={"STS12": 72.3, "STS13": 78.1},
        )

        assert metrics.baseline_cosine_similarity.successful.mean == 0.98
        assert metrics.self_consistency_score.successful.mean == 0.995
        assert metrics.mteb_main_score == 75.5
        assert len(metrics.mteb_task_scores) == 2

    @pytest.mark.smoke
    def test_marshalling(self):
        """Test model_dump / model_validate round-trip."""
        metrics = EmbeddingsQualityMetrics(
            mteb_main_score=75.5,
            mteb_task_scores={"STS12": 72.3},
        )
        dumped = metrics.model_dump()
        rebuilt = EmbeddingsQualityMetrics.model_validate(dumped)
        assert rebuilt.mteb_main_score == metrics.mteb_main_score
        assert rebuilt.mteb_task_scores == metrics.mteb_task_scores


class TestEmbeddingsMetrics:
    """Tests for EmbeddingsMetrics schema."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Validate public surface and key properties."""
        fields = EmbeddingsMetrics.model_fields
        for field_name in (
            "request_totals",
            "requests_per_second",
            "request_concurrency",
            "request_latency",
            "input_tokens_count",
            "input_tokens_per_second",
            "quality",
            "encoding_format_breakdown",
        ):
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization_minimal(self):
        """Test initialization with required fields."""
        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=10, incomplete=0, errored=0, total=10
            ),
            requests_per_second=StatusDistributionSummary(),
            request_concurrency=StatusDistributionSummary(),
            request_latency=StatusDistributionSummary(),
            input_tokens_count=StatusBreakdown(
                successful=500, incomplete=0, errored=0, total=500
            ),
            input_tokens_per_second=StatusDistributionSummary(),
        )

        assert metrics.request_totals.successful == 10
        assert metrics.input_tokens_count.successful == 500
        assert metrics.quality is None
        assert metrics.encoding_format_breakdown == {}

    @pytest.mark.sanity
    def test_initialization_with_quality_metrics(self):
        """Test initialization with quality validation metrics."""
        quality = EmbeddingsQualityMetrics(
            mteb_main_score=75.5,
            mteb_task_scores={"STS12": 72.3},
        )

        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=10, incomplete=0, errored=0, total=10
            ),
            requests_per_second=StatusDistributionSummary(),
            request_concurrency=StatusDistributionSummary(),
            request_latency=StatusDistributionSummary(),
            input_tokens_count=StatusBreakdown(
                successful=500, incomplete=0, errored=0, total=500
            ),
            input_tokens_per_second=StatusDistributionSummary(),
            quality=quality,
        )

        assert metrics.quality is not None
        assert metrics.quality.mteb_main_score == 75.5

    @pytest.mark.sanity
    def test_initialization_with_encoding_breakdown(self):
        """Test initialization with encoding format breakdown."""
        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=15, incomplete=0, errored=0, total=15
            ),
            requests_per_second=StatusDistributionSummary(),
            request_concurrency=StatusDistributionSummary(),
            request_latency=StatusDistributionSummary(),
            input_tokens_count=StatusBreakdown(
                successful=750, incomplete=0, errored=0, total=750
            ),
            input_tokens_per_second=StatusDistributionSummary(),
            encoding_format_breakdown={"float": 10, "base64": 5},
        )

        assert metrics.encoding_format_breakdown == {"float": 10, "base64": 5}
        assert sum(metrics.encoding_format_breakdown.values()) == 15

    @pytest.mark.sanity
    def test_initialization_all_fields(self):
        """Test initialization with all fields populated."""
        quality = EmbeddingsQualityMetrics(
            mteb_main_score=75.5,
            mteb_task_scores={"STS12": 72.3, "STS13": 78.1},
        )

        dist = create_distribution_summary(
            mean=0.15,
            median=0.14,
            mode=0.14,
            variance=0.0004,
            std_dev=0.02,
            min_val=0.10,
            max_val=0.20,
            count=100,
            total_sum=15.0,
        )

        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=100, incomplete=5, errored=2, total=107
            ),
            requests_per_second=StatusDistributionSummary(
                successful=dist, errored=None, incomplete=None, total=None
            ),
            request_concurrency=StatusDistributionSummary(
                successful=dist, errored=None, incomplete=None, total=None
            ),
            request_latency=StatusDistributionSummary(
                successful=dist, errored=None, incomplete=None, total=None
            ),
            input_tokens_count=StatusBreakdown(
                successful=5000, incomplete=200, errored=100, total=5300
            ),
            input_tokens_per_second=StatusDistributionSummary(
                successful=dist, errored=None, incomplete=None, total=None
            ),
            quality=quality,
            encoding_format_breakdown={"float": 80, "base64": 20},
        )

        assert metrics.request_totals.successful == 100
        assert metrics.request_totals.total == 107
        assert metrics.input_tokens_count.successful == 5000
        assert metrics.quality.mteb_main_score == 75.5
        assert metrics.encoding_format_breakdown["float"] == 80

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Missing required fields should fail validation."""
        with pytest.raises(ValidationError):
            EmbeddingsMetrics()  # type: ignore[call-arg]

    @pytest.mark.smoke
    def test_marshalling(self):
        """Test model_dump / model_validate round-trip."""
        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=10, incomplete=0, errored=0, total=10
            ),
            requests_per_second=StatusDistributionSummary(),
            request_concurrency=StatusDistributionSummary(),
            request_latency=StatusDistributionSummary(),
            input_tokens_count=StatusBreakdown(
                successful=500, incomplete=0, errored=0, total=500
            ),
            input_tokens_per_second=StatusDistributionSummary(),
            encoding_format_breakdown={"float": 10},
        )

        dumped = metrics.model_dump()
        rebuilt = EmbeddingsMetrics.model_validate(dumped)
        assert (
            rebuilt.request_totals.successful
            == metrics.request_totals.successful
        )
        assert (
            rebuilt.input_tokens_count.successful
            == metrics.input_tokens_count.successful
        )
        assert (
            rebuilt.encoding_format_breakdown
            == metrics.encoding_format_breakdown
        )

    @pytest.mark.regression
    def test_no_output_tokens(self):
        """Verify embeddings have dummy output token fields for compatibility."""
        fields = EmbeddingsMetrics.model_fields
        # Embeddings have dummy output token fields for progress tracker compatibility
        # They exist but are always zero
        assert "output_token_count" in fields
        assert "output_tokens_per_second" in fields

    @pytest.mark.regression
    def test_no_streaming_metrics(self):
        """Verify embeddings metrics do not have streaming-related fields."""
        fields = EmbeddingsMetrics.model_fields
        # Embeddings should NOT have streaming metrics
        assert "time_to_first_token" not in fields
        assert "inter_token_latency" not in fields
        assert "time_per_output_token" not in fields
