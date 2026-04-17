"""Unit tests for embeddings benchmark schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.profiles import SynchronousProfile
from guidellm.benchmark.schemas.base import BenchmarkConfig
from guidellm.benchmark.schemas.embeddings import (
    EmbeddingsBenchmark,
    EmbeddingsBenchmarkMetadata,
    EmbeddingsBenchmarksReport,
    EmbeddingsMetrics,
)
from guidellm.benchmark.schemas.embeddings.entrypoints import BenchmarkEmbeddingsArgs
from guidellm.benchmark.schemas.embeddings.metrics import SchedulerMetrics
from guidellm.scheduler import SchedulerState, SynchronousStrategy
from guidellm.schemas import (
    DistributionSummary,
    EmbeddingsRequestStats,
    Percentiles,
    RequestInfo,
    StatusBreakdown,
    StatusDistributionSummary,
    UsageMetrics,
)


def create_percentiles(p50=0.5) -> Percentiles:
    """Helper to create Percentiles."""
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
    mean=0.5,
    median=0.5,
    count=10,
) -> DistributionSummary:
    """Helper to create DistributionSummary."""
    return DistributionSummary(
        mean=mean,
        median=median,
        mode=median,
        variance=0.01,
        std_dev=0.1,
        min=median * 0.5,
        max=median * 1.5,
        count=count,
        total_sum=mean * count,
        percentiles=create_percentiles(median),
    )


class TestEmbeddingsBenchmark:
    """Tests for EmbeddingsBenchmark schema."""

    @pytest.fixture
    def valid_benchmark(self) -> EmbeddingsBenchmark:
        """Create a valid embeddings benchmark for testing."""
        # Create minimal valid config
        config = BenchmarkConfig(
            run_id="test-run",
            run_index=0,
            strategy=SynchronousStrategy(rate=10),
            constraints={},
            profile=SynchronousProfile(rate=10),
            requests={"type": "embeddings", "model": "test-model"},
            backend={"type": "openai_http", "url": "http://localhost:8000"},
            environment={"platform": "test", "python_version": "3.12"},
        )

        scheduler_state = SchedulerState(
            request_count=10,
            successful_count=10,
            incomplete_count=0,
            errored_count=0,
        )

        scheduler_metrics = SchedulerMetrics(
            start_time=0.0,
            request_start_time=0.1,
            measure_start_time=1.0,
            measure_end_time=9.0,
            request_end_time=9.9,
            end_time=10.0,
            requests_made=StatusBreakdown(
                successful=10, incomplete=0, errored=0, total=10
            ),
            queued_time_avg=0.01,
            resolve_start_delay_avg=0.005,
            resolve_targeted_start_delay_avg=0.002,
            request_start_delay_avg=0.003,
            resolve_time_avg=0.15,
        )

        metrics = EmbeddingsMetrics(
            request_totals=StatusBreakdown(
                successful=10, incomplete=0, errored=0, total=10
            ),
            requests_per_second=StatusDistributionSummary(
                successful=create_distribution_summary(mean=20.0),
                errored=None,
                incomplete=None,
                total=create_distribution_summary(mean=20.0),
            ),
            request_concurrency=StatusDistributionSummary(
                successful=create_distribution_summary(mean=2.0),
                errored=None,
                incomplete=None,
                total=create_distribution_summary(mean=2.0),
            ),
            request_latency=StatusDistributionSummary(
                successful=create_distribution_summary(mean=0.15),
                errored=None,
                incomplete=None,
                total=create_distribution_summary(mean=0.15),
            ),
            input_tokens_count=StatusBreakdown(
                successful=500, incomplete=0, errored=0, total=500
            ),
            input_tokens_per_second=StatusDistributionSummary(
                successful=create_distribution_summary(mean=100.0),
                errored=None,
                incomplete=None,
                total=create_distribution_summary(mean=100.0),
            ),
            encoding_format_breakdown={"float": 10},
        )

        # Create sample requests
        requests_list = []
        for i in range(10):
            info = RequestInfo(request_id=f"req-{i}", status="completed")
            info.timings.request_start = float(i)
            info.timings.request_end = float(i) + 0.15
            info.timings.resolve_end = float(i) + 0.15

            stats = EmbeddingsRequestStats(
                request_id=f"req-{i}",
                info=info,
                input_metrics=UsageMetrics(text_tokens=50),
                encoding_format="float",
            )
            requests_list.append(stats)

        requests = StatusBreakdown(
            successful=requests_list,
            incomplete=[],
            errored=[],
            total=None,
        )

        return EmbeddingsBenchmark(
            config=config,
            scheduler_state=scheduler_state,
            scheduler_metrics=scheduler_metrics,
            metrics=metrics,
            requests=requests,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            warmup_duration=1.0,
            cooldown_duration=1.0,
        )

    @pytest.mark.smoke
    def test_class_signature(self):
        """Test that EmbeddingsBenchmark has correct fields."""
        # Check critical fields exist (inherited from base Benchmark class)
        bench = EmbeddingsBenchmark
        fields = bench.model_fields
        for field_name in [
            "type_",
            "config",
            "scheduler_state",
            "scheduler_metrics",
            "metrics",
            "requests",
        ]:
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_benchmark: EmbeddingsBenchmark):
        """Test initialization from valid data."""
        assert isinstance(valid_benchmark, EmbeddingsBenchmark)
        assert valid_benchmark.type_ == "embeddings_benchmark"

    @pytest.mark.sanity
    def test_type_field_immutable(self, valid_benchmark: EmbeddingsBenchmark):
        """Test that type_ field is always 'embeddings_benchmark'."""
        assert valid_benchmark.type_ == "embeddings_benchmark"

        # Type field is a Literal, so wrong values should raise ValidationError
        dumped = valid_benchmark.model_dump()
        dumped["type_"] = "wrong_type"
        with pytest.raises(ValidationError):
            EmbeddingsBenchmark.model_validate(dumped)

    @pytest.mark.sanity
    def test_marshalling(self, valid_benchmark: EmbeddingsBenchmark):
        """Test model_dump / model_validate round-trip."""
        dumped = valid_benchmark.model_dump()
        assert dumped["type_"] == "embeddings_benchmark"

        rebuilt = EmbeddingsBenchmark.model_validate(dumped)
        assert rebuilt.type_ == "embeddings_benchmark"
        assert rebuilt.duration == valid_benchmark.duration

    @pytest.mark.regression
    def test_required_fields_missing(self):
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            EmbeddingsBenchmark()  # type: ignore[call-arg]

    @pytest.mark.sanity
    def test_embeddings_specific_metrics(self, valid_benchmark: EmbeddingsBenchmark):
        """Test embeddings-specific metrics are present."""
        metrics = valid_benchmark.metrics

        # Should have embeddings-specific fields
        assert hasattr(metrics, "input_tokens_count")
        assert hasattr(metrics, "input_tokens_per_second")
        assert hasattr(metrics, "encoding_format_breakdown")

        # Should NOT have generative-specific fields
        assert not hasattr(metrics, "output_tokens_count")
        assert not hasattr(metrics, "time_to_first_token")


class TestEmbeddingsBenchmarksReport:
    """Tests for EmbeddingsBenchmarksReport schema."""

    @pytest.fixture
    def valid_report(self) -> EmbeddingsBenchmarksReport:
        """Create a valid embeddings benchmarks report."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-embedding-model",
            data=["prompt_tokens=100"],
            backend="openai_http",
            encoding_format="float",
        )

        return EmbeddingsBenchmarksReport(
            benchmarks=[],
            args=args,
            metadata=EmbeddingsBenchmarkMetadata(),
        )

    @pytest.mark.smoke
    def test_class_signature(self):
        """Test that EmbeddingsBenchmarksReport has correct signature."""
        fields = EmbeddingsBenchmarksReport.model_fields
        for field_name in ["type_", "benchmarks", "args", "metadata"]:
            assert field_name in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_report: EmbeddingsBenchmarksReport):
        """Test initialization from valid data."""
        assert isinstance(valid_report, EmbeddingsBenchmarksReport)
        assert valid_report.type_ == "embeddings_benchmarks_report"

    @pytest.mark.sanity
    def test_type_field_immutable(self, valid_report: EmbeddingsBenchmarksReport):
        """Test that type_ field is always 'embeddings_benchmarks_report'."""
        assert valid_report.type_ == "embeddings_benchmarks_report"

    @pytest.mark.sanity
    def test_marshalling(self, valid_report: EmbeddingsBenchmarksReport):
        """Test model_dump / model_validate round-trip."""
        dumped = valid_report.model_dump()
        assert dumped["type_"] == "embeddings_benchmarks_report"

        rebuilt = EmbeddingsBenchmarksReport.model_validate(dumped)
        assert rebuilt.type_ == "embeddings_benchmarks_report"
        assert len(rebuilt.benchmarks) == len(valid_report.benchmarks)

    @pytest.mark.sanity
    def test_empty_benchmarks_allowed(self):
        """Test that report can have empty benchmarks list."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
        )

        report = EmbeddingsBenchmarksReport(
            benchmarks=[],
            args=args,
            metadata=EmbeddingsBenchmarkMetadata(),
        )

        assert isinstance(report, EmbeddingsBenchmarksReport)
        assert len(report.benchmarks) == 0

    @pytest.mark.sanity
    def test_multiple_benchmarks(self):
        """Test report with multiple benchmarks."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            data=["prompt_tokens=100"],
        )

        report = EmbeddingsBenchmarksReport(
            benchmarks=[],
            args=args,
            metadata=EmbeddingsBenchmarkMetadata(),
        )

        assert len(report.benchmarks) == 0

    @pytest.mark.regression
    def test_required_fields_missing(self):
        """Test that missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            EmbeddingsBenchmarksReport()  # type: ignore[call-arg]


class TestBenchmarkEmbeddingsArgs:
    """Tests for BenchmarkEmbeddingsArgs entrypoint schema."""

    @pytest.mark.smoke
    def test_class_signature(self):
        """Test that BenchmarkEmbeddingsArgs has correct fields."""
        fields = BenchmarkEmbeddingsArgs.model_fields
        for field_name in [
            "target",
            "data",
            "encoding_format",
            "data_column_mapper",
            "data_finalizer",
            "data_collator",
        ]:
            assert field_name in fields

    @pytest.mark.sanity
    def test_initialization_minimal(self):
        """Test initialization with minimal required fields."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
        )

        assert args.target == "http://localhost:8000"
        assert args.model == "test-model"

    @pytest.mark.sanity
    def test_embeddings_specific_defaults(self):
        """Test that embeddings-specific defaults are set correctly."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
        )

        # Embeddings defaults
        assert args.data_column_mapper == "embeddings_column_mapper"
        assert args.data_finalizer == "embeddings"
        assert args.data_collator == "embeddings"
        assert args.data_num_workers == 0  # macOS compatibility
        assert args.encoding_format == "float"
        assert "json" in args.outputs

    @pytest.mark.sanity
    def test_encoding_format_validation(self):
        """Test encoding format is validated."""
        # Valid values
        args_float = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            encoding_format="float",
        )
        assert args_float.encoding_format == "float"

        args_base64 = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            encoding_format="base64",
        )
        assert args_base64.encoding_format == "base64"

        # Invalid value should raise
        with pytest.raises(ValidationError):
            BenchmarkEmbeddingsArgs(
                target="http://localhost:8000",
                model="test-model",
                encoding_format="invalid",  # type: ignore[arg-type]
            )

    @pytest.mark.sanity
    def test_marshalling(self):
        """Test model_dump / model_validate round-trip."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            data=["prompt_tokens=100"],
            encoding_format="base64",
        )

        dumped = args.model_dump()
        rebuilt = BenchmarkEmbeddingsArgs.model_validate(dumped)

        assert rebuilt.target == args.target
        assert rebuilt.encoding_format == args.encoding_format

    @pytest.mark.regression
    def test_max_duration_instead_of_max_seconds(self):
        """Test that embeddings uses max_duration not max_seconds."""
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
            max_duration=30.0,
        )

        assert args.max_duration == 30.0


class TestEmbeddingsBenchmarkMetadata:
    """Tests for EmbeddingsBenchmarkMetadata schema."""

    @pytest.mark.smoke
    def test_initialization(self):
        """Test metadata initialization."""
        metadata = EmbeddingsBenchmarkMetadata()
        assert isinstance(metadata, EmbeddingsBenchmarkMetadata)

    @pytest.mark.sanity
    def test_auto_populated_fields(self):
        """Test that metadata auto-populates fields."""
        metadata = EmbeddingsBenchmarkMetadata()

        # Should have version info
        assert hasattr(metadata, "guidellm_version")
        assert hasattr(metadata, "python_version")
        assert hasattr(metadata, "start_time")

    @pytest.mark.sanity
    def test_marshalling(self):
        """Test model_dump / model_validate round-trip."""
        metadata = EmbeddingsBenchmarkMetadata()
        dumped = metadata.model_dump()
        rebuilt = EmbeddingsBenchmarkMetadata.model_validate(dumped)

        assert rebuilt.guidellm_version == metadata.guidellm_version
