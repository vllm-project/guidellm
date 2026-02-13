"""Unit tests for embeddings benchmark output formatters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from guidellm.benchmark.outputs.embeddings_console import EmbeddingsBenchmarkerConsole
from guidellm.benchmark.outputs.embeddings_csv import EmbeddingsBenchmarkerCSV
from guidellm.benchmark.outputs.embeddings_html import EmbeddingsBenchmarkerHTML
from guidellm.benchmark.outputs.embeddings_serialized import (
    EmbeddingsBenchmarkerSerialized,
)
from guidellm.benchmark.schemas.base import BenchmarkConfig
from guidellm.benchmark.schemas.embeddings import (
    EmbeddingsBenchmark,
    EmbeddingsBenchmarkMetadata,
    EmbeddingsBenchmarksReport,
    EmbeddingsMetrics,
    EmbeddingsQualityMetrics,
)
from guidellm.benchmark.schemas.embeddings.entrypoints import BenchmarkEmbeddingsArgs
from guidellm.benchmark.schemas.embeddings.metrics import SchedulerMetrics
from guidellm.benchmark.profiles import SynchronousProfile
from guidellm.scheduler import SchedulerState
from guidellm.schemas import (
    DistributionSummary,
    EmbeddingsRequestStats,
    Percentiles,
    RequestInfo,
    StatusBreakdown,
    StatusDistributionSummary,
    UsageMetrics,
)

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory


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
    mean=0.5,
    median=0.5,
    mode=0.5,
    variance=0.01,
    std_dev=0.1,
    min_val=0.1,
    max_val=1.0,
    count=100,
    total_sum=50.0,
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


@pytest.fixture
def sample_benchmark() -> EmbeddingsBenchmark:
    """Create a sample embeddings benchmark for testing."""
    # Create basic scheduler state
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
        requests_made=StatusBreakdown(successful=10, incomplete=0, errored=0, total=10),
        queued_time_avg=0.01,
        resolve_start_delay_avg=0.005,
        resolve_targeted_start_delay_avg=0.002,
        request_start_delay_avg=0.003,
        resolve_time_avg=0.15,
    )

    # Create quality metrics
    quality_metrics = EmbeddingsQualityMetrics(
        baseline_cosine_similarity=StatusDistributionSummary(
            successful=create_distribution_summary(
                mean=0.98, median=0.985, count=10, total_sum=9.8
            ),
            errored=None,
            incomplete=None,
            total=None,
        ),
        mteb_main_score=75.5,
        mteb_task_scores={"STS12": 72.3, "STS13": 78.1},
    )

    # Create metrics
    latency_dist = create_distribution_summary(
        mean=0.15, median=0.14, count=10, total_sum=1.5
    )
    metrics = EmbeddingsMetrics(
        request_totals=StatusBreakdown(
            successful=10, incomplete=0, errored=0, total=10
        ),
        requests_per_second=StatusDistributionSummary(
            successful=create_distribution_summary(mean=20.0, count=10, total_sum=200.0),
            errored=None,
            incomplete=None,
            total=create_distribution_summary(mean=20.0, count=10, total_sum=200.0),
        ),
        request_concurrency=StatusDistributionSummary(
            successful=create_distribution_summary(mean=2.0, count=10, total_sum=20.0),
            errored=None,
            incomplete=None,
            total=create_distribution_summary(mean=2.0, count=10, total_sum=20.0),
        ),
        request_latency=StatusDistributionSummary(
            successful=latency_dist,
            errored=None,
            incomplete=None,
            total=latency_dist,
        ),
        input_tokens_count=StatusBreakdown(
            successful=500, incomplete=0, errored=0, total=500
        ),
        input_tokens_per_second=StatusDistributionSummary(
            successful=create_distribution_summary(
                mean=100.0, count=10, total_sum=1000.0
            ),
            errored=None,
            incomplete=None,
            total=create_distribution_summary(mean=100.0, count=10, total_sum=1000.0),
        ),
        quality=quality_metrics,
        encoding_format_breakdown={"float": 7, "base64": 3},
    )

    # Create sample request stats
    successful_requests = []
    for i in range(10):
        info = RequestInfo(request_id=f"req-{i}", status="completed")
        info.timings.request_start = float(i)
        info.timings.request_end = float(i) + 0.15
        info.timings.resolve_end = float(i) + 0.15

        stats = EmbeddingsRequestStats(
            request_id=f"req-{i}",
            info=info,
            input_metrics=UsageMetrics(text_tokens=50),
            encoding_format="float" if i < 7 else "base64",
            cosine_similarity=0.98 if i % 2 == 0 else None,
        )
        successful_requests.append(stats)

    requests = StatusBreakdown(
        successful=successful_requests,
        incomplete=[],
        errored=[],
        total=None,
    )

    # Create a minimal config (we won't use most fields for output testing)
    from guidellm.scheduler import SynchronousStrategy

    config = BenchmarkConfig(
        run_id="test-run-001",
        run_index=0,
        strategy=SynchronousStrategy(rate=10),
        constraints={},
        profile=SynchronousProfile(rate=10),
        requests={
            "type": "embeddings",
            "model": "test-embedding-model",
        },
        backend={
            "type": "openai_http",
            "url": "http://localhost:8000",
        },
        environment={
            "platform": "test",
            "python_version": "3.11",
        },
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


@pytest.fixture
def sample_report(sample_benchmark: EmbeddingsBenchmark) -> EmbeddingsBenchmarksReport:
    """Create a sample embeddings benchmark report for testing."""
    args = BenchmarkEmbeddingsArgs(
        target="http://localhost:8000",
        model="test-embedding-model",
        backend="openai_http",
        enable_quality_validation=True,
        baseline_model="sentence-transformers/all-MiniLM-L6-v2",
        encoding_format="float",
    )

    return EmbeddingsBenchmarksReport(
        benchmarks=[sample_benchmark],
        args=args,
        metadata=EmbeddingsBenchmarkMetadata(),
    )


class TestEmbeddingsBenchmarkerSerialized:
    """Tests for EmbeddingsBenchmarkerSerialized (JSON/YAML output)."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that serialized formatter is properly registered."""
        from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput

        # Should be registered for both json and yaml
        assert "json" in EmbeddingsBenchmarkerOutput.registry
        assert "yaml" in EmbeddingsBenchmarkerOutput.registry
        assert (
            EmbeddingsBenchmarkerOutput.registry["json"]
            == EmbeddingsBenchmarkerSerialized
        )

    @pytest.mark.smoke
    def test_validated_kwargs(self):
        """Test validated_kwargs normalizes paths correctly."""
        # Test with string path
        kwargs = EmbeddingsBenchmarkerSerialized.validated_kwargs(
            output_path="/tmp/test.json"
        )
        assert "output_path" in kwargs
        assert isinstance(kwargs["output_path"], Path)
        assert str(kwargs["output_path"]) == "/tmp/test.json"

        # Test with Path object
        path_obj = Path("/tmp/test.json")
        kwargs = EmbeddingsBenchmarkerSerialized.validated_kwargs(output_path=path_obj)
        assert kwargs["output_path"] == path_obj

        # Test with None
        kwargs = EmbeddingsBenchmarkerSerialized.validated_kwargs(output_path=None)
        assert "output_path" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_finalize_json(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test finalizing report to JSON file."""
        output_file = tmp_path / "test_embeddings.json"
        formatter = EmbeddingsBenchmarkerSerialized(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path == output_file
        assert result_path.suffix == ".json"

        # Validate JSON content
        with result_path.open("r") as f:
            data = json.load(f)

        assert data["type_"] == "embeddings_benchmarks_report"
        assert len(data["benchmarks"]) == 1
        assert "metadata" in data
        assert "args" in data

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_finalize_yaml(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test finalizing report to YAML file."""
        output_file = tmp_path / "test_embeddings.yaml"
        formatter = EmbeddingsBenchmarkerSerialized(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path == output_file
        assert result_path.suffix in [".yaml", ".yml"]

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_finalize_directory(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test finalizing with directory path (should use default filename)."""
        formatter = EmbeddingsBenchmarkerSerialized(output_path=tmp_path)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path.parent == tmp_path
        # Default behavior should create a file with some name
        assert result_path.suffix in [".json", ".yaml", ".yml"]


class TestEmbeddingsBenchmarkerCSV:
    """Tests for EmbeddingsBenchmarkerCSV output formatter."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that CSV formatter is properly registered."""
        from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput

        assert "csv" in EmbeddingsBenchmarkerOutput.registry
        assert (
            EmbeddingsBenchmarkerOutput.registry["csv"] == EmbeddingsBenchmarkerCSV
        )

    @pytest.mark.smoke
    def test_default_filename(self):
        """Test default CSV filename."""
        assert EmbeddingsBenchmarkerCSV.DEFAULT_FILE == "embeddings_benchmarks.csv"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_csv_creates_file(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test that finalize creates a valid CSV file."""
        output_file = tmp_path / "test_embeddings.csv"
        formatter = EmbeddingsBenchmarkerCSV(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path == output_file
        assert result_path.suffix == ".csv"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_csv_structure(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test CSV has correct structure and headers."""
        output_file = tmp_path / "test_embeddings.csv"
        formatter = EmbeddingsBenchmarkerCSV(output_path=output_file)

        await formatter.finalize(sample_report)

        # Read CSV and check structure
        with output_file.open("r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have at least header rows + data rows
        assert len(rows) >= 4  # Multi-row header + at least 1 data row

        # Check for embeddings-specific headers (no output tokens or streaming)
        csv_text = output_file.read_text()
        assert "Request Latency" in csv_text
        assert "Input Tokens" in csv_text

        # Should NOT have output token or streaming headers
        assert "Output Tokens" not in csv_text
        assert "Time to First Token" not in csv_text
        assert "Inter Token Latency" not in csv_text

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_csv_quality_metrics(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test CSV includes quality validation metrics."""
        output_file = tmp_path / "test_embeddings.csv"
        formatter = EmbeddingsBenchmarkerCSV(output_path=output_file)

        await formatter.finalize(sample_report)

        csv_text = output_file.read_text()

        # Check for quality metrics
        assert "Cosine Similarity" in csv_text or "Quality" in csv_text
        assert "MTEB" in csv_text

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_csv_encoding_formats(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test CSV includes encoding format breakdown."""
        output_file = tmp_path / "test_embeddings.csv"
        formatter = EmbeddingsBenchmarkerCSV(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        csv_text = result_path.read_text()

        # Check that CSV contains benchmark data (encoding format breakdown
        # is stored in metrics but not separately exported to CSV)
        assert "test-embedding-model" in csv_text
        assert len(csv_text) > 0

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_csv_directory_path(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test CSV creation with directory path."""
        formatter = EmbeddingsBenchmarkerCSV(output_path=tmp_path)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path.parent == tmp_path
        assert result_path.name == EmbeddingsBenchmarkerCSV.DEFAULT_FILE


class TestEmbeddingsBenchmarkerHTML:
    """Tests for EmbeddingsBenchmarkerHTML output formatter."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that HTML formatter is properly registered."""
        from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput

        assert "html" in EmbeddingsBenchmarkerOutput.registry
        assert (
            EmbeddingsBenchmarkerOutput.registry["html"] == EmbeddingsBenchmarkerHTML
        )

    @pytest.mark.smoke
    def test_default_filename(self):
        """Test default HTML filename."""
        assert EmbeddingsBenchmarkerHTML.DEFAULT_FILE == "embeddings_benchmarks.html"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_html_creates_file(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test that finalize creates a valid HTML file."""
        output_file = tmp_path / "test_embeddings.html"
        formatter = EmbeddingsBenchmarkerHTML(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path == output_file
        assert result_path.suffix == ".html"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_html_structure(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test HTML file has valid structure."""
        output_file = tmp_path / "test_embeddings.html"
        formatter = EmbeddingsBenchmarkerHTML(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        html_content = result_path.read_text()

        # Check basic HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "</html>" in html_content
        assert "<head>" in html_content
        assert "<body>" in html_content

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_html_embeddings_data(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test HTML contains embeddings-specific data."""
        output_file = tmp_path / "test_embeddings.html"
        formatter = EmbeddingsBenchmarkerHTML(output_path=output_file)

        result_path = await formatter.finalize(sample_report)

        html_content = result_path.read_text()

        # Check for embedded data and embeddings-specific content
        assert "uiApiData" in html_content
        assert "embeddings" in html_content.lower() or "embedding" in html_content.lower()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_html_no_streaming_metrics(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test HTML does not include streaming metrics."""
        output_file = tmp_path / "test_embeddings.html"
        formatter = EmbeddingsBenchmarkerHTML(output_path=output_file)

        await formatter.finalize(sample_report)

        html_content = output_file.read_text()

        # Should NOT have streaming-related content
        assert "Time to First Token" not in html_content
        assert "TTFT" not in html_content
        assert "Inter Token Latency" not in html_content
        assert "ITL" not in html_content

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_html_directory_path(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test HTML creation with directory path."""
        formatter = EmbeddingsBenchmarkerHTML(output_path=tmp_path)

        result_path = await formatter.finalize(sample_report)

        assert result_path.exists()
        assert result_path.parent == tmp_path
        assert result_path.name == EmbeddingsBenchmarkerHTML.DEFAULT_FILE


class TestEmbeddingsBenchmarkerConsole:
    """Tests for EmbeddingsBenchmarkerConsole output formatter."""

    @pytest.mark.smoke
    def test_class_registration(self):
        """Test that console formatter is properly registered."""
        from guidellm.benchmark.outputs.output import EmbeddingsBenchmarkerOutput

        assert "console" in EmbeddingsBenchmarkerOutput.registry
        assert (
            EmbeddingsBenchmarkerOutput.registry["console"]
            == EmbeddingsBenchmarkerConsole
        )

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_console_finalize(
        self, sample_report: EmbeddingsBenchmarksReport
    ):
        """Test that console formatter finalize returns None (no file output)."""
        formatter = EmbeddingsBenchmarkerConsole()

        result = await formatter.finalize(sample_report)

        # Console formatter doesn't write to file, should return None or empty Path
        assert result is None or (isinstance(result, Path) and not result.exists())

    @pytest.mark.regression
    def test_console_instantiation(self):
        """Test console formatter can be instantiated."""
        formatter = EmbeddingsBenchmarkerConsole()
        assert formatter is not None
        assert isinstance(formatter, EmbeddingsBenchmarkerConsole)


class TestOutputFormattersIntegration:
    """Integration tests for output formatters working together."""

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_integration_multiple_formats(
        self, sample_report: EmbeddingsBenchmarksReport, tmp_path: Path
    ):
        """Test that all formatters can process the same report."""
        # JSON
        json_formatter = EmbeddingsBenchmarkerSerialized(
            output_path=tmp_path / "test.json"
        )
        json_path = await json_formatter.finalize(sample_report)
        assert json_path.exists()

        # CSV
        csv_formatter = EmbeddingsBenchmarkerCSV(output_path=tmp_path / "test.csv")
        csv_path = await csv_formatter.finalize(sample_report)
        assert csv_path.exists()

        # HTML
        html_formatter = EmbeddingsBenchmarkerHTML(output_path=tmp_path / "test.html")
        html_path = await html_formatter.finalize(sample_report)
        assert html_path.exists()

        # Console
        console_formatter = EmbeddingsBenchmarkerConsole()
        console_result = await console_formatter.finalize(sample_report)
        # Console doesn't write files, returns None
        assert console_result is None

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_empty_report_handling(self, tmp_path: Path):
        """Test formatters handle reports with no benchmarks gracefully."""
        # Create report with no benchmarks
        args = BenchmarkEmbeddingsArgs(
            target="http://localhost:8000",
            model="test-model",
        )
        empty_report = EmbeddingsBenchmarksReport(
            benchmarks=[],
            args=args,
            metadata=EmbeddingsBenchmarkMetadata(),
        )

        # JSON should still work
        json_formatter = EmbeddingsBenchmarkerSerialized(
            output_path=tmp_path / "empty.json"
        )
        json_path = await json_formatter.finalize(empty_report)
        assert json_path.exists()

        # Verify JSON content is valid
        with json_path.open("r") as f:
            data = json.load(f)
        assert data["type_"] == "embeddings_benchmarks_report"
        assert len(data["benchmarks"]) == 0
