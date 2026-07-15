## WRITTEN BY AI ##
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import reimport_benchmarks_report
from guidellm.benchmark.outputs.plot import (
    GenerativeBenchmarkerPlot,
    PlotBenchmarkOutputArgs,
)
from guidellm.benchmark.schemas import BenchmarkScenario, GenerativeBenchmarksReport
from guidellm.benchmark.schemas.output import (
    BenchmarkOutputArgs,
)


class MockPercentiles:
    def __init__(self, p95: float = 10.0, p90: float = 9.0, p50: float = 5.0):
        self.p95 = p95
        self.p90 = p90
        self.p50 = p50


class MockDistribution:
    def __init__(self, mean: float = 5.0, median: float = 4.0, p95: float = 10.0):
        self.mean = mean
        self.median = median
        self.percentiles = MockPercentiles(p95=p95)


class MockStatusDistribution:
    def __init__(self, mean: float = 5.0, median: float = 4.0, p95: float = 10.0):
        self.successful = MockDistribution(mean=mean, median=median, p95=p95)
        self.incomplete = MockDistribution(mean=0.0, median=0.0, p95=0.0)
        self.errored = MockDistribution(mean=0.0, median=0.0, p95=0.0)
        self.total = MockDistribution(mean=mean, median=median, p95=p95)


class MockRequestTotals:
    def __init__(
        self,
        successful: int = 10,
        incomplete: int = 0,
        errored: int = 0,
    ):
        self.successful = successful
        self.incomplete = incomplete
        self.errored = errored
        self.total = successful + incomplete + errored


class MockMetrics:
    def __init__(self):
        self.request_totals = MockRequestTotals()
        self.requests_per_second = MockStatusDistribution(mean=1.5)
        self.time_to_first_token_ms = MockStatusDistribution(
            mean=100.0,
            median=90.0,
            p95=150.0,
        )
        self.time_to_first_output_token_ms = MockStatusDistribution(
            mean=100.0,
            median=90.0,
            p95=150.0,
        )
        self.request_latency = MockStatusDistribution(mean=2.5, median=2.2, p95=3.0)
        self.time_per_output_token_ms = MockStatusDistribution(
            mean=20.0,
            median=18.0,
            p95=25.0,
        )
        self.inter_token_latency_ms = MockStatusDistribution(
            mean=15.0,
            median=14.0,
            p95=18.0,
        )
        self.request_concurrency = MockStatusDistribution(mean=4.0)
        self.tokens_per_second = MockStatusDistribution(mean=30.0)
        self.prompt_tokens_per_second = MockStatusDistribution(mean=10.0)
        self.output_tokens_per_second = MockStatusDistribution(mean=20.0)


class MockBenchmark:
    def __init__(self):
        self.metrics = MockMetrics()


@pytest.mark.smoke
def test_from_args_creates_correct_instance():
    """
    Test that GenerativeBenchmarkerPlot is created correctly from plot args.

    ## WRITTEN BY AI ##
    """
    args = PlotBenchmarkOutputArgs(path=Path("test.png"), dpi=120)
    plot_output = GenerativeBenchmarkerPlot.from_args(args)
    assert plot_output.output_path == Path("test.png")
    assert plot_output.dpi == 120


@pytest.mark.smoke
def test_from_args_with_wrong_type_raises_value_error():
    """
    Test that passing non-Plot args to from_args raises a ValueError.

    ## WRITTEN BY AI ##
    """

    class DummyArgs(BenchmarkOutputArgs):
        kind: str = "dummy"

    args = DummyArgs(kind="dummy")
    with pytest.raises(ValueError) as excinfo:
        GenerativeBenchmarkerPlot.from_args(args)
    assert "Expected PlotBenchmarkOutputArgs" in str(excinfo.value)


@pytest.mark.sanity
def test_path_validation_with_png():
    """
    Test that a valid PNG path remains unchanged.

    ## WRITTEN BY AI ##
    """
    args = PlotBenchmarkOutputArgs(path=Path("test.png"))
    assert args.path == Path("test.png")


@pytest.mark.sanity
def test_path_validation_coerces_missing_suffix():
    """
    Test that plot paths with missing suffix are coerced to .png.

    ## WRITTEN BY AI ##
    """
    args = PlotBenchmarkOutputArgs(path=Path("test"))
    assert args.path == Path("test.png")


@pytest.mark.sanity
def test_path_validation_raises_unsupported_suffix():
    """
    Test that unsupported plot suffixes raise a validation error.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError) as excinfo:
        PlotBenchmarkOutputArgs(path=Path("test.unsupported"))
    assert "Plot output type .unsupported is not supported" in str(excinfo.value)


@pytest.mark.sanity
def test_path_validation_case_insensitive():
    """
    Test that upper-case PNG extensions are preserved correctly.

    ## WRITTEN BY AI ##
    """
    args = PlotBenchmarkOutputArgs(path=Path("test.PNG"))
    assert args.path == Path("test.PNG")


@pytest.mark.sanity
def test_invalid_args_raises_validation_error():
    """
    Test that invalid args like negative DPI raise Pydantic ValidationError.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValidationError):
        # dpi should ideally be positive (we pass invalid type or structure)
        PlotBenchmarkOutputArgs(dpi="invalid_dpi")  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_writes_valid_image(tmp_path: Path):
    """
    Test that finalize writes a valid PNG plot image when benchmark data is present.

    ## WRITTEN BY AI ##
    """
    benchmarks = [MockBenchmark(), MockBenchmark()]
    benchmarks[1].metrics.requests_per_second = MockStatusDistribution(mean=2.5)
    benchmarks[1].metrics.tokens_per_second = MockStatusDistribution(mean=50.0)

    report = SimpleNamespace(benchmarks=benchmarks)
    output_file = tmp_path / "my_plot.png"
    plot_output = GenerativeBenchmarkerPlot(output_path=output_file, dpi=100)

    path = await plot_output.finalize(report)  # type: ignore[arg-type]

    assert path == output_file
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_resolves_directory_path(tmp_path: Path):
    """
    Test that if output_path is a directory, it resolves to benchmarks.png inside it.

    ## WRITTEN BY AI ##
    """
    report = SimpleNamespace(benchmarks=[MockBenchmark()])
    plot_output = GenerativeBenchmarkerPlot(output_path=tmp_path, dpi=80)

    path = await plot_output.finalize(report)  # type: ignore[arg-type]

    expected_path = tmp_path / "benchmarks.png"
    assert path == expected_path
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_handles_empty_benchmarks_gracefully(tmp_path: Path):
    """
    Test that finalize handles empty benchmarks list gracefully without raising errors.

    ## WRITTEN BY AI ##
    """
    report = SimpleNamespace(benchmarks=[])
    output_file = tmp_path / "empty_plot.png"
    plot_output = GenerativeBenchmarkerPlot(output_path=output_file, dpi=80)

    path = await plot_output.finalize(report)  # type: ignore[arg-type]

    assert path == output_file
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_finalize_with_partially_missing_metrics(tmp_path: Path):
    """
    Test that finalize runs successfully when some metrics are None/missing.

    ## WRITTEN BY AI ##
    """
    b1 = MockBenchmark()
    # Simulate completely missing metrics for requests_per_second
    b1.metrics.requests_per_second = None  # type: ignore[assignment]
    # Simulate missing successful distribution
    b1.metrics.time_to_first_token_ms.successful = None  # type: ignore[assignment]

    b2 = MockBenchmark()
    # Simulate missing percentiles
    b2.metrics.request_latency.successful.percentiles = None  # type: ignore[assignment]
    # Simulate None values inside distribution
    b2.metrics.time_per_output_token_ms.successful.mean = None  # type: ignore[assignment]
    b2.metrics.time_per_output_token_ms.successful.median = None  # type: ignore[assignment]

    report = SimpleNamespace(benchmarks=[b1, b2])
    output_file = tmp_path / "missing_metrics_plot.png"
    plot_output = GenerativeBenchmarkerPlot(output_path=output_file, dpi=80)

    path = await plot_output.finalize(report)  # type: ignore[arg-type]

    assert path == output_file
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.fixture
def minimal_report() -> GenerativeBenchmarksReport:
    """
    Create a GenerativeBenchmarksReport for testing.
    """
    config = BenchmarkScenario.model_validate(
        {
            "spec": {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000/v1",
                    "model": "test-model",
                },
                "data": [{"kind": "huggingface", "source": "test_data.jsonl"}],
                "profile": {"kind": "constant", "rate": 10.0},
            },
        }
    )
    return GenerativeBenchmarksReport(config=config)


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_reimport_benchmarks_report_custom_extension_plot(
    tmp_path: Path, minimal_report: GenerativeBenchmarksReport
):
    """
    Test that reimport preserves a custom filename and resolves plot to .pdf.

    ## WRITTEN BY AI ##
    """
    # Save the report to a temp JSON file
    report_file = tmp_path / "report.json"
    report_file.write_text(minimal_report.model_dump_json(), encoding="utf-8")

    # Re-import and save specifically to a named .pdf file
    expected_pdf = tmp_path / "my_report.pdf"

    await reimport_benchmarks_report(
        file=report_file, outputs=[{"kind": "plot", "path": expected_pdf}]
    )

    # Assert that it resolved and created the correct file: my_report.pdf
    assert expected_pdf.exists()
    assert expected_pdf.is_file()
    assert expected_pdf.stat().st_size > 0


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_reimport_benchmarks_report_multiple_sibling_file_path(
    tmp_path: Path, minimal_report: GenerativeBenchmarksReport
):
    """
    Test that reimport uses a sibling path preserving stem for other formats.

    ## WRITTEN BY AI ##
    """
    # Save the report to a temp JSON file
    report_file = tmp_path / "report.json"
    report_file.write_text(minimal_report.model_dump_json(), encoding="utf-8")

    expected_json = tmp_path / "my_report.json"
    expected_png = tmp_path / "my_report.png"

    await reimport_benchmarks_report(
        file=report_file,
        outputs=[
            {"kind": "json", "path": expected_json},
            {"kind": "plot", "path": expected_png},
        ],
    )

    assert expected_json.exists()
    assert expected_json.is_file()
    assert expected_png.exists()
    assert expected_png.is_file()
