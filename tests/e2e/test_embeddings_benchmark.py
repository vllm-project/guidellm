# E2E tests for embeddings benchmark scenarios

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from loguru import logger


class EmbeddingsMockServer:
    """Mock server for embeddings E2E tests using guidellm mock-server."""

    def __init__(self, port: int, model: str = "BAAI/bge-base-en-v1.5"):
        self.port = port
        self.model = model
        self.server_url = f"http://127.0.0.1:{self.port}"
        self.health_url = f"{self.server_url}/health"
        self.process: subprocess.Popen | None = None

    def get_guidellm_executable(self) -> str:
        """Get the path to the guidellm executable in the current environment."""
        python_bin_dir = Path(sys.executable).parent
        guidellm_path = python_bin_dir / "guidellm"
        if guidellm_path.exists():
            return str(guidellm_path)
        return "guidellm"

    def start(self):
        """Start the mock embeddings server."""
        guidellm_exe = self.get_guidellm_executable()

        logger.info(f"Starting embeddings mock server on {self.server_url}...")
        command = [
            guidellm_exe,
            "mock-server",
            "--port",
            str(self.port),
            "--model",
            self.model,
        ]
        logger.info(f"Server command: {' '.join(command)}")

        self.process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to become healthy
        max_retries = 30
        retry_delay_sec = 0.5
        for i in range(max_retries):
            try:
                response = requests.get(self.health_url, timeout=1)
                if response.status_code == 200:
                    logger.info(f"Embeddings mock server started at {self.server_url}")
                    return
            except requests.RequestException:
                pass

            if i < max_retries - 1:
                time.sleep(retry_delay_sec)

        # Server didn't start, terminate and raise
        self.stop()
        raise RuntimeError(
            f"Embeddings mock server failed to start after {max_retries} retries"
        )

    def stop(self):
        """Stop the mock server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping embeddings mock server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing it...")
                self.process.kill()
                self.process.wait()
            logger.info("Embeddings mock server stopped.")

    def get_url(self) -> str:
        """Get the server URL."""
        return self.server_url


class EmbeddingsClient:
    """Wrapper for running guidellm embeddings benchmark commands."""

    def __init__(
        self, target: str, output_dir: Path, outputs: str = "embeddings_benchmarks.json"
    ):
        self.target = target
        self.output_dir = output_dir
        self.outputs = outputs
        self.process: subprocess.Popen | None = None
        self.stdout: str | None = None
        self.stderr: str | None = None

    def get_guidellm_executable(self) -> str:
        """Get the path to the guidellm executable."""
        python_bin_dir = Path(sys.executable).parent
        guidellm_path = python_bin_dir / "guidellm"
        if guidellm_path.exists():
            return str(guidellm_path)
        return "guidellm"

    def start_benchmark(
        self,
        data: str = "Benchmark this text for embeddings quality",
        profile: str = "constant",
        rate: int = 10,
        max_requests: int | None = None,
        max_duration: int | None = None,
        encoding_format: str = "float",
        enable_quality_validation: bool = False,
        baseline_model: str | None = None,
        quality_tolerance: float | None = None,
        processor: str | None = None,
        additional_args: str = "",
    ):
        """Start embeddings benchmark command."""
        guidellm_exe = self.get_guidellm_executable()

        # Build command components
        cmd_parts = [
            f"HF_HOME={self.output_dir / 'huggingface_cache'}",
            f"{guidellm_exe} benchmark embeddings",
            f"--target {self.target}",
            f"--data '{data}'",
            f"--profile {profile}",
            f"--rate {rate}",
            f"--encoding-format {encoding_format}",
            f"--output-dir {self.output_dir}",
            f"--outputs {self.outputs}",
        ]

        if max_requests is not None:
            cmd_parts.append(f"--max-requests {max_requests}")

        if max_duration is not None:
            cmd_parts.append(f"--max-duration {max_duration}")

        if enable_quality_validation:
            cmd_parts.append("--enable-quality-validation")

        if baseline_model is not None:
            cmd_parts.append(f"--baseline-model {baseline_model}")

        if quality_tolerance is not None:
            cmd_parts.append(f"--quality-tolerance {quality_tolerance}")

        if processor is not None:
            cmd_parts.append(f"--processor {processor}")

        if additional_args:
            cmd_parts.append(additional_args)

        command = " \\\n  ".join(cmd_parts)
        logger.info(f"Embeddings benchmark command: {command}")

        self.process = subprocess.Popen(  # noqa: S603
            ["/bin/sh", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_for_completion(self, timeout: int = 30):
        """Wait for the benchmark to complete."""
        if self.process is None:
            raise RuntimeError("No process started. Call start_benchmark() first.")

        try:
            logger.info("Waiting for embeddings benchmark to complete...")
            self.stdout, self.stderr = self.process.communicate(timeout=timeout)
            logger.debug(f"Benchmark stdout:\n{self.stdout}")
            logger.debug(f"Benchmark stderr:\n{self.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("Benchmark did not complete within timeout, terminating...")
            self.process.terminate()
            try:
                self.stdout, self.stderr = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Benchmark did not terminate gracefully, killing it...")
                self.process.kill()
                self.stdout, self.stderr = self.process.communicate()


@pytest.fixture(scope="module")
def embeddings_server():
    """Pytest fixture to start and stop embeddings mock server."""
    server = EmbeddingsMockServer(port=8001, model="test-embedding-model")
    try:
        server.start()
        yield server
    finally:
        server.stop()


def assert_no_python_exceptions(stderr: str | None) -> None:
    """Assert that stderr does not contain Python exception indicators."""
    if stderr is None:
        return

    python_exception_indicators = [
        "Traceback (most recent call last):",
        "AttributeError:",
        "ValueError:",
        "TypeError:",
        "KeyError:",
        "IndexError:",
        "NameError:",
        "ImportError:",
        "RuntimeError:",
    ]

    for indicator in python_exception_indicators:
        assert indicator not in stderr, f"Python exception detected: {indicator}"


def load_embeddings_report(report_path: Path) -> dict:
    """Load and validate embeddings benchmark report."""
    assert report_path.exists(), f"Report file does not exist: {report_path}"

    with report_path.open("r") as f:
        report = json.load(f)

    assert "type_" in report, "Report missing 'type_' field"
    assert report["type_"] == "embeddings_benchmarks_report", (
        f"Expected embeddings_benchmarks_report, got {report['type_']}"
    )
    assert "benchmarks" in report, "Report missing 'benchmarks' field"
    assert len(report["benchmarks"]) > 0, "Report contains no benchmarks"

    return report


def assert_embeddings_request_fields(requests: list) -> None:
    """Assert that embeddings requests contain expected fields."""
    assert len(requests) >= 1, "No requests found"

    for request in requests:
        # Basic fields
        assert "request_id" in request, "Missing 'request_id' field"
        assert "request_latency" in request, "Missing 'request_latency' field"
        assert request["request_latency"] > 0, "request_latency should be > 0"

        # Input token metrics (no output tokens for embeddings)
        assert "prompt_tokens" in request, "Missing 'prompt_tokens' field"
        assert request["prompt_tokens"] > 0, "prompt_tokens should be > 0"

        assert "total_tokens" in request, "Missing 'total_tokens' field"
        assert request["total_tokens"] > 0, "total_tokens should be > 0"

        # Should NOT have output token fields
        assert "output_tokens" not in request or request["output_tokens"] is None, (
            "Embeddings should not have output_tokens"
        )

        # Should NOT have streaming fields
        assert "time_to_first_token_ms" not in request, (
            "Embeddings should not have time_to_first_token_ms"
        )
        assert "inter_token_latency_ms" not in request, (
            "Embeddings should not have inter_token_latency_ms"
        )

        # Encoding format
        assert "encoding_format" in request, "Missing 'encoding_format' field"
        assert request["encoding_format"] in ["float", "base64"], (
            f"Invalid encoding_format: {request['encoding_format']}"
        )


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_basic_embeddings_benchmark(embeddings_server: EmbeddingsMockServer, tmp_path: Path):
    """Test basic embeddings benchmark execution."""
    report_name = "basic_embeddings.json"
    report_path = tmp_path / report_name

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        data="Test embeddings benchmark",
        max_requests=10,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)

    # Assert no Python exceptions
    assert_no_python_exceptions(client.stderr)

    # Load and validate report
    report = load_embeddings_report(report_path)
    benchmark = report["benchmarks"][0]

    # Validate requests
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == 10, (
        f"Expected 10 successful requests, got {len(successful_requests)}"
    )
    assert_embeddings_request_fields(successful_requests)

    # Validate metrics structure
    metrics = benchmark["metrics"]
    assert "request_totals" in metrics
    assert "input_tokens_count" in metrics
    assert "encoding_format_breakdown" in metrics

    # Should NOT have output token metrics
    assert "output_tokens_count" not in metrics, (
        "Embeddings metrics should not have output_tokens_count"
    )


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_embeddings_float_encoding(embeddings_server: EmbeddingsMockServer, tmp_path: Path):
    """Test embeddings benchmark with float encoding format."""
    report_name = "float_encoding_embeddings.json"
    report_path = tmp_path / report_name

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        data="Test float encoding",
        max_requests=5,
        encoding_format="float",
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_embeddings_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check encoding format
    successful_requests = benchmark["requests"]["successful"]
    for request in successful_requests:
        assert request["encoding_format"] == "float"

    # Check encoding_format_breakdown in metrics
    metrics = benchmark["metrics"]
    assert "float" in metrics["encoding_format_breakdown"]
    assert metrics["encoding_format_breakdown"]["float"] == 5


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_embeddings_base64_encoding(embeddings_server: EmbeddingsMockServer, tmp_path: Path):
    """Test embeddings benchmark with base64 encoding format."""
    report_name = "base64_encoding_embeddings.json"
    report_path = tmp_path / report_name

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        data="Test base64 encoding",
        max_requests=5,
        encoding_format="base64",
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_embeddings_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check encoding format
    successful_requests = benchmark["requests"]["successful"]
    for request in successful_requests:
        assert request["encoding_format"] == "base64"

    # Check encoding_format_breakdown in metrics
    metrics = benchmark["metrics"]
    assert "base64" in metrics["encoding_format_breakdown"]
    assert metrics["encoding_format_breakdown"]["base64"] == 5


@pytest.mark.timeout(60)
@pytest.mark.sanity
def test_embeddings_csv_output(embeddings_server: EmbeddingsMockServer, tmp_path: Path):
    """Test embeddings benchmark CSV output generation."""
    report_name = "embeddings_csv_test"

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs="json,csv",
    )

    client.start_benchmark(
        data="Test CSV output",
        max_requests=5,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=60)
    assert_no_python_exceptions(client.stderr)

    # Check both JSON and CSV files exist
    json_path = tmp_path / "embeddings_benchmarks.json"
    csv_path = tmp_path / "embeddings_benchmarks.csv"

    assert json_path.exists(), "JSON output file not created"
    assert csv_path.exists(), "CSV output file not created"

    # Validate CSV has content
    csv_content = csv_path.read_text()
    assert len(csv_content) > 0, "CSV file is empty"
    assert "request_latency" in csv_content, "CSV missing request_latency column"
    assert "prompt_tokens" in csv_content, "CSV missing prompt_tokens column"


@pytest.mark.timeout(60)
@pytest.mark.sanity
def test_embeddings_html_output(embeddings_server: EmbeddingsMockServer, tmp_path: Path):
    """Test embeddings benchmark HTML output generation."""
    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs="json,html",
    )

    client.start_benchmark(
        data="Test HTML output",
        max_requests=5,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=60)
    assert_no_python_exceptions(client.stderr)

    # Check both JSON and HTML files exist
    json_path = tmp_path / "embeddings_benchmarks.json"
    html_path = tmp_path / "embeddings_benchmarks.html"

    assert json_path.exists(), "JSON output file not created"
    assert html_path.exists(), "HTML output file not created"

    # Validate HTML has content
    html_content = html_path.read_text()
    assert len(html_content) > 0, "HTML file is empty"
    assert "<html" in html_content.lower(), "HTML file is not valid HTML"


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_embeddings_max_duration_constraint(
    embeddings_server: EmbeddingsMockServer, tmp_path: Path
):
    """Test that max_duration constraint works for embeddings."""
    report_name = "max_duration_embeddings.json"
    report_path = tmp_path / report_name

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Run for 3 seconds at 5 requests/sec
    client.start_benchmark(
        data="Test max duration",
        rate=5,
        max_duration=3,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_embeddings_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check duration is approximately 3 seconds
    assert benchmark["duration"] <= 5.0, (
        f"Benchmark duration {benchmark['duration']} exceeded expected range"
    )


@pytest.mark.timeout(30)
@pytest.mark.sanity
def test_embeddings_max_requests_constraint(
    embeddings_server: EmbeddingsMockServer, tmp_path: Path
):
    """Test that max_requests constraint works for embeddings."""
    report_name = "max_requests_embeddings.json"
    report_path = tmp_path / report_name
    max_requests = 8

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        data="Test max requests",
        max_requests=max_requests,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_embeddings_report(report_path)
    benchmark = report["benchmarks"][0]

    # Check request count matches constraint
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} requests, got {len(successful_requests)}"
    )


@pytest.mark.timeout(30)
@pytest.mark.regression
def test_embeddings_report_metadata(
    embeddings_server: EmbeddingsMockServer, tmp_path: Path
):
    """Test that embeddings report includes proper metadata."""
    report_name = "metadata_embeddings.json"
    report_path = tmp_path / report_name

    client = EmbeddingsClient(
        target=embeddings_server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        data="Test metadata",
        max_requests=3,
        processor="gpt2",
    )

    client.wait_for_completion(timeout=30)
    assert_no_python_exceptions(client.stderr)

    report = load_embeddings_report(report_path)

    # Check metadata
    assert "metadata" in report, "Report missing metadata"
    metadata = report["metadata"]

    assert "guidellm_version" in metadata, "Missing guidellm_version"
    assert "python_version" in metadata, "Missing python_version"
    assert "start_time" in metadata, "Missing start_time"

    # Check args
    assert "args" in report, "Report missing args"
    args = report["args"]

    assert "target" in args, "Missing target in args"
    assert "encoding_format" in args, "Missing encoding_format in args"
    assert args["encoding_format"] == "float", (
        f"Expected encoding_format='float', got {args['encoding_format']}"
    )
