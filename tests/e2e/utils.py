"""Utilities for E2E tests."""

import json
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger


def get_guidellm_executable() -> str:
    """Get the path to the guidellm executable in the current environment."""
    # Get the directory where the current Python executable is located
    python_bin_dir = Path(sys.executable).parent
    guidellm_path = python_bin_dir / "guidellm"
    if guidellm_path.exists():
        return str(guidellm_path)
    else:
        # Fallback to just "guidellm" if not found
        return "guidellm"


class GuidellmClient:
    """Wrapper class for running guidellm benchmark commands."""

    def __init__(self, target: str, output_path: Path):
        """
        Initialize the guidellm client.

        :param target: The target URL for the benchmark
        :param output_path: Path where the benchmark report will be saved
        """
        self.target = target
        self.output_path = output_path
        self.process: subprocess.Popen | None = None
        self.stdout: str | None = None
        self.stderr: str | None = None

    def start_benchmark(
        self,
        profile: str = "constant",
        rate: int = 10,
        max_seconds: int | None = None,
        max_requests: int | None = None,
        max_error_rate: float | None = None,
        stop_over_saturated: bool | None = False,
        data: str = "prompt_tokens=256,output_tokens=128",
        processor: str = "gpt2",
        additional_args: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """
        Start a guidellm benchmark command.

        :param rate_type: Type of rate control (constant, etc.)
        :param rate: Request rate
        :param max_seconds: Maximum duration in seconds
        :param max_requests: Maximum number of requests
        :param max_error_rate: Maximum error rate before stopping
        :param stop_over_saturated: Whether to stop the benchmark if the model is
                                    over-saturated.
        :param data: Data configuration string
        :param processor: Processor/tokenizer to use
        :param additional_args: Additional command line arguments
        """
        guidellm_exe = get_guidellm_executable()

        # Build command components
        cmd_parts = [
            *([f"{k}={v}" for k, v in extra_env.items()] if extra_env else []),
            "HF_HOME=/tmp/huggingface_cache",
            f"{guidellm_exe} benchmark",
            f'--target "{self.target}"',
            f"--profile {profile}",
            f"--rate {rate}",
        ]

        if max_seconds is not None:
            cmd_parts.append(f"--max-seconds {max_seconds}")

        if max_requests is not None:
            cmd_parts.append(f"--max-requests {max_requests}")

        if max_error_rate is not None:
            cmd_parts.append(f"--max-error-rate {max_error_rate}")

        if stop_over_saturated:
            cmd_parts.append("--stop-over-saturated")

        cmd_parts.extend(
            [
                f'--data "{data}"',
                f'--processor "{processor}"',
                f"--output-path {self.output_path}",
            ]
        )

        if additional_args:
            cmd_parts.append(additional_args)

        command = " \\\n  ".join(cmd_parts)

        logger.info(f"Client command: {command}")

        self.process = subprocess.Popen(  # noqa: S603
            ["/bin/bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_for_completion(
        self, timeout: int = 30, stop_server_after: int | None = None, server=None
    ) -> None:
        """
        Wait for the benchmark to complete.

        :param timeout: Maximum time to wait for completion
        :param stop_server_after: If provided, stop the server after this many seconds
        :param server: Server object to stop (if stop_server_after is provided)
        """
        if self.process is None:
            raise RuntimeError("No process started. Call start_benchmark() first.")

        if stop_server_after is not None and server is not None:
            logger.info(
                f"Waiting {stop_server_after} seconds before stopping server..."
            )
            time.sleep(stop_server_after)
            server.stop()

        try:
            logger.info("Fetching client output")
            self.stdout, self.stderr = self.process.communicate(timeout=timeout)
            logger.debug(f"Client stdout:\n{self.stdout}")
            logger.debug(f"Client stderr:\n{self.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning("Client did not complete within timeout, terminating...")
            self.process.terminate()
            try:
                self.stdout, self.stderr = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Client did not terminate gracefully, killing it...")
                self.process.kill()
                self.stdout, self.stderr = self.process.communicate()
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                    logger.info("Client stopped successfully.")
                except subprocess.TimeoutExpired:
                    logger.warning("Client did not terminate gracefully, killing it...")
                    self.process.kill()
                    self.process.wait()


def assert_no_python_exceptions(stderr: str | None) -> None:
    """
    Assert that stderr does not contain any Python exception indicators.

    :param stderr: The stderr string to check (can be None)
    :raises AssertionError: If Python exceptions are detected
    """
    if stderr is None:
        return  # No stderr to check

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
        assert indicator not in stderr, (
            f"Python exception detected in stderr: {indicator}"
        )


def load_benchmark_report(report_path: Path) -> dict:
    """
    Load and validate a benchmark report JSON file.

    :param report_path: Path to the report file
    :return: The loaded report dictionary
    :raises AssertionError: If the file doesn't exist or is invalid
    """
    assert report_path.exists(), f"Report file does not exist: {report_path}"

    with report_path.open("r") as f:
        report = json.load(f)

    assert "benchmarks" in report, "Report missing 'benchmarks' field"
    benchmarks = report["benchmarks"]
    assert len(benchmarks) > 0, "Report contains no benchmarks"

    return report


def assert_successful_requests_fields(successful_requests: list) -> None:
    """
    Assert that successful requests contain all expected timing and token fields.

    :param successful_requests: List of successful request objects
    :raises AssertionError: If required fields are missing or invalid
    """
    assert len(successful_requests) >= 1, "No successful requests found"

    for request in successful_requests:
        # Basic latency
        assert "request_latency" in request, "Missing 'request_latency' field"
        assert request["request_latency"] > 0, "request_latency should be > 0"

        # Streaming timing fields
        assert "time_to_first_token_ms" in request, (
            "Missing 'time_to_first_token_ms' field"
        )
        assert request["time_to_first_token_ms"] is not None, (
            "time_to_first_token_ms should not be None"
        )
        assert request["time_to_first_token_ms"] > 0, (
            "time_to_first_token_ms should be > 0"
        )

        assert "time_per_output_token_ms" in request, (
            "Missing 'time_per_output_token_ms' field"
        )
        assert request["time_per_output_token_ms"] is not None, (
            "time_per_output_token_ms should not be None"
        )
        assert request["time_per_output_token_ms"] > 0, (
            "time_per_output_token_ms should be > 0"
        )

        assert "inter_token_latency_ms" in request, (
            "Missing 'inter_token_latency_ms' field"
        )
        assert request["inter_token_latency_ms"] is not None, (
            "inter_token_latency_ms should not be None"
        )
        assert request["inter_token_latency_ms"] > 0, (
            "inter_token_latency_ms should be > 0"
        )

        # Token throughput fields
        assert "tokens_per_second" in request, "Missing 'tokens_per_second' field"
        assert request["tokens_per_second"] > 0, "tokens_per_second should be > 0"

        assert "output_tokens_per_second" in request, (
            "Missing 'output_tokens_per_second' field"
        )
        assert request["output_tokens_per_second"] > 0, (
            "output_tokens_per_second should be > 0"
        )

        # Token count fields
        assert "total_tokens" in request, "Missing 'total_tokens' field"
        assert request["total_tokens"] > 0, "total_tokens should be > 0"

        assert "prompt_tokens" in request, "Missing 'prompt_tokens' field"
        assert request["prompt_tokens"] > 0, "prompt_tokens should be > 0"

        assert "output_tokens" in request, "Missing 'output_tokens' field"
        assert request["output_tokens"] > 0, "output_tokens should be > 0"


def assert_constraint_triggered(
    benchmark: dict, constraint_name: str, expected_metadata: dict
) -> None:
    """
    Assert that a specific constraint was triggered with expected metadata.

    :param benchmark: The benchmark object
    :param constraint_name: Name of the constraint (e.g., 'max_seconds', 'max_requests',
                            'max_error_rate')
    :param expected_metadata: Dictionary of expected metadata fields and values
    :raises AssertionError: If constraint was not triggered or metadata is incorrect
    """
    assert "scheduler_state" in benchmark, "Benchmark missing 'scheduler_state' field"
    scheduler_state = benchmark["scheduler_state"]

    assert "end_processing_constraints" in scheduler_state, (
        "Scheduler state missing 'end_processing_constraints' field"
    )
    constraints = scheduler_state["end_processing_constraints"]

    assert constraint_name in constraints, (
        f"Constraint '{constraint_name}' was not triggered"
    )
    constraint = constraints[constraint_name]

    assert "metadata" in constraint, (
        f"Constraint '{constraint_name}' missing 'metadata' field"
    )
    metadata = constraint["metadata"]

    for key, expected_value in expected_metadata.items():
        assert key in metadata, (
            f"Constraint '{constraint_name}' metadata missing '{key}' field"
        )
        actual_value = metadata[key]

        if isinstance(expected_value, bool):
            assert actual_value is expected_value, (
                f"Expected {key}={expected_value}, got {actual_value}"
            )
        elif callable(expected_value):
            # Allow callable predicates for complex validation
            assert expected_value(actual_value), (
                f"Predicate failed for {key}={actual_value}"
            )
        else:
            assert actual_value == expected_value, (
                f"Expected {key}={expected_value}, got {actual_value}"
            )


def cleanup_report_file(report_path: Path) -> None:
    """
    Clean up the report file if it exists.

    :param report_path: Path to the report file to remove
    """
    if report_path.exists():
        report_path.unlink()
