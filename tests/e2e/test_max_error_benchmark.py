# test_server_interaction.py

import json
import subprocess
import time
from pathlib import Path

import pytest
from loguru import logger

from tests.e2e.vllm_sim_server import VllmSimServer


@pytest.fixture(scope="module")
def server():
    """
    Pytest fixture to start and stop the server for the entire module
    using the TestServer class.
    """
    server = VllmSimServer(port=8000, model="databricks/dolly-v2-12b", mode="echo")
    try:
        server.start()
        yield server  # Yield the URL for tests to use
    finally:
        server.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(30)
def test_max_error_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/max_error_benchmarks.json")
    rate = 10
    max_error_rate = 0.1
    command = f"""guidellm benchmark \
  --target "{server.get_url()}" \
  --rate-type constant \
  --rate {rate} \
  --max-seconds 60 \
  --max-error {max_error_rate} \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-path {report_path}
              """
    logger.info(f"Client command: {command}")
    process = subprocess.Popen(  # noqa: S603
        ["/bin/bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    logger.info("Waiting for client to start...")
    time.sleep(10)
    server.stop()

    try:
        logger.info("Fetching client output")
        stdout, stderr = process.communicate()
        logger.debug(f"Client stdout:\n{stdout}")
        logger.debug(f"Client stderr:\n{stderr}")

        assert report_path.exists()
        with report_path.open("r") as f:
            report = json.load(f)

        assert "benchmarks" in report
        benchmarks = report["benchmarks"]
        assert len(benchmarks) > 0
        benchmark = benchmarks[0]
        assert "run_stats" in benchmark
        run_stats = benchmark["run_stats"]
        assert "status" in run_stats
        status = run_stats["status"]
        assert status == "error"
        assert "termination_reason" in run_stats
        termination_reason = run_stats["termination_reason"]
        assert termination_reason == "max_error_reached"
        assert "window_error_rate" in run_stats
        window_error_rate = run_stats["window_error_rate"]
        assert window_error_rate > max_error_rate
    finally:
        process.terminate()  # Send SIGTERM
        try:
            process.wait(timeout=5)  # Wait for the process to terminate
            logger.info("Client stopped successfully.")
        except subprocess.TimeoutExpired:
            logger.warning("Client did not terminate gracefully, killing it...")
            process.kill()  # Send SIGKILL if it doesn't terminate
            process.wait()

    if report_path.exists():
        report_path.unlink()
