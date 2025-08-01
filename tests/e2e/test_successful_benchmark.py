# test_server_interaction.py

import json
import os
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
def test_max_seconds_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/max_duration_benchmarks.json")
    rate = 10
    command = f"""
guidellm benchmark \
  --target "{server.get_url()}" \
  --rate-type constant \
  --rate {rate} \
  --max-seconds 1 \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-path {report_path}
              """

    logger.info(f"Client command: {command}")
    os.system(command)  # noqa: S605

    assert report_path.exists()
    with report_path.open("r") as f:
        report = json.load(f)

    assert "benchmarks" in report
    benchmarks = report["benchmarks"]
    assert len(benchmarks) > 0
    benchmark = benchmarks[0]
    assert "requests" in benchmark
    requests = benchmark["requests"]
    assert "successful" in requests
    successful = requests["successful"]
    assert len(successful) > rate

    assert "run_stats" in benchmark
    run_stats = benchmark["run_stats"]
    assert "status" in run_stats
    status = run_stats["status"]
    assert status == "success"
    assert "termination_reason" in run_stats
    termination_reason = run_stats["termination_reason"]
    assert termination_reason == "max_seconds_reached"

    if report_path.exists():
        report_path.unlink()


@pytest.mark.timeout(30)
def test_max_requests_benchmark(server: VllmSimServer):
    """
    Another example test interacting with the server.
    """
    report_path = Path("tests/e2e/max_number_benchmarks.json")
    rate = 10
    command = f"""
guidellm benchmark \
  --target "{server.get_url()}" \
  --rate-type constant \
  --rate {rate} \
  --max-requests {rate} \
  --data "prompt_tokens=256,output_tokens=128" \
  --output-path {report_path}
              """

    logger.info(f"Client command: {command}")
    os.system(command)  # noqa: S605

    assert report_path.exists()
    with report_path.open("r") as f:
        report = json.load(f)

    assert "benchmarks" in report
    benchmarks = report["benchmarks"]
    assert len(benchmarks) > 0
    benchmark = benchmarks[0]
    assert "requests" in benchmark
    requests = benchmark["requests"]
    assert "successful" in requests
    successful = requests["successful"]
    assert len(successful) == rate

    assert "run_stats" in benchmark
    run_stats = benchmark["run_stats"]
    assert "status" in run_stats
    status = run_stats["status"]
    assert status == "success"
    assert "termination_reason" in run_stats
    termination_reason = run_stats["termination_reason"]
    assert termination_reason == "max_requests_reached"

    if report_path.exists():
        report_path.unlink()
