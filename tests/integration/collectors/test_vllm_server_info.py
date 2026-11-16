"""Exercise server-info collection across a real process boundary and reports."""

import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from guidellm.benchmark.schemas import GenerativeBenchmarksReport
from guidellm.collectors import CollectorProcess
from guidellm.scheduler import ConcurrentStrategy
from guidellm.schemas.benchmark import BenchmarkScenario, VLLMServerInfoCollectorArgs
from tests.unit.benchmark.html_report_fixtures import make_benchmark


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("second_status", [200, 404])
async def test_sub_benchmark_snapshots_survive_process_and_report_round_trip(
    tmp_path: Path, second_status: int
):
    """Capture once per benchmark, preserve failures and reload JSON/YAML reports.

    ## WRITTEN BY AI ##
    """
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            requests.append(self.path)
            status = 200 if len(requests) == 1 else second_status
            body = json.dumps(
                {
                    "vllm_config": {
                        "max_model_len": len(requests),
                        "api_key": "private-value",
                    },
                    "system_env": {"SECRET": "private-value"},
                }
            ).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/server_info"
    args = VLLMServerInfoCollectorArgs(url=url)
    process = CollectorProcess([args])
    benchmarks = []
    try:
        await process.start()
        for _ in range(2):
            start = time.time()
            await process.benchmark_started()
            assert len(requests) == len(benchmarks)
            await process.benchmark_stopped()
            benchmark = make_benchmark(
                strategy=ConcurrentStrategy(streams=1), rps=1.0, tps=16.0
            )
            benchmark.collectors = await process.results(start, time.time())
            benchmarks.append(benchmark)
    finally:
        await process.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert requests == ["/server_info?config_format=json"] * 2
    first, second = [benchmark.collectors[0] for benchmark in benchmarks]
    assert first["kind"] == second["kind"] == "vllm_server_info"
    assert first["vllm_config"] == {"max_model_len": 1, "api_key": "[REDACTED]"}
    if second_status == 200:
        assert second["vllm_config"]["max_model_len"] == 2
    else:
        assert second["error"] == "HTTPStatusError"
        assert "vllm_config" not in second
    assert first["collected_at"] <= second["collected_at"]

    report = GenerativeBenchmarksReport(
        config=BenchmarkScenario(
            spec={
                "backend": {
                    "kind": "openai_http",
                    "target": "http://different-backend:8000",
                },
                "data": [
                    {"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}
                ],
                "profile": {"kind": "synchronous"},
                "collectors": [args.model_dump()],
            }
        ),
        benchmarks=benchmarks,
    )
    for extension in ("json", "yaml"):
        path = report.save_file(tmp_path / f"benchmarks.{extension}")
        assert "private-value" not in path.read_text()
        restored = GenerativeBenchmarksReport.load_file(path)
        assert [item.collectors for item in restored.benchmarks] == [[first], [second]]
