"""End-to-end generated-audio benchmark coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)


@pytest.mark.timeout(60)
@pytest.mark.regression
def test_speech_endpoint_audio_metrics_reach_benchmark_report(
    server: E2EServer,
    e2e_server_kind: str,
    tmp_path: Path,
):
    """Benchmark generated speech through MockServer and serialize its metrics.

    :param server: Running E2E server fixture.
    :param e2e_server_kind: Selected E2E server type.
    :param tmp_path: Temporary directory for benchmark output.
    :return: None.
    ## WRITTEN BY AI ##
    """
    if e2e_server_kind != "mock":
        pytest.skip("The llm-d simulator does not provide the speech endpoint.")

    report_name = "audio_speech_benchmark.json"
    report_path = tmp_path / report_name
    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )
    client.start_benchmark(
        rate=10,
        max_requests=2,
        data="kind=synthetic_text,prompt_tokens=8,output_tokens=16",
        backend=(
            f"kind=openai_http,target={server.get_url()},"
            "request_format=/v1/audio/speech,"
            "extras.body.response_format=pcm,"
            "extras.body.stream_format=audio"
        ),
    )
    client.wait_for_completion(timeout=45)

    assert_no_python_exceptions(client.stderr)
    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]
    audio_seconds = benchmark["metrics"]["audio"]["seconds"]

    assert len(benchmark["requests"]["successful"]) == 2
    assert audio_seconds["output"]["successful"]["mean"] == pytest.approx(0.4)
