"""Plain text progress through the real CLI, scheduler, and HTTP backend."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer
from tests.e2e.utils import assert_no_python_exceptions, load_benchmark_report
from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR


@pytest.mark.regression
@pytest.mark.timeout(60)
def test_simple_progress_with_redirected_stdout(server: E2EServer, tmp_path: Path):
    """A real benchmark emits periodic lines and correct final counts into a pipe.

    ## WRITTEN BY AI ##
    """
    report_path = tmp_path / "benchmarks.json"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "guidellm",
            "run",
            "--backend",
            f"kind=openai_http,target={server.get_url()}",
            "--profile",
            "kind=constant,rate=4",
            "--constraint",
            "kind=max_duration,seconds=3",
            "--data",
            "kind=synthetic_text,prompt_tokens=64,output_tokens=16",
            "--tokenizer",
            f"kind=huggingface_auto,model={MINIMAL_TOKENIZER_DIR}",
            "--output",
            f"kind=json,path={report_path}",
            "--console",
            "kind=simple,interval=0.5",
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert_no_python_exceptions(result.stderr)
    lines = [
        line for line in result.stdout.splitlines() if line.startswith("Benchmark 1 (")
    ]
    assert len(lines) >= 3, result.stdout
    assert ": started |" in lines[0]
    assert ": completed |" in lines[-1]
    assert any(": active |" in line for line in lines[1:-1])
    assert all("\x1b" not in line for line in lines)
    report = load_benchmark_report(report_path)
    totals = report["benchmarks"][0]["metrics"]["request_totals"]
    assert totals["successful"] > 0
    assert f"successful={totals['successful']}" in lines[-1]
    assert f"errored={totals['errored']}" in lines[-1]
