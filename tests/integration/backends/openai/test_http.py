"""HTTP backend metadata capture through the CLI and saved report."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.fixtures.tokenizers import MINIMAL_TOKENIZER_DIR


@pytest.mark.regression
@pytest.mark.parametrize(
    ("selection", "server_info_status"),
    [
        (["vllm_config"], 200),
        (["vllm_env", "system_env"], 200),
        ("all", 200),
        ("all", 404),
        (None, 200),
        ([], 200),
    ],
)
def test_server_config_capture_in_cli_report(
    tmp_path: Path, selection, server_info_status: int
):
    """Run real HTTP requests and workers, preserving optional report metadata.

    The local protocol fixture performs no model inference.

    """
    requests: list[str] = []
    diagnostic = "capture-only-system-diagnostic " * 1000

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            requests.append(self.path)
            if self.path == "/health":
                self.reply(200, {})
            elif self.path == "/v1/models":
                self.reply(200, {"data": [{"id": "test-model"}]})
            elif self.path == "/server_info?config_format=json":
                self.reply(
                    server_info_status,
                    {
                        "vllm_config": {
                            "parallel_config": {"tensor_parallel_size": 2},
                            "model_config": {
                                "hf_token": "private-token",
                                "model": "capture-only-model-name",
                                "endpoint": "postgres://user:private-url-secret@localhost/db",
                                "args": [
                                    "--api-key",
                                    "private-argument",
                                    "--max-tokens",
                                    "32",
                                ],
                            },
                        },
                        "vllm_env": {
                            "VLLM_USE_V1": True,
                            "VLLM_API_KEY": "private-key",
                            "VLLM_EC_SIDE_CHANNEL_HOST": "private-host.internal",
                        },
                        "system_env": {
                            "cuda_runtime_version": "12.8",
                            "HF_TOKEN": "private-token",
                            "cpu_info": diagnostic,
                            "env_vars": (
                                "CUDA_VERSION=13.0\nVLLM_API_KEY=private-inline-secret"
                            ),
                        },
                    },
                )
            else:
                self.reply(404, {})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            self.reply(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "hello world",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                        "total_tokens": 10,
                    },
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report_path = tmp_path / "benchmark.json"
    target = f"http://127.0.0.1:{server.server_port}"
    backend_arg = (
        f"kind=openai_http,target={target},stream=false,capture_server_config=all"
        if selection == "all"
        else json.dumps(
            {
                "kind": "openai_http",
                "target": target,
                "stream": False,
                "capture_server_config": selection,
            }
        )
    )
    try:
        result = subprocess.run(  # noqa: S603 - fixed CLI and local fixture inputs
            [
                sys.executable,
                "-m",
                "guidellm",
                "run",
                "--backend",
                backend_arg,
                "--profile",
                "kind=synchronous",
                "--data",
                "kind=synthetic_text,prompt_tokens=8,output_tokens=2",
                "--tokenizer",
                f"kind=huggingface_auto,model={MINIMAL_TOKENIZER_DIR}",
                "--constraint",
                "kind=max_requests,count=2",
                "--output",
                f"kind=json,path={report_path}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    console_output = result.stdout + result.stderr
    assert result.returncode == 0, console_output
    assert "backend validated" in console_output
    assert "capture-only-" not in console_output
    assert "private-" not in console_output
    report_text = report_path.read_text()
    report = json.loads(report_text)
    benchmark = report["benchmarks"][0]
    assert len(benchmark["requests"]["successful"]) == 2
    if not selection:
        assert "/server_info?config_format=json" not in requests
        assert "server_info" not in benchmark["config"]["backend"]
        return
    assert requests.count("/server_info?config_format=json") == 1
    assert requests.index("/server_info?config_format=json") < requests.index(
        "/v1/chat/completions"
    )
    assert "private-" not in report_text
    if server_info_status == 200:
        expected = {
            "vllm_config": {
                "parallel_config": {"tensor_parallel_size": 2},
                "model_config": {
                    "hf_token": "[REDACTED]",
                    "model": "capture-only-model-name",
                    "endpoint": "postgres://[REDACTED]@localhost/db",
                    "args": ["--api-key", "[REDACTED]", "--max-tokens", "32"],
                },
            },
            "vllm_env": {
                "VLLM_USE_V1": True,
                "VLLM_API_KEY": "[REDACTED]",
                "VLLM_EC_SIDE_CHANNEL_HOST": "[REDACTED]",
            },
            "system_env": {
                "cuda_runtime_version": "12.8",
                "HF_TOKEN": "[REDACTED]",
                "cpu_info": diagnostic,
                "env_vars": "[REDACTED]",
            },
        }
        sections = set(expected) if selection == "all" else set(selection)
        assert benchmark["config"]["backend"]["server_info"] == {
            key: value for key, value in expected.items() if key in sections
        }
    else:
        assert "server_info" not in benchmark["config"]["backend"]
