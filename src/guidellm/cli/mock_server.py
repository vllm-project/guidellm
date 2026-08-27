"""Mock server command for testing."""

from __future__ import annotations

import click

import guidellm.entrypoints as entry
from guidellm.schemas.mock_server import MockServerConfig
from guidellm.utils.console import Console

__all__ = ["mock_server"]


@click.command(
    "mock-server",
    help=(
        "Start a mock OpenAI/vLLM-compatible server for testing. "
        "Simulates model inference with configurable latency and token generation."
    ),
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host address to bind the server to.",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port number to bind the server to.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes.",
)
@click.option(
    "--model",
    default="llama-3.1-8b-instruct",
    help="Name of the model to mock.",
)
@click.option(
    "--processor",
    default=None,
    help="Processor or tokenizer to use for requests.",
)
@click.option(
    "--request-latency",
    default=3,
    type=float,
    help="Request latency in seconds for non-streaming requests.",
)
@click.option(
    "--request-latency-std",
    default=0,
    type=float,
    help="Request latency standard deviation in seconds (normal distribution).",
)
@click.option(
    "--ttft-ms",
    default=150,
    type=float,
    help="Time to first token in milliseconds for streaming requests.",
)
@click.option(
    "--ttft-ms-std",
    default=0,
    type=float,
    help="Time to first token standard deviation in milliseconds.",
)
@click.option(
    "--itl-ms",
    default=10,
    type=float,
    help="Inter-token latency in milliseconds for streaming requests.",
)
@click.option(
    "--itl-ms-std",
    default=0,
    type=float,
    help="Inter-token latency standard deviation in milliseconds.",
)
@click.option(
    "--output-tokens",
    default=128,
    type=int,
    help="Number of output tokens for streaming requests.",
)
@click.option(
    "--output-tokens-std",
    default=0,
    type=float,
    help="Output tokens standard deviation (normal distribution).",
)
@click.option(
    "--fail-after-requests",
    default=None,
    type=int,
    help=(
        "After this many generation requests, return HTTP 500 for subsequent "
        "generation requests. Useful for testing error-rate constraints."
    ),
)
@click.option(
    "--max-concurrent-requests",
    default=None,
    type=int,
    help=(
        "Limit in-flight generation requests (others wait). Useful for testing "
        "over-saturation detection."
    ),
)
def mock_server(
    host: str,
    port: int,
    workers: int,
    model: str,
    processor: str | None,
    request_latency: float,
    request_latency_std: float,
    ttft_ms: float,
    ttft_ms_std: float,
    itl_ms: float,
    itl_ms_std: float,
    output_tokens: int,
    output_tokens_std: float,
    fail_after_requests: int | None,
    max_concurrent_requests: int | None,
):
    config = MockServerConfig(
        host=host,
        port=port,
        workers=workers,
        model=model,
        processor=processor,
        request_latency=request_latency,
        request_latency_std=request_latency_std,
        ttft_ms=ttft_ms,
        ttft_ms_std=ttft_ms_std,
        itl_ms=itl_ms,
        itl_ms_std=itl_ms_std,
        output_tokens=output_tokens,
        output_tokens_std=output_tokens_std,
        fail_after_requests=fail_after_requests,
        max_concurrent_requests=max_concurrent_requests,
    )

    server = entry.MockServer(config)
    console = Console()
    console.print_update(
        title="GuideLLM mock server starting...",
        details=f"Listening on http://{host}:{port} for model {model}",
        status="success",
    )
    server.run()
