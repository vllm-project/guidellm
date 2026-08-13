# E2E tests for client-side tool calling benchmark scenarios

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.conftest import E2EServer, start_mock_server
from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    load_benchmark_report,
)

# macOS workers segfault with fork; use spawn for maximum compatibility
_BENCHMARK_ENV = {"GUIDELLM__MP_CONTEXT_TYPE": "spawn"}


@pytest.fixture(scope="module")
def server() -> Iterator[E2EServer]:
    """
    Start and stop a MockServer for tool calling E2E tests.

    Uses the built-in MockServer which supports tool_calls responses
    when tool_choice="required" is present in the request. Uses an
    ephemeral port so a leftover listener on a fixed port cannot mask
    a failed bind.
    """
    handle = start_mock_server(
        model="test-tool-model",
        ttft_ms=5.0,
        itl_ms=1.0,
        output_tokens=32,
    )
    try:
        yield handle  # Yield the URL for tests to use
    finally:
        handle.stop()  # Teardown: Stop the server after tests are done


@pytest.mark.timeout(90)
@pytest.mark.sanity
def test_client_tool_call_single_turn(server: E2EServer, tmp_path: Path):
    """
    Test single-turn client tool calling: turns=1, tool_call_turns=1.

    Produces 2 requests per conversation (tool_call + injection).
    Validates that the benchmark completes, tool_call metrics are present,
    and timing fields are valid.

    ## WRITTEN BY AI ##
    """
    report_name = "tool_call_single_turn.json"
    report_path = tmp_path / report_name
    max_requests = 10

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Single-turn tool calling: each conversation = 2 requests
    client.start_benchmark(
        rate=4,
        max_requests=max_requests,
        data=(
            "kind=synthetic_text,prompt_tokens=64,output_tokens=32,"
            "turns=1,tool_call_turns=1"
        ),
        extra_env=_BENCHMARK_ENV,
    )

    client.wait_for_completion(timeout=60)

    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) > 0, "No successful requests found"

    # With max_requests=10 and 2 requests per conversation, we expect
    # exactly 10 successful requests (5 full conversations)
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} successful requests, got {len(successful_requests)}"
    )

    # Verify tool_calls are present in at least some requests
    # (the tool_call turns should have a non-empty tool_calls list)
    tool_call_requests = [r for r in successful_requests if r.get("tool_calls")]
    assert len(tool_call_requests) > 0, "No requests with tool_calls found"

    # Verify timing fields are present on all successful requests
    for request in successful_requests:
        assert request.get("request_latency", 0) > 0, "request_latency should be > 0"
        assert request.get("output_tokens", 0) > 0, "output_tokens should be > 0"


@pytest.mark.timeout(90)
@pytest.mark.sanity
def test_client_tool_call_multi_turn(server: E2EServer, tmp_path: Path):
    """
    Test multi-turn client tool calling: turns=3, tool_call_turns=2.

    Produces 5 requests per conversation:
    tool_call, injection, tool_call, injection, standard.
    Validates request count and that max_requests constraint triggers.

    ## WRITTEN BY AI ##
    """
    report_name = "tool_call_multi_turn.json"
    report_path = tmp_path / report_name
    # 5 requests per conversation, use 10 total
    max_requests = 10

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    client.start_benchmark(
        rate=4,
        max_requests=max_requests,
        data=(
            "kind=synthetic_text,prompt_tokens=64,output_tokens=32,"
            "turns=3,tool_call_turns=2"
        ),
        extra_env=_BENCHMARK_ENV,
    )

    client.wait_for_completion(timeout=60)

    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) > 0, "No successful requests found"

    # With 5 requests per conversation and max_requests=10, we expect
    # exactly 10 successful requests (2 full conversations)
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} successful requests, got {len(successful_requests)}"
    )

    # Verify the max_requests constraint was triggered
    scheduler_state = benchmark["scheduler_state"]
    constraints = scheduler_state.get("end_processing_constraints", {})
    assert "max_requests" in constraints, "max_requests constraint was not triggered"

    # Verify timing fields
    for request in successful_requests:
        assert request.get("request_latency", 0) > 0, "request_latency should be > 0"


@pytest.mark.timeout(90)
def test_client_tool_call_with_tool_response_tokens(server: E2EServer, tmp_path: Path):
    """
    Test tool calling with configured tool_response_tokens.

    Uses turns=2, tool_call_turns=1, tool_response_tokens=50 to verify
    that the benchmark completes successfully when tool response injection
    uses a specific token count for the mocked tool output.

    ## WRITTEN BY AI ##
    """
    report_name = "tool_call_response_tokens.json"
    report_path = tmp_path / report_name
    max_requests = 6

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # turns=2, tool_call_turns=1 -> 3 requests per conversation:
    # tool_call, injection, standard
    client.start_benchmark(
        rate=4,
        max_requests=max_requests,
        data=(
            "kind=synthetic_text,prompt_tokens=64,output_tokens=32,"
            "turns=2,tool_call_turns=1,tool_response_tokens=50"
        ),
        extra_env=_BENCHMARK_ENV,
    )

    client.wait_for_completion(timeout=60)

    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) > 0, "No successful requests found"

    # 3 requests per conversation, max_requests=6 -> 2 full conversations
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} successful requests, got {len(successful_requests)}"
    )

    # Verify timing fields
    for request in successful_requests:
        assert request.get("request_latency", 0) > 0, "request_latency should be > 0"
        assert request.get("output_tokens", 0) > 0, "output_tokens should be > 0"


@pytest.mark.timeout(90)
@pytest.mark.sanity
def test_client_tool_call_responses_api(server: E2EServer, tmp_path: Path):
    """
    Test client-side tool calling over the Responses API (/v1/responses).

    Uses turns=1, tool_call_turns=1 with request_format=/v1/responses.
    Validates that the benchmark completes with function_call items and
    correct request count (2: tool_call + injection).

    ## WRITTEN BY AI ##
    """
    report_name = "tool_call_responses_api.json"
    report_path = tmp_path / report_name
    max_requests = 10

    client = GuidellmClient(
        target=server.get_url(),
        output_dir=tmp_path,
        outputs=report_name,
    )

    # Override the default backend to use Responses API format
    # Single --backend including Responses API format (duplicates are illegal).
    client.start_benchmark(
        rate=4,
        max_requests=max_requests,
        data=(
            "kind=synthetic_text,prompt_tokens=64,output_tokens=32,"
            "turns=1,tool_call_turns=1"
        ),
        backend=(
            f"kind=openai_http,target={server.get_url()},request_format=/v1/responses"
        ),
        extra_env=_BENCHMARK_ENV,
    )

    client.wait_for_completion(timeout=60)

    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) > 0, "No successful requests found"

    # With max_requests=10 and 2 requests per conversation, we expect
    # exactly 10 successful requests (5 full conversations)
    assert len(successful_requests) == max_requests, (
        f"Expected {max_requests} successful requests, got {len(successful_requests)}"
    )

    # Verify tool_calls are present in at least some requests
    tool_call_requests = [r for r in successful_requests if r.get("tool_calls")]
    assert len(tool_call_requests) > 0, "No requests with tool_calls found"

    # Verify timing fields
    for request in successful_requests:
        assert request.get("request_latency", 0) > 0, "request_latency should be > 0"
        assert request.get("output_tokens", 0) > 0, "output_tokens should be > 0"
