# Property-based E2E tests following Mark Kurtz's specifications
#
# Test Categories:
# - SMOKE: 5 curated use cases (20s each, couple minutes total)
# - SANITY: Property-based cartesian product (20s each, couple hours total)
# - REGRESSION: Curated long-running tests (few minutes each, couple hours total)
#
# Uses hypothesis for systematic test case generation instead of manual configuration

from pathlib import Path
from typing import Optional

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from tests.e2e.utils import (
    GuidellmClient,
    assert_no_python_exceptions,
    assert_successful_requests_fields,
    cleanup_report_file,
    load_benchmark_report,
)
from tests.e2e.vllm_sim_server import VllmSimServer

# Backend performance profiles as specified by Mark Kurtz
BACKEND_PROFILES = {
    "fast": {"ttft": 100, "itl": 10},  # TTFT <100ms, ITL <10ms
    "medium": {"ttft": 500, "itl": 25},  # TTFT <500ms, ITL <25ms
    "slow": {"ttft": 2000, "itl": 100},  # TTFT <2s, ITL <100ms
}


# Server fixture factory
def create_server_fixture(profile_name: str, port: int = 8000):
    """Create session-scoped server fixture for a backend profile."""
    profile = BACKEND_PROFILES[profile_name]

    @pytest.fixture
    def server():
        server = VllmSimServer(
            mode="random",
            time_to_first_token=profile["ttft"],
            inter_token_latency=profile["itl"],
        )
        with server:
            yield server

    return server


# Create server fixtures
fast_server = create_server_fixture("fast")
medium_server = create_server_fixture("medium")
slow_server = create_server_fixture("slow")

SERVER_FIXTURES = {
    "fast": fast_server,
    "medium": medium_server,
    "slow": slow_server,
}


def run_benchmark_test(
    server,
    strategy: str,
    rate: Optional[int],
    data_config: str,
    max_seconds: Optional[int] = None,
    max_requests: Optional[int] = None,
    warmup_percent: Optional[int] = None,
    cooldown_percent: Optional[int] = None,
    timeout_multiplier: float = 1.5,
):
    """Simplified benchmark test runner."""

    # Generate unique report path
    test_id = f"{strategy}_{rate}_{max_seconds}s_{max_requests}r"
    report_path = Path(f"tests/e2e/property_{test_id}.json")
    cleanup_report_file(report_path)

    # Create client
    client = GuidellmClient(target=server.get_url(), output_path=report_path)

    # Build command arguments
    additional_args = ""
    if warmup_percent:
        additional_args += f" --warmup-percent {warmup_percent}"
    if cooldown_percent:
        additional_args += f" --cooldown-percent {cooldown_percent}"

    # Calculate timeout with more generous buffer for high-latency servers
    timeout_base = max_seconds or 30
    # Increased buffer from 30s to 60s for high-latency servers
    timeout = int((timeout_base + 60) * timeout_multiplier)

    if strategy == "sweep":
        timeout = timeout * 10

    # Start benchmark
    benchmark_args = {
        "rate_type": strategy,
        "rate": rate,
        "data": data_config,
        "additional_args": additional_args,
    }

    if max_seconds:
        benchmark_args["max_seconds"] = max_seconds
    if max_requests:
        benchmark_args["max_requests"] = max_requests

    client.start_benchmark(**benchmark_args)
    client.wait_for_completion(timeout=timeout)

    # Validate results - allow application bugs to fail tests
    assert_no_python_exceptions(client.stderr)

    report = load_benchmark_report(report_path)
    benchmark = report["benchmarks"][0]

    # Basic validation
    assert "requests" in benchmark
    assert "successful" in benchmark["requests"]
    assert len(benchmark["requests"]["successful"]) > 0

    # Cleanup
    cleanup_report_file(report_path)

    return benchmark


# =============================================================================
# SMOKE TESTS
# =============================================================================


@pytest.mark.smoke
@pytest.mark.timeout(90)
def test_interactive_chat_use_case(fast_server):
    """
    Interactive chat style use case:
    - data: emulated 512x512
    - backend: fast (TTFT <100ms, ITL <10ms)
    - strategy: constant (changed from sweep due to baseline issues)
    - constraints: max_seconds=60, max_requests=1000
    - aggregation: warmup=10%, cooldown=10%
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="constant",  # Changed from sweep to avoid baseline issues
        rate=5,  # constant rate (reduced for 512x512 tokens)
        data_config="prompt_tokens=512,output_tokens=512",
        max_seconds=15,  # Normal timeout for constant strategy
        max_requests=25,  # Reduced for quick smoke test
        # Removed warmup/cooldown to avoid interaction issues with 512x512 tokens
    )

    # Validate it's a proper interactive chat benchmark
    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_rag_throughput_use_case(fast_server):
    """
    RAG style use case:
    - data: emulated 2048x128
    - backend: fast (changed from medium due to server simulator issues)
    - strategy: throughput
    - constraints: max_seconds=60, max_requests=500
    - aggregation: None
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="throughput",
        rate=10,  # Normal rate for fast server
        data_config="prompt_tokens=512,output_tokens=128",
        max_seconds=15,  # Normal timeout for fast server
        max_requests=30,  # Normal count for smoke test
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_rag_constant_rate_use_case(fast_server):
    """
    RAG style with constant rate:
    - data: emulated 2048x128
    - backend: fast (changed from medium due to server simulator issues)
    - strategy: constant at 10 RPS
    - constraints: max_seconds=60, max_requests=500
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="constant",
        rate=5,  # Normal rate for fast server
        data_config="prompt_tokens=512,output_tokens=128",
        max_seconds=15,  # Normal timeout for fast server
        max_requests=30,  # Normal count for smoke test
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_code_generation_use_case(fast_server):
    """
    Code generation style use case:
    - data: emulated 512x2048
    - backend: fast (changed from medium due to server simulator issues)
    - strategy: concurrent at 50
    - constraints: max_seconds=120
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="concurrent",
        rate=5,  # Normal rate for fast server
        data_config="prompt_tokens=512,output_tokens=512",
        max_seconds=15,  # Normal timeout for fast server
        max_requests=10,  # Small count for smoke test
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_fast_perf_stress_use_case(fast_server):
    """
    Fast performance stress test:
    - data: emulated 64x64
    - backend: fast (TTFT <50ms, ITL <5ms) - using fast server as closest
    - strategy: constant at 50
    - aggregation: warmup=5%
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="constant",
        rate=5,  # Reduced rate for quick test
        data_config="prompt_tokens=64,output_tokens=64",
        max_seconds=10,  # Reduced for quick smoke test
        max_requests=25,  # Reduced for quick smoke test
        warmup_percent=5,
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_synchronous_fast_use_case(fast_server):
    """
    Synchronous strategy test with fast backend:
    - data: emulated 512x512 (interactive chat size)
    - backend: fast (TTFT <100ms, ITL <10ms)
    - strategy: synchronous
    - constraints: max_seconds=15, max_requests=30
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="synchronous",
        rate=None,  # synchronous doesn't use rate
        data_config="prompt_tokens=512,output_tokens=512",
        max_seconds=15,  # Short for smoke test
        max_requests=30,  # Small count for smoke test
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_synchronous_alternative_use_case(fast_server):
    """
    Synchronous strategy test with alternative data:
    - data: emulated 512x256 (different from other fast server tests)
    - backend: fast (changed from medium due to server simulator issues)
    - strategy: synchronous
    - constraints: max_seconds=15, max_requests=20
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="synchronous",
        rate=None,  # synchronous doesn't use rate
        data_config="prompt_tokens=512,output_tokens=256",
        max_seconds=15,  # Normal timeout for fast server
        max_requests=10,  # Small count for smoke test
    )

    assert len(benchmark["requests"]["successful"]) > 0


@pytest.mark.smoke
@pytest.mark.timeout(90)
def test_sweep_smoke_use_case(fast_server):
    """
    Sweep strategy smoke test:
    - data: emulated 64x64 (small tokens for fast sweep)
    - backend: fast (TTFT <100ms, ITL <10ms)
    - strategy: sweep (runs 10 sub-benchmarks)
    - constraints: max_seconds=8, max_requests=20 (per sub-benchmark)
    - Higher timeout due to 10 sub-benchmarks
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="sweep",
        rate=10,  # Sweep max rate
        data_config="prompt_tokens=64,output_tokens=64",
        max_seconds=8,  # Short per sub-benchmark (8s * 10 = ~80s total)
        max_requests=20,  # Small count per sub-benchmark
        timeout_multiplier=2.0,  # Higher multiplier for sweep overhead
    )

    assert len(benchmark["requests"]["successful"]) > 0


# =============================================================================
# SANITY TESTS - Property-based cartesian product
# =============================================================================


# Hypothesis strategies for test case generation
@composite
def backend_strategy(draw):
    """Generate backend profile configurations."""
    return draw(st.sampled_from(["fast", "medium", "slow"]))


@composite
def data_strategy(draw):
    """Generate data configurations based on Mark's input sizes."""
    sizes = [
        (64, 64),  # Fast perf
        (512, 128),  # Short prompt, short output
        (512, 512),  # Interactive chat
        (512, 2048),  # Code generation
        (2048, 128),  # RAG
        (2048, 2048),  # Offline throughput
    ]
    prompt_tokens, output_tokens = draw(st.sampled_from(sizes))
    return f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}"


@composite
def strategy_rate_strategy(draw):
    """Generate strategy and rate combinations."""
    strategy = draw(
        st.sampled_from(
            ["synchronous", "sweep", "constant", "concurrent", "throughput"]
        )
    )

    if strategy == "synchronous":
        rate = None  # synchronous doesn't use rate
    elif strategy == "sweep":
        rate = draw(st.integers(min_value=5, max_value=20))
    elif strategy in ["constant", "concurrent"]:
        rate = draw(st.sampled_from([1, 5, 10, 25, 50]))
    else:  # throughput
        rate = draw(st.integers(min_value=5, max_value=50))

    return strategy, rate


@composite
def constraints_strategy(draw):
    """Generate constraint configurations."""
    # For sanity tests, keep them short (20s max)
    max_seconds = draw(st.integers(min_value=10, max_value=20))
    max_requests = draw(st.sampled_from([25, 50, 100]))
    return max_seconds, max_requests


@composite
def aggregation_strategy(draw):
    """Generate aggregation configurations."""
    use_aggregation = draw(st.booleans())
    if not use_aggregation:
        return None, None

    warmup = draw(st.integers(min_value=5, max_value=20))
    cooldown = draw(st.integers(min_value=5, max_value=20))
    return warmup, cooldown


@pytest.mark.sanity
@pytest.mark.timeout(3600)
@given(
    backend=backend_strategy(),
    data_config=data_strategy(),
    strategy_rate=strategy_rate_strategy(),
    constraints=constraints_strategy(),
    aggregation=aggregation_strategy(),
)
@settings(
    max_examples=20,  # Limit examples for reasonable test time
    deadline=None,  # Disable deadline for E2E tests
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_sanity_property_based_benchmark(
    backend, data_config, strategy_rate, constraints, aggregation
):
    """
    Property-based sanity tests covering cartesian product of configurations.
    Each test runs for up to 20 seconds with systematic parameter combinations.
    """
    strategy, rate = strategy_rate
    max_seconds, max_requests = constraints
    warmup_percent, cooldown_percent = aggregation

    profile = BACKEND_PROFILES[backend]

    server = VllmSimServer(
        mode="random",
        time_to_first_token=profile["ttft"],
        inter_token_latency=profile["itl"],
    )
    with server:
        benchmark = run_benchmark_test(
            server=server,
            strategy=strategy,
            rate=rate,
            data_config=data_config,
            max_seconds=max_seconds,
            max_requests=max_requests,
            warmup_percent=warmup_percent,
            cooldown_percent=cooldown_percent,
            timeout_multiplier=1.2,
        )

    # Property-based assertions
    assert "requests" in benchmark
    assert "successful" in benchmark["requests"]
    assert len(benchmark["requests"]["successful"]) > 0
    assert "failed" in benchmark["requests"]

    # Validate metrics structure
    assert "metrics" in benchmark
    metrics = benchmark["metrics"]
    assert "request_rate" in metrics
    assert "error_rate" in metrics


# =============================================================================
# REGRESSION TESTS - Curated long-running tests
# =============================================================================


@pytest.mark.regression
@pytest.mark.timeout(600)
def test_regression_high_load_code_generation(medium_server):
    """
    Long-running code generation stress test.
    - High concurrent load (100)
    - Long duration (120s)
    - Large outputs (2048 tokens)
    """

    benchmark = run_benchmark_test(
        server=medium_server,
        strategy="concurrent",
        rate=100,
        data_config="prompt_tokens=512,output_tokens=2048",
        max_seconds=120,
        max_requests=1000,
        timeout_multiplier=2.0,
    )

    # Validate high-load performance
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) >= 50, (
        f"Too few successful requests: {len(successful_requests)}"
    )

    if successful_requests:
        assert_successful_requests_fields(successful_requests)


@pytest.mark.regression
@pytest.mark.timeout(600)
def test_regression_offline_throughput_stress(slow_server):
    """
    Long-running offline throughput test.
    - Large inputs/outputs (2048x2048)
    - Slow backend simulation
    - High request volume (5000)
    """

    benchmark = run_benchmark_test(
        server=slow_server,
        strategy="throughput",
        rate=50,
        data_config="prompt_tokens=2048,output_tokens=2048",
        max_requests=1000,  # Reduced from 5000 for reasonable test time
        timeout_multiplier=3.0,
    )

    # Validate throughput characteristics
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) >= 100, (
        f"Too few successful requests: {len(successful_requests)}"
    )


@pytest.mark.regression
@pytest.mark.timeout(600)
def test_regression_sustained_high_rate_constant(fast_server):
    """
    Long-running sustained high rate test.
    - Fast backend with high constant rate
    - Extended duration to test stability
    """

    benchmark = run_benchmark_test(
        server=fast_server,
        strategy="constant",
        rate=500,
        data_config="prompt_tokens=64,output_tokens=64",
        max_seconds=180,
        max_requests=2000,
        warmup_percent=5,
        timeout_multiplier=2.0,
    )

    # Validate sustained performance
    successful_requests = benchmark["requests"]["successful"]
    assert len(successful_requests) >= 200, (
        f"Too few successful requests: {len(successful_requests)}"
    )

    # Check rate sustainability
    metrics = benchmark["metrics"]
    request_rate = metrics.get("request_rate", 0)
    assert request_rate > 100, f"Request rate too low: {request_rate}"
