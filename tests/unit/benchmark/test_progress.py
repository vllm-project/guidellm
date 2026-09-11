"""Tests for default benchmark logging and Rich progress."""

import asyncio
import sys
from io import StringIO

import pytest
from rich.console import Console

from guidellm.benchmark import progress as progress_module
from guidellm.benchmark.profiles import ProfileFactory
from guidellm.benchmark.progress import (
    GenerativeConsoleBenchmarkerProgress,
    GenerativeLoggingBenchmarkerProgress,
)
from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
)
from guidellm.scheduler import SchedulerState, SynchronousStrategy
from guidellm.schemas.benchmark.profiles import SynchronousProfileArgs


@pytest.fixture
def accumulator():
    """Create a real accumulator without a model server."""
    result = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="progress-test",
            run_index=0,
            strategy=SynchronousStrategy(),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
        )
    )
    result.timings.request_start = 1.0
    result.timings.measure_start = 1.0
    result.timings.measure_end = 3.0
    result.timings.request_end = 3.0
    result.timings.current_update = 2.0
    return result


@pytest.mark.regression
@pytest.mark.asyncio
async def test_rich_lifecycle_still_renders(accumulator):
    """The existing Rich display still initializes, updates, and cleans up.

    ## WRITTEN BY AI ##
    """
    progress = GenerativeConsoleBenchmarkerProgress()
    profile = ProfileFactory.create(SynchronousProfileArgs(), random_seed=0)
    state = SchedulerState()
    with progress.console.capture() as captured:
        await progress.on_initialize(profile)
        try:
            await progress.on_benchmark_start(SynchronousStrategy())
            await progress.on_benchmark_update(accumulator, state)
            await progress.on_benchmark_complete(
                GenerativeBenchmark.compile(accumulator, state)
            )
        finally:
            await progress.on_finalize()
    assert "Benchmarks" in captured.get()
    assert progress.tasks_progress is None


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("with_rich", [True, False])
async def test_logs_are_periodic_with_or_without_rich(
    monkeypatch, accumulator, with_rich
):
    """Keep Rich rendering and queued log records independent.

    ## WRITTEN BY AI ##
    """
    now = [100.0]
    monkeypatch.setattr(progress_module, "monotonic", lambda: now[0])
    records = []
    sink = progress_module.logger.add(
        lambda message: records.append(message.record),
        enqueue=True,
        filter=lambda record: "progress_status" in record["extra"],
    )
    display = GenerativeConsoleBenchmarkerProgress() if with_rich else None
    output = StringIO()
    if display:
        display.console = Console(file=output, force_terminal=True, width=120)
    progress = GenerativeLoggingBenchmarkerProgress(interval=10)
    trackers = [progress, *([display] if display else [])]
    profile = ProfileFactory.create(SynchronousProfileArgs(), random_seed=0)
    state = SchedulerState(successful_requests=12, errored_requests=2)
    try:
        await asyncio.gather(*(p.on_initialize(profile) for p in trackers))
        await asyncio.gather(
            *(p.on_benchmark_start(SynchronousStrategy()) for p in trackers)
        )
        for timestamp in (100.0, 101.0, 109.9, 110.0):
            now[0] = timestamp
            await asyncio.gather(
                *(p.on_benchmark_update(accumulator, state) for p in trackers)
            )
        await asyncio.gather(
            *(
                p.on_benchmark_complete(GenerativeBenchmark.compile(accumulator, state))
                for p in trackers
            )
        )
    finally:
        await asyncio.gather(*(p.on_finalize() for p in trackers))
        progress_module.logger.remove(sink)
    assert [r["extra"]["progress_status"] for r in records] == [
        "started",
        "active",
        "completed",
    ]
    assert "successful=12 errored=2" in records[1]["message"]
    assert "elapsed=10.0s" in records[1]["message"]
    if display:
        assert "Benchmarks" in output.getvalue()
        assert display.tasks_progress is None


@pytest.mark.sanity
@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_log_intervals(interval):
    """Reject non-finite or non-positive logging intervals.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError):
        GenerativeLoggingBenchmarkerProgress(interval=interval)


@pytest.mark.regression
@pytest.mark.asyncio
async def test_queued_logs_share_rich_terminal(monkeypatch, accumulator):
    """Queued stderr records coexist with Rich and restore its stream on exit.

    ## WRITTEN BY AI ##
    """
    output = StringIO()
    monkeypatch.setattr(sys, "stderr", output)
    monkeypatch.setattr(progress_module, "stderr_eq_stdout", lambda: True)
    display = GenerativeConsoleBenchmarkerProgress()
    display.console = Console(file=output, force_terminal=True, width=160)
    sink = progress_module.logger.add(
        lambda message: sys.stderr.write(str(message)),
        enqueue=True,
        format="{message}",
        filter=lambda record: "progress_status" in record["extra"],
    )
    progress = GenerativeLoggingBenchmarkerProgress()
    trackers = [progress, *([display] if display else [])]
    profile = ProfileFactory.create(SynchronousProfileArgs(), random_seed=0)
    try:
        await asyncio.gather(*(p.on_initialize(profile) for p in trackers))
        await asyncio.gather(
            *(p.on_benchmark_start(SynchronousStrategy()) for p in trackers)
        )
        await asyncio.gather(
            *(
                p.on_benchmark_complete(
                    GenerativeBenchmark.compile(accumulator, SchedulerState())
                )
                for p in trackers
            )
        )
    finally:
        await asyncio.gather(*(p.on_finalize() for p in trackers))
        progress_module.logger.remove(sink)
    assert "Benchmarks" in output.getvalue()
    assert ": started |" in output.getvalue()
    assert ": completed |" in output.getvalue()
    assert sys.stderr is output
