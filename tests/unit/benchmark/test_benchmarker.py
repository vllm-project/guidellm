"""Regression tests for independent concurrent progress observers."""

import asyncio
import threading
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from guidellm.benchmark import CompositeBenchmarkerProgress
from guidellm.benchmark import benchmarker as module
from guidellm.benchmark.progress import BenchmarkerProgress


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "update", "scheduler", "initialize"])
async def test_progress_observers_run_concurrently_and_finalize(monkeypatch, failure):
    """Notify observers concurrently and finalize only when execution completes.

    ## WRITTEN BY AI ##
    """
    hooks = [
        "on_initialize",
        "on_benchmark_start",
        "on_benchmark_update",
        "on_benchmark_compile",
        "on_benchmark_complete",
        "on_finalize",
    ]
    arrivals = dict.fromkeys(hooks, 0)
    gates = {hook: asyncio.Event() for hook in hooks}
    finished = []

    observers = [Mock(spec=BenchmarkerProgress) for _ in range(2)]
    for index, observer in enumerate(observers):
        for hook in hooks:
            setattr(
                observer,
                hook,
                AsyncMock(
                    side_effect=partial(
                        _callback, arrivals, gates, finished, failure, index, hook
                    )
                ),
            )

    strategy = Mock()
    strategy.prefetch_count = Mock(return_value=0)

    def strategies():
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    monkeypatch.setattr(module.InfoMixin, "extract_from_obj", lambda _: {})
    monkeypatch.setattr(module, "BenchmarkConfig", Mock())
    accumulator_class = Mock()
    benchmark_class = Mock()
    benchmark_class.compile.side_effect = partial(
        _compile_benchmark, observers, finished, benchmark_class.compile.return_value
    )

    async def schedule(**kwargs):
        yield None, None, None, Mock()
        if failure == "scheduler":
            raise RuntimeError("scheduler failure")

    monkeypatch.setattr(module, "Scheduler", Mock(return_value=Mock(run=schedule)))

    async def consume():
        return [
            result
            async for result in module.Benchmarker().run(
                accumulator_class=accumulator_class,
                benchmark_class=benchmark_class,
                requests=[],
                backend=Mock(),
                profile=profile,
                environment=Mock(),
                warmup=Mock(),
                cooldown=Mock(),
                progress=CompositeBenchmarkerProgress(observers),
            )
        ]

    if failure in ("initialize", "scheduler"):
        with pytest.raises(RuntimeError, match="failure"):
            await asyncio.wait_for(consume(), timeout=3)
    else:
        assert await asyncio.wait_for(consume(), timeout=3) == [
            benchmark_class.compile.return_value
        ]
        assert [o.on_benchmark_complete.await_count for o in observers] == [1, 1]
    for observer in observers:
        observer.on_initialize.assert_awaited_once()
        assert observer.on_benchmark_compile.await_count == int(
            failure not in ("initialize", "scheduler")
        )
        assert observer.on_finalize.await_count == int(
            failure not in ("initialize", "scheduler")
        )
    assert [(index, "on_finalize") in finished for index in range(2)] == [
        failure not in ("initialize", "scheduler")
    ] * 2


def _compile_benchmark(observers, finished, result, **kwargs):
    for observer in observers:
        observer.on_benchmark_compile.assert_awaited_once()
        observer.on_benchmark_complete.assert_not_awaited()
    assert set(finished[-2:]) == {(index, "on_benchmark_compile") for index in range(2)}
    return result


async def _callback(arrivals, gates, finished, failure, index, hook, *args):
    arrivals[hook] += 1
    if arrivals[hook] == 2:
        gates[hook].set()
    await gates[hook].wait()
    # A sequential dispatcher would deadlock waiting for the second observer.
    await asyncio.sleep(0)
    finished.append((index, hook))
    failing_hook = {
        "update": "on_benchmark_update",
        "initialize": "on_initialize",
    }.get(failure)
    if index == 0 and hook == failing_hook:
        raise RuntimeError("observer failure")


class _CompiledBenchmark(BaseModel):
    start_time: float
    end_time: float
    collectors: list[Any] | None = None


def _collector_process(events: list[str]) -> Mock:
    process = Mock()
    process.benchmark_started = AsyncMock(side_effect=lambda: events.append("start"))
    process.benchmark_stopped = AsyncMock(side_effect=lambda: events.append("stop"))
    process.results = AsyncMock(side_effect=lambda start, end: [("result", start, end)])
    return process


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_collectors_wrap_each_benchmark(monkeypatch):
    """Start and stop collectors around each benchmark and attach their results.

    ## WRITTEN BY AI ##
    """
    events: list[str] = []
    collectors = _collector_process(events)
    strategy = Mock()
    strategy.prefetch_count.return_value = 0

    def strategies():
        yield strategy, {}
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    monkeypatch.setattr(module.InfoMixin, "extract_from_obj", lambda _: {})
    monkeypatch.setattr(module, "BenchmarkConfig", Mock())
    benchmark_class = Mock()
    benchmark_class.compile.return_value = _CompiledBenchmark(
        start_time=10.0, end_time=20.0
    )

    async def schedule(**kwargs):
        events.append("requests")
        yield None, None, None, Mock()

    monkeypatch.setattr(module, "Scheduler", Mock(return_value=Mock(run=schedule)))

    results = [
        result
        async for result in module.Benchmarker().run(
            accumulator_class=Mock(),
            benchmark_class=benchmark_class,
            requests=[],
            backend=Mock(),
            profile=profile,
            environment=Mock(),
            warmup=Mock(),
            cooldown=Mock(),
            progress=Mock(spec=BenchmarkerProgress),
            collectors=collectors,
        )
    ]

    assert events == ["start", "requests", "stop"] * 2
    assert [result.collectors for result in results] == [[("result", 10.0, 20.0)]] * 2


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_collectors_stop_when_requests_fail(monkeypatch):
    """Stop the collectors even when the benchmark's requests raise.

    ## WRITTEN BY AI ##
    """
    events: list[str] = []
    collectors = _collector_process(events)
    strategy = Mock()
    strategy.prefetch_count.return_value = 0

    def strategies():
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    monkeypatch.setattr(module.InfoMixin, "extract_from_obj", lambda _: {})
    monkeypatch.setattr(module, "BenchmarkConfig", Mock())

    async def schedule(**kwargs):
        yield None, None, None, Mock()
        raise RuntimeError("scheduler failure")

    monkeypatch.setattr(module, "Scheduler", Mock(return_value=Mock(run=schedule)))

    with pytest.raises(RuntimeError, match="scheduler failure"):
        async for _ in module.Benchmarker().run(
            accumulator_class=Mock(),
            benchmark_class=Mock(),
            requests=[],
            backend=Mock(),
            profile=profile,
            environment=Mock(),
            warmup=Mock(),
            cooldown=Mock(),
            progress=Mock(spec=BenchmarkerProgress),
            collectors=collectors,
        ):
            pass

    assert events == ["start", "stop"]
    collectors.results.assert_not_awaited()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_prefetching_stops_when_benchmark_task_is_cancelled(monkeypatch):
    """Cancelling a benchmark task signals prefetching to stop loading graphs.

    ## WRITTEN BY AI ##
    """
    real_prefetch_requests = module.prefetch_requests
    stop_events = []
    first_callback = threading.Event()
    prefetch_done = threading.Event()

    class _Graph:
        nodes = ["request"]

    def requests_source():
        yield _Graph()
        if stop_events[0].wait(timeout=2):
            return
        yield _Graph()

    def prefetch_wrapper(requests, count, on_fetched=None, stop_event=None):
        stop_events.append(stop_event)
        try:
            return real_prefetch_requests(
                requests,
                count,
                on_fetched=on_fetched,
                stop_event=stop_event,
            )
        finally:
            prefetch_done.set()

    monkeypatch.setattr(module, "prefetch_requests", prefetch_wrapper)

    strategy = Mock()
    strategy.prefetch_count.return_value = "all"

    def strategies():
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    progress = Mock(spec=BenchmarkerProgress)
    progress.on_initialize = AsyncMock()
    progress.on_benchmark_start = AsyncMock()
    progress.on_requests_prefetched.side_effect = lambda *_: first_callback.set()

    async def consume():
        async for _ in module.Benchmarker().run(
            accumulator_class=Mock(),
            benchmark_class=Mock(),
            requests=requests_source(),
            backend=Mock(),
            profile=profile,
            environment=Mock(),
            warmup=Mock(),
            cooldown=Mock(),
            progress=progress,
        ):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(first_callback.wait, 2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert prefetch_done.wait(timeout=2)
    assert progress.on_requests_prefetched.call_count == 1


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_process_response_runs_before_accumulation(monkeypatch):
    """Apply the response hook before the accumulator records the response.

    ## WRITTEN BY AI ##
    """
    events: list[str] = []
    response = Mock()
    strategy = Mock()
    strategy.prefetch_count.return_value = 0

    def strategies():
        yield strategy, {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    monkeypatch.setattr(module.InfoMixin, "extract_from_obj", lambda _: {})
    monkeypatch.setattr(module, "BenchmarkConfig", Mock())

    async def schedule(**kwargs):
        yield response, None, None, Mock()

    monkeypatch.setattr(module, "Scheduler", Mock(return_value=Mock(run=schedule)))
    accumulator = Mock()
    accumulator.update_estimate.side_effect = lambda *args: events.append("update")
    process_response = AsyncMock(side_effect=lambda _: events.append("process"))

    async for _ in module.Benchmarker().run(
        accumulator_class=Mock(return_value=accumulator),
        benchmark_class=Mock(),
        requests=[],
        backend=Mock(),
        profile=profile,
        environment=Mock(),
        warmup=Mock(),
        cooldown=Mock(),
        progress=Mock(spec=BenchmarkerProgress),
        process_response=process_response,
    ):
        pass

    process_response.assert_awaited_once_with(response)
    assert events == ["process", "update"]
