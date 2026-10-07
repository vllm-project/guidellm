"""Regression tests for independent concurrent progress observers."""

import asyncio
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
    server_metrics: list[Any] | None = None


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_server_metrics_collectors_wrap_each_benchmark(monkeypatch):
    """Scrape around each benchmark's requests and attach its window summary.

    ## WRITTEN BY AI ##
    """
    events: list[str] = []
    collector = Mock()
    collector.start = AsyncMock(side_effect=lambda: events.append("start"))
    collector.stop = AsyncMock(side_effect=lambda: events.append("stop"))
    collector.summarize.side_effect = lambda start, end: ("summary", start, end)

    def strategies():
        yield Mock(), {}
        yield Mock(), {}

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
            server_metrics=[collector],
        )
    ]

    assert events == ["start", "requests", "stop"] * 2
    assert [result.server_metrics for result in results] == [
        [("summary", 10.0, 20.0)]
    ] * 2


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_server_metrics_collectors_stop_when_requests_fail(monkeypatch):
    """Stop scraping even when the benchmark's requests raise.

    ## WRITTEN BY AI ##
    """
    collector = Mock(start=AsyncMock(), stop=AsyncMock())

    def strategies():
        yield Mock(), {}

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
            server_metrics=[collector],
        ):
            pass

    collector.start.assert_awaited_once()
    collector.stop.assert_awaited_once()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_server_metrics_collectors_are_all_stopped_after_errors(monkeypatch):
    """Stop every started collector even when one fails to start or stop.

    ## WRITTEN BY AI ##
    """
    first = Mock(start=AsyncMock(), stop=AsyncMock(side_effect=RuntimeError("stop")))
    second = Mock(start=AsyncMock(), stop=AsyncMock())
    failing = Mock(start=AsyncMock(side_effect=RuntimeError("start")), stop=AsyncMock())

    def strategies():
        yield Mock(), {}

    profile = Mock(completed_strategies=[])
    profile.strategies_generator.side_effect = strategies
    monkeypatch.setattr(module.InfoMixin, "extract_from_obj", lambda _: {})
    monkeypatch.setattr(module, "BenchmarkConfig", Mock())
    monkeypatch.setattr(module, "Scheduler", Mock())

    with pytest.raises(RuntimeError, match="start"):
        async for _ in module.Benchmarker().run(
            accumulator_class=Mock(),
            benchmark_class=Mock(),
            requests=[],
            backend=Mock(),
            profile=profile,
            environment=Mock(),
            warmup=Mock(),
            cooldown=Mock(),
            server_metrics=[first, second, failing],
        ):
            pass

    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()
    failing.stop.assert_not_awaited()
