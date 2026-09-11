"""Regression tests for independent concurrent progress observers."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, Mock

import pytest

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
                progress=observers,
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
        assert observer.on_finalize.await_count == int(
            failure not in ("initialize", "scheduler")
        )
    assert [(index, "on_finalize") in finished for index in range(2)] == [
        failure not in ("initialize", "scheduler")
    ] * 2


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
