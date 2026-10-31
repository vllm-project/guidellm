from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import pytest

from guidellm.scheduler import loop_lag
from guidellm.scheduler.loop_lag import LoopLagHistograms
from tests.unit.testing_utils import async_timeout


def _block_event_loop(seconds: float) -> None:
    """Hold the event loop the way a CPU-starved worker does."""
    time.sleep(seconds)


def _histograms(worker_count: int = 1, interval: float = 0.05) -> LoopLagHistograms:
    return LoopLagHistograms(mp.get_context("spawn"), worker_count, interval)


@pytest.mark.sanity
def test_sample_value_is_within_bucket_error():
    """Report each lag from 1 us to 100 s within 4.5% through its bucket midpoint.

    ## WRITTEN BY AI ##
    """
    for lag in np.geomspace(1e-6, 100.0, 997):
        value = loop_lag._bucket_value(loop_lag._bucket(float(lag)))

        assert abs(value - lag) / lag <= 2 ** (1 / 16) - 1 + 1e-9


@pytest.mark.sanity
def test_samples_merge_workers_and_count_early_timers_as_zero():
    """Merge counts across workers and count lag under a microsecond as zero.

    ## WRITTEN BY AI ##
    """
    histograms = _histograms(worker_count=3)
    histograms.record(0, 0.001)
    histograms.record(1, 0.001)
    histograms.record(2, -0.0001)
    histograms.record(2, 0.0)
    histograms.record(2, 5e-7)
    histograms.record(2, 0.5)

    samples = histograms.samples()

    assert [count for _, count in samples] == [3.0, 2.0, 1.0]
    assert samples[0][0] == 0.0
    assert samples[1][0] == pytest.approx(0.001, rel=0.045)
    assert samples[2][0] == pytest.approx(0.5, rel=0.045)


@pytest.mark.sanity
def test_each_worker_records_in_its_own_row():
    """Keep each worker's counts in its own row of the shared array.

    ## WRITTEN BY AI ##
    """
    histograms = _histograms(worker_count=2)
    histograms.record(1, 0.001)

    bucket = loop_lag._bucket(0.001)
    assert histograms._counts[bucket] == 0
    assert histograms._counts[loop_lag._BUCKETS + bucket] == 1


@pytest.mark.sanity
def test_lag_beyond_last_bucket_is_counted():
    """Count a lag past the last bucket in that bucket.

    ## WRITTEN BY AI ##
    """
    histograms = _histograms()
    histograms.record(0, 1000.0)

    [(value, count)] = histograms.samples()

    assert count == 1
    assert 100.0 < value < 1000.0


@pytest.mark.regression
@pytest.mark.asyncio
@async_timeout(5)
@pytest.mark.parametrize(("block", "low", "high"), [(0.0, 0.0, 0.04), (0.2, 0.1, 1.0)])
async def test_monitor_records_how_long_the_loop_was_held(
    block: float, low: float, high: float
):
    """Record lag when the event loop is held past a timer, and none when it is free.

    ## WRITTEN BY AI ##
    """
    histograms = _histograms(interval=0.01)
    monitor = asyncio.create_task(histograms.monitor(0))
    await asyncio.sleep(0.06)
    _block_event_loop(block)
    await asyncio.sleep(0.06)
    monitor.cancel()

    longest = max(value for value, _ in histograms.samples())

    assert low <= longest < high


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_monitor_counts_lag_within_timer_resolution_as_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """Count lag up to the 1 ms timer resolution as zero and subtract it beyond.

    ## WRITTEN BY AI ##
    """
    clock = iter([0.0, 0.2508, 0.5, 0.7530, 0.9])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        loop_lag, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    monkeypatch.setattr(loop_lag, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(
        loop_lag,
        "random",
        SimpleNamespace(
            Random=lambda: SimpleNamespace(expovariate=lambda rate: 1 / rate)
        ),
    )
    histograms = _histograms(interval=0.25)

    with pytest.raises(asyncio.CancelledError):
        await histograms.monitor(0)

    [(zero, zero_count), (lag, lag_count)] = histograms.samples()
    assert sleeps == [0.25, 0.25, 0.25]
    assert (zero, zero_count, lag_count) == (0.0, 1.0, 1.0)
    assert lag == pytest.approx(0.002, rel=0.045)


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_monitor_spaces_samples_exponentially(monkeypatch: pytest.MonkeyPatch):
    """Draw the time between samples from an exponential with mean ``interval``.

    ## WRITTEN BY AI ##
    """
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4000:
            raise asyncio.CancelledError

    monkeypatch.setattr(loop_lag, "asyncio", SimpleNamespace(sleep=fake_sleep))

    with pytest.raises(asyncio.CancelledError):
        await _histograms(interval=0.1).monitor(0)

    assert np.mean(sleeps) == pytest.approx(0.1, rel=0.1)
    assert np.std(sleeps) / np.mean(sleeps) == pytest.approx(1.0, rel=0.1)
