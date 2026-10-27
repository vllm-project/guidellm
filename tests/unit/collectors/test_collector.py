"""Unit tests for the collector base classes."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from guidellm.collectors import Collector, OneshotCollector, PeriodicCollector
from guidellm.schemas.benchmark import CollectorArgs


class _Oneshot(OneshotCollector):
    def __init__(self) -> None:
        super().__init__("oneshot")
        self.calls = 0

    @classmethod
    def from_args(cls, args: CollectorArgs) -> _Oneshot:
        return cls()

    async def collect(self) -> dict[str, Any]:
        self.calls += 1
        return {"calls": self.calls}


class _UnregisteredArgs(CollectorArgs):
    kind: str = "missing"


class _Periodic(PeriodicCollector):
    def __init__(self, interval: float) -> None:
        super().__init__("periodic", interval)
        self.samples: list[str] = []

    @classmethod
    def from_args(cls, args: CollectorArgs) -> _Periodic:
        return cls(1.0)

    async def reset(self) -> None:
        self.samples = []

    async def sample(self) -> None:
        self.samples.append("sample")

    def result(self, start_time: float, end_time: float) -> dict[str, Any]:
        return {"samples": len(self.samples)}


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_oneshot_collects_once_per_benchmark():
    """Collect when each benchmark stops, and only for that benchmark.

    ## WRITTEN BY AI ##
    """
    collector = _Oneshot()

    await collector.start()
    assert collector.result(0.0, 1.0) == {}
    await collector.stop()
    assert collector.result(0.0, 1.0) == {"calls": 1}

    await collector.start()
    assert collector.result(0.0, 1.0) == {}
    await collector.stop()
    assert collector.result(0.0, 1.0) == {"calls": 2}


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_periodic_samples_at_both_ends_and_between():
    """Sample at start, every interval, and once more at stop.

    ## WRITTEN BY AI ##
    """
    collector = _Periodic(interval=0.01)

    await collector.start()
    await asyncio.sleep(0.05)
    await collector.stop()
    sampled = len(collector.samples)
    await asyncio.sleep(0.03)

    assert sampled >= 3
    assert len(collector.samples) == sampled


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_periodic_start_discards_previous_benchmark():
    """Reset the samples when the next benchmark starts.

    ## WRITTEN BY AI ##
    """
    collector = _Periodic(interval=3600.0)

    await collector.start()
    await collector.stop()
    await collector.start()
    await collector.stop()

    assert collector.result(0.0, 1.0) == {"samples": 2}


@pytest.mark.sanity
def test_resolve_rejects_unregistered_kind():
    """Name the registered kinds when the requested kind is unknown.

    ## WRITTEN BY AI ##
    """
    with pytest.raises(ValueError, match="'missing' is not registered.*prometheus"):
        Collector.resolve(_UnregisteredArgs())
