"""
Base classes for collectors that gather data while each benchmark runs.

A collector is created from its :class:`CollectorArgs`, started when a benchmark
begins and stopped when it ends, then asked for one result covering that
benchmark's measurement window. Collectors run in their own process, managed by
:class:`~guidellm.collectors.process.CollectorProcess`, so they never take time
from the benchmark's main process.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Any

from guidellm.schemas.benchmark import CollectorArgs
from guidellm.utils.registry import RegistryMixin

__all__ = ["Collector", "OneshotCollector", "PeriodicCollector"]


class Collector(RegistryMixin[type["Collector"]], ABC):
    """
    Gather data from outside GuideLLM for each benchmark of a run.

    A collector is reused across the benchmarks of a run. For each benchmark,
    :meth:`start` is called before its requests begin, :meth:`stop` after they
    end, and :meth:`result` once the measurement window is known. Every collector
    returns a result for every benchmark, even an empty one, so results line up
    with the order the collectors were configured in.
    """

    def __init__(self, kind: str) -> None:
        """
        :param kind: The kind this collector was configured with
        """
        self.kind = kind

    @classmethod
    @abstractmethod
    def from_args(cls, args: CollectorArgs) -> Collector:
        """
        Create a collector from its arguments.

        :param args: Arguments for this collector's kind
        :return: The configured collector
        """
        ...

    @classmethod
    def resolve(cls, args: CollectorArgs) -> Collector:
        """
        Resolve collector arguments into a collector.

        :param args: Collector arguments with a registered kind
        :return: The configured collector
        :raises ValueError: If the kind is not registered
        """
        collector_class = cls.get_registered_object(args.kind)
        if collector_class is None:
            available = list(cls.registry.keys()) if cls.registry else []
            raise ValueError(
                f"Collector kind '{args.kind}' is not registered. "
                f"Available kinds: {available}"
            )

        return collector_class.from_args(args)

    @abstractmethod
    async def start(self) -> None:
        """Begin collecting for a new benchmark."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Finish collecting once the benchmark's requests have ended."""
        ...

    @abstractmethod
    def result(self, start_time: float, end_time: float) -> dict[str, Any]:
        """
        Return the result for the latest benchmark.

        :param start_time: Start of the measurement window in seconds since epoch
        :param end_time: End of the measurement window in seconds since epoch
        :return: JSON-serializable result, which may be empty
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Release any resources held across benchmarks."""


class OneshotCollector(Collector, ABC):
    """
    Collect once per benchmark, after its requests have ended.

    Suited to data that does not change during a benchmark, such as a server's
    configuration. It is still collected for every benchmark, since a run's
    benchmarks may be configured differently.
    """

    def __init__(self, kind: str) -> None:
        """
        :param kind: The kind this collector was configured with
        """
        super().__init__(kind)
        self._result: dict[str, Any] = {}

    @abstractmethod
    async def collect(self) -> dict[str, Any]:
        """
        Collect this benchmark's data.

        :return: JSON-serializable result, which may be empty
        """
        ...

    async def start(self) -> None:
        """Discard the previous benchmark's result."""
        self._result = {}

    async def stop(self) -> None:
        """Collect the result for the benchmark that just ended."""
        self._result = await self.collect()

    def result(self, start_time: float, end_time: float) -> dict[str, Any]:  # noqa: ARG002
        """
        :param start_time: Start of the measurement window in seconds since epoch
        :param end_time: End of the measurement window in seconds since epoch
        :return: The result collected when the benchmark ended
        """
        return self._result


class PeriodicCollector(Collector, ABC):
    """
    Sample at a fixed interval while a benchmark runs.

    Samples once when the benchmark starts, every ``interval`` seconds while it
    runs, and once more when it stops, so the measurement window is covered at
    both ends. Subclasses summarize the samples in :meth:`result`.
    """

    def __init__(self, kind: str, interval: float) -> None:
        """
        :param kind: The kind this collector was configured with
        :param interval: Seconds between samples while a benchmark runs
        """
        super().__init__(kind)
        self.interval = interval
        self._task: asyncio.Task[None] | None = None

    @abstractmethod
    async def reset(self) -> None:
        """Discard the previous benchmark's samples before a new one starts."""
        ...

    @abstractmethod
    async def sample(self) -> None:
        """Take one sample. Failures should be recorded, not raised."""
        ...

    async def start(self) -> None:
        """Reset, sample once, then sample every interval."""
        await self.reset()
        await self.sample()
        self._task = asyncio.create_task(self._sample_periodically())

    async def stop(self) -> None:
        """Stop the periodic samples and take a final sample."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.sample()

    async def _sample_periodically(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            await self.sample()
