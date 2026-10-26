"""
Run a benchmark's collectors in a separate process.

The main process already drives the scheduler and compiles results, so all
collectors share one child process with its own event loop. The main process
only tells it when each benchmark starts and stops, and asks for the results
once the benchmark's measurement window is known.
"""

from __future__ import annotations

import asyncio
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from guidellm.collectors.collector import Collector
from guidellm.logger import logger
from guidellm.schemas.benchmark import CollectorArgs
from guidellm.settings import settings

__all__ = ["CollectorProcess"]

_SHUTDOWN_TIMEOUT = 10.0


class CollectorProcess:
    """
    Own the child process that runs a benchmark's collectors.

    Example:
        ::

            collectors = CollectorProcess(args)
            await collectors.start()
            try:
                await collectors.benchmark_started()
                ...  # run the benchmark
                await collectors.benchmark_stopped()
                results = await collectors.results(start_time, end_time)
            finally:
                await collectors.shutdown()
    """

    def __init__(self, args: list[CollectorArgs]):
        """
        :param args: Arguments for each collector, in the order results are returned
        """
        self.args = args
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None

    async def start(self) -> None:
        """
        Start the child process and create the collectors in it.

        :raises RuntimeError: If a collector cannot be created
        """
        context = get_context(settings.mp_context_type)
        parent_connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_run_collectors,
            args=(child_connection, [arg.model_dump() for arg in self.args]),
            name="guidellm-collectors",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        self._connection = parent_connection
        await self._request("ready")

    async def benchmark_started(self) -> None:
        """Start every collector for a new benchmark."""
        await self._request("start")

    async def benchmark_stopped(self) -> None:
        """Stop every collector once the benchmark's requests have ended."""
        await self._request("stop")

    async def results(self, start_time: float, end_time: float) -> list[dict[str, Any]]:
        """
        Return each collector's result for the latest benchmark.

        :param start_time: Start of the measurement window in seconds since epoch
        :param end_time: End of the measurement window in seconds since epoch
        :return: One result per collector, in configuration order
        """
        return await self._request("result", start_time, end_time)

    async def shutdown(self) -> None:
        """Close the collectors and wait for the child process to exit."""
        if self._process is None or self._connection is None:
            return
        try:
            await self._request("close")
        except (EOFError, OSError, RuntimeError) as err:
            logger.warning("Collector process did not close cleanly: {}", err)
        self._connection.close()
        await asyncio.to_thread(self._process.join, _SHUTDOWN_TIMEOUT)
        if self._process.is_alive():
            self._process.terminate()
        self._process = None
        self._connection = None

    async def _request(self, command: str, *params: Any) -> Any:
        if self._connection is None:
            raise RuntimeError("Collector process has not been started")
        if command != "ready":
            self._connection.send((command, *params))
        status, payload = await asyncio.to_thread(self._connection.recv)
        if status != "ok":
            raise RuntimeError(f"Collector process failed on {command!r}: {payload}")

        return payload


def _run_collectors(connection: Connection, args: list[dict[str, Any]]) -> None:
    asyncio.run(_serve_collectors(connection, args))


async def _serve_collectors(connection: Connection, args: list[dict[str, Any]]) -> None:
    try:
        parsed = [CollectorArgs.model_validate(arg) for arg in args]
        host = _CollectorHost([Collector.resolve(arg) for arg in parsed])
    except Exception as err:  # noqa: BLE001
        connection.send(("error", repr(err)))
        return
    connection.send(("ok", None))

    while True:
        command, *params = await asyncio.to_thread(connection.recv)
        if command == "start":
            await host.start()
            connection.send(("ok", None))
        elif command == "stop":
            await host.stop()
            connection.send(("ok", None))
        elif command == "result":
            connection.send(("ok", host.results(*params)))
        elif command == "close":
            await host.close()
            connection.send(("ok", None))
            return
        else:
            connection.send(("error", f"unknown command {command!r}"))


class _CollectorHost:
    """
    Drive the collectors inside the child process.

    A collector that fails to start or stop still reports a result, an error
    message, so the results keep lining up with the configured collectors. Each
    result contains its collector's ``kind``.
    """

    def __init__(self, collectors: list[Collector]):
        self.collectors = collectors
        self.errors: list[str | None] = [None] * len(collectors)

    async def start(self) -> None:
        for index, collector in enumerate(self.collectors):
            self.errors[index] = None
            try:
                await collector.start()
            except Exception as err:  # noqa: BLE001
                logger.warning("Collector {} failed to start: {}", index, err)
                self.errors[index] = f"start failed: {err}"

    async def stop(self) -> None:
        for index, collector in enumerate(self.collectors):
            if self.errors[index] is not None:
                continue
            try:
                await collector.stop()
            except Exception as err:  # noqa: BLE001
                logger.warning("Collector {} failed to stop: {}", index, err)
                self.errors[index] = f"stop failed: {err}"

    def results(self, start_time: float, end_time: float) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, collector in enumerate(self.collectors):
            kind = collector.kind
            if (error := self.errors[index]) is not None:
                results.append({"kind": kind, "error": error})
                continue
            try:
                results.append({"kind": kind, **collector.result(start_time, end_time)})
            except Exception as err:  # noqa: BLE001
                logger.warning("Collector {} failed to report: {}", index, err)
                results.append({"kind": kind, "error": f"result failed: {err}"})
        return results

    async def close(self) -> None:
        for collector in self.collectors:
            try:
                await collector.close()
            except Exception as err:  # noqa: BLE001
                logger.warning("Collector failed to close: {}", err)
