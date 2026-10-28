"""
Event-loop lag of worker processes, recorded in shared memory.

Each worker samples how late a timer fires on its event loop and counts the
lag in a base-2 exponential histogram with eight buckets per doubling, the
bucket width of OpenTelemetry's exponential histograms at scale 3. A bucket's
midpoint is within 4.5% of every lag from 1 microsecond to 134 seconds counted
in it. Every worker writes only its own row of a shared array, and the main
process merges the rows once the workers exit.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from multiprocessing.context import BaseContext

__all__ = ["LoopLagHistograms"]

_BUCKETS_PER_DOUBLING = 8
# Bucket 0 holds lags under 1 microsecond. The last bucket ends at 2**27
# microseconds, about 134 seconds, and also holds anything longer.
_BUCKETS = 1 + 27 * _BUCKETS_PER_DOUBLING
# uvloop timers fire to the millisecond, so lag up to 1 ms is not starvation
_TIMER_RESOLUTION = 0.001  # seconds


def _bucket(lag: float) -> int:
    """Return the histogram bucket for a lag in seconds."""
    micros = lag * 1e6
    if micros < 1.0:
        return 0
    index = 1 + int(math.log2(micros) * _BUCKETS_PER_DOUBLING)
    return min(index, _BUCKETS - 1)


def _bucket_value(bucket: int) -> float:
    """Return the geometric midpoint of a bucket, in seconds."""
    if bucket == 0:
        return 0.0
    return 2.0 ** ((bucket - 0.5) / _BUCKETS_PER_DOUBLING) / 1e6


class LoopLagHistograms:
    """
    One histogram of event-loop lag per worker process, in shared memory.

    Create it in the main process before starting the workers, so each worker
    receives the same shared array.
    """

    def __init__(self, mp_context: BaseContext, worker_count: int, interval: float):
        """
        :param mp_context: Multiprocessing context used to start the workers
        :param worker_count: Number of worker processes that will record lag
        :param interval: Mean seconds between samples in each worker
        """
        self.worker_count = worker_count
        self.interval = interval
        self._counts = mp_context.RawArray("d", worker_count * _BUCKETS)

    def record(self, worker_index: int, lag: float) -> None:
        """
        Count one lag sample for a worker.

        :param worker_index: Index of the worker the sample came from
        :param lag: Seconds the event loop ran past a timer's due time
        """
        self._counts[worker_index * _BUCKETS + _bucket(lag)] += 1

    def samples(self) -> list[tuple[float, float]]:
        """
        Merge the workers' histograms.

        :return: ``(lag, count)`` for each non-empty bucket, where ``lag`` is
            the bucket midpoint in seconds
        """
        merged = [0.0] * _BUCKETS
        for worker in range(self.worker_count):
            offset = worker * _BUCKETS
            for bucket in range(_BUCKETS):
                merged[bucket] += self._counts[offset + bucket]
        return [
            (_bucket_value(bucket), count)
            for bucket, count in enumerate(merged)
            if count > 0
        ]

    async def monitor(self, worker_index: int) -> None:
        """
        Sample a worker's event-loop lag until cancelled.

        The time between samples is exponential with mean ``interval``, so samples
        do not stay in phase with a periodic stall such as a CPU quota period. The
        1 ms timer resolution is subtracted from each sample.

        :param worker_index: Index of the worker running this coroutine
        """
        rng = random.Random()
        while True:
            delay = rng.expovariate(1 / self.interval)
            start = time.perf_counter()
            await asyncio.sleep(delay)
            elapsed = time.perf_counter() - start
            self.record(worker_index, elapsed - delay - _TIMER_RESOLUTION)
