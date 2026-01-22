"""
Request scheduling strategies for controlling benchmark request processing patterns.

Provides timing implementations and concrete strategies that control request
concurrency, timing patterns, and throughput characteristics to simulate real-world
usage scenarios. Strategies define how requests are distributed across worker processes,
when they should be scheduled, and what constraints apply to concurrent processing.
The scheduling system separates timing logic from strategy constraints, enabling
flexible combination of timing behaviors with process and concurrency limits.

Available strategies include synchronous (sequential), concurrent (fixed streams),
throughput (maximum load), constant-rate (steady intervals), and Poisson-distributed
(realistic variance) patterns for comprehensive benchmarking scenarios.
"""

from __future__ import annotations

import asyncio
import math
import random
from abc import abstractmethod
from multiprocessing import Event, Value, synchronize
from multiprocessing.sharedctypes import Synchronized
from typing import Annotated, ClassVar, Literal, TypeVar

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, PrivateAttr

from guidellm.schemas import PydanticClassRegistryMixin, RequestInfo
from guidellm.utils import InfoMixin

__all__ = [
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
    "ConcurrentStrategy",
    "SchedulingStrategy",
    "StrategyT",
    "StrategyType",
    "SynchronousStrategy",
    "ThroughputStrategy",
]


StrategyType = Annotated[
    Literal["synchronous", "concurrent", "throughput", "constant", "poisson"],
    "Valid strategy type identifiers for scheduling request patterns",
]


class SchedulingStrategy(PydanticClassRegistryMixin["SchedulingStrategy"], InfoMixin):
    """
    Base class for scheduling strategies controlling request processing patterns.

    Defines the interface for strategies that combine timing implementations with
    process and concurrency constraints to enable various benchmark scenarios.
    Strategies manage request timing, worker process coordination, and concurrency
    limits across distributed execution environments.

    :cvar schema_discriminator: Field name used for polymorphic deserialization
    """

    schema_discriminator: ClassVar[str] = "type_"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[SchedulingStrategy]:
        if cls.__name__ == "SchedulingStrategy":
            return cls

        return SchedulingStrategy

    type_: Literal["strategy"] = Field(
        description="Scheduling strategy type identifier for polymorphic dispatch",
    )
    worker_count: PositiveInt | None = Field(
        default=None,
        description="Number of worker processes to use for this strategy",
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum number of concurrent requests to allow",
    )

    _processes_init_event: synchronize.Event | None = PrivateAttr(None)
    _processes_request_index: Synchronized[int] | None = PrivateAttr(None)
    _processes_start_time: Synchronized[float] | None = PrivateAttr(None)
    _cached_processes_start_time: float | None = PrivateAttr(None)

    @property
    def processes_limit(self) -> PositiveInt | None:
        """
        Get the maximum number of worker processes supported by this strategy.

        :return: Maximum number of worker processes, None if unlimited
        """
        return None

    @property
    def requests_limit(self) -> PositiveInt | None:
        """
        Get the maximum number of concurrent requests supported by this strategy.

        :return: Maximum number of concurrent requests, None if unlimited
        """
        return None

    def init_processes_timings(
        self, worker_count: PositiveInt, max_concurrency: PositiveInt
    ):
        """
        Initialize shared timing state for multi-process coordination.

        Sets up synchronized counters and locks for coordinating request timing
        across distributed worker processes.

        :param worker_count: Number of worker processes to coordinate
        :param max_concurrency: Maximum number of concurrent requests allowed
        """
        self.worker_count = worker_count
        self.max_concurrency = max_concurrency

        self._processes_init_event = Event()
        self._processes_request_index = Value("i", 0)
        self._processes_start_time = Value("d", -1.0)

    def init_processes_start(self, start_time: float):
        """
        Set the synchronized start time for all worker processes.

        Updates shared state with the benchmark start time to coordinate request
        scheduling across all workers.

        :param start_time: Unix timestamp when request processing should begin
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_init_event is None:
            raise RuntimeError(
                "SchedulingStrategy init_processes_start called before "
                "init_processes_timings"
            )
        if self._processes_start_time is None:
            raise RuntimeError(
                "_processes_lock is not None but _processes_start_time is None"
            )

        with self._processes_start_time.get_lock():
            self._processes_start_time.value = start_time
            self._processes_init_event.set()

    async def get_processes_start_time(self) -> float:
        """
        Get the synchronized start time, waiting if not yet set.

        Blocks until the main process sets the start time via init_processes_start,
        enabling synchronized request scheduling across all workers.

        :return: Unix timestamp when request processing began
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_init_event is None:
            raise RuntimeError(
                "SchedulingStrategy get_processes_start_time called before "
                "init_processes_timings"
            )
        if self._processes_start_time is None:
            raise RuntimeError(
                "_processes_lock is not None but _processes_start_time is None"
            )

        if self._cached_processes_start_time is None:
            # Wait for the init event to be set by the main process
            await asyncio.gather(asyncio.to_thread(self._processes_init_event.wait))
            self._cached_processes_start_time = self._processes_start_time.value

        return self._cached_processes_start_time

    def next_request_index(self) -> PositiveInt:
        """
        Get the next sequential request index across all worker processes.

        Thread-safe counter providing globally unique indices for request timing
        calculations in distributed environments.

        :return: Globally unique request index for timing calculations
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_request_index is None:
            raise RuntimeError(
                "SchedulingStrategy next_request_index called before "
                "init_processes_timings"
            )

        with self._processes_request_index.get_lock():
            self._processes_request_index.value += 1
            return self._processes_request_index.value

    @abstractmethod
    async def next_request_time(self, worker_index: NonNegativeInt) -> float:
        """
        Calculate the scheduled start time for the next request.

        Strategy-specific implementation determining when requests should be
        processed based on timing patterns and worker distribution.

        :param worker_index: Worker process index for distributing request timing
        :return: Unix timestamp when the request should be processed
        """

    @abstractmethod
    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion and update internal timing state.

        Strategy-specific handling of completed requests to maintain timing
        coordination and schedule subsequent requests.

        :param request_info: Completed request metadata including timing details
            and completion status
        """


StrategyT = TypeVar("StrategyT", bound=SchedulingStrategy)
"Type variable bound to SchedulingStrategy for generic strategy operations"


@SchedulingStrategy.register("synchronous")
class SynchronousStrategy(SchedulingStrategy):
    """
    Sequential request processing with strict single-request-at-a-time execution.

    Processes requests one at a time in strict sequential order, providing predictable
    timing behavior ideal for measuring maximum sequential throughput and ensuring
    complete request isolation. Each request completes before the next begins.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]
    _process_last_request_time: float | None = PrivateAttr(None)

    def __str__(self) -> str:
        """
        :return: String identifier for synchronous strategy
        """
        return "synchronous"

    @property
    def processes_limit(self) -> PositiveInt:
        """
        :return: Always 1 to enforce single-process constraint
        """
        return 1

    @property
    def requests_limit(self) -> PositiveInt:
        """
        :return: Always 1 to enforce single-request constraint
        """
        return 1

    async def next_request_time(self, worker_index: NonNegativeInt) -> float:
        """
        Calculate next request time based on previous completion.

        :param worker_index: Unused for synchronous strategy
        :return: Time of last completion or start time if first request
        """
        _ = worker_index  # unused for synchronous strategy

        if self._process_last_request_time is not None:
            return self._process_last_request_time

        return await self.get_processes_start_time()

    def request_completed(self, request_info: RequestInfo):
        """
        Update timing state with completed request information.

        :param request_info: Completed request metadata including timing
        """
        if request_info.completed_at is not None:
            self._process_last_request_time = request_info.completed_at


@SchedulingStrategy.register("concurrent")
class ConcurrentStrategy(SchedulingStrategy):
    """
    Parallel request processing with fixed concurrency limits.

    Enables concurrent request processing up to a specified number of streams,
    providing balanced throughput while maintaining predictable resource usage.
    Requests are distributed across streams with completion-based timing coordination.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: PositiveInt = Field(
        description="Number of concurrent streams for scheduling requests",
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds to spread initial requests up to max_concurrency "
            "at the beginning of each strategy run"
        ),
    )

    _process_last_request_time: float | None = PrivateAttr(None)

    def __str__(self) -> str:
        """
        :return: String identifier with stream count
        """
        return f"concurrent@{self.streams}"

    @property
    def processes_limit(self) -> PositiveInt:
        """
        :return: Number of streams as maximum worker processes
        """
        return self.streams

    @property
    def requests_limit(self) -> PositiveInt:
        """
        :return: Number of streams as maximum concurrent requests
        """
        return self.streams

    async def next_request_time(self, worker_index: PositiveInt) -> float:
        """
        Calculate next request time with stream-based distribution.

        Initial requests are staggered across streams during rampup, subsequent
        requests scheduled after previous completion within each stream.

        :param worker_index: Worker process index for distributing initial requests
        :return: Time of last completion or staggered start time if first request
        """
        _ = worker_index  # unused
        current_index = self.next_request_index()
        start_time = await self.get_processes_start_time()

        if current_index < self.streams and self.rampup_duration > 0:
            # linearly spread start times for first concurrent requests across rampup
            return start_time + self.rampup_duration * (current_index / self.streams)

        if self._process_last_request_time is not None:
            return self._process_last_request_time

        return start_time

    def request_completed(self, request_info: RequestInfo):
        """
        Update timing state with completed request information.

        Tracks completion time to schedule next request in the same stream.

        :param request_info: Completed request metadata including timing
        """
        if request_info.completed_at is not None:
            self._process_last_request_time = request_info.completed_at


@SchedulingStrategy.register("throughput")
class ThroughputStrategy(SchedulingStrategy):
    """
    Maximum throughput scheduling with optional concurrency limits.

    Schedules requests to maximize system throughput by allowing unlimited concurrent
    processing with optional constraints. Supports startup ramping to gradually
    distribute initial requests for controlled system ramp-up.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds to spread initial requests up to max_concurrency "
            "at the beginning of each strategy run"
        ),
    )

    def __str__(self) -> str:
        """
        :return: String identifier for throughput strategy
        """
        return f"throughput@{self.max_concurrency or 'unlimited'}"

    @property
    def processes_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    async def next_request_time(self, worker_index: int) -> float:
        """
        Calculate next request time with optional startup ramping.

        Spreads initial requests linearly during rampup period, then schedules
        all subsequent requests immediately.

        :param worker_index: Unused for throughput strategy
        :return: Immediate start or ramped start time during startup period
        """
        _ = worker_index  # unused for throughput strategy
        start_time = await self.get_processes_start_time()

        if self.max_concurrency is not None and self.rampup_duration > 0:
            current_index = self.next_request_index()
            delay = (
                self.rampup_duration
                if current_index >= self.max_concurrency
                else self.rampup_duration
                * (current_index / float(self.max_concurrency))
            )

            return start_time + delay
        else:
            return start_time

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for throughput strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for throughput strategy


@SchedulingStrategy.register("constant")
class AsyncConstantStrategy(SchedulingStrategy):
    """
    Constant-rate scheduling for predictable load patterns.

    Schedules requests at a fixed rate distributed evenly across worker processes,
    providing predictable timing behavior for steady-state load simulation and
    consistent system performance measurement. Requests arrive at uniform intervals.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description="Request scheduling rate in requests per second",
        gt=0,
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
    )
    rampup_duration: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Duration in seconds to linearly ramp up from 0 to target rate "
            "at the beginning of each strategy run"
        ),
    )

    def __str__(self) -> str:
        """
        :return: String identifier with rate value
        """
        return f"constant@{self.rate:.2f}"

    @property
    def processes_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    async def next_request_time(self, worker_index: PositiveInt) -> float:
        """
        Calculate next request time at fixed intervals with optional linear rampup.

        Schedules requests at uniform intervals determined by the configured rate,
        independent of request completion times. If rampup_duration is set, the rate
        increases linearly from 0 to the target rate during the rampup period, then
        continues at the constant rate.

        :param worker_index: Unused for constant strategy
        :return: Start time plus interval based on request index and
            rampup configuration
        """
        _ = worker_index  # unused
        current_index = self.next_request_index()
        start_time = await self.get_processes_start_time()

        if self.rampup_duration > 0:
            # Calculate number of requests that would be sent during rampup
            # Cumulative requests by time t during rampup:
            # n = rate * t² / (2 * rampup_duration)
            # At end of rampup (t = rampup_duration), n_rampup is calculated below
            n_rampup = self.rate * self.rampup_duration / 2.0

            if current_index == 1:
                # First request at start_time
                return start_time
            elif current_index <= n_rampup:
                # During rampup: solve for t where
                # n = rate * t² / (2 * rampup_duration)
                time_offset = math.sqrt(
                    2.0 * current_index * self.rampup_duration / self.rate
                )
                return start_time + time_offset
            else:
                # After rampup: continue at constant rate
                time_offset = (
                    self.rampup_duration + (current_index - n_rampup) / self.rate
                )
                return start_time + time_offset
        else:
            # No rampup: uniform intervals
            return start_time + current_index / self.rate

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for constant strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for async constant strategy


@SchedulingStrategy.register("poisson")
class AsyncPoissonStrategy(SchedulingStrategy):
    """
    Poisson-distributed scheduling for realistic load simulation.

    Schedules requests following a Poisson process with exponentially distributed
    inter-arrival times, providing realistic simulation of user behavior and network
    traffic patterns. Request arrivals have random variance around the target rate.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description="Request scheduling rate in requests per second",
        gt=0,
    )
    max_concurrency: PositiveInt | None = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for Poisson distribution reproducibility",
    )

    _random: random.Random | None = PrivateAttr(None)
    _offset: Synchronized[float] | None = PrivateAttr(None)

    def __str__(self) -> str:
        """
        :return: String identifier with rate value
        """
        return f"poisson@{self.rate:.2f}"

    @property
    def processes_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> PositiveInt | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    def init_processes_timings(self, worker_count: int, max_concurrency: int):
        """
        Initialize Poisson-specific timing state.

        Sets up shared offset value for coordinating exponentially distributed
        request timing across worker processes.

        :param worker_count: Number of worker processes to coordinate
        :param max_concurrency: Maximum number of concurrent requests allowed
        """
        self._offset = Value("d", -1.0)
        # Call base implementation last to avoid
        # setting Event before offset is ready
        super().init_processes_timings(worker_count, max_concurrency)

    def init_processes_start(self, start_time: float):
        """
        Initialize the offset time for Poisson timing calculations.

        Sets the initial timing offset from which exponentially distributed
        intervals are calculated.

        :param start_time: Unix timestamp when request processing should begin
        """
        ThroughputStrategy.init_processes_start(self, start_time)

        if self._offset is None:
            raise RuntimeError(
                "_offset is None in init_processes_start; was "
                "init_processes_timings not called?"
            )
        with self._offset.get_lock():
            self._offset.value = start_time

    async def next_request_time(self, worker_index: PositiveInt) -> float:
        """
        Calculate next request time using exponential distribution.

        Generates inter-arrival times following exponential distribution,
        accumulating delays to produce Poisson-distributed request arrivals.

        :param worker_index: Unused for Poisson strategy
        :return: Next arrival time based on Poisson process
        """
        _ = worker_index  # unused
        _ = await self.get_processes_start_time()  # ensure offset is initialized

        if self._random is None:
            self._random = random.Random(self.random_seed)

        next_delay = self._random.expovariate(self.rate)

        if self._offset is None:
            raise RuntimeError(
                "_offset is None in next_request_time; was "
                "init_processes_timings not called?"
            )
        with self._offset.get_lock():
            self._offset.value += next_delay

            return self._offset.value

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for Poisson strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for async poisson strategy
