"""
Request scheduling strategies for controlling benchmark request processing patterns.

Provides timing implementations and concrete strategies that control request
concurrency, timing patterns, and throughput characteristics to simulate real-world
usage scenarios. Strategies define how requests are distributed across worker processes,
when they should be scheduled, and what constraints apply to concurrent processing.
The scheduling system separates timing logic from strategy constraints, enabling
flexible combination of timing behaviors with process and concurrency limits.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import abstractmethod
from multiprocessing import Lock, Value
from typing import Annotated, ClassVar, Literal, TypeVar

from pydantic import Field, PrivateAttr

from guidellm.schemas import RequestInfo
from guidellm.utils import InfoMixin, PydanticClassRegistryMixin

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
        description="The type of scheduling strategy to schedule requests with",
    )
    worker_count: int = Field(
        default=0,
        description="Number of worker processes to use for this strategy",
        ge=0,
    )
    max_concurrency: int = Field(
        default=0,
        description="Maximum number of concurrent requests to allow",
        ge=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for startup request distribution",
        ge=0,
    )

    _processes_lock = PrivateAttr(None)
    _processes_request_index = PrivateAttr(None)
    _processes_start_time = PrivateAttr(None)
    _cached_processes_start_time: float | None = PrivateAttr(None)

    @property
    def processes_limit(self) -> int | None:
        """
        Get the maximum number of worker processes supported by this strategy.

        :return: Maximum number of worker processes, None if unlimited
        """
        return None

    @property
    def requests_limit(self) -> int | None:
        """
        Get the maximum number of concurrent requests supported by this strategy.

        :return: Maximum number of concurrent requests, None if unlimited
        """
        return None

    def init_processes_timings(
        self,
        worker_count: int,
        max_concurrency: int,
        startup_duration: float,
    ):
        """
        Initialize shared timing state for multi-process coordination.

        :param worker_count: Number of worker processes to coordinate
        :param max_concurrency: Maximum number of concurrent requests allowed
        :param startup_duration: Duration in seconds for request startup ramping
        """
        self.worker_count = worker_count
        self.max_concurrency = max_concurrency
        self.startup_duration = startup_duration

        self._processes_request_index = Value("i", 0)
        self._processes_lock = Lock()
        self._processes_start_time = Value("d", -1.0)

    def init_processes_start(self, start_time: float):
        """
        Set the synchronized start time for all worker processes.

        :param start_time: Unix timestamp when request processing should begin
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_lock is None:
            raise RuntimeError(
                "SchedulingStrategy init_processes_start called before "
                "init_processes_timings"
            )

        with self._processes_lock:
            self._processes_start_time.value = start_time

    async def get_processes_start_time(self) -> float:
        """
        Get the synchronized start time, waiting if not yet set.

        :return: Unix timestamp when request processing began
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_lock is None:
            raise RuntimeError(
                "SchedulingStrategy get_processes_start_time called before "
                "init_processes_timings"
            )

        while self._cached_processes_start_time is None:
            with self._processes_lock:
                if self._processes_start_time.value != -1.0:
                    self._cached_processes_start_time = self._processes_start_time.value
                else:
                    await asyncio.sleep(0.01)  # wait for start time to be set by main

        return self._cached_processes_start_time

    def next_request_index(self) -> int:
        """
        Get the next sequential request index across all worker processes.

        :return: Globally unique request index for timing calculations
        :raises RuntimeError: If called before init_processes_timings
        """
        if self._processes_lock is None:
            raise RuntimeError(
                "SchedulingStrategy next_request_index called before "
                "init_processes_timings"
            )

        with self._processes_lock:
            self._processes_request_index.value += 1
            return self._processes_request_index.value

    @abstractmethod
    async def next_request_time(self, offset: int) -> float:
        """
        Calculate the scheduled start time for the next request.

        :param offset: Worker process offset for distributing request timing
        :return: Unix timestamp when the request should be processed
        """

    @abstractmethod
    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion and update internal timing state.

        :param request_info: Information about the completed request including
            timing details and completion status
        """


StrategyT = TypeVar("StrategyT", bound=SchedulingStrategy)


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
    def processes_limit(self) -> int | None:
        """
        :return: Always 1 to enforce single-process constraint
        """
        return 1

    @property
    def requests_limit(self) -> int | None:
        """
        :return: Always 1 to enforce single-request constraint
        """
        return 1

    async def next_request_time(self, offset: int) -> float:
        """
        Calculate next request time based on previous completion.

        :param offset: Unused for synchronous strategy
        :return: Time of last completion or start time if first request
        """
        _ = offset  # offset unused for synchronous strategy

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
    streams: int = Field(
        description="Number of concurrent streams for scheduling requests",
        gt=0,
    )

    _process_last_request_time: float | None = PrivateAttr(None)

    def __str__(self) -> str:
        """
        :return: String identifier with stream count
        """
        return f"concurrent@{self.streams}"

    @property
    def processes_limit(self) -> int:
        """
        :return: Number of streams as maximum worker processes
        """
        return self.streams

    @property
    def requests_limit(self) -> int:
        """
        :return: Number of streams as maximum concurrent requests
        """
        return self.streams

    async def next_request_time(self, offset: int) -> float:
        """
        Calculate next request time with stream-based distribution.

        :param offset: Worker process offset for distributing initial requests
        :return: Time of last completion or staggered start time if first request
        """
        if self._process_last_request_time is not None:
            return self._process_last_request_time

        start_time = await self.get_processes_start_time()

        return start_time + (offset / self.worker_count)

    def request_completed(self, request_info: RequestInfo):
        """
        Update timing state with completed request information.

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
    max_concurrency: int | None = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
        gt=0,
    )

    def __str__(self) -> str:
        """
        :return: String identifier for throughput strategy
        """
        return "throughput"

    @property
    def processes_limit(self) -> int | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> int | None:
        """
        :return: Max concurrency if set, otherwise None for unlimited
        """
        return self.max_concurrency

    async def next_request_time(self, offset: int) -> float:
        """
        Calculate next request time with optional startup ramping.

        :param offset: Unused for throughput strategy
        :return: Immediate start or ramped start time during startup period
        """
        _ = offset  # offset unused for throughput strategy
        start_time = await self.get_processes_start_time()

        if (
            self.startup_duration > 0
            and (time.time() - start_time) < self.startup_duration
            and (current_index := self.next_request_index()) <= self.max_concurrency
        ):
            # linearly ramp start times to spread max_concurrency requests evenly
            # over startup_duration
            return start_time + self.startup_duration * (
                current_index / self.max_concurrency
            )

        return start_time + self.startup_duration

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for throughput strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for throughput strategy


@SchedulingStrategy.register("constant")
class AsyncConstantStrategy(ThroughputStrategy):
    """
    Constant-rate scheduling for predictable load patterns.

    Schedules requests at a fixed rate distributed evenly across worker processes,
    providing predictable timing behavior for steady-state load simulation and
    consistent system performance measurement. Requests arrive at uniform intervals.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description="Rate for scheduling requests asynchronously in requests/second",
        gt=0,
    )

    def __str__(self) -> str:
        """
        :return: String identifier with rate value
        """
        return f"constant@{self.rate:.2f}"

    async def next_request_time(self, offset: int) -> float:
        """
        Calculate next request time at fixed intervals.

        :param offset: Unused for constant strategy
        :return: Start time plus constant interval based on request index
        """
        _ = offset  # offset unused for throughput strategy
        current_index = self.next_request_index()
        start_time = await self.get_processes_start_time()

        return start_time + current_index / self.rate

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for constant strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for async constant strategy


@SchedulingStrategy.register("poisson")
class AsyncPoissonStrategy(ThroughputStrategy):
    """
    Poisson-distributed scheduling for realistic load simulation.

    Schedules requests following a Poisson process with exponentially distributed
    inter-arrival times, providing realistic simulation of user behavior and network
    traffic patterns. Request arrivals have random variance around the target rate.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description="Rate for scheduling requests asynchronously in requests/second",
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for Poisson distribution",
    )

    _random: random.Random | None = PrivateAttr(None)
    _offset = PrivateAttr(None)

    def __str__(self) -> str:
        """
        :return: String identifier with rate value
        """
        return f"poisson@{self.rate:.2f}"

    def init_processes_timings(
        self,
        worker_count: int,
        max_concurrency: int,
        startup_duration: float,
    ):
        """
        Initialize Poisson-specific timing state.

        :param worker_count: Number of worker processes to coordinate
        :param max_concurrency: Maximum number of concurrent requests allowed
        :param startup_duration: Duration in seconds for request startup ramping
        """
        super().init_processes_timings(worker_count, max_concurrency, startup_duration)
        with self._processes_lock:
            self._offset = Value("d", -1.0)

    def init_processes_start(self, start_time: float):
        """
        Initialize the offset time for Poisson timing calculations.

        :param start_time: Unix timestamp when request processing should begin
        """
        ThroughputStrategy.init_processes_start(self, start_time)
        with self._processes_lock:
            self._offset.value = start_time

    async def next_request_time(self, offset: int) -> float:
        """
        Calculate next request time using exponential distribution.

        :param offset: Unused for Poisson strategy
        :return: Next arrival time based on Poisson process
        """
        _ = offset  # offset unused for throughput strategy
        _ = await self.get_processes_start_time()  # ensure offset is initialized

        if self._random is None:
            self._random = random.Random(self.random_seed)

        next_delay = self._random.expovariate(self.rate)

        with self._processes_lock:
            self._offset.value += next_delay

            return self._offset.value

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no-op for Poisson strategy).

        :param request_info: Completed request metadata (unused)
        """
        _ = request_info  # request_info unused for async poisson strategy
