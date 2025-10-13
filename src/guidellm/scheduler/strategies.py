"""
Request scheduling strategies for controlling how benchmark requests are processed.

This module provides timing implementations and concrete strategies that control request
concurrency, timing patterns, and throughput characteristics to simulate real-world
usage scenarios. The scheduling system separates timing logic from strategy constraints,
enabling flexible combination of timing behaviors with process and concurrency limits.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, Literal, TypeVar

from pydantic import Field, PrivateAttr

from guidellm.schemas import RequestInfo
from guidellm.utils import InfoMixin, PydanticClassRegistryMixin, StandardBaseModel

__all__ = [
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
    "ConcurrentStrategy",
    "ConstantRateRequestTimings",
    "LastCompletionRequestTimings",
    "NoDelayRequestTimings",
    "PoissonRateRequestTimings",
    "ScheduledRequestTimings",
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


def _exponential_decay_tau(max_progress: float, convergence: float = 0.99) -> float:
    """
    Calculate tau value for exponential decay to reach target progress level.

    :param max_progress: The max progress value to reach
    :param convergence: The target convergence level for reaching max_progress
    :return: The calculated tau value for the given max_progress and convergence
    """
    return max_progress / (-math.log(1 - convergence))


def _exponential_decay_fraction(progress: float, tau: float = 1.0) -> float:
    """
    Calculate completion fraction based on exponential decay curve.

    :param progress: The current progress value (>=0)
    :param tau: The scale factor for the exponential decay
    :return: The fraction of completion based on exponential decay (0 -> 1)
    """
    return 1 - math.exp(-progress / tau)


class ScheduledRequestTimings(StandardBaseModel, ABC):
    """
    Abstract base class for controlling when requests are scheduled.

    Defines the interface for timing implementations that determine request scheduling
    behavior. Different implementations provide various patterns like synchronous,
    constant-rate, or stochastic scheduling to simulate real-world scenarios.
    """

    @abstractmethod
    def next_offset(self) -> float:
        """
        Calculate the time offset for the next request to be scheduled.

        :return: The offset in seconds from scheduler start time for next request
        """

    @abstractmethod
    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion and update internal timing state.

        :param request_info: Information about the completed request including
            timing details and completion status
        """


class LastCompletionRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for synchronous and concurrent scheduling strategies.

    Schedules the next request immediately after the last request completes, enabling
    sequential or limited concurrent processing with completion-based timing control.
    """

    offset: float = Field(
        default=0.0,
        description="Current time offset in seconds from scheduler start time",
    )
    startup_requests: int = Field(
        default=0,
        description="Number of initial requests to schedule with equal spacing",
        ge=0,
    )
    startup_requests_delay: float = Field(
        default=0.0,
        description="Delay in seconds between startup requests",
        ge=0,
    )
    _requests_count: int = PrivateAttr(0)

    def next_offset(self) -> float:
        """
        Get the current offset value and apply startup delay if applicable.

        :return: The current offset value in seconds from scheduler start time
        """
        self._requests_count += 1

        if self._requests_count <= self.startup_requests:
            self.offset += self.startup_requests_delay

        return self.offset

    def request_completed(self, request_info: RequestInfo):
        """
        Update timing state based on the completed request.

        :param request_info: Information about the completed request
        """
        if (
            self._requests_count > self.startup_requests
            and request_info.completed_at is not None
        ):
            # set the next sync offset to the time when the previous request completed
            self.offset = request_info.completed_at - request_info.scheduler_start_time


class NoDelayRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for throughput-maximizing scheduling strategies.

    Schedules requests with minimal delay to achieve maximum throughput, with optional
    startup ramping to gradually increase request processing during initialization.
    """

    offset: float = Field(
        default=0.0,
        description="Base time offset in seconds from scheduler start time",
        ge=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for gradual startup ramp",
        ge=0,
    )
    startup_target_requests: int = Field(
        default=1,
        description="Target number of requests to converge to during startup",
        gt=0,
    )
    startup_convergence: float = Field(
        default=0.99,
        description="Target convergence rate during startup phase",
    )
    _start_time: float | None = PrivateAttr(None)
    _requests_count: int = PrivateAttr(0)

    def next_offset(self) -> float:
        """
        Calculate offset with optional startup adjustment.

        :return: Static offset plus any startup adjustment
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._requests_count += 1
        elapsed = time.time() - self._start_time

        if self.startup_duration > 0 and elapsed < self.startup_duration:
            startup_percent = _exponential_decay_fraction(
                self._requests_count,
                _exponential_decay_tau(
                    self.startup_target_requests, self.startup_convergence
                ),
            )
        else:
            startup_percent = 1.0

        return self.offset + startup_percent * self.startup_duration

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no action needed for throughput strategy).

        :param request_info: Information about the completed request (unused)
        """


class ConstantRateRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for constant-rate scheduling strategies.

    Schedules requests at a fixed rate with evenly spaced intervals to provide
    predictable timing behavior for steady-state load simulation.
    """

    rate: float = Field(
        description="Target rate in requests per second",
        gt=0,
    )
    offset: float = Field(
        default=0.0,
        description="Base time offset in seconds from scheduler start time",
        ge=0,
    )
    _requests_count: int = PrivateAttr(0)

    def next_offset(self) -> float:
        """
        Calculate the offset for the next request at a constant rate.

        :return: The offset in seconds for the next request
        """
        num_requests = self._requests_count
        self._requests_count += 1
        interval = 1.0 / self.rate

        return self.offset + interval * num_requests

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no action needed for constant rate strategy).

        :param request_info: Information about the completed request (unused)
        """


class PoissonRateRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for Poisson-distributed scheduling strategies.

    Schedules requests following a Poisson process with exponentially distributed
    inter-arrival times to simulate realistic traffic patterns with random variance.
    """

    rate: float = Field(
        description="Target average rate in requests per second",
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description="Seed for random number generator for reproducible behavior",
    )
    offset: float = Field(
        default=0.0,
        description="Base time offset in seconds from scheduler start time",
    )
    _requests_count: int = PrivateAttr(0)
    _random: random.Random | None = PrivateAttr(None)

    def next_offset(self) -> float:
        """
        Calculate the offset for the next request using Poisson distribution.

        :return: The cumulative offset in seconds for the next request
        """
        self._requests_count += 1

        if self._random is None:
            self._random = random.Random(self.random_seed)
        else:
            next_delay = self._random.expovariate(self.rate)
            self.offset += next_delay

        return self.offset

    def request_completed(self, request_info: RequestInfo):
        """
        Handle request completion (no action needed for Poisson rate strategy).

        :param request_info: Information about the completed request (unused)
        """


class SchedulingStrategy(PydanticClassRegistryMixin["SchedulingStrategy"], InfoMixin):
    """
    Abstract base class for scheduling strategies controlling request processing.

    Defines the interface for strategies that combine timing implementations with
    process and concurrency constraints to enable various benchmark scenarios.
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

    def create_request_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int | float
    ) -> ScheduledRequestTimings:
        """
        Create a timing instance to define scheduling behavior for a worker process.

        :param local_rank: The rank of the worker process within local world size
        :param local_world_size: Total number of worker processes in local world
        :param local_max_concurrency: Maximum concurrent requests for the worker
        :return: A ScheduledRequestTimings instance for the worker process
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError(
            "create_worker_timings method must be implemented by subclasses."
        )


StrategyT = TypeVar("StrategyT", bound=SchedulingStrategy)


@SchedulingStrategy.register("synchronous")
class SynchronousStrategy(SchedulingStrategy):
    """
    Sequential request processing strategy with single-process constraint.

    Processes requests one at a time in strict sequential order, providing predictable
    timing behavior ideal for measuring maximum sequential throughput and ensuring
    request isolation.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    def __str__(self) -> str:
        """
        Return string representation of the strategy.

        :return: String identifier for synchronous strategy
        """
        return "synchronous"

    @property
    def processes_limit(self) -> int | None:
        """
        Get maximum number of worker processes for synchronous scheduling.

        :return: Always returns 1 to enforce single-process constraint
        """
        return 1

    @property
    def requests_limit(self) -> int | None:
        """
        Get maximum number of concurrent requests for synchronous scheduling.

        :return: Always returns 1 to enforce single-request constraint
        """
        return 1

    def create_request_timings(
        self,
        local_rank: int,
        local_world_size: int,
        local_max_concurrency: int,  # noqa: ARG002
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for synchronous request scheduling.

        :param local_rank: The rank of the worker process (must be 0)
        :param local_world_size: Total number of worker processes (must be 1)
        :param local_max_concurrency: Maximum concurrent requests (unused)
        :return: LastCompletionRequestTimings instance for sequential processing
        :raises ValueError: If multiple workers or non-zero rank specified
        """
        if local_world_size > 1 or local_rank != 0:
            raise ValueError(
                "SynchronousStrategy can only be used with a single worker process."
            )

        return LastCompletionRequestTimings()


@SchedulingStrategy.register("concurrent")
class ConcurrentStrategy(SchedulingStrategy):
    """
    Parallel request processing strategy with controlled concurrency limits.

    Enables concurrent request processing up to a specified number of streams,
    providing balanced throughput while maintaining predictable resource usage
    and completion-based timing coordination.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: int = Field(
        description="Number of concurrent streams for scheduling requests",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for distributing startup requests",
        ge=0,
    )

    def __str__(self) -> str:
        """
        Return string representation of the strategy.

        :return: String identifier with stream count
        """
        return f"concurrent@{self.streams}"

    @property
    def processes_limit(self) -> int:
        """
        Get maximum number of worker processes for concurrent scheduling.

        :return: Number of streams as maximum worker processes
        """
        return self.streams

    @property
    def requests_limit(self) -> int:
        """
        Get maximum number of concurrent requests for concurrent scheduling.

        :return: Number of streams as maximum concurrent requests
        """
        return self.streams

    def create_request_timings(
        self,
        local_rank: int,
        local_world_size: int,
        local_max_concurrency: int,  # noqa: ARG002
    ) -> LastCompletionRequestTimings:
        """
        Create timing implementation for concurrent request scheduling.

        :param local_rank: The rank of the worker process (must be < streams)
        :param local_world_size: Total worker processes (must not exceed streams)
        :param local_max_concurrency: Maximum concurrent requests (unused)
        :return: LastCompletionRequestTimings instance for stream-based processing
        :raises ValueError: If worker configuration exceeds stream limits
        """
        if local_world_size > self.streams:
            raise ValueError(
                "ConcurrentStrategy can only be used with up to "
                f"{self.streams} worker processes."
            )

        if local_rank >= self.streams:
            raise ValueError(
                f"Local rank {local_rank} exceeds the number of streams {self.streams}."
            )

        if self.startup_duration > 0:
            # Ensure equal global distribution of the start up for concurrent streams
            # Ex: for 10 streams, 2 workers, and 8 seconds start up duration,
            # the first worker should start at 0.0, 1.6, 3.2, 4.8, 6.4
            # and the second worker should start at 0.8, 2.4, 4.0, 5.6, 7.2
            delay_per_stream = self.startup_duration / self.streams
            streams_per_worker = self.streams // local_world_size

            offset = local_rank * streams_per_worker * delay_per_stream
            startup_requests = streams_per_worker + (
                1
                if local_world_size > 1 and local_rank < self.streams % local_world_size
                else 0
            )
            startup_requests_delay = delay_per_stream * local_world_size
        else:
            offset = 0.0
            startup_requests = 0
            startup_requests_delay = 0.0

        return LastCompletionRequestTimings(
            offset=offset,
            startup_requests=startup_requests,
            startup_requests_delay=startup_requests_delay,
        )


@SchedulingStrategy.register("throughput")
class ThroughputStrategy(SchedulingStrategy):
    """
    Maximum throughput strategy with optional concurrency limits.

    Schedules requests to maximize system throughput by allowing unlimited concurrent
    processing with optional constraints and startup ramping for controlled ramp-up.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: int | None = Field(
        default=None,
        description="Maximum number of concurrent requests to schedule",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for startup request distribution",
        ge=0,
    )

    def __str__(self) -> str:
        """
        Return string representation of the strategy.

        :return: String identifier for throughput strategy
        """
        return "throughput"

    @property
    def processes_limit(self) -> int | None:
        """
        Get maximum number of worker processes for throughput scheduling.

        :return: The max_concurrency value if set, otherwise None for unlimited
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> int | None:
        """
        Get maximum number of concurrent requests for throughput scheduling.

        :return: The max_concurrency value if set, otherwise None for unlimited
        """
        return self.max_concurrency

    def create_request_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for throughput request scheduling.

        :param local_rank: The rank of the worker process
        :param local_world_size: Total number of worker processes
        :param local_max_concurrency: Maximum concurrent requests for the worker
        :return: NoDelayRequestTimings instance for immediate request scheduling
        """
        if self.startup_duration > 0:
            # Vary offset by up to 5% of the startup duration for a bit of variance
            offset = 0.05 * self.startup_duration * (local_rank / local_world_size)
            # Use local_max_concurrency as the target requests for startup convergence
            startup_target_requests = local_max_concurrency
        else:
            offset = 0.0
            startup_target_requests = 1

        return NoDelayRequestTimings(
            startup_duration=self.startup_duration,
            startup_target_requests=startup_target_requests,
            offset=offset,
        )


@SchedulingStrategy.register("constant")
class AsyncConstantStrategy(ThroughputStrategy):
    """
    Asynchronous constant-rate scheduling strategy for predictable load patterns.

    Schedules requests at a fixed rate distributed evenly across worker processes,
    providing predictable timing behavior for steady-state load simulation and
    consistent system performance measurement.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description="Rate for scheduling requests asynchronously in requests/second",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for startup request distribution",
        ge=0,
    )

    def __str__(self) -> str:
        """
        Return string representation of the strategy.

        :return: String identifier with rate value
        """
        return f"constant@{self.rate:.2f}"

    def create_request_timings(
        self,
        local_rank: int,
        local_world_size: int,
        local_max_concurrency: int,  # noqa: ARG002
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for constant-rate request scheduling.

        :param local_rank: The rank of the worker process
        :param local_world_size: Total number of worker processes for rate division
        :param local_max_concurrency: Maximum concurrent requests for the worker
        :return: ConstantRateRequestTimings instance with per-worker rate
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size
        # Start each worker with an offset to interleave rates
        worker_offset = (1 / self.rate) * local_rank

        return ConstantRateRequestTimings(
            rate=worker_rate,
            offset=worker_offset,
        )


@SchedulingStrategy.register("poisson")
class AsyncPoissonStrategy(ThroughputStrategy):
    """
    Asynchronous Poisson-distributed scheduling strategy for realistic load simulation.

    Schedules requests following a Poisson process with exponentially distributed
    inter-arrival times, providing realistic simulation of user behavior and network
    traffic patterns with random variance around the target rate.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description="Rate for scheduling requests asynchronously in requests/second",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description="Duration in seconds for startup request distribution",
        ge=0,
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for Poisson distribution",
    )

    def __str__(self) -> str:
        """
        Return string representation of the strategy.

        :return: String identifier with rate value
        """
        return f"poisson@{self.rate:.2f}"

    def create_request_timings(
        self,
        local_rank: int,
        local_world_size: int,
        local_max_concurrency: int,  # noqa: ARG002
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for Poisson-distributed request scheduling.

        :param local_rank: The rank of the worker process for seed generation
        :param local_world_size: Total number of worker processes for rate division
        :param local_max_concurrency: Maximum concurrent requests for the worker
        :return: PoissonRateRequestTimings instance with per-worker rate and unique seed
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size
        # Use a different seed for each worker to ensure different sequences
        worker_seed = self.random_seed + local_rank
        # Start each worker with an offset to interleave rates
        worker_offset = (1 / self.rate) * local_rank

        return PoissonRateRequestTimings(
            rate=worker_rate,
            random_seed=worker_seed,
            offset=worker_offset,
        )
