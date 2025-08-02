"""
Request scheduling strategies for the GuideLLM toolkit.

This module provides a comprehensive set of scheduling strategies that control how
requests are processed and timed within the GuideLLM benchmarking system. These
strategies enable fine-grained control over request concurrency, timing patterns,
and throughput characteristics to simulate various real-world usage scenarios.

The scheduling system is built around abstract timing implementations that define
when requests should be executed, and concrete strategy classes that combine
timing behaviors with process and concurrency limits.

Classes:
    ScheduledRequestTimings: Abstract base class for request timing implementations
    LastCompletionRequestTimings: Timing implementation for synchronous/concurrent
        strategies
    NoDelayRequestTimings: Timing implementation for throughput-maximizing strategies
    ConstantRateRequestTimings: Timing implementation for constant-rate request
        scheduling
    PoissonRateRequestTimings: Timing implementation for Poisson-distributed request
        scheduling
    SchedulingStrategy: Abstract base class for all scheduling strategies
    SynchronousStrategy: Sequential request processing with maximum throughput
    ConcurrentStrategy: Parallel request processing with limited concurrency
    ThroughputStrategy: Unrestricted request processing for maximum system throughput
    AsyncConstantStrategy: Asynchronous request scheduling at a constant rate
    AsyncPoissonStrategy: Asynchronous request scheduling with Poisson distribution

Functions:
    strategy_display_str: Generate human-readable string representations of strategies
"""

import math
import random
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

from pydantic import Field

from guidellm.objects import StandardBaseModel
from guidellm.scheduler.objects import ScheduledRequestInfo

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
    "StrategyType",
    "SynchronousStrategy",
    "ThroughputStrategy",
    "strategy_display_str",
]


StrategyType = Literal["synchronous", "concurrent", "throughput", "constant", "poisson"]


class ScheduledRequestTimings(StandardBaseModel, ABC):
    """
    Abstract base class for request timing implementations in scheduling strategies.

    This class defines the interface for controlling when requests are scheduled
    and how timing offsets are calculated. Different implementations provide
    various timing behaviors such as synchronous, constant-rate, or stochastic
    request scheduling patterns.

    Implementations must provide logic for calculating the next request offset
    and handling request completion events that may affect future timing decisions.
    """

    @abstractmethod
    def next_offset(self) -> float:
        """
        Calculate the time offset for the next request to be scheduled.

        :return: The offset in seconds from the scheduler start time when the
            next request should be scheduled.
        """

    @abstractmethod
    def request_completed(self, request_info: ScheduledRequestInfo):
        """
        Handle the completion of a request and update internal timing state.

        This method is called when a request completes (successfully or with error)
        and allows the timing implementation to update its internal state based on
        the completion information.

        :param request_info: Information about the completed request including
            timing details and completion status.
        """


class LastCompletionRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for synchronous and concurrent scheduling strategies.

    This implementation schedules the next request immediately after the last
    request has completed, enabling sequential or limited concurrent processing.
    It maintains an internal offset based on completion times to ensure proper
    scheduling behavior.
    """

    offset: float = Field(
        default=0.0,
        description="The current time offset in seconds from scheduler start time.",
    )
    startup_requests: int = Field(
        default=0,
        description=(
            "Number of initial requests to schedule during startup phase with equal "
            "spacing of startup_requests_delay before going to last request times."
        ),
        ge=0,
    )
    startup_requests_delay: float = Field(
        default=0.0,
        description=(
            "Delay in seconds used to add to the offset for each request "
            "within the startup phase (_num_requests <= startup_requests)."
        ),
        ge=0,
    )
    _requests_count: int = Field(
        default=0,
        description="Internal counter tracking the number of offsets generated.",
        ge=0,
    )

    def next_offset(self) -> float:
        """
        :return: The current offset value in seconds from scheduler start time.
        """
        self._num_request_offsets += 1

        if self._num_request_offsets <= self.startup_requests:
            self.offset += self.startup_requests_delay

        return self.offset

    def request_completed(self, request_info: ScheduledRequestInfo):
        """
        Update timing state and offset based on the completed request.

        :param request_info: Information about the completed request including
            timing details and completion status.
        """
        if (
            self._num_request_offsets > self.startup_requests
            and request_info.completed_at is not None
        ):
            # set the next sync offset to the time when the previous request completed
            self.offset = request_info.completed_at - request_info.scheduler_start_time


class NoDelayRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for throughput-maximizing scheduling strategies.

    This implementation schedules requests with no delay, allowing the system
    to process requests as quickly as possible. It always returns a zero offset,
    enabling maximum throughput by scheduling requests immediately without
    waiting for previous requests to complete.
    """

    offset: float = Field(
        default=0.0,
        description="The current time offset in seconds from scheduler start time.",
        ge=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "The duration of the startup phase in seconds to gradually ramp up "
            "request processing."
        ),
        ge=0,
    )
    startup_tau: float = Field(
        default=1.0,
        description=(
            "The target average time between requests during the startup phase."
        ),
        gt=0,
    )
    _start_time: float = Field(
        default=None,
        description="The start time of the request processing phase.",
    )
    _requests_count: int = Field(
        default=0,
        description="Internal counter tracking the number of offsets generated.",
        ge=0,
    )

    def next_offset(self) -> float:
        """
        Get the offset for the next request with no delay.

        :return: Always returns 0.0 to schedule requests immediately.
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._requests_count += 1

        if (
            self.startup_duration > 0
            and time.time() < self._start_time + self.startup_duration
        ):
            # Gradually ramp up the request processing during start up phase
            # using exponential decay to the startup_duration
            return self.offset + self.startup_duration * (
                1 - math.exp(-self._requests_count / self.startup_tau)
            )
        elif self.startup_duration > 0:
            return self.offset + self.startup_duration
        else:
            return self.offset

    def request_completed(self, request_info: ScheduledRequestInfo):
        """
        Handle request completion (no action needed for throughput strategy).

        :param request_info: Information about the completed request (unused).
        """


class ConstantRateRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for constant-rate scheduling strategies.

    This implementation schedules requests at a constant rate defined in requests
    per second. The offset for each subsequent request is calculated as a multiple
    of the interval between requests, ensuring evenly spaced request scheduling.
    """

    rate: float = Field(
        description="The target rate in requests per second. Must be positive.",
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "to converge quickly to the desired rate before switching to "
            "constant-rate scheduling."
        ),
        ge=0,
    )
    _start_time: float = Field(
        default=None,
        description="The start time of the request processing phase.",
    )
    _requests_count: int = Field(
        default=0,
        description="Internal counter tracking the number of offsets generated.",
        ge=0,
    )

    def next_offset(self) -> float:
        """
        Calculate the offset for the next request at a constant rate.

        Each request is scheduled at a fixed interval based on the target rate,
        with offsets increasing linearly: 0, 1/rate, 2/rate, 3/rate, etc.

        :return: The offset in seconds for the next request.
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._requests_count += 1
        interval = 1.0 / self.rate

        if (
            self.startup_duration > 0
            and time.time() - self._start_time < self.startup_duration
        ):
            # Adjust the rate during the startup phase to exponentially decay
            # to the desired rate
            tau = (self.rate * self.startup_duration) / (
                -1 * math.log(0.01)
            )  # target 99% convergence
            adjust_fraction = 1 - math.exp(-self._requests_count / tau)
            interval *= adjust_fraction

        return self.offset + interval * self._requests_count

    def request_completed(self, request_info: ScheduledRequestInfo):
        """
        Handle request completion (no action needed for constant rate strategy).

        :param request_info: Information about the completed request (unused).
        """


class PoissonRateRequestTimings(ScheduledRequestTimings):
    """
    Timing implementation for Poisson-distributed scheduling strategies.

    This implementation schedules requests following a Poisson process with
    exponentially distributed inter-arrival times. The average rate is specified
    in requests per second, but individual intervals vary randomly according to
    the exponential distribution, simulating realistic traffic patterns.
    """

    rate: float = Field(
        description="The target average rate in requests per second. Must be positive.",
        gt=0,
    )
    random_seed: int = Field(
        default=42,
        description=(
            "Seed for the random number generator to ensure reproducible behavior."
        ),
    )
    offset: float = Field(
        default=0.0,
        description="The cumulative time offset in seconds from scheduler start time.",
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "to converge quickly to the desired rate before switching to "
            "constant-rate scheduling."
        ),
        ge=0,
    )
    _start_time: float = Field(
        default=None,
        description="The start time of the request processing phase.",
    )
    _requests_count: int = Field(
        default=0,
        description="Internal counter tracking the number of offsets generated.",
        ge=0,
    )
    _random: Any = Field(
        default=None,
        description="Random number generator instance for Poisson distribution.",
        exclude=True,  # Don't include in serialization
    )

    def next_offset(self) -> float:
        """
        Calculate the offset for the next request using Poisson distribution.

        Uses exponential distribution to generate inter-arrival times that
        follow a Poisson process. Each call advances the cumulative offset
        by a randomly generated delay.

        :return: The cumulative offset in seconds for the next request.
        """
        if self._start_time is None:
            self._start_time = time.time()

        if self._random is None:
            self._random = random.Random(self.random_seed)

        self._requests_count += 1

        next_delay = self._random.expovariate(self.rate)

        if (
            self.startup_duration > 0
            and time.time() - self._start_time < self.startup_duration
        ):
            # Adjust the rate during the startup phase to exponentially decay
            # to the desired rate
            tau = (self.rate * self.startup_duration) / (
                -1 * math.log(0.01)
            )  # target 99% convergence
            adjust_fraction = 1 - math.exp(-self._requests_count / tau)
            next_delay *= adjust_fraction

        self.offset += next_delay

        return self.offset

    def request_completed(self, request_info: ScheduledRequestInfo):
        """
        Handle request completion (no action needed for Poisson rate strategy).

        :param request_info: Information about the completed request (unused).
        """


class SchedulingStrategy(StandardBaseModel):
    """
    An abstract base class for scheduling strategies enabling control over how
    requests are processed by the scheduler.
    """

    type_: Literal["strategy"] = Field(
        description="The type of scheduling strategy to schedule requests with.",
    )

    @property
    def processes_limit(self) -> Optional[int]:
        """
        :return: The maximum number of worker processes supported by the
            scheduling strategy. None if not limited.
        """
        return None

    @property
    def requests_limit(self) -> Optional[int]:
        """
        :return: The maximum number of concurrent requests that can be processed
            at once by the scheduling strategy. None if not limited.
        """
        return None

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create a ScheduledRequestTimings instance to define the timing behavior
        for the worker process to schedule requests.

        :param local_rank: The rank of the worker process within the local world size.
        :param local_world_size: The total num of worker processes in the local world.
        :param local_max_concurrency: The maximum number of concurrent requests
            for the worker process.
        :return: A ScheduledRequestTimings instance for the worker process.
        """
        raise NotImplementedError(
            "create_worker_timings method must be implemented by subclasses."
        )


class SynchronousStrategy(SchedulingStrategy):
    """
    Sequential request processing strategy with maximum throughput constraints.

    This strategy processes requests one at a time in strict sequential order,
    waiting for each request to complete before starting the next. It provides
    the most predictable timing behavior and is useful for measuring maximum
    achievable throughput under sequential processing constraints.

    The strategy enforces a limit of one worker process and one concurrent request,
    making it ideal for scenarios where request ordering and isolation are critical.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @property
    def processes_limit(self) -> Optional[int]:
        """
        Get the maximum number of worker processes for synchronous scheduling.

        :return: Always returns 1 to enforce single-process constraint.
        """
        return 1

    @property
    def requests_limit(self) -> Optional[int]:
        """
        Get the maximum number of concurrent requests for synchronous scheduling.

        :return: Always returns 1 to enforce single-request constraint.
        """
        return 1

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for synchronous request scheduling.

        :param local_rank: The rank of the worker process. Must be 0.
        :param local_world_size: Total number of worker processes. Must be 1.
        :param local_max_concurrency: The maximum number of concurrent requests
            for the worker process.
        :return: LastCompletionRequestTimings instance for sequential processing.
        :raises ValueError: If multiple workers or non-zero rank is specified.
        """
        if local_world_size > 1 or local_rank != 0:
            raise ValueError(
                "SynchronousStrategy can only be used with a single worker process."
            )

        return LastCompletionRequestTimings()


class ConcurrentStrategy(SchedulingStrategy):
    """
    Parallel request processing strategy with controlled concurrency limits.

    This strategy enables concurrent request processing up to a specified number
    of streams, allowing multiple requests to be processed simultaneously while
    maintaining predictable resource usage. It provides a balance between
    throughput and resource control.

    The number of concurrent streams determines both the maximum number of worker
    processes and the maximum number of requests that can be processed in parallel.
    Each worker process handles one stream and waits for request completion before
    processing the next request in that stream.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: int = Field(
        description=(
            "The number of concurrent streams to use for scheduling requests. "
            "This must be a positive integer."
        ),
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "before switching to completion-based timing."
        ),
        ge=0,
    )

    @property
    def processes_limit(self) -> int:
        """
        Get the maximum number of worker processes for concurrent scheduling.

        :return: The number of streams, which equals the maximum worker processes.
        """
        return self.streams

    @property
    def requests_limit(self) -> int:
        """
        Get the maximum number of concurrent requests for concurrent scheduling.

        :return: The number of streams, which equals the maximum concurrent requests.
        """
        return self.streams

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> LastCompletionRequestTimings:
        """
        Create timing implementation for concurrent request scheduling.

        :param local_rank: The rank of the worker process. Must be less than streams.
        :param local_world_size: Total number of worker processes. Must not exceed
            streams.
        :param local_max_concurrency: The maximum number of concurrent requests
            for the worker process.
        :return: LastCompletionRequestTimings instance for stream-based processing.
        :raises ValueError: If worker configuration exceeds stream limits.
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


class ThroughputStrategy(SchedulingStrategy):
    """
    Maximum throughput strategy with optional concurrency limits.

    This strategy schedules requests to maximize system throughput by allowing
    unlimited concurrent request processing. Requests are scheduled immediately
    without waiting for previous requests to complete, enabling the system to
    achieve its maximum processing capacity.

    An optional maximum concurrency limit can be set to prevent resource
    exhaustion while still allowing high-throughput processing patterns.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of concurrent requests to schedule. "
            "This must be a positive integer greater than 0."
        ),
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "before switching to full throughput scheduling."
        ),
        ge=0,
    )

    @property
    def processes_limit(self) -> Optional[int]:
        """
        Get the maximum number of worker processes for throughput scheduling.

        :return: The max_concurrency value if set, otherwise None for unlimited
            worker processes.
        """
        return self.max_concurrency

    @property
    def requests_limit(self) -> Optional[int]:
        """
        Get the maximum number of concurrent requests for throughput scheduling.

        :return: The max_concurrency value if set, otherwise None for unlimited
            concurrent requests.
        """
        return self.max_concurrency

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for throughput request scheduling.

        :param local_rank: The rank of the worker process (unused for throughput).
        :param local_world_size: Total number of worker processes (unused for
            throughput).
        :return: NoDelayRequestTimings instance for immediate request scheduling.
        """
        if self.startup_duration > 0:
            # Vary offset by up to 5% of the startup duration for a bit of variance
            offset = 0.05 * self.startup_duration * (local_rank / local_world_size)
            # set convergence of tau to target reaching 99% of the startup_duration
            # at local_max_concurrency
            tau = local_max_concurrency / (-1 * math.log(0.01))
        else:
            offset = 0.0
            tau = 1.0

        return NoDelayRequestTimings(
            startup_duration=self.startup_duration,
            startup_tau=tau,
            offset=offset,
        )


class AsyncConstantStrategy(ThroughputStrategy):
    """
    Asynchronous constant-rate scheduling strategy for predictable load patterns.

    This strategy schedules requests at a fixed rate specified in requests per
    second, distributed evenly across all worker processes. It provides predictable
    timing behavior while allowing asynchronous processing, making it ideal for
    simulating steady-state load conditions and measuring system performance
    under consistent request rates.

    The total rate is divided equally among all worker processes, ensuring the
    aggregate rate matches the specified value regardless of the number of workers.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "to converge quickly to the desired rate before switching to "
            "constant-rate scheduling."
        ),
        ge=0,
    )

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for constant-rate request scheduling.

        Divides the total rate evenly across all worker processes to maintain
        the specified aggregate rate.

        :param local_rank: The rank of the worker process (unused for rate calculation).
        :param local_world_size: Total number of worker processes for rate division.
        :param local_max_concurrency: The maximum number of concurrent requests
            for the worker process.
        :return: ConstantRateRequestTimings instance with per-worker rate.
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size

        return ConstantRateRequestTimings(
            rate=worker_rate,
            startup_duration=self.startup_duration,
        )


class AsyncPoissonStrategy(ThroughputStrategy):
    """
    Asynchronous Poisson-distributed scheduling strategy for realistic load simulation.

    This strategy schedules requests following a Poisson process with exponentially
    distributed inter-arrival times. The average rate is specified in requests per
    second, but individual intervals vary randomly, providing a more realistic
    simulation of user behavior and network traffic patterns.

    The total rate is divided equally among all worker processes, with each worker
    using a different random seed to ensure independent request streams that
    collectively achieve the target rate.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    startup_duration: float = Field(
        default=0.0,
        description=(
            "Duration in seconds over which startup requests are distributed "
            "to converge quickly to the desired rate before switching to "
            "constant-rate scheduling."
        ),
        ge=0,
    )
    random_seed: int = Field(
        default=42,
        description=("The random seed to use for the Poisson distribution."),
    )

    def create_worker_timings(
        self, local_rank: int, local_world_size: int, local_max_concurrency: int
    ) -> ScheduledRequestTimings:
        """
        Create timing implementation for Poisson-distributed request scheduling.

        Divides the total rate evenly across all worker processes and assigns
        unique random seeds to ensure independent but coordinated request streams.

        :param local_rank: The rank of the worker process for seed generation.
        :param local_world_size: Total number of worker processes for rate division.
        :param local_max_concurrency: The maximum number of concurrent requests
            for the worker process.
        :return: PoissonRateRequestTimings instance with per-worker rate and
            unique seed.
        """
        # Divide the rate evenly across all worker processes
        worker_rate = self.rate / local_world_size
        # Use a different seed for each worker to ensure different sequences
        worker_seed = self.random_seed + local_rank
        return PoissonRateRequestTimings(
            rate=worker_rate,
            random_seed=worker_seed,
            startup_duration=self.startup_duration,
        )


def strategy_display_str(strategy: Union[StrategyType, SchedulingStrategy]) -> str:
    """
    Generate a human-readable string representation of a scheduling strategy.

    Creates concise string representations that include the strategy type and
    relevant configuration parameters (e.g., rate for async strategies, streams
    for concurrent strategies). Useful for logging, debugging, and user interfaces.

    :param strategy: A strategy type string or SchedulingStrategy instance to format.
    :return: A formatted string representation of the strategy with configuration
        details when available.

    Examples:
        >>> strategy_display_str("synchronous")
        "synchronous"
        >>> strategy_display_str(ConcurrentStrategy(streams=4))
        "concurrent@4"
        >>> strategy_display_str(AsyncConstantStrategy(rate=10.5))
        "constant@10.50"
    """
    strategy_type = strategy if isinstance(strategy, str) else strategy.type_
    strategy_instance = strategy if isinstance(strategy, SchedulingStrategy) else None

    if strategy_type == "concurrent":
        rate = f"@{strategy_instance.streams}" if strategy_instance else "@##"  # type: ignore[attr-defined]
    elif strategy_type in ("constant", "poisson"):
        rate = f"@{strategy_instance.rate:.2f}" if strategy_instance else "@#.##"  # type: ignore[attr-defined]
    else:
        rate = ""

    return f"{strategy_type}{rate}"
