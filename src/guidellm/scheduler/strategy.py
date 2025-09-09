import math
import random
import time
from collections.abc import Generator
from typing import (
    Literal,
    Optional,
    Union,
)

from pydantic import Field

from guidellm.config import settings
from guidellm.objects import StandardBaseModel

__all__ = [
    "AsyncConstantStrategy",
    "AsyncPoissonStrategy",
    "ConcurrentStrategy",
    "SchedulingStrategy",
    "StrategyType",
    "SynchronousStrategy",
    "ThroughputStrategy",
    "strategy_display_str",
]


StrategyType = Literal["synchronous", "concurrent", "throughput", "constant", "poisson"]


class SchedulingStrategy(StandardBaseModel):
    """
    An abstract base class for scheduling strategies.
    This class defines the interface for scheduling requests and provides
    a common structure for all scheduling strategies.
    Subclasses should implement the `request_times` method to provide
    specific scheduling behavior.

    :param type_: The type of scheduling strategy to use.
        This should be one of the predefined strategy types.
    """

    type_: Literal["strategy"] = Field(
        description="The type of scheduling strategy schedule requests with.",
    )
    start_time: float = Field(
        default_factory=time.time,
        description="The start time for the scheduling strategy.",
    )

    @property
    def processing_mode(self) -> Literal["sync", "async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.
        This property should be implemented by subclasses to return
        the appropriate processing mode.

        :return: The processing mode for the scheduling strategy,
            either 'sync' or 'async'.
        """
        return "async"

    @property
    def processes_limit(self) -> int:
        """
        The limit on the number of worker processes for the scheduling strategy.
        It determines how many worker processes are created
        for the scheduling strategy and must be implemented by subclasses.

        :return: The number of processes for the scheduling strategy.
        """
        return settings.max_worker_processes

    @property
    def queued_requests_limit(self) -> Optional[int]:
        """
        The maximum number of queued requests for the scheduling strategy.
        It determines how many requests can be queued at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: The maximum number of queued requests for the scheduling strategy.
        """
        return settings.max_concurrency

    @property
    def processing_requests_limit(self) -> int:
        """
        The maximum number of processing requests for the scheduling strategy.
        It determines how many requests can be processed at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: The maximum number of processing requests for the scheduling strategy.
        """
        return settings.max_concurrency

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method should be implemented by subclasses to provide specific
        scheduling behavior.

        :return: A generator that yields timestamps for request scheduling
            or -1 for requests that should be sent immediately.
        """
        raise NotImplementedError("Subclasses must implement request_times() method.")


class SynchronousStrategy(SchedulingStrategy):
    """
    A class representing a synchronous scheduling strategy.
    This strategy schedules requests synchronously, one at a time,
    with the maximum rate possible.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for synchronous scheduling.

    :param type_: The synchronous StrategyType to schedule requests synchronously.
    """

    type_: Literal["synchronous"] = "synchronous"  # type: ignore[assignment]

    @property
    def processing_mode(self) -> Literal["sync"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'sync' for synchronous scheduling strategy
            for the single worker process.
        """
        return "sync"

    @property
    def processes_limit(self) -> int:
        """
        The limit on the number of worker processes for the scheduling strategy.
        It determines how many worker processes are created
        for the scheduling strategy and must be implemented by subclasses.

        :return: 1 for the synchronous scheduling strategy to limit
            the worker processes to one.
        """
        return 1

    @property
    def queued_requests_limit(self) -> int:
        """
        The maximum number of queued requests for the scheduling strategy.
        It determines how many requests can be queued at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: 1 for the synchronous scheduling strategy to limit
            the queued requests to one that is ready to be processed.
        """
        return 1

    @property
    def processing_requests_limit(self) -> int:
        """
        The maximum number of processing requests for the scheduling strategy.
        It determines how many requests can be processed at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: 1 for the synchronous scheduling strategy to limit
            the processing requests to one that is ready to be processed.
        """
        return 1

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields time.time() so requests are sent immediately,
            while scheduling them synchronously.

        :return: A generator that yields time.time() for immediate request scheduling.
        """
        init_time = self.start_time
        while True:
            yield max(init_time, time.time())


class ConcurrentStrategy(SchedulingStrategy):
    """
    A class representing a concurrent scheduling strategy.
    This strategy schedules requests concurrently with the specified
    number of streams.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for concurrent scheduling.

    :param type_: The concurrent StrategyType to schedule requests concurrently.
    :param streams: The number of concurrent streams to use for scheduling requests.
        Each stream runs synchronously with the maximum rate possible.
        This must be a positive integer.
    """

    type_: Literal["concurrent"] = "concurrent"  # type: ignore[assignment]
    streams: int = Field(
        description=(
            "The number of concurrent streams to use for scheduling requests. "
            "Each stream runs sychronously with the maximum rate possible. "
            "This must be a positive integer."
        ),
        gt=0,
    )

    @property
    def processing_mode(self) -> Literal["sync"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'sync' for synchronous scheduling strategy
            for the multiple worker processes equal to streams.
        """
        return "sync"

    @property
    def processes_limit(self) -> int:
        """
        The limit on the number of worker processes for the scheduling strategy.
        It determines how many worker processes are created
        for the scheduling strategy and must be implemented by subclasses.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the worker processes to the number of streams.
        """

        return min(self.streams, settings.max_worker_processes)

    @property
    def queued_requests_limit(self) -> int:
        """
        The maximum number of queued requests for the scheduling strategy.
        It determines how many requests can be queued at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the queued requests to the number of streams that are ready to be processed.
        """
        return self.streams

    @property
    def processing_requests_limit(self) -> int:
        """
        The maximum number of processing requests for the scheduling strategy.
        It determines how many requests can be processed at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: {self.streams} for the concurrent scheduling strategy to limit
            the processing requests to the number of streams that ready to be processed.
        """
        return self.streams

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields time.time() so requests are sent
        immediately, while scheduling them concurrently with the specified
        number of streams.

        :return: A generator that yields time.time() for immediate request scheduling.
        """
        init_time = self.start_time
        while True:
            yield max(init_time, time.time())


class ThroughputStrategy(SchedulingStrategy):
    """
    A class representing a throughput scheduling strategy.
    This strategy schedules as many requests asynchronously as possible,
    with the maximum rate possible.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for throughput scheduling.

    :param type_: The throughput StrategyType to schedule requests asynchronously.
    """

    type_: Literal["throughput"] = "throughput"  # type: ignore[assignment]
    max_concurrency: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of concurrent requests to schedule. "
            "If set to None, the concurrency value from settings will be used. "
            "This must be a positive integer greater than 0."
        ),
        gt=0,
    )

    @property
    def processing_mode(self) -> Literal["async"]:
        """
        The processing mode for the scheduling strategy, either 'sync' or 'async'.
        This property determines how the worker processes are setup:
        either to run synchronously with one request at a time or asynchronously.

        :return: 'async' for asynchronous scheduling strategy
            for the multiple worker processes handling requests.
        """
        return "async"

    @property
    def queued_requests_limit(self) -> int:
        """
        The maximum number of queued requests for the scheduling strategy.
        It determines how many requests can be queued at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: The processing requests limit to ensure that there are enough
            requests even for the worst case scenario where the max concurrent
            requests are pulled at once for processing.
        """
        return self.processing_requests_limit

    @property
    def processing_requests_limit(self) -> int:
        """
        The maximum number of processing requests for the scheduling strategy.
        It determines how many requests can be processed at one time
        for the scheduling strategy and must be implemented by subclasses.

        :return: {self.max_concurrency} for the throughput scheduling strategy to limit
            the processing requests to the maximum concurrency.
            If max_concurrency is None, then the default processing requests limit
            will be used.
        """
        return self.max_concurrency or super().processing_requests_limit

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields the start time.time() so requests are sent
        immediately, while scheduling as many asynchronously as possible.

        :return: A generator that yields the start time.time()
            for immediate request scheduling.
        """
        init_time = self.start_time
        while True:
            yield init_time


class AsyncConstantStrategy(ThroughputStrategy):
    """
    A class representing an asynchronous constant scheduling strategy.
    This strategy schedules requests asynchronously at a constant request rate
    in requests per second.
    If initial_burst is set, it will send an initial burst of math.floor(rate)
    requests to reach the target rate.
    This is useful to ensure that the target rate is reached quickly
    and then maintained.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for asynchronous constant scheduling.

    :param type_: The constant StrategyType to schedule requests asynchronously.
    :param rate: The rate at which to schedule requests asynchronously in
        requests per second. This must be a positive float.
    :param initial_burst: True to send an initial burst of requests
        (math.floor(self.rate)) to reach target rate.
        False to not send an initial burst.
    """

    type_: Literal["constant"] = "constant"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    initial_burst: bool = Field(
        default=True,
        description=(
            "True to send an initial burst of requests (math.floor(self.rate)) "
            "to reach target rate. False to not send an initial burst."
        ),
    )

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method schedules requests asynchronously at a constant rate
        in requests per second.
        If burst_time is set, it will send an initial burst of requests
        to reach the target rate.
        This is useful to ensure that the target rate is reached quickly
        and then maintained.

        :return: A generator that yields timestamps for request scheduling.
        """
        constant_increment = 1.0 / self.rate

        init_time = self.start_time
        # handle bursts first to get to the desired rate
        if self.initial_burst is not None:
            # send an initial burst equal to the rate
            # to reach the target rate
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield init_time

            init_time += constant_increment

        counter = 0

        # continue with constant rate after bursting
        while True:
            yield init_time + constant_increment * counter
            counter += 1


class AsyncPoissonStrategy(ThroughputStrategy):
    """
    A class representing an asynchronous Poisson scheduling strategy.
    This strategy schedules requests asynchronously at a Poisson request rate
    in requests per second.
    If initial_burst is set, it will send an initial burst of math.floor(rate)
    requests to reach the target rate.
    It inherits from the `SchedulingStrategy` base class and
    implements the `request_times` method to provide the specific
    behavior for asynchronous Poisson scheduling.

    :param type_: The Poisson StrategyType to schedule requests asynchronously.
    :param rate: The rate at which to schedule requests asynchronously in
        requests per second. This must be a positive float.
    :param initial_burst: True to send an initial burst of requests
        (math.floor(self.rate)) to reach target rate.
        False to not send an initial burst.
    """

    type_: Literal["poisson"] = "poisson"  # type: ignore[assignment]
    rate: float = Field(
        description=(
            "The rate at which to schedule requests asynchronously in "
            "requests per second. This must be a positive float."
        ),
        gt=0,
    )
    initial_burst: bool = Field(
        default=True,
        description=(
            "True to send an initial burst of requests (math.floor(self.rate)) "
            "to reach target rate. False to not send an initial burst."
        ),
    )
    random_seed: int = Field(
        default=42,
        description=("The random seed to use for the Poisson distribution. "),
    )

    def request_times(self) -> Generator[float, None, None]:
        """
        A generator that yields timestamps for when requests should be sent.
        This method schedules requests asynchronously at a Poisson rate
        in requests per second.
        The inter arrival time between requests is exponentially distributed
        based on the rate.

        :return: A generator that yields timestamps for request scheduling.
        """
        init_time = self.start_time
        if self.initial_burst is not None:
            # send an initial burst equal to the rate
            # to reach the target rate
            burst_count = math.floor(self.rate)
            for _ in range(burst_count):
                yield init_time
        else:
            yield init_time

        # set the random seed for reproducibility
        rand = random.Random(self.random_seed)  # noqa: S311

        while True:
            inter_arrival_time = rand.expovariate(self.rate)
            init_time += inter_arrival_time
            yield init_time


def strategy_display_str(strategy: Union[StrategyType, SchedulingStrategy]) -> str:
    strategy_type = strategy if isinstance(strategy, str) else strategy.type_
    strategy_instance = strategy if isinstance(strategy, SchedulingStrategy) else None

    if strategy_type == "concurrent":
        rate = f"@{strategy_instance.streams}" if strategy_instance else "@##"  # type: ignore[attr-defined]
    elif strategy_type in ("constant", "poisson"):
        rate = f"@{strategy_instance.rate:.2f}" if strategy_instance else "@#.##"  # type: ignore[attr-defined]
    else:
        rate = ""

    return f"{strategy_type}{rate}"
