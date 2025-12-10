"""
Environment abstractions for coordinating scheduler execution across distributed nodes.

Provides abstractions that handle synchronization, timing coordination, error
propagation, and lifecycle management for scheduler execution across single or
multiple nodes. The Environment protocol defines the interface for distributed
coordination while NonDistributedEnvironment provides a minimal implementation
for single-node execution. Environments manage the complete execution lifecycle
from parameter distribution through result aggregation.

Execution Flow:
1. sync_run_params() - Distribute workload and synchronize parameters
2. sync_run_start() - Coordinate synchronized start time
3. update_run_iteration() - Update state after each request iteration
4. sync_run_error() - Handle and propagate errors across nodes
5. sync_run_end() - Aggregate results and finalize execution
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Generic

from guidellm.scheduler.constraints import Constraint
from guidellm.scheduler.schemas import (
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    SchedulerState,
)
from guidellm.scheduler.strategies import SchedulingStrategy
from guidellm.schemas import RequestInfo
from guidellm.settings import settings
from guidellm.utils import InfoMixin

__all__ = ["Environment", "NonDistributedEnvironment"]


class Environment(ABC, Generic[RequestT, ResponseT], InfoMixin):
    """
    Abstract interface for coordinating scheduler execution across distributed nodes.

    Defines the protocol for managing distributed scheduler execution including
    parameter synchronization, timing coordination, state updates, error propagation,
    and result aggregation. Implementations handle distributed coordination complexity
    while providing a unified interface for scheduler orchestration.
    """

    @abstractmethod
    async def sync_run_params(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
    ) -> tuple[
        Iterable[RequestT | MultiTurnRequestT[RequestT]],
        SchedulingStrategy,
        dict[str, Constraint],
    ]:
        """
        Synchronize execution parameters across nodes and resolve local scope.

        :param requests: Complete set of requests to process across all nodes
        :param strategy: Scheduling strategy to apply during execution
        :param constraints: Runtime constraints to enforce during execution
        :return: Tuple of (local_requests, strategy, constraints) for this node
        :raises Exception: If parameter synchronization fails or nodes inconsistent
        """
        ...

    @abstractmethod
    async def sync_run_start(self) -> float:
        """
        Coordinate synchronized start time across all nodes.

        :return: Unix timestamp when all nodes should begin processing
        :raises Exception: If startup synchronization fails across nodes
        """
        ...

    @abstractmethod
    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT | MultiTurnRequestT[RequestT],
        request_info: RequestInfo,
        state: SchedulerState,
    ):
        """
        Update environment state with completed request iteration results.

        :param response: Response generated for the request, if successful
        :param request: The processed request
        :param request_info: Metadata about request processing including timings
        :param state: Current scheduler state with metrics and progress
        :raises Exception: If state update fails or indicates critical errors
        """
        ...

    @abstractmethod
    async def sync_run_error(self, err: list[Exception] | Exception):
        """
        Handle and propagate errors across all active nodes.

        :param err: The exception(s) that occurred during execution
        """
        ...

    @abstractmethod
    async def sync_run_end(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT | None,
            RequestT,
            RequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Finalize execution and aggregate results from all nodes.

        :return: Iterator of (response, request, request_info, state) tuples from
            remote nodes in distributed environments, empty for non-distributed
        :raises Exception: Any errors that occurred during execution
        """
        yield None  # type: ignore[misc]


class NonDistributedEnvironment(Environment[RequestT, ResponseT]):
    """
    Single-node scheduler execution environment with minimal coordination overhead.

    Implements the Environment interface with no-op synchronization for local testing,
    development, and single-machine benchmarking. All synchronization methods return
    immediately without distributed coordination logic.

    Example:
    ::
        from guidellm.scheduler import (
            MaxNumberConstraint,
            NonDistributedEnvironment,
            RequestInfo,
            SchedulerState,
            SynchronousStrategy,
        )

        env = NonDistributedEnvironment()
        requests = [f"req_{ind}" for ind in range(5)]
        strategy = SynchronousStrategy()
        constraints = {"max_num": MaxNumberConstraint(max_num=5)}
        state = SchedulerState()

        local_req, local_strat, local_const = await env.sync_run_params(
            requests, strategy, constraints
        )
        start_time = await env.sync_run_start()
        for req in local_req:
            state.processed_requests += 1
            await env.update_run_iteration(f"resp_{req}", req, RequestInfo(), state)
        async for nonlocal_req in env.sync_run_end():
            state.processed_requests += 1
    """

    def __init__(self):
        """
        Initialize single-node environment with empty error storage.
        """
        self.run_errors: list[Exception] = []

    async def sync_run_params(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
    ) -> tuple[
        Iterable[RequestT | MultiTurnRequestT[RequestT]],
        SchedulingStrategy,
        dict[str, Constraint],
    ]:
        """
        Return parameters unchanged for single-node execution.

        :param requests: Requests to process locally
        :param strategy: Scheduling strategy to apply during execution
        :param constraints: Runtime constraints to enforce during execution
        :return: Original (requests, strategy, constraints) tuple unchanged
        """
        return requests, strategy, constraints

    async def sync_run_start(self) -> float:
        """
        Return current time plus configured delay for single-node startup.

        :return: Unix timestamp when execution should begin
        """
        return time.time() + settings.scheduler_start_delay_non_distributed

    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT | MultiTurnRequestT[RequestT],
        request_info: RequestInfo,
        state: SchedulerState,
    ):
        """
        No-op for single-node execution with no distributed state synchronization.

        :param response: Response generated for the request, if successful
        :param request: The processed request
        :param request_info: Metadata about request processing including timings
        :param state: Current scheduler state with metrics and progress
        """

    async def sync_run_error(self, err: Exception | list[Exception]):
        """
        Store error for later propagation during run finalization.

        :param err: The exception(s) that occurred during execution
        """
        err = [err] if not isinstance(err, list) else err
        self.run_errors.extend(err)

    async def sync_run_end(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT | None,
            RequestT,
            RequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Finalize single-node execution and propagate any stored errors.

        :return: Empty iterator as there are no remote nodes
        :raises Exception: Any error stored during execution via sync_run_error
        """
        if self.run_errors:
            if len(self.run_errors) == 1:
                raise self.run_errors[0]
            else:
                raise RuntimeError(
                    f"Errors occurred during execution: {self.run_errors}"
                )

        if False:
            # Force compiler to recognize as generator
            yield None  # type: ignore[misc]
