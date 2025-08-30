"""
Environment abstractions for coordinating scheduler execution across distributed nodes.

Provides environment abstractions that handle synchronization, timing coordination,
error propagation, and lifecycle management for scheduler execution across single
or multiple nodes. The Environment protocol defines the interface for distributed
coordination while NonDistributedEnvironment provides a minimal implementation
for single-node execution.

Environment Execution Flow:
1. sync_run_params() - Distribute workload and synchronize parameters across nodes
2. sync_run_start() - Coordinate synchronized start time for all nodes
3. update_run_iteration() - Update state after each request (called per iteration)
4. sync_run_error() - Handle and propagate errors across nodes
5. sync_run_end() - Aggregate results and cleanup at completion
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import (
    Generic,
)

from guidellm.scheduler.constraints import Constraint
from guidellm.scheduler.objects import (
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.settings import settings
from guidellm.utils import InfoMixin

__all__ = ["Environment", "NonDistributedEnvironment"]


class Environment(ABC, Generic[RequestT, ResponseT], InfoMixin):
    """
    Abstract base for coordinating scheduler execution across distributed nodes.

    Defines the interface for managing distributed scheduler execution including
    parameter synchronization, timing coordination, state updates, error propagation,
    and result aggregation. Implementations handle the complexity of distributed
    coordination while providing a unified interface for scheduler orchestration.
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

        Coordinates parameter distribution and validation across active nodes.
        In distributed environments, handles node assignment and workload partitioning.
        In non-distributed environments, typically returns parameters unchanged.

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

        Ensures all nodes begin processing simultaneously for accurate benchmarking
        and consistent timing measurements across distributed execution.

        :return: Unix timestamp when all nodes should begin processing
        :raises Exception: If startup synchronization fails across nodes
        """
        ...

    @abstractmethod
    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo,
        state: SchedulerState,
    ):
        """
        Update environment state with completed request iteration results.

        Called after each request processing to update execution progress and
        synchronize any required state across nodes in distributed environments.
        Generally, distributed is expected to store the iteration updates until
        all nodes have processed and sync_run_end is called to retrieve them.

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

        Coordinates error handling when failures occur, ensuring all nodes are
        notified for appropriate cleanup or shutdown procedures.

        :param err: The exception(s) that occurred during execution
        """
        ...

    @abstractmethod
    async def sync_run_end(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT,
            RequestT | MultiTurnRequestT[RequestT],
            ScheduledRequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Finalize execution and aggregate results from all nodes.

        Handles cleanup, result synchronization, and error propagation at execution
        completion. Collects and yields results from worker nodes in distributed
        environments.

        :return: Iterator of (response, request, request_info, state) tuples from
            remote nodes in distributed environments, empty for non-distributed
        :raises Exception: Any errors that occurred during execution
        """
        ...


class NonDistributedEnvironment(Environment):
    """
    Single-node scheduler execution environment with minimal coordination overhead.

    Simplified environment for running schedulers on a single node without distributed
    coordination requirements. Implements the Environment interface with no-op
    synchronization for local testing, development, and single-machine benchmarking.

    Example:
    ::
        from guidellm.scheduler import (
            MaxNumberConstraint,
            NonDistributedEnvironment,
            ScheduledRequestInfo,
            SchedulerState,
            SynchronousStrategy,
        )


        # Definitions
        requests = [f"req_{ind}" for ind in range(5)]
        strategy = SynchronousStrategy()
        constraints = {"max_num": MaxNumberConstraint(max_num=5)}
        state = SchedulerState()

        # Run environment
        local_req, local_strat, local_const = await env.sync_run_params(
            requests, strategy, constraints
        )
        start_time = await env.sync_run_start()
        for req in local_req:
            state.processed_requests += 1
            await env.update_run_iteration(
                f"resp_{req}", req, ScheduledRequestInfo(), state
            )
        async for nonlocal_req in env.sync_run_end():
            state.processed_requests += 1
    """

    def __init__(self):
        """Initialize with empty error storage for single-node execution."""
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
        :return: Tuple containing the original (requests, strategy, constraints)
        """
        return requests, strategy, constraints

    async def sync_run_start(self) -> float:
        """
        Return current time plus configured delay for single-node startup.

        :return: Unix timestamp for when the run should start
        """
        return time.time() + settings.scheduler_start_delay_non_distributed

    async def update_run_iteration(
        self,
        response: ResponseT | None,
        request: RequestT,
        request_info: ScheduledRequestInfo,
        state: SchedulerState,
    ):
        """
        No-op for single-node execution with no distributed state synchronization.

        :param response: Response generated for the request, if successful
        :param request: The request that was processed
        :param request_info: Metadata about request processing including timings
        :param state: Current scheduler state with metrics and progress
        """

    async def sync_run_error(self, err: Exception):
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
            ResponseT,
            RequestT | MultiTurnRequestT[RequestT],
            ScheduledRequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Finalize single-node execution and propagate any stored errors.

        :return: Empty iterator since there are no remote nodes
        :raises Exception: Any error stored during execution via sync_run_error
        """
        if self.run_errors:
            if len(self.run_errors) == 1:
                raise self.run_errors[0]
            else:
                raise RuntimeError(
                    f"Errors occurred during execution: {self.run_errors}"
                )

        return
        yield  # needed to force generator compilation
