"""
Scheduler for coordinating distributed load testing and benchmarking workloads.

This module provides the main Scheduler class responsible for orchestrating benchmarking
and evaluation operations across multiple worker processes and environments.
The scheduler manages request distribution, timing coordination,
and result aggregation while ensuring thread-safe singleton behavior.

Classes:
    Scheduler: Singleton scheduler for coordinating distributed request processing
        workloads with generic support for backends, request types, and responses.
"""

import threading
from collections.abc import AsyncIterator, Iterable
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Union,
)

from guidellm.scheduler.constraints import ConstraintsFactory
from guidellm.scheduler.environment import Environment
from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker_group import WorkerProcessGroup
from guidellm.utils.singleton import ThreadSafeSingletonMixin

__all__ = ["Scheduler"]


class Scheduler(
    Generic[BackendT, RequestT, RequestTimingsT, ResponseT], ThreadSafeSingletonMixin
):
    """
    A generic singleton scheduler for coordinating distributed load testing workloads.

    The Scheduler orchestrates benchmarking operations by managing request distribution
    across worker processes, coordinating timing with distributed environments, and
    aggregating results. It implements the singleton pattern to ensure consistent
    state management across the application.

    The scheduler supports generic backend types, request formats, and response types,
    making it adaptable to various testing scenarios including LLM inference,
    API testing, and other distributed workload patterns.

    Example:
    ::
        from guidellm.scheduler import Scheduler
        from guidellm.backend import (
            OpenAIBackend,
            GenerationRequest,
            GenerationResponse,
            GenerationRequestTimings
        )

        scheduler = Scheduler[
            OpenAIBackend,GenerationRequest,GenerationRequestTimings,GenerationResponse
        ]()
        async for response, request, info, state in scheduler.run(
            requests=request_list,
            backend=backend,
            strategy=strategy,
            env=environment,
            max_requests=1000
        ):
            print(f"Resp: {response}, Req: {request}, Info: {info}, State: {state}")
    """

    def __init__(self):
        """
        Initialize the scheduler singleton instance.
        """
        if not self.initialized:
            self.run_lock = threading.Lock()
        super().__init__()

    async def run(
        self,
        requests: Iterable[
            Union[RequestT, Iterable[Union[RequestT, tuple[RequestT, float]]]]
        ],
        backend: BackendT[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        env: Environment,
        **constraints: dict[
            str, Union[int, float, str, Callable[[SchedulerState], Any]]
        ],
    ) -> AsyncIterator[
        tuple[
            Optional[ResponseT],
            RequestT,
            ScheduledRequestInfo[RequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Execute a request processing run with the provided configuration.

        Coordinates the execution of requests across worker processes with the specified
        backend and scheduling strategy, on the targeted environment, and until
        completion as defined by the constraints.
        It does this while managing timing, synchronization, and resource cleanup.
        The method yields request updates as they become available,
        including requeust queued, request processing start, and request completion,
        allowing for real-time monitoring and processing.

        :param requests: Iterable of the requests to process with multiple formats;
            Iterable[RequestT] for single requests,
            Iterable[Iterable] for multi-turn requests where each item is either
            a RequestT for immediately processing the next request in the sequence,
            or a tuple of (RequestT, float) where the float is the delay in seconds
            before processing the next request in the sequence.
        :param backend: Backend instance for processing requests, must be compatible
            with the request and response types.
        :param strategy: Scheduling strategy defining how requests are timed
            for processing within the backend.
        :param env: Environment for coordinating optional distributed execution,
            handling synchronization and parameter sharing.
        :param **constraints: Required constraints for controlling execution behavior
            to define stopping conditions,
            such as maximum requests, duration limits, or custom.
            Values can be primitives matching to available keys in ConstraintsFactory,
            or callable functions receiving scheduler and request state and returning
            the action to take for the scheduler.
        :yields: Tuples containing (response, request, scheduling_info, scheduler_state)
            for each processed request. Response may be None for failed requests.
        :raises: Any exceptions from worker processes, environment coordination, or
            constraint evaluation are propagated after proper cleanup.
        """
        with self.run_lock:
            worker_group: Optional[
                WorkerProcessGroup[BackendT, RequestT, RequestTimingsT, ResponseT]
            ] = None

            # Any issues during the run will raise an error (local or remote),
            # be caught and passed to the environment,
            # and will ensure clean up before raising the error.
            try:
                # Setup local run parameters, sync with the environment
                constraints = ConstraintsFactory.resolve_constraints(constraints)
                (
                    local_requests,
                    local_strategy,
                    local_constraints,
                ) = await env.sync_run_params(requests, strategy, constraints)

                # Setup the worker group, sync start with the environment
                worker_group = WorkerProcessGroup[
                    BackendT, RequestT, RequestTimingsT, ResponseT
                ](
                    backend=backend,
                    requests=local_requests,
                    strategy=local_strategy,
                    constraints=local_constraints,
                )
                await worker_group.create_processes()
                local_start_time = await env.sync_run_start()
                await worker_group.start(local_start_time)

                # Yield any updates and sync with the environment for non-local updates
                async for (
                    response,
                    request,
                    request_info,
                    state,
                ) in worker_group.request_updates():
                    await env.update_run_iteration(
                        response, request, request_info, state
                    )
                    yield response, request, request_info, state
            except Exception as err:  # noqa: BLE001
                env.sync_run_error(err)
            finally:
                # Ensure all worker processes are cleaned up for error or completion
                if worker_group is not None:
                    err = await worker_group.shutdown()
                    if err is not None:
                        env.sync_run_error(err)

            # Ensure any errors are raised and all responses
            # are yielded for aggregation on the primary node
            async for (
                response,
                request,
                request_info,
                state,
            ) in env.sync_run_end():
                yield response, request, request_info, state
