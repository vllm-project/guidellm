"""
Thread-safe singleton scheduler for distributed load generation workload coordination.

Provides the core orchestration engine that coordinates request processing across
worker processes and distributed environments. Manages timing synchronization,
resource allocation, constraint enforcement, and result aggregation for
load generation operations. Integrates with backends, environments, and strategies
to enable scalable load testing across various scenarios including LLM inference.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Generic

from guidellm.scheduler.constraints import (
    Constraint,
    ConstraintsInitializerFactory,
)
from guidellm.scheduler.environment import Environment, NonDistributedEnvironment
from guidellm.scheduler.objects import (
    BackendInterface,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker_group import WorkerProcessGroup
from guidellm.utils.singleton import ThreadSafeSingletonMixin

__all__ = ["Scheduler"]


class Scheduler(
    Generic[RequestT, ResponseT],
    ThreadSafeSingletonMixin,
):
    """
    Thread-safe singleton scheduler for distributed benchmarking workload coordination.

    Orchestrates request processing across worker processes with distributed timing
    coordination, constraint enforcement, and result aggregation. Provides a unified
    interface for executing benchmarking operations while abstracting the complexity
    of multi-process coordination, environment synchronization, and resource management.
    Implements singleton pattern to ensure consistent execution state across concurrent
    benchmark operations.

    Example:
    ::
        from guidellm.scheduler import Scheduler
        from guidellm.backend import OpenAIBackend
        from guidellm.scheduler import NonDistributedEnvironment, SynchronousStrategy

        scheduler = Scheduler()
        async for response, request, info, state in scheduler.run(
            requests=request_list,
            backend=backend,
            strategy=SynchronousStrategy(),
            env=NonDistributedEnvironment(),
            max_requests=1000
        ):
            print(f"Processed: {request} with info: {info} and response: {response}")
    """

    async def run(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        backend: BackendInterface[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        env: Environment | None,
        **constraints: dict[str, Any | dict[str, Any] | Constraint],
    ) -> AsyncIterator[
        tuple[
            ResponseT | None,
            RequestT,
            ScheduledRequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Execute distributed request processing with coordinated timing and constraints.

        Orchestrates the complete benchmarking workflow across worker processes with
        environment synchronization, constraint enforcement, and error handling.
        Manages resource lifecycle from initialization through cleanup while yielding
        real-time processing updates for monitoring and aggregation.

        :param requests: Request collection to process. Supports single requests or
            multi-turn sequences with optional inter-request delays
        :param backend: Backend interface for request processing and response generation
        :param strategy: Scheduling strategy controlling request timing and distribution
        :param env: Environment interface for distributed coordination and
            synchronization
        :param constraints: Runtime constraints for execution control (max_requests,
            max_duration, max_error_rate, etc.). Values can be primitives, dictionaries,
            or constraint instances
        :yields: Requests udpates as (response, request, request_info, scheduler_state)
        tuples. Each request will generate three ordered updates:
            queued, in_progress, completed | errored | cancelled.
        :raises Exception: Worker process errors, environment synchronization failures,
            or constraint evaluation errors are propagated after cleanup
        """
        with self.thread_lock:
            if env is None:
                env = NonDistributedEnvironment()

            worker_group: WorkerProcessGroup[RequestT, ResponseT] | None = None

            # Any issues during the run will raise an error (local or remote),
            # be caught and passed to the environment,
            # and will ensure clean up before raising the error.
            try:
                # Setup local run parameters, sync with the environment
                constraints = ConstraintsInitializerFactory.resolve_constraints(
                    constraints
                )
                (
                    local_requests,
                    local_strategy,
                    local_constraints,
                ) = await env.sync_run_params(requests, strategy, constraints)

                # Setup the worker group, sync start with the environment
                worker_group = WorkerProcessGroup[RequestT, ResponseT](
                    requests=None,
                    cycle_requests=local_requests,
                    backend=backend,
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
                await env.sync_run_error(err)
            finally:
                # Ensure all worker processes are cleaned up for error or completion
                if worker_group is not None:
                    err = await worker_group.shutdown()
                    if err is not None:
                        await env.sync_run_error(err)

            # Ensure any errors are raised and all responses
            # are yielded for aggregation on the primary node
            async for (
                response,
                request,
                request_info,
                state,
            ) in env.sync_run_end():
                yield response, request, request_info, state
