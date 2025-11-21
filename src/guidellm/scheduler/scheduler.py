"""
Thread-safe singleton scheduler for distributed benchmarking workload coordination.

Orchestrates request processing across worker processes with distributed timing
coordination, constraint enforcement, and result aggregation. Integrates with
backends, environments, and strategies to enable scalable load testing across
various scenarios including LLM inference benchmarking.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Generic

from guidellm.scheduler.constraints import Constraint, ConstraintsInitializerFactory
from guidellm.scheduler.environments import Environment, NonDistributedEnvironment
from guidellm.scheduler.schemas import (
    BackendInterface,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    SchedulerState,
)
from guidellm.scheduler.strategies import SchedulingStrategy
from guidellm.scheduler.worker_group import WorkerProcessGroup
from guidellm.schemas import RequestInfo
from guidellm.utils import ThreadSafeSingletonMixin

__all__ = ["Scheduler"]


class Scheduler(
    Generic[RequestT, ResponseT],
    ThreadSafeSingletonMixin,
):
    """
    Thread-safe singleton scheduler for distributed benchmarking workload coordination.

    Orchestrates request processing across worker processes with distributed timing
    coordination, constraint enforcement, and result aggregation. Abstracts the
    complexity of multi-process coordination, environment synchronization, and
    resource management while providing a unified interface for executing benchmarking
    operations. Implements singleton pattern to ensure consistent execution state.

    Example:
    ::
        from guidellm.scheduler import Scheduler
        from guidellm.scheduler import NonDistributedEnvironment, SynchronousStrategy

        scheduler = Scheduler()
        async for response, request, info, state in scheduler.run(
            requests=request_list,
            backend=backend,
            strategy=SynchronousStrategy(),
            env=NonDistributedEnvironment(),
            max_requests=1000
        ):
            print(f"Processed: {request}")
    """

    async def run(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        backend: BackendInterface[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        env: Environment[RequestT, ResponseT] | None,
        **constraints: Any | dict[str, Any] | Constraint,
    ) -> AsyncIterator[
        tuple[
            ResponseT | None,
            RequestT | MultiTurnRequestT[RequestT],
            RequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Execute distributed request processing with coordinated timing and constraints.

        Orchestrates the complete benchmarking workflow across worker processes with
        environment synchronization, constraint enforcement, and error handling. Manages
        resource lifecycle from initialization through cleanup while yielding real-time
        processing updates for monitoring and aggregation.

        :param requests: Request collection to process, supporting single requests or
            multi-turn sequences with optional inter-request delays
        :param backend: Backend interface for request processing and response generation
        :param strategy: Scheduling strategy controlling request timing and distribution
        :param env: Environment interface for distributed coordination and
            synchronization. Defaults to NonDistributedEnvironment if None
        :param constraints: Runtime constraints for execution control (max_requests,
            max_duration, max_error_rate, etc.) as primitives, dictionaries, or
            constraint instances
        :yields: Request updates as (response, request, request_info, scheduler_state)
            tuples. Each request generates three ordered updates: queued, in_progress,
            completed | errored | cancelled
        :raises Exception: Worker process errors, environment synchronization failures,
            or constraint evaluation errors are propagated after cleanup
        """
        with self.thread_lock:
            if env is None:
                env = NonDistributedEnvironment[RequestT, ResponseT]()

            worker_group: WorkerProcessGroup[RequestT, ResponseT] | None = None

            # Any issues during the run will raise an error (local or remote),
            # be caught and passed to the environment,
            # and will ensure clean up before raising the error.
            try:
                # Setup local run parameters, sync with the environment
                resolved_constraints = (
                    ConstraintsInitializerFactory.resolve_constraints(constraints)
                )
                (
                    local_requests,
                    local_strategy,
                    local_constraints,
                ) = await env.sync_run_params(requests, strategy, resolved_constraints)

                # Setup the worker group, sync start with the environment
                worker_group = WorkerProcessGroup[RequestT, ResponseT](
                    requests=local_requests,
                    backend=backend,
                    strategy=local_strategy,
                    **local_constraints,
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
                raise err
            finally:
                # Ensure all worker processes are cleaned up for error or completion
                if worker_group is not None:
                    err = await worker_group.shutdown()  # type: ignore[misc]
                    if err is not None:
                        await env.sync_run_error(err)

            # Ensure any errors are raised and all responses
            # are yielded for aggregation on the primary node
            async for (
                dist_response,
                dist_request,
                dist_request_info,
                dist_state,
            ) in env.sync_run_end():
                yield dist_response, dist_request, dist_request_info, dist_state
