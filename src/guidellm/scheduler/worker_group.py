"""
Multi-process worker group orchestration for distributed request scheduling.

Provides infrastructure for coordinating worker processes with shared state
management, inter-process communication, and lifecycle coordination. Handles
dynamic scaling, load balancing, constraint evaluation, and graceful shutdown
across distributed workers processing concurrent requests.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections.abc import AsyncIterator, Generator, Iterable, Iterator
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier, Event
from typing import Generic, Literal

from guidellm.scheduler.constraints import Constraint, RequestsExhaustedConstraint
from guidellm.scheduler.objects import (
    BackendInterface,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerMessagingPydanticRegistry,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker import WorkerProcess
from guidellm.settings import settings
from guidellm.utils import (
    InterProcessMessaging,
    InterProcessMessagingManagerQueue,
    InterProcessMessagingPipe,
    InterProcessMessagingQueue,
    synchronous_to_exitable_async,
)

__all__ = ["WorkerProcessGroup"]


class WorkerProcessGroup(Generic[RequestT, ResponseT]):
    """
    Orchestrates multiple worker processes for distributed request processing.

    Manages process lifecycle, request distribution, response collection, and state
    synchronization across workers. Handles dynamic scaling, load balancing, and
    constraint evaluation with graceful shutdown coordination for high-throughput
    request processing workloads.

    Example:
    ::
        from guidellm.scheduler.worker_group import WorkerProcessGroup

        group = WorkerProcessGroup(
            requests=request_iterable,
            cycle_requests=None,
            backend=backend_instance,
            strategy=scheduling_strategy,
            constraints={"max_time": time_constraint}
        )

        await group.create_processes()
        await group.start(time.time())

        async for response, request, info, state in group.request_updates():
            if response is not None:
                # Process completed request
                handle_response(response)

        await group.shutdown()
    """

    def __init__(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]] | None,
        cycle_requests: Iterable[RequestT | MultiTurnRequestT[RequestT]] | None,
        backend: BackendInterface[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
    ):
        """
        Initialize a worker process group for distributed request processing.

        :param requests: Finite iterable of requests to process sequentially
        :param cycle_requests: Iterable of requests to cycle through indefinitely
        :param backend: Backend interface for processing requests
        :param strategy: Scheduling strategy for request timing and distribution
        :param constraints: Named constraints for controlling execution behavior
        :raises ValueError: If neither requests nor cycle_requests are provided,
            or if cycle_requests is an Iterator rather than Iterable
        """
        if not requests and not cycle_requests:
            raise ValueError(
                "At least one of 'requests' or 'cycle_requests' must be provided. "
                f"Got requests: {requests}, cycle_requests: {cycle_requests}"
            )

        if isinstance(cycle_requests, Iterator):
            raise ValueError(
                f"cycle_requests must be an Iterable or None, not an Iterator. "
                f"Got {type(cycle_requests)}"
            )

        self.requests = requests
        self.cycle_requests = cycle_requests
        self.backend = backend
        self.strategy = strategy
        self.constraints = constraints

        # Multiprocessing contexts and primitives, created in create_processes
        self.mp_context = None
        self.mp_manager = None
        self.processes: list[BaseProcess] = None
        self.requests_completed_event: Event = None
        self.startup_barrier: Barrier = None
        self.shutdown_event: Event = None
        self.error_event: Event = None

        # Scheduler and messaging state, created in start
        self._state: _WorkerGroupState[ResponseT, RequestT] = None
        self.messaging: InterProcessMessaging[
            tuple[
                RequestT | MultiTurnRequestT[RequestT],
                ScheduledRequestInfo,
            ],
            tuple[
                ResponseT | None,
                RequestT | MultiTurnRequestT[RequestT],
                ScheduledRequestInfo,
                SchedulerState,
            ],
        ] = None

    async def create_processes(self):
        """
        Create and initialize worker processes for distributed request processing.

        Sets up multiprocessing infrastructure and worker processes based on
        strategy constraints, backend capabilities, and system configuration.
        Determines optimal process count and concurrency limits, then spawns
        worker processes with distributed request handling capabilities.

        :raises RuntimeError: If process initialization or startup fails
        """
        # Processes limits and params
        max_conc: int = min(
            self.strategy.requests_limit or math.inf,
            self.backend.requests_limit or math.inf,
        )
        if max_conc == math.inf:
            # if concurrency not specified, use settings
            max_conc = settings.max_concurrency
        if max_conc <= 0:
            raise RuntimeError("max_concurrency resolved to 0; increase limits/config")

        num_processes = int(
            min(
                max_conc,  # Only spawn as many processes as we need for max_concurrency
                self.strategy.processes_limit or math.inf,
                self.backend.processes_limit or math.inf,
                settings.max_worker_processes,
            )
        )
        if num_processes <= 0:
            raise RuntimeError("num_processes resolved to 0; increase limits/config")

        per_proc_max_conc = max_conc // num_processes
        max_pending_size = max(
            1, math.floor(max_conc * settings.mp_max_pending_buffer_percent)
        )
        per_proc_max_buffer_size = max(
            1, math.floor(per_proc_max_conc * settings.mp_max_worker_buffer_percent)
        )

        # Initialize multiprocessing components
        self.mp_context: BaseContext = get_context(settings.mp_context_type)
        self.mp_manager = self.mp_context.Manager()
        self.startup_barrier = self.mp_context.Barrier(num_processes + 1)
        self.shutdown_event = self.mp_context.Event()
        self.error_event = self.mp_context.Event()
        self.requests_completed_event = self.mp_context.Event()

        if settings.mp_messaging_object == "queue":
            self.messaging = InterProcessMessagingQueue(
                mp_context=self.mp_context,
                serialization=settings.mp_serialization,
                encoding=settings.mp_encoding,
                max_pending_size=max_pending_size,
                max_buffer_send_size=settings.mp_requests_send_buffer_size,
                poll_interval=settings.mp_poll_interval,
            )
        elif settings.mp_messaging_object == "manager_queue":
            self.messaging = InterProcessMessagingManagerQueue(
                manager=self.mp_manager,
                mp_context=self.mp_context,
                serialization=settings.mp_serialization,
                encoding=settings.mp_encoding,
                max_pending_size=max_pending_size,
                max_buffer_send_size=settings.mp_requests_send_buffer_size,
                poll_interval=settings.mp_poll_interval,
            )
        elif settings.mp_messaging_object == "pipe":
            self.messaging = InterProcessMessagingPipe(
                num_workers=num_processes,
                mp_context=self.mp_context,
                serialization=settings.mp_serialization,
                encoding=settings.mp_encoding,
                max_pending_size=max_pending_size,
                max_buffer_send_size=settings.mp_requests_send_buffer_size,
                poll_interval=settings.mp_poll_interval,
            )

        # Initialize worker processes
        self.processes = []
        for rank in range(num_processes):
            # Distribute any remainder across the first N ranks
            async_limit = per_proc_max_conc + (
                1 if rank < (max_conc % num_processes) else 0
            )

            worker = WorkerProcess[RequestT, ResponseT](
                messaging=self.messaging.create_worker_copy(
                    worker_index=rank,
                    max_buffer_send_size=None,
                    max_buffer_receive_size=per_proc_max_buffer_size,
                ),
                async_limit=async_limit,
                startup_barrier=self.startup_barrier,
                shutdown_event=self.shutdown_event,
                error_event=self.error_event,
                requests_completed_event=self.requests_completed_event,
                backend=self.backend,
                request_timings=self.strategy.create_request_timings(
                    local_rank=rank,
                    local_world_size=num_processes,
                    local_max_concurrency=async_limit,
                ),
            )
            proc = self.mp_context.Process(target=worker.run, daemon=False)
            proc.start()
            self.processes.append(proc)

        reason, _ = await synchronous_to_exitable_async(
            synchronous=None,
            exit_events={
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            exit_barrier=self.startup_barrier,
            poll_interval=settings.mp_poll_interval,
        )
        if reason != "barrier":
            raise RuntimeError(
                f"Worker process group startup failed with exit reason: {reason}"
            )

    async def start(self, start_time: float):
        """
        Begin request processing at the specified start time.

        Initializes scheduler state and background tasks, then waits until the
        specified start time before beginning operations. Sets up inter-process
        communication and coordinates synchronized startup across all workers.

        :param start_time: Unix timestamp when processing should begin
        :raises RuntimeError: If workers encounter errors during startup or
            if create_processes() was not called first
        """
        if not self.processes:
            raise RuntimeError("create_processes() must be called before start()")

        self._state = _WorkerGroupState[RequestT, ResponseT](
            start_time=start_time,
            num_processes=len(self.processes),
            processes=self.processes,
            constraints=self.constraints,
            shutdown_event=self.shutdown_event,
        )
        await self.messaging.start(
            send_items=self._state.requests_generator(
                self.requests, self.cycle_requests
            ),
            receive_callback=self._state.update_callback_receive,
            send_stop_criteria=[self.shutdown_event, self.error_event],
            send_stopped_event=self.requests_completed_event,
            receive_stop_criteria=[self.error_event, self._state.stop_callback_receive],
            pydantic_models=list(SchedulerMessagingPydanticRegistry.registry.values()),
        )

        if (wait_time := start_time - time.time()) > 0:
            await asyncio.sleep(wait_time)
        if self.error_event.is_set():
            raise RuntimeError(
                "error_event is set in WorkerProcessGroup, "
                "indicating an error occurred in one of the worker processes."
            )

    async def request_updates(
        self,
    ) -> AsyncIterator[
        tuple[
            ResponseT | None,
            RequestT,
            ScheduledRequestInfo,
            SchedulerState,
        ]
    ]:
        """
        Yield request processing updates as they become available.

        Returns an async iterator of request updates including response, request,
        request scheduling info, and scheduler state. Updates occur on request queued,
        processing start, and completion. Response is None until processing completes.

        :return: Async iterator yielding (response, request, request_info, state)
            tuples where response is None until processing is complete
        :raises RuntimeError: If workers encounter unrecoverable errors
        """
        while (
            not self.messaging.receive_stopped_event.is_set()
            or not self.messaging.send_stopped_event.is_set()
            or not self.messaging.buffer_receive_queue.empty()
        ):
            if self.error_event.is_set():
                raise RuntimeError(
                    "error_event is set in WorkerProcessGroup, "
                    "indicating an error occurred in one of the worker processes."
                )

            try:
                (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) = await self.messaging.get(timeout=settings.mp_poll_interval)

                yield response, request, request_info, scheduler_state
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> list[Exception]:  # noqa: C901
        """
        Gracefully shut down the worker process group and clean up resources.

        Performs safe shutdown of worker processes, background tasks, and
        multiprocessing resources. Coordinates orderly termination across
        all workers and collects any exceptions encountered during shutdown.

        :return: List of exceptions encountered during shutdown; empty if no errors
        """
        exceptions: list[Exception] = []
        if self.shutdown_event is not None:
            self.shutdown_event.set()

        # Clear out start values
        if self.messaging is not None:
            await self.messaging.stop()
        self.messaging = None
        self._state = None

        # Clear out create processes values
        if self.processes is not None:
            for proc in self.processes:
                try:
                    await asyncio.to_thread(proc.join, timeout=5.0)
                    if proc.exitcode is not None and proc.exitcode > 0:
                        exceptions.append(
                            RuntimeError(
                                f"Worker {proc.pid} exited with code {proc.exitcode}"
                            )
                        )
                except Exception as err:  # noqa: BLE001
                    exceptions.append(err)
        self.processes = None
        self.startup_barrier = None
        self.shutdown_event = None
        self.error_event = None
        if self.mp_manager is not None:
            self.mp_manager.shutdown()
        self.mp_manager = None
        self.mp_context = None

        return exceptions


class _WorkerGroupState(Generic[RequestT, ResponseT]):
    """
    Manages scheduler state and synchronization for worker process groups.

    Handles request generation, state updates, constraint evaluation, and
    coordination between worker processes. Provides thread-safe state management
    with request lifecycle tracking and constraint-based termination logic.
    """

    def __init__(
        self,
        start_time: float,
        num_processes: int,
        processes: list[BaseProcess],
        constraints: dict[str, Constraint],
        shutdown_event: Event,
    ):
        """
        Initialize worker group state management.

        :param start_time: Unix timestamp when processing should begin
        :param num_processes: Number of worker processes in the group
        :param processes: List of worker process instances
        :param constraints: Named constraints for controlling execution behavior
        :param shutdown_event: Multiprocessing event for coordinated shutdown
        """
        self._start_time = start_time
        self._update_lock: threading.Lock = threading.Lock()
        self._state: SchedulerState = SchedulerState(
            node_id=0,
            num_processes=num_processes,
            start_time=start_time,
        )
        self.processes = processes
        self._constraints = constraints
        self._internal_constraints: dict[str, Constraint] = {}
        self._shutdown_event = shutdown_event
        self._shutdown_set = False

    def requests_generator(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]] | None,
        cycle_requests: Iterable[RequestT | MultiTurnRequestT[RequestT]] | None,
    ) -> Generator[tuple[RequestT | MultiTurnRequestT[RequestT],], None, None]:
        """
        Generate request-info pairs for worker processing with constraint evaluation.

        Processes finite requests sequentially then cycles through repeating requests
        indefinitely. Creates scheduling metadata for each request and evaluates
        constraints to determine when to stop request generation.

        :param requests: Finite iterable of requests to process sequentially
        :param cycle_requests: Iterable of requests to cycle through indefinitely
        :return: Generator yielding (request, request_info) tuples
        """

        def _iter():
            if requests:
                yield from requests

            if cycle_requests:
                while True:
                    yield from cycle_requests

        count = 0
        request_info: ScheduledRequestInfo = None
        for request in _iter():
            count += 1

            if hasattr(request, "request_id"):
                request_id = request.request_id
            elif hasattr(request, "id"):
                request_id = request.id
            elif hasattr(request, "id_"):
                request_id = request.id_
            elif hasattr(request, "uuid"):
                request_id = request.uuid
            else:
                request_id = str(uuid.uuid4())
            request_info: ScheduledRequestInfo = ScheduledRequestInfo(
                request_id=request_id,
                status="queued",
                scheduler_node_id=0,
                scheduler_process_id=-1,
                scheduler_start_time=self._start_time,
            )
            _, stop = self._locked_update(request_info, source="generator")
            yield (request, request_info)

            if stop:
                return

        # Reached the end, inject a RequestsExhaustedConstraint and update to record
        self._locked_update(
            info=request_info,
            source="generator",
            update_counts=False,
            requests_exhausted=RequestsExhaustedConstraint(num_requests=count),
        )

    def update_callback_receive(
        self,
        update: tuple[
            ResponseT | None,
            RequestT | MultiTurnRequestT,
            ScheduledRequestInfo,
        ],
    ) -> tuple[
        ResponseT | None,
        RequestT | MultiTurnRequestT,
        ScheduledRequestInfo,
        SchedulerState,
    ]:
        """
        Process received request updates and inject current scheduler state.

        Updates internal state tracking based on request status changes and
        evaluates constraints to determine if processing should be terminated.
        Triggers shutdown when stop conditions are met.

        :param update: Tuple containing response, request, and request info
        :return: Updated tuple with injected scheduler state
        """
        response, request, request_info = update
        state, stop = self._locked_update(info=request_info, source="updates")

        if stop:
            self._shutdown_event.set()

        return (
            response,
            request,
            request_info,
            state,  # inject state for updates to be yielded back
        )

    def stop_callback_receive(
        self, messaging: InterProcessMessaging, pending: bool, queue_empty: int
    ) -> bool:
        """
        Determine if message receiving should stop based on system state.

        Evaluates completion conditions including pending operations, queue state,
        and shutdown signals to coordinate graceful termination of message processing.

        :param messaging: Inter-process messaging instance
        :param pending: Whether operations are still pending
        :param queue_empty: The number of times the queue has reported empty in a row
        :return: True if message receiving should stop, False otherwise
        """
        return (
            not pending
            and queue_empty >= InterProcessMessaging.STOP_REQUIRED_QUEUE_EMPTY
            and messaging.send_stopped_event.is_set()  # No more requests will be added
            and self._shutdown_event.is_set()  # processing should stop
            and all(
                not proc.is_alive() for proc in self.processes
            )  # no more updates will be added by workers
        )

    def _locked_update(
        self,
        info: ScheduledRequestInfo,
        source: Literal["generator", "updates"],
        update_counts: bool = True,
        update_constraints: bool = True,
        **add_constraints: dict[str, Constraint],
    ) -> tuple[SchedulerState | None, bool]:
        with self._update_lock:
            if update_counts:
                if source == "generator":
                    self._update_new_request()
                elif source == "updates":
                    self._update_new_response(info)
                else:
                    raise ValueError(f"Unknown source: {source}")

            if add_constraints:
                self._internal_constraints.update(add_constraints)
            if update_constraints:
                self._update_with_constraints(info)
            self._state.end_time = time.time()
            state_copy: SchedulerState = self._state.model_copy()

        return (
            state_copy,
            (
                (source == "generator" and state_copy.end_queuing_time is not None)
                or (source == "updates" and state_copy.end_processing_time is not None)
            ),
        )

    def _update_new_request(self):
        self._state.created_requests += 1
        self._state.queued_requests += 1

    def _update_new_response(self, info: ScheduledRequestInfo):
        if info.status == "in_progress" or (
            info.status == "cancelled" and info.scheduler_timings.resolve_start is None
            # Cancelled request that never sent a progress update
        ):
            self._state.queued_requests -= 1
            self._state.processing_requests += 1

        if info.status in ("completed", "errored", "cancelled"):
            self._state.processing_requests -= 1
            self._state.processed_requests += 1
            self._state.successful_requests += 1 if info.status == "completed" else 0
            self._state.errored_requests += 1 if info.status == "errored" else 0
            self._state.cancelled_requests += 1 if info.status == "cancelled" else 0

    def _update_with_constraints(self, info: ScheduledRequestInfo):
        actions: dict[str, SchedulerUpdateAction] = {
            name: const(self._state, info) for name, const in self._constraints.items()
        }
        if self._internal_constraints:
            actions.update(
                {
                    name: const(self._state, info)
                    for name, const in self._internal_constraints.items()
                }
            )
        self._state.scheduler_constraints = actions

        if self._state.end_queuing_time is None and (
            stop_queuing_actions := {
                key: action
                for key, action in actions.items()
                if action.request_queuing == "stop"
            }
        ):
            # Queuing not stopped and actions returned to stop it
            self._state.end_queuing_constraints = stop_queuing_actions
            self._state.end_queuing_time = time.time()

        if self._state.end_processing_time is None and (
            stop_processing_actions := {
                key: action
                for key, action in actions.items()
                if action.request_processing in ("stop_local", "stop_all")
            }
        ):
            # Processing not stopped and actions returned to stop it
            self._state.end_processing_constraints = stop_processing_actions
            self._state.end_processing_time = time.time()
