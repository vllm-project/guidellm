"""
Worker process group management for multi-process request scheduling and execution.

This module provides the infrastructure for orchestrating multiple worker processes
that handle request scheduling, processing, and coordination in the GuideLLM toolkit.
It implements a multiprocessing-based architecture with asyncio/await patterns for
efficient concurrent request handling and distributed load balancing.

The module centers around the WorkerProcessGroup class, which manages the complete
lifecycle of worker processes including initialization, request distribution,
response collection, state synchronization, and graceful shutdown coordination.

Classes:
    WorkerProcessGroup: Orchestrates multiple worker processes with shared state
        management, inter-process communication, and centralized coordination for
        distributed request processing.
"""

import asyncio
import contextlib
import math
import queue
import threading
import time
from asyncio import Task
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager, Queue
from multiprocessing.managers import BaseManager
from multiprocessing.synchronize import Barrier, Event
from typing import Generic, Optional

from guidellm.config import settings
from guidellm.scheduler.constraints import CallableConstraint
from guidellm.scheduler.objects import (
    BackendT,
    RequestT,
    RequestTimingsT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker import WorkerProcess, worker_sync_iterable_to_async

__all__ = ["WorkerProcessGroup"]


class WorkerProcessGroup(Generic[BackendT, RequestT, RequestTimingsT, ResponseT]):
    """
    Orchestrates multiple worker processes with shared state management and
    coordination.

    This class manages a group of worker processes that collectively handle request
    processing for a distributed scheduling system. It provides centralized control
    over process lifecycle, request distribution, response collection, and state
    synchronization across all workers.

    The process group handles dynamic scaling, load balancing, constraint evaluation,
    and graceful shutdown coordination. It maintains shared queues for inter-process
    communication and tracks the overall system state including throughput metrics
    and constraint satisfaction.

    Example:
    ::
        from guidellm.scheduler.worker import WorkerProcessGroup

        group = WorkerProcessGroup(
            backend=my_backend,
            requests=request_iterable,
            strategy=scheduling_strategy,
            constraints={"max_requests": max_requests_constraint}
        )

        await group.create_processes()
        await group.start(time.time())

        async for response, request, info, state in group.request_updates():
            # Process each completed request
            handle_response(response, request, info, state)

        await group.shutdown()

    :param backend: Backend instance for processing requests.
    :param requests: Iterable of requests to process.
    :param strategy: Scheduling strategy configuration.
    :param constraints: Dictionary of named constraints for controlling execution.
    """

    def __init__(
        self,
        backend: BackendT[RequestT, RequestTimingsT, ResponseT],
        requests: Iterable[RequestT],
        strategy: SchedulingStrategy,
        constraints: dict[str, CallableConstraint],
    ):
        self.backend = backend
        self.requests = requests
        self.strategy = strategy
        self.constraints = constraints

        # multiprocessing attributes
        self.manager: BaseManager = None
        self.executor: ProcessPoolExecutor = None
        self.processes: list[Future] = None

        # synchronization primitives
        self.startup_barrier: Barrier = None
        self.shutdown_event: Event = None
        self.error_event: Event = None

        # queues for communication
        self.process_requests_queue: Queue[
            tuple[RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None
        self.process_updates_queue: Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None
        self.async_updates_queue: queue.Queue[
            tuple[Optional[ResponseT], RequestT, ScheduledRequestInfo[RequestTimingsT]]
        ] = None

        # scheduler state and request management
        self.state_update_lock: threading.Lock = None
        self.scheduler_state: SchedulerState = None
        self.populate_requests_task: Task = None
        self.populate_updates_task: Task = None

    async def create_processes(self):
        """
        Initialize and start the worker process group.

        Sets up the multiprocessing infrastructure including process pool executor,
        synchronization primitives, communication queues, and individual worker
        processes. Determines optimal process count and concurrency limits based
        on strategy constraints, backend capabilities, and system configuration.

        :raises RuntimeError: If process initialization fails or if workers encounter
            errors during startup.
        """
        # Processes limits and params
        num_processes = min(
            self.strategy.processes_limit or math.inf,
            self.backend.processes_limit or math.inf,
            settings.max_worker_processes,
        )
        max_request_concurrency = min(
            self.strategy.requests_limit or math.inf,
            self.backend.requests_limit or math.inf,
            settings.max_concurrency,
        )
        per_process_max_concurrency = math.ceil(max_request_concurrency / num_processes)
        max_queued_requests = (  # Add one for each process to ensure readiness
            max_request_concurrency + num_processes
        )

        # Initialize multiprocessing components
        self.manager = Manager()
        self.executor = ProcessPoolExecutor()
        self.startup_barrier = self.manager.Barrier(num_processes + 1)
        self.shutdown_event = self.manager.Event()
        self.error_event = self.manager.Event()
        self.process_requests_queue = self.manager.Queue(maxsize=max_queued_requests)
        self.process_updates_queue = self.manager.Queue()
        self.async_updates_queue = asyncio.Queue()

        # Initialize worker processes
        self.processes = []
        for process_rank in range(num_processes):
            worker = WorkerProcess[BackendT, RequestT, RequestTimingsT, ResponseT](
                local_rank=process_rank,
                local_world_size=num_processes,
                async_limit=(
                    per_process_max_concurrency
                    if process_rank < num_processes - 1
                    else max_request_concurrency
                    - (per_process_max_concurrency * (num_processes - 1))
                ),
                startup_barrier=self.startup_barrier,
                shutdown_event=self.shutdown_event,
                error_event=self.error_event,
                requests_queue=self.process_requests_queue,
                updates_queue=self.process_updates_queue,
                backend=self.backend,
                request_timings=self.strategy.create_worker_timings(
                    local_rank=process_rank,
                    local_world_size=num_processes,
                    local_max_concurrency=per_process_max_concurrency,
                ),
                poll_intervals=settings.scheduler_poll_interval,
            )
            future = self.executor.submit(worker.run)
            self.processes.append(future)

        startup_exit_reason, _ = await worker_sync_iterable_to_async(
            iter_func="infinite",
            exit_events={
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            exit_barrier=self.startup_barrier,
            poll_interval=settings.scheduler_poll_interval,
        )
        if startup_exit_reason != "barrier":
            raise RuntimeError(
                "Worker process group startup failed with exit reason: "
                f"{startup_exit_reason}"
            )

    async def start(self, start_time: float):
        """
        Begin request processing at the specified start time.

        Initializes the scheduler state, creates background tasks for request
        population and response handling, and waits until the specified start
        time before beginning operations.

        :param start_time: Unix timestamp when processing should begin.
        :raises RuntimeError: If workers encounter errors during startup or if
            initialization fails.
        """
        self.state_update_lock = threading.Lock()
        self.scheduler_state = SchedulerState(
            node_id=0,  # Process group node identifier
            num_processes=len(self.processes),
            start_time=start_time,
        )
        self.populate_requests_task = asyncio.create_task(
            self._populate_requests(start_time)
        )
        self.populate_updates_task = asyncio.create_task(self._populate_updates())

        await asyncio.sleep(start_time - time.time())
        self._raise_if_error()

    async def request_updates(
        self,
    ) -> AsyncIterator[
        tuple[
            Optional[ResponseT],
            RequestT,
            ScheduledRequestInfo[RequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Yield request processing updates as they become available.

        Returns an async iterator that yields tuples containing request processing
        updates, including the response (if available), the original request,
        scheduling metadata, and the current scheduler state.
        Updates occur on request queued, request processing start, and
        request completion, allowing for real-time monitoring and processing.
        The iterator continues until all requests have been processed or an error
        occurs.

        :return: Async iterator yielding (response, request, request_info, state)
            tuples for each request update; response is None until the request
            processing is complete.
        :raises RuntimeError: If workers encounter unrecoverable errors during
            processing.
        """
        last_state: SchedulerState = None
        last_check_time = time.time()
        shutdown = self.shutdown_event.is_set()

        while (
            not shutdown
            or last_state is None
            or last_state.processed_requests < last_state.created_requests
        ):
            try:
                (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) = self.async_updates_queue.get_nowait()
                last_state = scheduler_state

                yield response, request, request_info, scheduler_state
            except queue.Empty:
                await asyncio.sleep(settings.scheduler_poll_interval)

            if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                self._raise_if_error()
                shutdown = self.shutdown_event.is_set()
                last_check_time = time.time()

    async def shutdown(self) -> list[Exception]:
        """
        Gracefully shut down the worker process group and clean up resources.

        Performs a safe shutdown of all worker processes, background tasks,
        and multiprocessing resources to release system resources.
        Returns any errors encountered during the shutdown process,
        either during shutdown or previously occurred in worker processes.

        :return: A list of any exceptions raised while running or shutting down
            the worker processes. If no errors occurred, returns an empty list.
        """
        exceptions = []

        if self.shutdown_event is not None:
            self.shutdown_event.set()
            self.shutdown_event = None

        with contextlib.suppress(asyncio.CancelledError):
            try:
                for task in [
                    self.populate_requests_task,
                    self.populate_responses_task,
                ]:
                    if task is not None and not task.done():
                        task.cancel()
                        await task
            except Exception as err:
                exceptions.append(err)
            self.populate_requests_task = None
            self.populate_responses_task = None

            for process in self.processes:
                try:
                    asyncio.to_thread(process.result)
                except Exception as err:
                    exceptions.append(err)
            self.processes = None

        self.executor.shutdown(wait=True)
        self.executor = None
        self.manager.shutdown()
        self.manager = None

        self.startup_barrier = None
        self.error_event = None
        self.process_requests_queue = None
        self.process_updates_queue = None
        self.async_responses_queue = None

        return exceptions

    async def _raise_if_error(self):
        if self.error_event.is_set():
            raise RuntimeError(
                "error_event is set in WorkerProcessGroup, "
                "indicating an error occurred in one of the worker processes."
            )

    def _populate_requests(self, start_time: float):
        last_check_time = time.time()
        continue_requests = True

        while continue_requests:
            for request in self.requests:
                request_info = ScheduledRequestInfo[RequestTimingsT](
                    request_id=getattr(
                        request, "id_", getattr(request, "id", id(request))
                    ),
                    status="queued",
                    scheduler_start_time=start_time,
                )
                _, continue_requests, _ = self._update_scheduler_state(request_info)
                self.process_requests_queue.put((request, request_info))

                if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                    if self.shutdown_event.is_set() or self.error_event.is_set():
                        continue_requests = False
                    last_check_time = time.time()

                if not continue_requests:
                    break

    def _populate_updates(self):
        last_check_time = time.time()

        while True:
            try:
                (response, request, request_info) = self.process_updates_queue.get(
                    timeout=settings.scheduler_poll_interval
                )
                scheduler_state, _, continue_processing = self._update_scheduler_state(
                    request_info
                )
                self.async_updates_queue.put(
                    (response, request, request_info, scheduler_state)
                )
                if not continue_processing:
                    self.shutdown_event.set()
                    if (
                        scheduler_state.processed_requests
                        >= scheduler_state.created_requests
                    ):
                        # Ensure we've processed all updates before exiting
                        break
            except queue.Empty:
                continue

            if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                if self.error_event.is_set():
                    break
                last_check_time = time.time()

    def _update_scheduler_state(
        self, request_info: ScheduledRequestInfo[RequestTimingsT]
    ) -> tuple[SchedulerState, bool, bool]:
        """
        :param request_info: The request information to update the state with.
        :return: A tuple containing the updated scheduler state,
            whether to continue adding requests,
            and whether to continue processing requests.
        """
        with self.state_update_lock:
            if request_info.status == "queued":
                self.scheduler_state.created_requests += 1
                self.scheduler_state.queued_requests += 1
            elif request_info.status == "pending":
                self.scheduler_state.queued_requests -= 1
                self.scheduler_state.pending_requests += 1
            elif request_info.status == "in_progress":
                self.scheduler_state.pending_requests -= 1
                self.scheduler_state.processing_requests += 1
            elif request_info.status == "completed":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.successful_requests += 1
            elif request_info.status == "errored":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.errored_requests += 1
            elif request_info.status == "cancelled":
                self.scheduler_state.processing_requests -= 1
                self.scheduler_state.processed_requests += 1
                self.scheduler_state.cancelled_requests += 1
            else:
                raise ValueError(
                    f"Unknown request status: {request_info.status}. "
                    "Supported statuses are: queued, pending, in_progress, "
                    "completed, errored, cancelled."
                )

            queuing_stopped = self.scheduler_state.end_queuing_time is not None
            processing_stopped = self.scheduler_state.end_processing_time is not None

            for key, constraint in self.constraints.items():
                action = constraint(self.scheduler_state, request_info)
                if not queuing_stopped and action.request_queuing == "stop":
                    # Ensure metadata for final state only includes the first
                    # constraints that signaled for stopping queuing
                    self.scheduler_state.end_queuing_constraints[key] = action.metadata
                    self.scheduler_state.end_queuing_time = time.time()
                if not processing_stopped and action.request_processing in (
                    "stop_local",
                    "stop_all",
                ):
                    # Ensure metadata for final state only includes the first
                    # constraints that signaled for stopping processing
                    self.scheduler_state.end_processing_constraints[key] = (
                        action.metadata
                    )
                    self.scheduler_state.end_processing_time = time.time()

            state_copy: SchedulerState = self.scheduler_state.model_copy(deep=True)

        return (
            state_copy,
            state_copy.end_queuing_time is None,
            state_copy.end_processing_time is None,
        )
