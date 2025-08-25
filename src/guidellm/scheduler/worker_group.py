"""
Multi-process worker group orchestration for distributed request scheduling.

Provides infrastructure for coordinating worker processes with shared state
management, inter-process communication, and lifecycle coordination.

Classes:
    WorkerProcessGroup: Orchestrates multiple worker processes for distributed
        request processing with centralized coordination.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import queue
import threading
import time
import uuid
from asyncio import Task
from collections.abc import AsyncIterator, Iterable, Iterator
from multiprocessing import Queue, get_context
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier, Event
from threading import Event as ThreadingEvent
from typing import Generic

import culsans

from guidellm.config import settings
from guidellm.scheduler.constraints import Constraint
from guidellm.scheduler.objects import (
    BackendInterface,
    MeasuredRequestTimingsT,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerState,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker import WorkerProcess
from guidellm.utils import MsgpackEncoding, synchronous_to_exitable_async

__all__ = ["WorkerProcessGroup"]


class WorkerProcessGroup(Generic[RequestT, MeasuredRequestTimingsT, ResponseT]):
    """
    Orchestrates multiple worker processes for distributed request processing.

    Manages process lifecycle, request distribution, response collection, and state
    synchronization across workers. Handles dynamic scaling, load balancing, and
    constraint evaluation with graceful shutdown coordination.
    """

    def __init__(
        self,
        requests: Iterable[RequestT | MultiTurnRequestT[RequestT]],
        backend: BackendInterface[RequestT, MeasuredRequestTimingsT, ResponseT],
        strategy: SchedulingStrategy,
        constraints: dict[str, Constraint],
        infinite_requests: bool | None = None,
    ):
        self.requests = requests
        self.backend = backend
        self.strategy = strategy
        self.constraints = constraints
        self.infinite_requests = infinite_requests

        # Multiprocessing contexts and primitives, created in create_processes
        self.mp_context = None
        self.processes: list[BaseProcess] = None
        self.startup_barrier: Barrier = None
        self.shutdown_event: Event = None
        self.error_event: Event = None
        self.requests_queue: Queue[
            tuple[
                RequestT | MultiTurnRequestT[RequestT],
                ScheduledRequestInfo[MeasuredRequestTimingsT],
            ]
        ] = None
        self.updates_queue: Queue[
            tuple[
                ResponseT | None,
                RequestT,
                ScheduledRequestInfo[MeasuredRequestTimingsT],
            ]
        ] = None

        # Local process async/threading bridges + signals
        self.pending_updates_queue: culsans.Queue[
            tuple[
                ResponseT | None,
                RequestT | MultiTurnRequestT[RequestT],
                ScheduledRequestInfo[MeasuredRequestTimingsT],
            ]
        ] = None
        self.pending_requests_complete: ThreadingEvent = None
        self.pending_updates_complete: ThreadingEvent = None
        self.populate_requests_task: Task = None
        self.populate_updates_task: Task = None

        # Scheduler state
        self.state_update_lock: threading.Lock = None
        self.scheduler_state: SchedulerState = None

    async def create_processes(self):
        """
        Initialize and start the worker process group.

        Sets up multiprocessing infrastructure and worker processes based on
        strategy constraints, backend capabilities, and system configuration.

        :param backend: Backend instance for processing requests.
        :param requests: Iterable of requests to process.
        :param strategy: Scheduling strategy configuration.
        :param constraints: Dictionary of named constraints for controlling execution.
        :raises RuntimeError: If process initialization or startup fails.
        """
        # Processes limits and params

        max_conc = int(
            min(
                self.strategy.requests_limit or math.inf,
                self.backend.requests_limit or math.inf,
                settings.max_concurrency,
            )
        )
        if max_conc <= 0:
            raise RuntimeError("max_concurrency resolved to 0; increase limits/config")

        num_processes = int(
            min(
                self.strategy.processes_limit or math.inf,
                self.backend.processes_limit or math.inf,
                settings.max_worker_processes,
                # Only spawn as many processes as we need for max_concurrency
                max_conc,
            )
        )
        if num_processes <= 0:
            raise RuntimeError("num_processes resolved to 0; increase limits/config")

        per_proc_max_conc = max_conc // num_processes
        per_proc_max_queue = math.floor(math.log(per_proc_max_conc + math.e))
        max_queued_requests = (  # Add queue buffer for each process
            max_conc + (num_processes * per_proc_max_queue)
        )

        # Initialize multiprocessing components
        self.mp_context = get_context("fork")
        self.startup_barrier = self.mp_context.Barrier(num_processes + 1)
        self.shutdown_event = self.mp_context.Event()
        self.error_event = self.mp_context.Event()
        self.requests_queue = self.mp_context.Queue(maxsize=max_queued_requests)
        self.updates_queue = self.mp_context.Queue()

        # Initialize worker processes
        self.processes = []
        for rank in range(num_processes):
            # Distribute any remainder across the first R ranks
            async_limit = per_proc_max_conc + (
                1 if rank < (max_conc % num_processes) else 0
            )

            worker = WorkerProcess[RequestT, MeasuredRequestTimingsT, ResponseT](
                local_rank=rank,
                local_world_size=num_processes,
                async_limit=async_limit,
                startup_barrier=self.startup_barrier,
                shutdown_event=self.shutdown_event,
                error_event=self.error_event,
                requests_queue=self.requests_queue,
                updates_queue=self.updates_queue,
                backend=self.backend,
                request_timings=self.strategy.create_request_timings(
                    local_rank=rank,
                    local_world_size=num_processes,
                    local_max_concurrency=async_limit,
                ),
                poll_intervals=settings.scheduler_poll_interval,
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
            poll_interval=settings.scheduler_poll_interval,
        )
        if reason != "barrier":
            raise RuntimeError(
                f"Worker process group startup failed with exit reason: {reason}"
            )

    async def start(self, start_time: float):
        """
        Begin request processing at the specified start time.

        Initializes scheduler state and background tasks, then waits until the
        specified start time before beginning operations.

        :param start_time: Unix timestamp when processing should begin.
        :raises RuntimeError: If workers encounter errors during startup.
        """
        if self.processes is None:
            raise RuntimeError("create_processes() must be called before start()")

        self.state_update_lock = threading.Lock()
        self.scheduler_state = SchedulerState(
            node_id=0,  # Process group node identifier
            num_processes=len(self.processes),
            start_time=start_time,
        )
        self.pending_updates_queue = culsans.Queue()
        self.pending_requests_complete = ThreadingEvent()
        self.pending_updates_complete = ThreadingEvent()

        self.populate_requests_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._populate_requests_generator(start_time),
                exit_events={"error_event": self.error_event},
                poll_interval=0.0,
            )
        )
        self.populate_updates_task = asyncio.create_task(
            synchronous_to_exitable_async(
                self._populate_updates_generator(),
                exit_events={"error_event": self.error_event},
                poll_interval=0.0,
            )
        )

        await asyncio.sleep(max(0, start_time - time.time()))
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
            ScheduledRequestInfo[MeasuredRequestTimingsT],
            SchedulerState,
        ]
    ]:
        """
        Yield request processing updates as they become available.

        Returns an async iterator of request updates including response, request,
        scheduling metadata, and scheduler state. Updates occur on request queued,
        processing start, and completion.

        :return: Async iterator yielding (response, request, request_info, state)
            tuples; response is None until processing is complete.
        :raises RuntimeError: If workers encounter unrecoverable errors.
        """
        last_check_time = -1 * math.inf

        while (
            not self.pending_updates_complete.is_set()
            or not self.pending_updates_queue.empty()
        ):
            try:
                (
                    response,
                    request,
                    request_info,
                    scheduler_state,
                ) = await asyncio.wait_for(
                    self.pending_updates_queue.async_get(),
                    timeout=settings.scheduler_poll_interval,
                )

                yield response, request, request_info, scheduler_state
            except asyncio.TimeoutError:
                pass

            if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                if self.error_event.is_set():
                    raise RuntimeError(
                        "error_event is set in WorkerProcessGroup, "
                        "indicating an error occurred in one of the worker processes."
                    )
                last_check_time = time.time()

    async def shutdown(self) -> list[Exception]:  # noqa: C901
        """
        Gracefully shut down the worker process group and clean up resources.

        Performs safe shutdown of worker processes, background tasks, and
        multiprocessing resources.

        :return: List of exceptions encountered during shutdown; empty if no errors.
        """
        exceptions: list[Exception] = []

        if self.shutdown_event is not None:
            self.shutdown_event.set()

        cancel_tasks = [
            task
            for task in (self.populate_requests_task, self.populate_updates_task)
            if task and not task.done()
        ]
        for task in cancel_tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            if cancel_tasks:
                try:
                    await asyncio.gather(*cancel_tasks, return_exceptions=True)
                except Exception as err:  # noqa: BLE001
                    exceptions.append(err)
        self.populate_requests_task = None
        self.populate_updates_task = None

        if self.processes:
            for proc in self.processes:
                await asyncio.to_thread(proc.join, 5)
                if proc.exitcode not in (0, None):
                    exceptions.append(
                        RuntimeError(
                            f"Worker {proc.pid} exited with code {proc.exitcode}"
                        )
                    )
        self.processes = None
        self.mp_context = None

        self.startup_barrier = None
        self.shutdown_event = None
        self.error_event = None
        self.requests_queue = None
        self.updates_queue = None
        self.pending_updates_queue = None

        return exceptions

    def _update_state(
        self, info: ScheduledRequestInfo[MeasuredRequestTimingsT]
    ) -> tuple[SchedulerState, bool, bool]:
        if not self.scheduler_state or not self.state_update_lock:
            raise RuntimeError("workerProcessGroup not started")

        with self.state_update_lock:
            state = self.scheduler_state
            if info.status == "queued":
                state.created_requests += 1
                state.queued_requests += 1
            elif info.status == "in_progress":
                state.queued_requests -= 1
                state.processing_requests += 1
            elif info.status in ("completed", "errored", "cancelled"):
                state.processing_requests -= 1
                state.processed_requests += 1
                state.successful_requests += 1 if info.status == "completed" else 0
                state.errored_requests += 1 if info.status == "errored" else 0
                state.cancelled_requests += 1 if info.status == "cancelled" else 0
            else:
                raise ValueError(
                    f"Unknown request status: {info.status}. "
                    "Supported statuses are: queued, pending, in_progress, "
                    "completed, errored, cancelled."
                )

            state.end_time = time.time()  # Always update for last time update received
            actions = {
                name: const(state, info) for name, const in self.constraints.items()
            }
            state.scheduler_constraints = actions

            if state.end_queuing_time is None and (
                stop_queueing_actions := {
                    key: action
                    for key, action in actions.items()
                    if action.request_queuing == "stop"
                }
            ):
                # Queuing not stopped and actions returned to stop it
                state.end_queuing_constraints.update(stop_queueing_actions)
                state.end_queuing_time = time.time()

            if state.end_processing_time is None and (
                stop_processing_actions := {
                    key: action
                    for key, action in actions.items()
                    if action.request_processing in ("stop_local", "stop_all")
                }
            ):
                # Processing not stopped and actions returned to stop it
                state.end_processing_constraints.update(stop_processing_actions)
                state.end_processing_time = time.time()

            state_copy: SchedulerState = state.model_copy()

        return (
            state_copy,
            state_copy.end_queuing_time is None,
            state_copy.end_processing_time is None,
        )

    def _populate_requests_generator(self, scheduler_start_time: float):
        last_check_time: float = time.time()
        continue_requests: bool = True
        message: bytes | None = None
        request_iter: Iterator[RequestT] | None = (
            self._populate_requests_create_iterator(first=True)
        )

        try:
            while continue_requests or message is not None:
                if request_iter is None:
                    request_iter = self._populate_requests_create_iterator(first=False)

                if request_iter is None and continue_requests:
                    # Out of requests so stop
                    continue_requests = False
                    # Update scheduler state that requests were exhausted
                    with self.state_update_lock:
                        self.scheduler_state.end_queuing_constraints["request_iter"] = {
                            "status": "exhausted",
                            "time": time.time(),
                        }
                        self.scheduler_state.end_queuing_time = time.time()

                if continue_requests and message is None:
                    message, continue_requests = self._populate_requests_next_message(
                        request_iter, scheduler_start_time
                    )
                    if message is None:
                        # No message returned because request_iter is exhausted
                        request_iter = None

                if message is not None:
                    with contextlib.suppress(queue.Full):
                        self.requests_queue.put(
                            message[0], timeout=settings.scheduler_poll_interval
                        )
                        self.pending_updates_queue.sync_put(message[1])
                        message = None

                if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                    last_check_time = time.time()
                    continue_requests = (
                        continue_requests and not self.shutdown_event.is_set()
                    )
                    yield None  # Yield to check for error in wrapper to stop
        except Exception as err:  # noqa: BLE001
            print(f"******EXCEPTION in _populate_requests_generator: {err}")
            self.error_event.set()
            raise err
        finally:
            self.pending_requests_complete.set()

    def _populate_requests_create_iterator(
        self, first: bool = False
    ) -> Iterator[RequestT] | None:
        if first:
            # First invocation, get a new iterator if not already one
            return (
                iter(self.requests)
                if not isinstance(self.requests, Iterator)
                else self.requests
            )

        if self.infinite_requests is True and isinstance(self.requests, Iterator):
            # Out of requests and infinite set to True, but request_iter is Iterator
            # Cannot create new, raise RuntimeError
            raise RuntimeError(
                f"Requests iterator {self.requests} exhausted and "
                "infinite_requests is set to True"
            )

        if self.infinite_requests is not False and isinstance(self.requests, Iterable):
            # Out of requests and infinite set to True or set to default
            # Create new iterator out of the Iterable
            return iter(self.requests)

        # Either infinite is False for Iterable or Iterator
        # or infinite is None (default) for Iterator
        # So, return None to stop
        return None

    def _populate_requests_next_message(
        self, request_iter: Iterator[RequestT], scheduler_start_time: float
    ) -> tuple[tuple[bytes, bytes] | None, bool]:
        try:
            request = next(request_iter)
            request_id = (
                request.request_id or request.id or request.id_ or str(uuid.uuid4())
            )
            request_info = ScheduledRequestInfo[MeasuredRequestTimingsT](
                request_id=request_id,
                status="queued",
                scheduler_node_id=-1,
                scheduler_process_id=0,
                scheduler_start_time=scheduler_start_time,
            )
            state, continue_requests, _ = self._update_state(request_info)

            request_msg = MsgpackEncoding.encode((request, request_info))
            update_msg = (None, request, request_info, state)

            return (request_msg, update_msg), continue_requests
        except StopIteration:
            return None, True

    def _populate_updates_generator(self):
        """Generator for populating updates from workers."""
        last_check_time = time.time()
        last_state: SchedulerState = None
        continue_processing = True
        shutdown_set = False
        canceled_remaining = False

        try:
            while (
                continue_processing
                or last_state is None
                or (last_state.processed_requests < last_state.created_requests)
            ):
                next_state, continue_updates = self._populate_updates_process_next()
                if next_state is not None:
                    last_state = next_state
                    continue_processing = continue_processing and continue_updates

                if not continue_processing and not shutdown_set:
                    self.shutdown_event.set()
                    shutdown_set = True
                    time.sleep(
                        settings.scheduler_poll_interval
                    )  # Ensure shut down propagates

                if not continue_processing and not canceled_remaining:
                    # We've shut down, no more requests will be added, cancel remaining
                    next_state = self._populate_updates_cancel_remaining()
                    if next_state is not None:
                        last_state = next_state
                    canceled_remaining = True

                if (time.time() - last_check_time) >= settings.scheduler_poll_interval:
                    last_check_time = time.time()
                    if not shutdown_set and self.shutdown_event.is_set():
                        shutdown_set = True
                        continue_processing = False
                        with self.state_update_lock:
                            self.scheduler_state.end_queuing_constraints[
                                "shutdown_event"
                            ] = {
                                "status": "set",
                                "time": time.time(),
                            }
                            self.scheduler_state.end_processing_time = time.time()

                    yield None  # Yield to check for error in wrapper to stop
        except Exception as err:  # noqa: BLE001
            print(f"******EXCEPTION in _populate_updates_generator: {err}")
            self.error_event.set()
            raise err
        finally:
            self.pending_updates_complete.set()

    def _populate_updates_process_next(
        self,
    ) -> tuple[SchedulerState | None, bool]:
        try:
            message = self.updates_queue.get(timeout=settings.scheduler_poll_interval)
            response, request, request_info = MsgpackEncoding.decode(message)

            scheduler_state, _, continue_updates = self._update_state(request_info)
            self.pending_updates_queue.sync_put(
                (response, request, request_info, scheduler_state)
            )

            return scheduler_state, continue_updates
        except queue.Empty:
            return None, True

    def _populate_updates_cancel_remaining(
        self,
    ) -> SchedulerState | None:
        last_state = None

        while True:
            try:
                message = self.requests_queue.get(
                    timeout=settings.scheduler_poll_interval
                )
                request, request_info = MsgpackEncoding.decode(message)

                # Send start first
                request_info.status = "in_progress"
                scheduler_state, _, _ = self._update_state(request_info)
                self.pending_updates_queue.sync_put(
                    (None, request, request_info.model_copy(), scheduler_state)
                )

                # Send canceled
                request_info.status = "cancelled"
                request_info.error = "Request was cancelled"
                request_info.scheduler_timings.resolve_end = time.time()
                scheduler_state, _, _ = self._update_state(request_info)
                self.pending_updates_queue.sync_put(
                    (None, request, request_info, scheduler_state)
                )

                last_state = scheduler_state
            except queue.Empty:
                if self.pending_requests_complete.is_set():
                    # no more requests being pushed to queue, safe to exit
                    break

        return last_state
