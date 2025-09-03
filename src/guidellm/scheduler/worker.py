"""
Individual worker process management for multi-process request execution.

Manages worker processes that handle request scheduling, backend processing, and
coordination in distributed benchmark environments. Workers consume requests from
queues, apply timing strategies, process requests through backends, and publish
status updates while maintaining synchronization across the process group.
"""

from __future__ import annotations

import asyncio
import time
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Event as ThreadingEvent
from typing import Generic, Literal

try:
    import uvloop

    HAS_UVLOOP = True
except ImportError:
    uvloop = None

    HAS_UVLOOP = False

import contextlib

from guidellm.scheduler.objects import (
    BackendInterface,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerMessagingPydanticRegistry,
)
from guidellm.scheduler.strategy import ScheduledRequestTimings
from guidellm.utils import InterProcessMessaging, synchronous_to_exitable_async

__all__ = ["WorkerProcess"]


class WorkerProcess(Generic[RequestT, ResponseT]):
    """
    Individual worker process for distributed request execution and coordination.

    Manages the complete request lifecycle from queue consumption through backend
    processing and status publication. Coordinates with other workers through
    barriers and events while maintaining configurable concurrency limits and
    timing strategies for request scheduling.

    Example:
    ::
        worker = WorkerProcess(
            messaging=messaging_interface,
            async_limit=10,
            startup_barrier=barrier,
            shutdown_event=shutdown,
            error_event=error,
            backend=backend_instance,
            request_timings=timing_strategy
        )
        worker.run()
    """

    def __init__(
        self,
        messaging: InterProcessMessaging[
            tuple[
                ResponseT | None,
                RequestT | MultiTurnRequestT[RequestT],
                ScheduledRequestInfo,
            ],
        ],
        async_limit: int,
        startup_barrier: ProcessingBarrier,
        shutdown_event: ProcessingEvent,
        error_event: ProcessingEvent,
        requests_completed_event: ProcessingEvent,
        backend: BackendInterface[RequestT, ResponseT],
        request_timings: ScheduledRequestTimings,
    ):
        """
        Initialize worker process instance.

        :param messaging: Inter-process communication interface for request coordination
        :param async_limit: Maximum concurrent requests this worker can handle
        :param startup_barrier: Multiprocessing barrier for coordinated startup
        :param shutdown_event: Event for signaling graceful shutdown
        :param error_event: Event for signaling error conditions across processes
        :param requests_completed_event: Event for signaling when the main process
            has stopped sending requests / all requests are added to the queue
        :param backend: Backend instance for processing requests
        :param request_timings: Timing strategy for request scheduling
        """
        self.messaging = messaging
        self.async_limit = async_limit
        self.startup_barrier = startup_barrier
        self.shutdown_event = shutdown_event
        self.error_event = error_event
        self.requests_completed_event = requests_completed_event
        self.backend = backend
        self.request_timings = request_timings
        self.startup_completed = False

    def run(self):
        """
        Main entry point for worker process execution.

        Initializes asyncio event loop with optional uvloop optimization and starts
        worker async operations. Handles event loop cleanup for forked processes.

        :raises RuntimeError: If worker encounters unrecoverable error during execution
        """
        try:
            loop = (
                asyncio.new_event_loop() if not HAS_UVLOOP else uvloop.new_event_loop()
            )
            asyncio.set_event_loop(loop)
            asyncio.run(self.run_async())
        except Exception as err:
            print(f"******EXCEPTION in worker {self.messaging.worker_index} run: {err}")
            self.error_event.set()
            raise RuntimeError(
                f"Worker process {self.messaging.worker_index} encountered an "
                f"error: {err}"
            ) from err

    async def run_async(self):
        """
        Execute main asynchronous worker process logic.

        Orchestrates concurrent execution of request processing and shutdown monitoring
        tasks. Handles task cleanup, error propagation, and cancellation coordination
        when any task completes or fails.

        :raises RuntimeError: If worker tasks encounter unrecoverable errors
        :raises asyncio.CancelledError: If worker process was cancelled
        """
        stop_task = asyncio.create_task(self._run_async_stop_processing())
        request_proc_task = asyncio.create_task(self._run_async_requests_processing())
        caller_cancelled = False

        try:
            await asyncio.wait(
                [stop_task, request_proc_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            caller_cancelled = True

        stop_task.cancel()
        request_proc_task.cancel()

        try:
            # Ensure all child tasks cancel correctly
            await asyncio.wait(
                [stop_task, request_proc_task], return_when=asyncio.ALL_COMPLETED
            )
        except asyncio.CancelledError:
            caller_cancelled = True

        if (
            task_err := (
                request_proc_task.exception()
                if not request_proc_task.cancelled()
                else stop_task.exception()
                if not stop_task.cancelled()
                else None
            )
        ) is not None:
            raise RuntimeError(
                f"Worker process {self.messaging.worker_index} encountered an "
                f"error: {task_err}"
            ) from task_err

        if caller_cancelled:
            raise asyncio.CancelledError("Worker process was cancelled")

    async def _run_async_stop_processing(
        self,
    ) -> Literal["error_event", "shutdown_event"]:
        exit_reason, _ = await synchronous_to_exitable_async(
            synchronous=None,
            exit_events={
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            poll_interval=self.messaging.poll_interval,
        )

        if exit_reason in {"shutdown_event", "canceled"}:
            raise asyncio.CancelledError("Worker process shutdown event set")

        if exit_reason == "error_event":
            raise RuntimeError(
                f"Worker process {self.messaging.worker_index} received error signal."
            )

        raise RuntimeError(
            f"Worker process {self.messaging.worker_index} received unknown exit: "
            f"{exit_reason}"
        )

    async def _run_async_requests_processing(self):
        try:
            # Get backend ready for reqeuests
            await self.backend.process_startup()
            await self.backend.validate()

            # Get messaging system ready
            all_requests_processed = ThreadingEvent()
            await self.messaging.start(
                send_stop_criteria=[all_requests_processed],
                receive_stop_criteria=[self.requests_completed_event, self.error_event],
                pydantic_models=list(
                    SchedulerMessagingPydanticRegistry.registry.values()
                ),
            )

            # Wait for all processes to be ready
            barrier_exit_reason, _ = await synchronous_to_exitable_async(
                synchronous=None,
                exit_barrier=self.startup_barrier,
                poll_interval=self.messaging.poll_interval,
            )

            if barrier_exit_reason not in ["barrier", "canceled"]:
                raise RuntimeError(
                    f"Worker process {self.messaging.worker_index} failed to "
                    f"synchronize at startup: {barrier_exit_reason}"
                )

            self.startup_completed = True

            # Run request processing
            async_semaphore = asyncio.Semaphore(self.async_limit)
            pending_tasks = set()

            def _task_done(task):
                pending_tasks.discard(task)
                async_semaphore.release()

                if not task.cancelled() and (exception := task.exception()):
                    raise exception

            # Main loop; loop until canceled
            while True:
                await async_semaphore.acquire()
                request_task = asyncio.create_task(self._process_next_request())
                pending_tasks.add(request_task)
                request_task.add_done_callback(_task_done)
        except (asyncio.CancelledError, Exception) as err:
            if self.startup_completed:
                await self._cancel_remaining_requests(
                    pending_tasks, all_requests_processed
                )
                await self.messaging.stop()
                await self.backend.process_shutdown()

            raise err

    async def _process_next_request(self):
        request: RequestT | MultiTurnRequestT[RequestT] | None = None
        request_info: ScheduledRequestInfo | None = None
        response: ResponseT | None = None

        try:
            # Pull request from the queue
            request, request_info = await self.messaging.get()
            current_time = time.time()
            request_info.status = "pending"
            request_info.scheduler_timings.dequeued = current_time

            if isinstance(request, (list, tuple)):
                raise NotImplementedError("Multi-turn requests are not yet supported")

            # Schedule the request for targeted time
            target_start = (
                request_info.scheduler_start_time + self.request_timings.next_offset()
            )
            request_info.scheduler_timings.targeted_start = target_start
            request_info.scheduler_timings.scheduled_at = current_time

            if target_start > current_time:
                await asyncio.sleep(target_start - current_time)
                # adapt delay so that scheduled at reflects the sleep time
                request_info.scheduler_timings.scheduled_at = target_start

            # Process the request with the backend
            request_info.scheduler_timings.resolve_start = time.time()
            self._send_update("in_progress", response, request, request_info)
            async for resp, info in self.backend.resolve(request, request_info, None):
                response = resp
                request_info = info

            # Complete the request
            request_info.scheduler_timings.resolve_end = time.time()
            self._send_update("completed", response, request, request_info)

            response = request = request_info = None
        except asyncio.CancelledError:
            # Handle cancellation
            if request is not None and request_info is not None:
                request_info.error = "Request was cancelled"
                request_info.scheduler_timings.resolve_end = time.time()
                self._send_update("cancelled", response, request, request_info)
            raise
        except Exception as exc:  # noqa: BLE001
            if request is not None and request_info is not None:
                request_info.error = str(exc)
                request_info.scheduler_timings.resolve_end = time.time()
                self._send_update("errored", response, request, request_info)

    def _send_update(
        self,
        new_status: Literal["in_progress", "completed", "errored", "cancelled"],
        response: ResponseT | None,
        request: RequestT | MultiTurnRequestT[RequestT],
        request_info: ScheduledRequestInfo,
    ):
        prev_status = request_info.status

        try:
            request_info.status = new_status
            request_info = (
                request_info.model_copy()
                if new_status not in {"completed", "errored", "cancelled"}
                else request_info  # last update, don't need to copy
            )
            self.messaging.put_sync(
                (response, request, request_info),
                timeout=-1,
            )
            prev_status = new_status
        except Exception as exc:
            # Reset status to last one that succeeded or started function with
            # Calling logic can retry after handling error, if possible
            request_info.status = prev_status
            raise exc

    async def _cancel_remaining_requests(
        self, pending_tasks: set[asyncio.Task], all_requests_processed: ThreadingEvent
    ):
        # Cancel any tasks that were active tasks
        cancel_tasks = []
        for task in pending_tasks:
            if not task.done():
                task.cancel()
                cancel_tasks.append(task)

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # Cancel any tasks pending on the queue
        while not self.messaging.receive_stopped_event.is_set():
            # Loop until we know nothing else will be added
            with contextlib.suppress((asyncio.TimeoutError, Exception)):
                request, request_info = await self.messaging.get(
                    timeout=self.messaging.poll_interval
                )
                request_info.error = "Request was cancelled"
                request_info.scheduler_timings.resolve_end = time.time()
                self._send_update("cancelled", None, request, request_info)

        all_requests_processed.set()
        await synchronous_to_exitable_async(
            synchronous=None,
            exit_events={"send_stopped": self.messaging.send_stopped_event},
            poll_interval=self.messaging.poll_interval,
        )
