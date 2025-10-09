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
from typing import Annotated, Generic, Literal

try:
    import uvloop

    HAS_UVLOOP: Annotated[
        bool, "Flag indicating if uvloop is available for event loop optimization"
    ] = True
except ImportError:
    uvloop = None

    HAS_UVLOOP: Annotated[
        bool, "Flag indicating if uvloop is available for event loop optimization"
    ] = False


from guidellm.scheduler.objects import (
    BackendInterface,
    MultiTurnRequestT,
    RequestT,
    ResponseT,
    ScheduledRequestInfo,
    SchedulerMessagingPydanticRegistry,
)
from guidellm.scheduler.strategies import ScheduledRequestTimings
from guidellm.utils import (
    InterProcessMessaging,
    wait_for_sync_barrier,
    wait_for_sync_event,
    wait_for_sync_objects,
)

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
        backend: BackendInterface[RequestT, ResponseT],
        request_timings: ScheduledRequestTimings,
        async_limit: int,
        startup_barrier: ProcessingBarrier,
        requests_generated_event: ProcessingEvent,
        constraint_reached_event: ProcessingEvent,
        shutdown_event: ProcessingEvent,
        error_event: ProcessingEvent,
    ):
        """
        Initialize worker process instance.

        :param messaging: Inter-process communication interface for request coordination
        :param backend: Backend instance for processing requests
        :param request_timings: Timing strategy for request scheduling
        :param async_limit: Maximum concurrent requests this worker can handle
        :param startup_barrier: Multiprocessing barrier for coordinated startup
        :param requests_generated_event: Event signaling when request generation is
            complete
        :param constraint_reached_event: Event signaling when processing constraints
            are met
        :param shutdown_event: Event for signaling graceful shutdown
        :param error_event: Event for signaling error conditions across processes
        """
        self.messaging = messaging
        self.backend = backend
        self.request_timings = request_timings
        self.async_limit = async_limit
        self.startup_barrier = startup_barrier
        self.requests_generated_event = requests_generated_event
        self.constraint_reached_event = constraint_reached_event
        self.shutdown_event = shutdown_event
        self.error_event = error_event

        # Internal states
        self.startup_completed = False
        self.backend_started = False
        self.messaging_started = False

    def run(self):
        """
        Main entry point for worker process execution.

        Initializes asyncio event loop with optional uvloop optimization and starts
        worker async operations. Handles event loop cleanup for forked processes.

        :raises RuntimeError: If worker encounters unrecoverable error during execution
        """
        try:
            if HAS_UVLOOP:
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            asyncio.run(self.run_async())
        except Exception as err:
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
        stop_task = asyncio.create_task(self._stop_monitor())
        request_proc_task = asyncio.create_task(self._process_requests())
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

    async def _stop_monitor(
        self,
    ) -> Literal["error_event", "shutdown_event"]:
        exit_key = await wait_for_sync_objects(
            {
                "error_event": self.error_event,
                "shutdown_event": self.shutdown_event,
            },
            poll_interval=self.messaging.poll_interval,
        )

        if exit_key == "error_event":
            raise RuntimeError(
                f"Worker process {self.messaging.worker_index} received error signal."
            )

    async def _process_requests(self):
        try:
            # 1. Start up synchronization (backend, messaging, and other processes)
            # 2. Messaging startup, receive requests until requests_generated event
            await self._processing_startup()

            # 3. Run process requests loop until constraint_reached event
            processing_task = asyncio.create_task(self._process_requests_loop())
            await wait_for_sync_event(
                self.constraint_reached_event,
                poll_interval=self.messaging.poll_interval,
            )
            processing_task.cancel()

            # 4. Cancel pending requests until proc canceled (manual, shutdown, error)
            await self._cancel_requests_loop()
        finally:
            # 5. On cancel, shut down event, error event, or internal error:
            #    attempt to shut down this worker cleanly (stop backend and messaging)
            await self._processing_shutdown()

    async def _processing_startup(self):
        # Get backend ready
        await self.backend.process_startup()
        self.backend_started = True
        await self.backend.validate()

        # Get messaging system ready
        await self.messaging.start(
            receive_stop_criteria=[self.requests_generated_event],
            pydantic_models=list(SchedulerMessagingPydanticRegistry.registry.values()),
        )
        self.messaging_started = True

        # Wait for all processes to be ready
        await wait_for_sync_barrier(
            self.startup_barrier,
            poll_interval=self.messaging.poll_interval,
        )

        self.startup_completed = True

    async def _processing_shutdown(self):
        if self.backend_started:
            await self.backend.process_shutdown()
            self.backend_started = False

        if self.messaging_started:
            await self.messaging.stop()
            self.messaging_started = False

        self.startup_completed = False

    async def _process_requests_loop(self):
        try:
            # Run request processing
            async_semaphore = asyncio.Semaphore(self.async_limit)
            pending_tasks: set[asyncio.Task] = set()

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
        except asyncio.CancelledError as err:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

            raise err

    async def _cancel_requests_loop(self):
        while True:
            try:
                request: RequestT
                request_info: ScheduledRequestInfo
                request, request_info = await self.messaging.get(
                    timeout=self.messaging.poll_interval
                )
            except asyncio.TimeoutError:
                continue

            request_info.scheduler_node_id = self.messaging.worker_index
            request_info.error = "Request was cancelled"
            request_info.scheduler_timings.resolve_end = time.time()
            self._send_update("cancelled", None, request, request_info)

    async def _process_next_request(self):
        request: RequestT | MultiTurnRequestT[RequestT] | None = None
        request_info: ScheduledRequestInfo | None = None
        response: ResponseT | None = None

        try:
            # Pull request from the queue
            request, request_info = await self.messaging.get()

            if isinstance(request, list | tuple):
                raise NotImplementedError("Multi-turn requests are not yet supported")

            # Calculate targeted start and set pending state for request
            request_info.scheduler_node_id = self.messaging.worker_index
            request_info.scheduler_timings.dequeued = time.time()
            target_start = (
                request_info.scheduler_start_time + self.request_timings.next_offset()
            )
            request_info.scheduler_timings.targeted_start = target_start
            self._send_update("pending", response, request, request_info)

            # Schedule the request
            current_time = time.time()
            request_info.scheduler_timings.scheduled_at = current_time
            if target_start > current_time:
                await asyncio.sleep(target_start - current_time)
                # Adapt delay so that scheduled at reflects the sleep time
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
        new_status: Literal[
            "pending", "in_progress", "completed", "errored", "cancelled"
        ],
        response: ResponseT | None,
        request: RequestT | MultiTurnRequestT[RequestT],
        request_info: ScheduledRequestInfo,
    ):
        prev_status = request_info.status

        if new_status == prev_status:
            # already sent this update, don't send again
            return

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
