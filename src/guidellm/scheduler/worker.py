"""
Worker process implementation for distributed request execution and coordination.

Manages individual worker processes within the scheduler system, handling request
lifecycle from queue consumption through backend processing and status publication.
Workers coordinate with other processes through barriers and events, apply timing
strategies for request scheduling, maintain concurrency limits, and publish real-time
status updates throughout request processing.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from typing import Annotated, Generic, Literal

try:
    import uvloop

    HAS_UVLOOP: Annotated[
        bool, "Flag indicating uvloop availability for event loop optimization"
    ] = True
except ImportError:
    uvloop = None  # type: ignore[assignment] # Optional dependency

    HAS_UVLOOP = False


from guidellm.logger import logger
from guidellm.scheduler.dag import DAGExecutionState
from guidellm.scheduler.schemas import (
    BackendInterface,
    ConversationGraph,
    HistoryT,
    RequestT,
    ResponseT,
)
from guidellm.scheduler.strategies import SchedulingStrategy
from guidellm.schemas import RequestInfo
from guidellm.utils.messaging import InterProcessMessaging
from guidellm.utils.synchronous import (
    wait_for_sync_barrier,
    wait_for_sync_event,
    wait_for_sync_objects,
)

__all__ = ["WorkerProcess"]


class WorkerProcess(Generic[RequestT, ResponseT]):
    """
    Worker process for distributed request execution in the scheduler system.

    Manages complete request lifecycle including queue consumption, backend processing,
    timing strategy application, and status publication. Coordinates with other workers
    through synchronization primitives while maintaining concurrency limits and handling
    graceful shutdown scenarios including errors and cancellations.

    Example:
    ::
        worker = WorkerProcess(
            worker_index=0,
            messaging=messaging_interface,
            backend=backend_instance,
            strategy=timing_strategy,
            async_limit=10,
            fut_scheduling_time_limit=5.0,
            startup_barrier=barrier,
            requests_generated_event=generated_event,
            constraint_reached_event=constraint_event,
            shutdown_event=shutdown,
            error_event=error,
        )
        worker.run()
    """

    def __init__(
        self,
        worker_index: int,
        messaging: InterProcessMessaging[
            tuple[ResponseT | None, RequestT, RequestInfo],
            ConversationGraph[RequestT],
        ],
        backend: BackendInterface[RequestT, ResponseT],
        strategy: SchedulingStrategy,
        async_limit: int,
        fut_scheduling_time_limit: float,
        startup_barrier: ProcessingBarrier,
        requests_generated_event: ProcessingEvent,
        constraint_reached_event: ProcessingEvent,
        shutdown_event: ProcessingEvent,
        error_event: ProcessingEvent,
    ):
        """
        Initialize worker process instance.

        :param worker_index: Unique identifier for this worker within the process group
        :param messaging: Inter-process messaging interface for request coordination
        :param backend: Backend interface for processing requests
        :param strategy: Scheduling strategy for determining request timing
        :param async_limit: Maximum concurrent requests this worker can process
        :param fut_scheduling_time_limit: Maximum time in seconds to schedule requests
            into the future
        :param startup_barrier: Synchronization barrier for coordinated startup
        :param requests_generated_event: Event signaling request generation completion
        :param constraint_reached_event: Event signaling processing constraint reached
        :param shutdown_event: Event signaling graceful shutdown request
        :param error_event: Event signaling error conditions across processes
        """
        self.worker_index = worker_index
        self.messaging = messaging
        self.backend = backend
        self.strategy = strategy
        self.async_limit = async_limit
        self.fut_scheduling_time_limit = fut_scheduling_time_limit
        self.startup_barrier = startup_barrier
        self.requests_generated_event = requests_generated_event
        self.constraint_reached_event = constraint_reached_event
        self.shutdown_event = shutdown_event
        self.error_event = error_event

        # Internal states
        self.startup_completed = False
        self.backend_started = False
        self.messaging_started = False
        self.turns_queue: list[DAGExecutionState[RequestT, ResponseT]] = []
        self.graph_request_infos: dict[
            str, dict[str, RequestInfo]
        ] = {}  # TODO: Is this the best design?

    def run(self):
        """
        Main entry point for worker process execution.

        Initializes asyncio event loop with optional uvloop optimization and executes
        worker async operations. Handles event loop cleanup and error propagation.

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

        Orchestrates concurrent execution of request processing and shutdown monitoring.
        Handles task cleanup, error propagation, and cancellation coordination when any
        task completes or encounters an error.

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
    ) -> None:
        """
        Monitor shutdown and error events for worker termination.
        :raises RuntimeError if the work process received an error signal.
        """
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
        """
        Manage request processing lifecycle from startup to shutdown.

        Coordinates startup synchronization, processes requests until constraints are
        reached, then cancels pending requests until shutdown or error occurs.
        """
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
        """Initialize backend, messaging, and synchronize with other workers."""
        # Get backend ready
        await self.backend.process_startup()
        self.backend_started = True
        await self.backend.validate()

        # Get messaging system ready
        await self.messaging.start(
            receive_stop_criteria=[self.requests_generated_event]
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
        """
        Process requests continuously until cancelled with concurrency limits.

        Schedules and processes requests according to the timing strategy while
        maintaining the configured concurrency limit through semaphore coordination.
        """
        pending_tasks: set[asyncio.Task] = set()
        try:
            # Run request processing
            async_semaphore = asyncio.Semaphore(self.async_limit)

            def _task_done(task: asyncio.Task):
                pending_tasks.discard(task)
                async_semaphore.release()

                if not task.cancelled():
                    if exception := task.exception():
                        raise exception

            # Main loop; loop until canceled
            while True:
                await async_semaphore.acquire()
                request_time = await self.strategy.next_request_time(
                    worker_index=self.worker_index
                )

                if (
                    time_until := request_time - time.time()
                ) >= self.fut_scheduling_time_limit:
                    await asyncio.sleep(time_until - self.fut_scheduling_time_limit)

                request_task = asyncio.create_task(
                    self._process_next_graph_node(target_start=request_time)
                )
                pending_tasks.add(request_task)
                request_task.add_done_callback(_task_done)
        except asyncio.CancelledError as err:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

            raise err

    async def _cancel_requests_loop(self):
        """Cancel all remaining queued graph nodes until worker process terminates."""
        for state in self.turns_queue:
            remaining = state.abort()
            infos = self.graph_request_infos.get(state.graph.graph_id, {})
            for node_id in remaining:
                node = state.graph.nodes[node_id]
                request_info = infos.get(node_id)
                if request_info is not None:
                    request_info.scheduler_node_id = self.messaging.worker_index or -1
                    request_info.error = "Request was cancelled"
                    request_info.timings.resolve_end = time.time()
                    self._send_update("cancelled", None, node.request, request_info)
        self.turns_queue.clear()
        self.graph_request_infos.clear()

        while True:
            try:
                graph: ConversationGraph[RequestT] = await self.messaging.get(
                    timeout=self.messaging.poll_interval
                )
            except asyncio.TimeoutError:
                continue
            for node_id, node in graph.nodes.items():
                request_info = graph.request_infos.get(node_id)
                if request_info is not None:
                    request_info.scheduler_node_id = self.messaging.worker_index or -1
                    request_info.error = "Request was cancelled"
                    request_info.timings.resolve_end = time.time()
                    self._send_update("cancelled", None, node.request, request_info)

    async def _get_next_ready_node(
        self,
    ) -> tuple[DAGExecutionState[RequestT, ResponseT], str]:
        """Get the next ready node from active graphs or dequeue a new graph.

        Preference order:
        1. A schedulable node from an in-flight graph (claim it).
        2. Wait until a think-time gate expires or a new graph arrives.
        3. Block on IPC for a new graph when nothing is delayed.
        """
        while True:
            for state in self.turns_queue:
                ready = state.get_ready_nodes()
                if ready:
                    node_id = ready[0]
                    state.claim_node(node_id)
                    return state, node_id

            earliest: float | None = None
            for state in self.turns_queue:
                next_at = state.next_delayed_ready_at()
                if next_at is not None and (earliest is None or next_at < earliest):
                    earliest = next_at

            if earliest is not None:
                timeout = max(0.0, earliest - time.time())
                try:
                    graph: ConversationGraph[RequestT] = await self.messaging.get(
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    continue
            else:
                graph = await self.messaging.get()

            state = DAGExecutionState(graph)
            self.turns_queue.append(state)
            self.graph_request_infos[graph.graph_id] = dict(graph.request_infos)

    async def _process_next_graph_node(self, target_start: float) -> None:
        """Process a single graph node from queue to completion."""
        request: RequestT | None = None
        request_info: RequestInfo | None = None
        response: ResponseT | None = None
        state: DAGExecutionState[RequestT, ResponseT] | None = None
        node_id: str | None = None

        try:
            state, node_id = await self._get_next_ready_node()
            request, request_info = self._prepare_node(state, node_id, target_start)
            response = await self._execute_node(
                state, node_id, request, request_info, target_start
            )
            self._finalize_node(state, node_id, request, request_info, response)
            response = request = None
        except asyncio.CancelledError:
            if request is not None and request_info is not None:
                request_info.error = "Request was cancelled"
                request_info.timings.resolve_end = time.time()
                self._send_update("cancelled", response, request, request_info)
            raise
        except Exception as exc:  # noqa: BLE001
            self._handle_node_error(
                exc, state, node_id, request, request_info, response
            )
        finally:
            if request_info is not None:
                self.strategy.request_completed(request_info)

    def _prepare_node(
        self,
        state: DAGExecutionState[RequestT, ResponseT],
        node_id: str,
        target_start: float,
    ) -> tuple[RequestT, RequestInfo]:
        """Look up request and RequestInfo for a node, set dequeue timing."""
        node = state.graph.nodes[node_id]
        request = node.request
        infos = self.graph_request_infos.get(state.graph.graph_id, {})
        request_info = infos.get(node_id)
        if request_info is None:
            raise RuntimeError(
                f"No RequestInfo for node '{node_id}' in graph '{state.graph.graph_id}'"
            )
        request_info.timings.dequeued = time.time()
        request_info.scheduler_node_id = self.messaging.worker_index or -1
        request_info.timings.targeted_start = target_start
        self._send_update("pending", None, request, request_info)
        return request, request_info

    async def _execute_node(
        self,
        state: DAGExecutionState[RequestT, ResponseT],
        node_id: str,
        request: RequestT,
        request_info: RequestInfo,
        target_start: float,
    ) -> ResponseT | None:
        """Schedule, assemble history, and execute a node via backend."""
        effective_target_start = await self.strategy.resolve_dequeued_target_start(
            self.worker_index,
            target_start,
            request_info.settings,
        )
        if effective_target_start != target_start:
            request_info.timings.targeted_start = effective_target_start
            self._send_update("pending", None, request, request_info)
        await self._schedule_request(request, request_info, effective_target_start)

        history_pairs = state.assemble_history(node_id)
        history: HistoryT[RequestT, ResponseT] | None = (
            history_pairs if history_pairs else None
        )
        response: ResponseT | None = None
        async for resp, info in self.backend.resolve(  # type: ignore[attr-defined]
            request, request_info, history
        ):
            if info is None:
                raise RuntimeError("Received invalid request info from backend")
            if resp is None and info.timings.first_token_iteration is not None:
                self._send_update("first_token", None, request, info)
            response = resp
        return response

    def _finalize_node(
        self,
        state: DAGExecutionState[RequestT, ResponseT],
        node_id: str,
        request: RequestT,
        request_info: RequestInfo,
        response: ResponseT | None,
    ) -> None:
        """Mark node completed and clean up finished graphs."""
        request_info.timings.resolve_end = time.time()
        self._send_update("completed", response, request, request_info)
        state.mark_completed(node_id, request, response)
        if state.is_complete:
            self.turns_queue.remove(state)
            self.graph_request_infos.pop(state.graph.graph_id, None)

    def _handle_node_error(  # noqa: PLR0913
        self,
        exc: Exception,
        state: DAGExecutionState[RequestT, ResponseT] | None,
        node_id: str | None,  # noqa: ARG002
        request: RequestT | None,
        request_info: RequestInfo | None,
        response: ResponseT | None,
    ) -> None:
        """Report error for the failed node and abort the entire graph."""
        if request is not None and request_info is not None:
            request_info.error = repr(exc)
            request_info.traceback = traceback.format_exc()
            request_info.timings.resolve_end = time.time()
            self._send_update("errored", response, request, request_info)
            logger.opt(exception=True).debug(
                f"Backend exception for request {request_info.request_id}"
            )
        else:
            logger.opt(exception=True).debug(
                "Graph node failed: worker={} graph={} node={}",
                self.worker_index,
                state.graph.graph_id if state is not None else None,
                node_id,
            )
        if state is not None:
            remaining = state.abort()
            infos = self.graph_request_infos.get(state.graph.graph_id, {})
            for rem_id in remaining:
                rem_node = state.graph.nodes[rem_id]
                rem_info = infos.get(rem_id)
                if rem_info is not None:
                    rem_info.error = "Request was cancelled"
                    rem_info.timings.resolve_end = time.time()
                    self._send_update("cancelled", None, rem_node.request, rem_info)
            if state in self.turns_queue:
                self.turns_queue.remove(state)
            self.graph_request_infos.pop(state.graph.graph_id, None)

    async def _schedule_request(
        self, request: RequestT, request_info: RequestInfo, target_start: float
    ):
        request_info.timings.scheduled_at = request_info.timings.dequeued
        if target_start > (current_time := time.time()):
            await asyncio.sleep(target_start - current_time)
            # Adapt delay so that scheduled at reflects the sleep time
            request_info.timings.scheduled_at = target_start

        # Process the request with the backend
        request_info.timings.resolve_start = time.time()
        self._send_update("in_progress", None, request, request_info)

    def _send_update(
        self,
        new_status: Literal[
            "pending",
            "in_progress",
            "first_token",
            "completed",
            "errored",
            "cancelled",
        ],
        response: ResponseT | None,
        request: RequestT,
        request_info: RequestInfo,
    ):
        """
        Publish request status update through messaging system.

        Updates request status and publishes to messaging queue for coordinator
        consumption. Prevents duplicate status updates for the same state.

        :param new_status: New status for the request
        :param response: Response object if available, None otherwise
        :param request: Request object being processed
        :param request_info: Request metadata and timing information
        :raises Exception: If messaging system fails to publish the update
        """
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
