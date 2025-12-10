"""
Inter-process messaging abstractions for distributed scheduler coordination.

Provides high-level interfaces for asynchronous message passing between worker
processes using various transport mechanisms including queues and pipes. Supports
configurable encoding, serialization, error handling, and flow control with
buffering and stop event coordination for distributed scheduler operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Event as ThreadingEvent
from typing import Any, Generic, Protocol, TypeVar, cast

import culsans
from pydantic import BaseModel

from guidellm.utils.encoding import (
    EncodingTypesAlias,
    MessageEncoding,
    SerializationTypesAlias,
)

__all__ = [
    "InterProcessMessaging",
    "InterProcessMessagingManagerQueue",
    "InterProcessMessagingPipe",
    "InterProcessMessagingQueue",
    "MessagingStopCallback",
    "ReceiveMessageT",
    "SendMessageT",
]

SendMessageT = TypeVar("SendMessageT", bound=Any)
"""Generic type variable for messages sent through the messaging system"""
ReceiveMessageT = TypeVar("ReceiveMessageT", bound=Any)
"""Generic type variable for messages received through the messaging system"""

CheckStopCallableT = Callable[[bool, int], bool]


class MessagingStopCallback(Protocol):
    """Protocol for evaluating stop conditions in messaging operations."""

    def __call__(
        self, messaging: InterProcessMessaging, pending: bool, queue_empty_count: int
    ) -> bool:
        """
        Evaluate whether messaging operations should stop.

        :param messaging: The messaging instance to evaluate
        :param pending: Whether there are pending operations
        :param queue_empty_count: The number of times in a row the queue has been empty
        :return: True if operations should stop, False otherwise
        """
        ...


class InterProcessMessaging(Generic[SendMessageT, ReceiveMessageT], ABC):
    """
    Abstract base for inter-process messaging in distributed scheduler coordination.

    Provides unified interface for asynchronous message passing between scheduler
    components using configurable transport mechanisms, encoding schemes, and
    flow control policies. Manages buffering, serialization, error handling,
    and coordinated shutdown across worker processes for distributed operations.

    Example:
    ::
        from guidellm.utils.messaging import InterProcessMessagingQueue

        messaging = InterProcessMessagingQueue(
            serialization="pickle",
            max_pending_size=100
        )

        await messaging.start()
        await messaging.put(request_data)
        response = await messaging.get(timeout=5.0)
        await messaging.stop()
    """

    STOP_REQUIRED_QUEUE_EMPTY_COUNT: int = 3

    def __init__(
        self,
        mp_context: BaseContext | None = None,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None,
        max_pending_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_done_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        poll_interval: float = 0.1,
        worker_index: int | None = None,
    ):
        """
        Initialize inter-process messaging coordinator.

        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_pending_size: Maximum items in send queue before blocking
        :param max_buffer_send_size: Maximum items in buffer send queue
        :param max_done_size: Maximum items in done queue before blocking
        :param max_buffer_receive_size: Maximum items in buffer receive queue
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        """
        self.worker_index: int | None = worker_index
        self.mp_context = mp_context or multiprocessing.get_context()
        self.serialization = serialization
        self.encoding = encoding
        self.max_pending_size = max_pending_size
        self.max_buffer_send_size = max_buffer_send_size
        self.max_done_size = max_done_size
        self.max_buffer_receive_size = max_buffer_receive_size
        self.poll_interval = poll_interval

        self.send_stopped_event: ThreadingEvent | ProcessingEvent | None = None
        self.receive_stopped_event: ThreadingEvent | ProcessingEvent | None = None
        self.shutdown_event: ThreadingEvent | None = None
        self.buffer_send_queue: culsans.Queue[SendMessageT] | None = None
        self.buffer_receive_queue: culsans.Queue[ReceiveMessageT] | None = None
        self.send_task: asyncio.Task | None = None
        self.receive_task: asyncio.Task | None = None
        self.running = False

    @abstractmethod
    def create_worker_copy(
        self, worker_index: int, **kwargs
    ) -> InterProcessMessaging[ReceiveMessageT, SendMessageT]:
        """
        Create worker-specific copy for distributed process coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured messaging instance for the specified worker
        """
        ...

    @abstractmethod
    def create_send_messages_threads(
        self,
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create send message processing threads for transport implementation.

        :param send_items: Optional collection of items to send during processing
        :param message_encoding: Message encoding configuration for serialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        ...

    @abstractmethod
    def create_receive_messages_threads(
        self,
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create receive message processing threads for transport implementation.

        :param receive_callback: Optional callback for processing received messages
        :param message_encoding: Message encoding configuration for deserialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        ...

    async def start(
        self,
        send_items: Iterable[Any] | None = None,
        receive_callback: Callable[[Any], Any] | None = None,
        send_stop_criteria: (
            list[ThreadingEvent | ProcessingEvent | MessagingStopCallback] | None
        ) = None,
        send_stopped_event: ThreadingEvent | ProcessingEvent | None = None,
        receive_stop_criteria: (
            list[ThreadingEvent | ProcessingEvent | MessagingStopCallback] | None
        ) = None,
        receive_stopped_event: ThreadingEvent | ProcessingEvent | None = None,
        pydantic_models: list[type[BaseModel]] | None = None,
    ):
        """
        Start asynchronous message processing tasks with buffering.

        :param send_items: Optional collection of items to send during processing
        :param receive_callback: Optional callback for processing received messages
        :param send_stop_criteria: Events and callables that trigger send task shutdown
        :param send_stopped_event: Event set when send task has fully stopped
        :param receive_stop_criteria: Events and callables that trigger receive shutdown
        :param receive_stopped_event: Event set when receive task has fully stopped
        :param pydantic_models: Optional list of Pydantic models for serialization
        """
        self.running = True
        self.send_stopped_event = send_stopped_event or ThreadingEvent()
        self.receive_stopped_event = receive_stopped_event or ThreadingEvent()
        self.shutdown_event = ThreadingEvent()
        self.buffer_send_queue = culsans.Queue[SendMessageT](
            maxsize=self.max_buffer_send_size or 0
        )
        self.buffer_receive_queue = culsans.Queue[ReceiveMessageT](
            maxsize=self.max_buffer_receive_size or 0
        )

        message_encoding: MessageEncoding = MessageEncoding(
            serialization=self.serialization,
            encoding=self.encoding,
            pydantic_models=pydantic_models,
        )
        send_stop_criteria = send_stop_criteria or []
        receive_stop_events = receive_stop_criteria or []

        self.send_task = asyncio.create_task(
            self.send_messages_coroutine(
                send_items=send_items,
                message_encoding=message_encoding,
                send_stop_criteria=send_stop_criteria,
            )
        )
        self.receive_task = asyncio.create_task(
            self.receive_messages_coroutine(
                receive_callback=receive_callback,
                message_encoding=message_encoding,
                receive_stop_criteria=receive_stop_events,
            )
        )

    async def stop(self):
        """
        Stop message processing tasks and clean up resources.
        """
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        else:
            raise RuntimeError(
                "shutdown_event is not set; was start() not called or "
                "is this a redundant stop() call?"
            )
        tasks = [self.send_task, self.receive_task]
        tasks_to_run: list[asyncio.Task[Any]] = [
            task for task in tasks if task is not None
        ]
        if len(tasks_to_run) > 0:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks_to_run, return_exceptions=True)
        self.send_task = None
        self.receive_task = None
        if self.worker_index is None:
            if self.buffer_send_queue is not None:
                self.buffer_send_queue.clear()
                await self.buffer_send_queue.aclose()
            if self.buffer_receive_queue is not None:
                self.buffer_receive_queue.clear()
                await self.buffer_receive_queue.aclose()
        self.buffer_send_queue = None
        self.buffer_receive_queue = None
        self.send_stopped_event = None
        self.receive_stopped_event = None
        self.shutdown_event = None
        self.running = False

    async def send_messages_coroutine(
        self,
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        send_stop_criteria: (
            list[ThreadingEvent | ProcessingEvent | MessagingStopCallback] | None
        ),
    ):
        """
        Execute send message processing with encoding and stop condition handling.

        :param send_items: Optional collection of items to send during processing
        :param message_encoding: Message encoding configuration for serialization
        :param send_stop_criteria: Events and callables that trigger send task shutdown
        """
        canceled_event = ThreadingEvent()

        try:
            await asyncio.gather(
                *[
                    asyncio.to_thread(thread, *args)
                    for (thread, args) in self.create_send_messages_threads(
                        send_items=send_items,
                        message_encoding=message_encoding,
                        check_stop=self._create_check_stop_callable(
                            send_stop_criteria, canceled_event
                        ),
                    )
                ]
            )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            if self.send_stopped_event is not None:
                self.send_stopped_event.set()

    async def receive_messages_coroutine(
        self,
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        receive_stop_criteria: (
            list[ThreadingEvent | ProcessingEvent | MessagingStopCallback] | None
        ),
    ):
        """
        Execute receive message processing with decoding and callback handling.

        :param receive_callback: Optional callback for processing received messages
        :param message_encoding: Message encoding configuration for deserialization
        :param receive_stop_criteria: Events and callables that trigger receive shutdown
        """
        canceled_event = ThreadingEvent()

        try:
            await asyncio.gather(
                *[
                    asyncio.to_thread(thread, *args)
                    for thread, args in self.create_receive_messages_threads(
                        receive_callback=receive_callback,
                        message_encoding=message_encoding,
                        check_stop=self._create_check_stop_callable(
                            receive_stop_criteria, canceled_event
                        ),
                    )
                ]
            )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            if self.receive_stopped_event is not None:
                self.receive_stopped_event.set()

    async def get(self, timeout: float | None = None) -> ReceiveMessageT:
        """
        Retrieve a message from receive buffer with optional timeout.

        :param timeout: Maximum time to wait for a message
        :return: Decoded message from the receive buffer
        """
        if self.buffer_receive_queue is None:
            raise RuntimeError(
                "buffer receive queue is None; check start()/stop() calls"
            )
        return await asyncio.wait_for(
            self.buffer_receive_queue.async_get(), timeout=timeout
        )

    def get_sync(self, timeout: float | None = None) -> ReceiveMessageT:
        """
        Retrieve message from receive buffer synchronously with optional timeout.

        :param timeout: Maximum time to wait for a message, if <=0 uses get_nowait
        :return: Decoded message from the receive buffer
        """
        if self.buffer_receive_queue is None:
            raise RuntimeError(
                "buffer receive queue is None; check start()/stop() calls"
            )
        if timeout is not None and timeout <= 0:
            return self.buffer_receive_queue.get_nowait()
        else:
            return self.buffer_receive_queue.sync_get(timeout=timeout)

    async def put(self, item: SendMessageT, timeout: float | None = None):
        """
        Add message to send buffer with optional timeout.

        :param item: Message item to add to the send buffer
        :param timeout: Maximum time to wait for buffer space
        """
        if self.buffer_send_queue is None:
            raise RuntimeError(
                "buffer receive queue is None; check start()/stop() calls"
            )
        await asyncio.wait_for(self.buffer_send_queue.async_put(item), timeout=timeout)

    def put_sync(self, item: SendMessageT, timeout: float | None = None):
        """
        Add message to send buffer synchronously with optional timeout.

        :param item: Message item to add to the send buffer
        :param timeout: Maximum time to wait for buffer space, if <=0 uses put_nowait
        """
        if self.buffer_send_queue is None:
            raise RuntimeError(
                "buffer receive queue is None; check start()/stop() calls"
            )
        if timeout is not None and timeout <= 0:
            self.buffer_send_queue.put_nowait(item)
        else:
            self.buffer_send_queue.sync_put(item, timeout=timeout)

    def _create_check_stop_callable(
        self,
        stop_criteria: (
            list[ThreadingEvent | ProcessingEvent | MessagingStopCallback] | None
        ),
        canceled_event: ThreadingEvent,
    ):
        stop_events = tuple(
            item
            for item in stop_criteria or []
            if isinstance(item, ThreadingEvent | ProcessingEvent)
        )
        stop_callbacks = tuple(item for item in stop_criteria or [] if callable(item))

        def check_stop(pending: bool, queue_empty_count: int) -> bool:
            if canceled_event.is_set():
                return True

            if stop_callbacks and any(
                cb(self, pending, queue_empty_count) for cb in stop_callbacks
            ):
                return True

            if self.shutdown_event is None:
                return True

            return (
                not pending
                and queue_empty_count >= self.STOP_REQUIRED_QUEUE_EMPTY_COUNT
                and (
                    self.shutdown_event.is_set()
                    or any(event.is_set() for event in stop_events)
                )
            )

        return check_stop


class InterProcessMessagingQueue(InterProcessMessaging[SendMessageT, ReceiveMessageT]):
    """
    Queue-based inter-process messaging for distributed scheduler coordination.

    Provides message passing using multiprocessing.Queue objects for communication
    between scheduler workers and main process. Handles message encoding, buffering,
    flow control, and coordinated shutdown with configurable queue behavior and
    error handling policies for distributed operations.

    Example:
    ::
        from guidellm.utils.messaging import InterProcessMessagingQueue

        messaging = InterProcessMessagingQueue(
            serialization="pickle",
            max_pending_size=100
        )

        # Create worker copy for distributed processing
        worker_messaging = messaging.create_worker_copy(worker_index=0)
    """

    pending_queue: multiprocessing.Queue | queue.Queue[Any] | None
    done_queue: multiprocessing.Queue | queue.Queue[Any] | None

    def __init__(
        self,
        mp_context: BaseContext | None = None,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None,
        max_pending_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_done_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        pending_queue: multiprocessing.Queue | queue.Queue[Any] | None = None,
        done_queue: multiprocessing.Queue | queue.Queue[Any] | None = None,
    ):
        """
        Initialize queue-based messaging for inter-process communication.

        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_pending_size: Maximum items in send queue before blocking
        :param max_buffer_send_size: Maximum items in buffer send queue
        :param max_done_size: Maximum items in receive queue before blocking
        :param max_buffer_receive_size: Maximum items in buffer receive queue
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param pending_queue: Multiprocessing queue for sending messages
        :param done_queue: Multiprocessing queue for receiving completed messages
        :param context: Multiprocessing context for creating queues
        """
        super().__init__(
            mp_context=mp_context,
            serialization=serialization,
            encoding=encoding,
            max_pending_size=max_pending_size,
            max_buffer_send_size=max_buffer_send_size,
            max_done_size=max_done_size,
            max_buffer_receive_size=max_buffer_receive_size,
            poll_interval=poll_interval,
            worker_index=worker_index,
        )
        self.pending_queue = pending_queue or self.mp_context.Queue(
            maxsize=max_pending_size or 0
        )
        self.done_queue = done_queue or self.mp_context.Queue(
            maxsize=max_done_size or 0
        )

    def create_worker_copy(
        self, worker_index: int, **kwargs
    ) -> InterProcessMessagingQueue[ReceiveMessageT, SendMessageT]:
        """
        Create worker-specific copy for distributed queue-based coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured queue messaging instance for the specified worker
        """
        copy_args = {
            "mp_context": self.mp_context,
            "serialization": self.serialization,
            "encoding": self.encoding,
            "max_pending_size": self.max_pending_size,
            "max_buffer_send_size": self.max_buffer_send_size,
            "max_done_size": self.max_done_size,
            "max_buffer_receive_size": self.max_buffer_receive_size,
            "poll_interval": self.poll_interval,
            "worker_index": worker_index,
            "pending_queue": self.pending_queue,
            "done_queue": self.done_queue,
        }
        final_args = {**copy_args, **kwargs}

        return InterProcessMessagingQueue[ReceiveMessageT, SendMessageT](**final_args)

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await super().stop()
        if self.worker_index is None:
            # only main process should close the queues
            if self.pending_queue is None:
                raise RuntimeError("pending_queue is None; was stop() already called?")
            with contextlib.suppress(queue.Empty):
                while True:
                    self.pending_queue.get_nowait()
            if hasattr(self.pending_queue, "close"):
                self.pending_queue.close()

            if self.done_queue is None:
                raise RuntimeError("done_queue is None; was stop() already called?")
            with contextlib.suppress(queue.Empty):
                while True:
                    self.done_queue.get_nowait()
            if hasattr(self.done_queue, "close"):
                self.done_queue.close()

        self.pending_queue = None
        self.done_queue = None

    def create_send_messages_threads(
        self,
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create send message processing threads for queue-based transport.

        :param send_items: Optional collection of items to send during processing
        :param message_encoding: Message encoding configuration for serialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        return [
            (
                self._send_messages_task_thread,
                (send_items, message_encoding, check_stop),
            )
        ]

    def create_receive_messages_threads(
        self,
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create receive message processing threads for queue-based transport.

        :param receive_callback: Optional callback for processing received messages
        :param message_encoding: Message encoding configuration for deserialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        return [
            (
                self._receive_messages_task_thread,
                (receive_callback, message_encoding, check_stop),
            )
        ]

    def _send_messages_task_thread(  # noqa: C901, PLR0912
        self,
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ):
        send_items_iter = iter(send_items) if send_items is not None else None
        pending_item = None
        queue_empty_count = 0

        while not check_stop(pending_item is not None, queue_empty_count):
            if pending_item is None:
                try:
                    if send_items_iter is not None:
                        item = next(send_items_iter)
                    else:
                        if self.buffer_send_queue is None:
                            raise RuntimeError(
                                "buffer_send_queue is None; was stop() already called?"
                            )
                        item = self.buffer_send_queue.sync_get(
                            timeout=self.poll_interval
                        )
                    pending_item = message_encoding.encode(item)
                    queue_empty_count = 0
                except (culsans.QueueEmpty, queue.Empty, StopIteration):
                    queue_empty_count += 1

            if pending_item is not None:
                try:
                    if self.worker_index is None:
                        # Main publisher
                        if self.pending_queue is None:
                            raise RuntimeError(
                                "pending_queue is None; was stop() already called?"
                            )
                        self.pending_queue.put(pending_item, timeout=self.poll_interval)
                    else:
                        # Worker
                        if self.done_queue is None:
                            raise RuntimeError(
                                "done_queue is None; was stop() already called?"
                            )
                        self.done_queue.put(pending_item, timeout=self.poll_interval)
                    if send_items_iter is None:
                        if self.buffer_send_queue is None:
                            raise RuntimeError(
                                "buffer_send_queue is None; was stop() already called?"
                            )
                        self.buffer_send_queue.task_done()
                    pending_item = None
                except (culsans.QueueFull, queue.Full):
                    pass

            time.sleep(0)  # Yield to other threads

    def _receive_messages_task_thread(  # noqa: C901
        self,
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ):
        pending_item = None
        received_item = None
        queue_empty_count = 0

        while not check_stop(pending_item is not None, queue_empty_count):
            if pending_item is None:
                try:
                    if self.worker_index is None:
                        # Main publisher
                        if self.done_queue is None:
                            raise RuntimeError(
                                "done_queue is None; check start()/stop() calls"
                            )
                        item = self.done_queue.get(timeout=self.poll_interval)
                    else:
                        # Worker
                        if self.pending_queue is None:
                            raise RuntimeError(
                                "pending_queue is None; check start()/stop() calls"
                            )
                        item = self.pending_queue.get(timeout=self.poll_interval)
                    pending_item = message_encoding.decode(item)
                    queue_empty_count = 0
                except (culsans.QueueEmpty, queue.Empty):
                    queue_empty_count += 1

            if pending_item is not None or received_item is not None:
                try:
                    if received_item is None:
                        received_item = (
                            pending_item
                            if not receive_callback
                            else receive_callback(pending_item)
                        )

                    if self.buffer_receive_queue is None:
                        raise RuntimeError(
                            "buffer_receive_queue is None; check start()/stop() calls"
                        )
                    self.buffer_receive_queue.sync_put(
                        cast("ReceiveMessageT", received_item)
                    )
                    pending_item = None
                    received_item = None
                except (culsans.QueueFull, queue.Full):
                    pass

            time.sleep(0)  # Yield to other threads


class InterProcessMessagingManagerQueue(
    InterProcessMessagingQueue[SendMessageT, ReceiveMessageT]
):
    """
    Manager-based queue messaging for inter-process scheduler coordination.

    Extends queue-based messaging with multiprocessing.Manager support for
    shared state coordination across worker processes. Provides managed queues
    for reliable message passing in distributed scheduler environments with
    enhanced process synchronization and resource management capabilities.

    Example:
    ::
        import multiprocessing
        from guidellm.utils.messaging import InterProcessMessagingManagerQueue

        manager = multiprocessing.Manager()
        messaging = InterProcessMessagingManagerQueue(
            manager=manager,
            serialization="pickle"
        )
    """

    def __init__(
        self,
        manager: SyncManager,
        mp_context: BaseContext | None = None,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None,
        max_pending_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_done_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        pending_queue: multiprocessing.Queue | None = None,
        done_queue: multiprocessing.Queue | None = None,
    ):
        """
        Initialize manager-based queue messaging for inter-process communication.

        :param manager: Multiprocessing manager for shared queue creation
        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_pending_size: Maximum items in send queue before blocking
        :param max_buffer_send_size: Maximum items in buffer send queue
        :param max_done_size: Maximum items in receive queue before blocking
        :param max_buffer_receive_size: Maximum items in buffer receive queue
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param pending_queue: Managed multiprocessing queue for sending messages
        :param done_queue: Managed multiprocessing queue for receiving completed
            messages
        """
        super().__init__(
            mp_context=mp_context,
            serialization=serialization,
            encoding=encoding,
            max_pending_size=max_pending_size,
            max_buffer_send_size=max_buffer_send_size,
            max_done_size=max_done_size,
            max_buffer_receive_size=max_buffer_receive_size,
            poll_interval=poll_interval,
            worker_index=worker_index,
            pending_queue=pending_queue or manager.Queue(maxsize=max_pending_size or 0),
            done_queue=done_queue or manager.Queue(maxsize=max_done_size or 0),
        )

    def create_worker_copy(
        self, worker_index: int, **kwargs
    ) -> InterProcessMessagingManagerQueue[ReceiveMessageT, SendMessageT]:
        """
        Create worker-specific copy for managed queue-based coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured manager queue messaging instance for the specified worker
        """
        copy_args = {
            "manager": None,
            "mp_context": self.mp_context,
            "serialization": self.serialization,
            "encoding": self.encoding,
            "max_pending_size": self.max_pending_size,
            "max_buffer_send_size": self.max_buffer_send_size,
            "max_done_size": self.max_done_size,
            "max_buffer_receive_size": self.max_buffer_receive_size,
            "poll_interval": self.poll_interval,
            "worker_index": worker_index,
            "pending_queue": self.pending_queue,
            "done_queue": self.done_queue,
        }
        final_args = {**copy_args, **kwargs}

        return InterProcessMessagingManagerQueue(**final_args)

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await InterProcessMessaging.stop(self)
        self.pending_queue = None
        self.done_queue = None


class InterProcessMessagingPipe(InterProcessMessaging[SendMessageT, ReceiveMessageT]):
    """
    Pipe-based inter-process messaging for distributed scheduler coordination.

    Provides message passing using multiprocessing.Pipe objects for direct
    communication between scheduler workers and main process. Offers lower
    latency than queue-based messaging with duplex communication channels
    for high-performance distributed operations.

    Example:
    ::
        from guidellm.utils.messaging import InterProcessMessagingPipe

        messaging = InterProcessMessagingPipe(
            num_workers=4,
            serialization="pickle",
            poll_interval=0.05
        )

        # Create worker copy for specific worker process
        worker_messaging = messaging.create_worker_copy(worker_index=0)
    """

    def __init__(
        self,
        num_workers: int,
        mp_context: BaseContext | None = None,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias | list[EncodingTypesAlias] = None,
        max_pending_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_done_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        pipe: tuple[Connection, Connection] | None = None,
    ):
        """
        Initialize pipe-based messaging for inter-process communication.

        :param num_workers: Number of worker processes requiring pipe connections
        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_pending_size: Maximum items in send queue before blocking
        :param max_buffer_send_size: Maximum items in buffer send queue
        :param max_done_size: Maximum items in receive queue before blocking
        :param max_buffer_receive_size: Maximum items in buffer receive queue
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param pipe: Existing pipe connection for worker-specific instances
        """
        super().__init__(
            mp_context=mp_context,
            serialization=serialization,
            encoding=encoding,
            max_pending_size=max_pending_size,
            max_buffer_send_size=max_buffer_send_size,
            max_done_size=max_done_size,
            max_buffer_receive_size=max_buffer_receive_size,
            poll_interval=poll_interval,
            worker_index=worker_index,
        )
        self.num_workers = num_workers

        self.pipes: list[tuple[Connection, Connection]]
        if pipe is None:
            self.pipes = [self.mp_context.Pipe(duplex=True) for _ in range(num_workers)]
        else:
            self.pipes = [pipe]

    def create_worker_copy(
        self, worker_index: int, **kwargs
    ) -> InterProcessMessagingPipe[ReceiveMessageT, SendMessageT]:
        """
        Create worker-specific copy for pipe-based coordination.

        :param worker_index: Index of the worker process for pipe routing
        :return: Configured pipe messaging instance for the specified worker
        """
        copy_args = {
            "num_workers": self.num_workers,
            "mp_context": self.mp_context,
            "serialization": self.serialization,
            "encoding": self.encoding,
            "max_pending_size": self.max_pending_size,
            "max_buffer_send_size": self.max_buffer_send_size,
            "max_done_size": self.max_done_size,
            "max_buffer_receive_size": self.max_buffer_receive_size,
            "poll_interval": self.poll_interval,
            "worker_index": worker_index,
            "pipe": self.pipes[worker_index],
        }

        final_args = {**copy_args, **kwargs}

        return InterProcessMessagingPipe(**final_args)

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await super().stop()
        if self.worker_index is None:
            # Only main process should close the pipes
            for main_con, worker_con in self.pipes:
                main_con.close()
                worker_con.close()

    def create_send_messages_threads(
        self,
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create send message processing threads for pipe-based transport.

        :param send_items: Optional collection of items to send during processing
        :param message_encoding: Message encoding configuration for serialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        if self.worker_index is None:
            # Create a separate task for each worker's pipe
            return [
                (
                    self._send_messages_task_thread,
                    (self.pipes[index], send_items, message_encoding, check_stop),
                )
                for index in range(self.num_workers)
            ]
        else:
            return [
                (
                    self._send_messages_task_thread,
                    (self.pipes[0], send_items, message_encoding, check_stop),
                )
            ]

    def create_receive_messages_threads(
        self,
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ) -> list[tuple[Callable, tuple[Any, ...]]]:
        """
        Create receive message processing threads for pipe-based transport.

        :param receive_callback: Optional callback for processing received messages
        :param message_encoding: Message encoding configuration for deserialization
        :param check_stop: Callable for evaluating stop conditions during processing
        :return: List of thread callables with their arguments for execution
        """
        if self.worker_index is None:
            # Create a separate task for each worker's pipe
            return [
                (
                    self._receive_messages_task_thread,
                    (self.pipes[index], receive_callback, message_encoding, check_stop),
                )
                for index in range(self.num_workers)
            ]
        else:
            return [
                (
                    self._receive_messages_task_thread,
                    (self.pipes[0], receive_callback, message_encoding, check_stop),
                )
            ]

    def _send_messages_task_thread(  # noqa: C901, PLR0912, PLR0915
        self,
        pipe: tuple[Connection, Connection],
        send_items: Iterable[Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ):
        local_stop = ThreadingEvent()
        send_connection: Connection = pipe[0] if self.worker_index is None else pipe[1]
        send_items_iter = iter(send_items) if send_items is not None else None
        pending_item = None
        queue_empty_count = 0
        pipe_item = None
        pipe_lock = threading.Lock()

        def _background_pipe_recv():
            nonlocal pipe_item

            while not local_stop.is_set():
                try:
                    with pipe_lock:
                        pending = pipe_item
                        pipe_item = None

                    if pending is not None:
                        send_connection.send(pending)
                except (EOFError, ConnectionResetError):
                    break

        if send_items_iter is None:
            threading.Thread(target=_background_pipe_recv, daemon=True).start()

        try:
            while not check_stop(pending_item is not None, queue_empty_count):
                if pending_item is None:
                    try:
                        if send_items_iter is not None:
                            item = next(send_items_iter)
                        else:
                            if self.buffer_send_queue is None:
                                raise RuntimeError(
                                    "buffer_send_queue is None; check start()/stop() calls"  # noqa: E501
                                )
                            item = self.buffer_send_queue.sync_get(
                                timeout=self.poll_interval
                            )
                        pending_item = message_encoding.encode(item)
                        queue_empty_count = 0
                    except (culsans.QueueEmpty, queue.Empty, StopIteration):
                        queue_empty_count += 1

                if pending_item is not None:
                    try:
                        with pipe_lock:
                            if pipe_item is not None:
                                time.sleep(self.poll_interval / 100)
                                raise queue.Full
                            else:
                                pipe_item = pending_item
                        if send_items_iter is None:
                            if self.buffer_send_queue is None:
                                raise RuntimeError(
                                    "buffer_send_queue is None; check start()/stop() calls"  # noqa: E501
                                )
                            self.buffer_send_queue.task_done()
                        pending_item = None
                    except (culsans.QueueFull, queue.Full):
                        pass
        finally:
            local_stop.set()

    def _receive_messages_task_thread(  # noqa: C901
        self,
        pipe: tuple[Connection, Connection],
        receive_callback: Callable[[Any], Any] | None,
        message_encoding: MessageEncoding,
        check_stop: CheckStopCallableT,
    ):
        receive_connection: Connection = (
            pipe[0] if self.worker_index is not None else pipe[1]
        )
        pending_item = None
        received_item = None
        queue_empty_count = 0

        while not check_stop(pending_item is not None, queue_empty_count):
            if pending_item is None:
                try:
                    if receive_connection.poll(self.poll_interval):
                        item = receive_connection.recv()
                        pending_item = message_encoding.decode(item)
                    else:
                        raise queue.Empty
                    queue_empty_count = 0
                except (culsans.QueueEmpty, queue.Empty):
                    queue_empty_count += 1

            if pending_item is not None or received_item is not None:
                try:
                    if received_item is None:
                        received_item = (
                            pending_item
                            if not receive_callback
                            else receive_callback(pending_item)
                        )
                    if self.buffer_receive_queue is None:
                        raise RuntimeError(
                            "buffer receive queue is None; check start()/stop() calls"
                        )
                    self.buffer_receive_queue.sync_put(
                        cast("ReceiveMessageT", received_item)
                    )
                    pending_item = None
                    received_item = None
                except (culsans.QueueFull, queue.Full):
                    pass
