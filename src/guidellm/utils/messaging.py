"""
Inter-process messaging abstractions for distributed scheduler coordination.

Provides high-level interfaces for asynchronous message passing between worker
processes using various transport mechanisms including queues and pipes. Supports
configurable encoding, serialization, error handling, and flow control with
buffering and stop event coordination for the scheduler's distributed operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from multiprocessing.connection import Connection
from multiprocessing.connection import Pipe as ProcessingPipe
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Event as ThreadingEvent
from typing import Any, Callable, Generic, Literal, TypeVar

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
]

MessageT = TypeVar("MessageT", bound=Any)
"""Generic type variable for messages processed by inter-process messaging systems."""


class InterProcessMessaging(Generic[MessageT], ABC):
    """
    Abstract base for inter-process messaging coordination in distributed scheduler.

    Provides unified interface for asynchronous message passing between scheduler
    components using configurable transport mechanisms, encoding schemes, and
    flow control policies. Manages buffering, serialization, error handling,
    and coordinated shutdown across worker processes for distributed load testing.

    Example:
    ::
        from guidellm.utils.messaging import InterProcessMessagingQueue

        messaging = InterProcessMessagingQueue(
            serialization="pickle",
            on_stop_action="stop_after_empty"
        )

        await messaging.start()
        await messaging.put(request_data)
        response = await messaging.get(timeout=5.0)
        await messaging.stop()
    """

    def __init__(
        self,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias = None,
        max_send_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_receive_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        on_stop_action: Literal[
            "ignore", "stop", "stop_after_empty", "error"
        ] = "stop_after_empty",
        on_empty_action: Literal["ignore", "stop", "error"] = "ignore",
        on_full_action: Literal["ignore", "stop", "error"] = "ignore",
        poll_interval: float = 0.1,
        worker_index: int | None = None,
    ):
        """
        Initialize inter-process messaging coordinator.

        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_send_size: Maximum number of items in send queue before blocking
        :param max_buffer_send_size: Maximum number of items in buffer send queue
        :param max_receive_size: Maximum number of items in receive queue before
            blocking
        :param max_buffer_receive_size: Maximum number of items in buffer receive queue
        :param on_stop_action: Behavior when stop events are triggered
        :param on_empty_action: Behavior when message queues become empty
        :param on_full_action: Behavior when message queues become full
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        """
        self.worker_index: int | None = worker_index
        self.serialization = serialization
        self.encoding = encoding
        self.max_send_size = max_send_size
        self.max_buffer_send_size = max_buffer_send_size
        self.max_receive_size = max_receive_size
        self.max_buffer_receive_size = max_buffer_receive_size
        self.on_stop_action = on_stop_action
        self.on_empty_action = on_empty_action
        self.on_full_action = on_full_action
        self.poll_interval = poll_interval

        self.message_encoding: MessageEncoding = None
        self.stop_events: list[ThreadingEvent | ProcessingEvent] = None
        self.stopped_event: ThreadingEvent = None
        self.shutdown_event: ThreadingEvent = None
        self.buffer_send_queue: culsans.Queue = None
        self.buffer_receive_queue: culsans.Queue = None
        self.send_task: asyncio.Task = None
        self.receive_task: asyncio.Task = None
        self.running = False

    @abstractmethod
    def create_worker_copy(self, worker_index: int) -> InterProcessMessaging[MessageT]:
        """
        Create worker-specific copy for distributed process coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured messaging instance for the specified worker
        """
        ...

    @abstractmethod
    async def send_messages_task(self, send_items: Iterable[Any] | None):
        """
        Execute asynchronous message sending task for process coordination.

        :param send_items: Optional collection of items to send to other processes
        """
        ...

    @abstractmethod
    async def receive_messages_task(
        self, receive_callback: Callable[[Any], None] | None
    ):
        """
        Execute asynchronous message receiving task for process coordination.

        :param receive_callback: Optional callback to process received messages
        """
        ...

    async def start(
        self,
        send_items: Iterable[Any] | None = None,
        receive_callback: Callable[[Any], None] | None = None,
        stop_events: list[ThreadingEvent | ProcessingEvent] | None = None,
        pydantic_models: list[type[BaseModel]] | None = None,
    ):
        """
        Start asynchronous message processing tasks with buffering.

        :param send_items: Optional collection of items to send during processing
        :param receive_callback: Optional callback for processing received messages
        :param stop_events: External events that trigger messaging shutdown
        :param pydantic_models: Optional list of Pydantic models for serialization
        """
        self.running = True
        self.message_encoding = MessageEncoding(
            serialization=self.serialization,
            encoding=self.encoding,
            pydantic_models=pydantic_models,
        )
        self.stop_events = stop_events if stop_events is not None else []
        self.stopped_event = ThreadingEvent()
        self.shutdown_event = ThreadingEvent()

        self.buffer_send_queue = culsans.Queue()
        self.buffer_receive_queue = culsans.Queue()

        self.send_task = asyncio.create_task(
            self.send_messages_task(send_items=send_items)
        )
        self.receive_task = asyncio.create_task(
            self.receive_messages_task(receive_callback=receive_callback)
        )

    async def stop(self):
        """
        Stop message processing tasks and clean up resources.
        """
        self.shutdown_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(
                self.send_task, self.receive_task, return_exceptions=True
            )
        self.send_task = None
        self.receive_task = None
        await self.buffer_send_queue.aclose()
        await self.buffer_receive_queue.aclose()
        self.buffer_send_queue = None
        self.buffer_receive_queue = None
        self.message_encoding = None
        self.stop_events = None
        self.stopped_event = None
        self.shutdown_event = None
        self.running = False

    async def get(self, timeout: float | None = None) -> Any:
        """
        Retrieve message from receive buffer with optional timeout.

        :param timeout: Maximum time to wait for a message
        :return: Decoded message from the receive buffer
        """
        return await asyncio.wait_for(
            self.buffer_receive_queue.async_get(), timeout=timeout
        )

    async def put(self, item: Any, timeout: float | None = None):
        """
        Add message to send buffer with optional timeout.

        :param item: Message item to add to the send buffer
        :param timeout: Maximum time to wait for buffer space
        """
        await asyncio.wait_for(self.buffer_send_queue.async_put(item), timeout=timeout)

    def check_on_stop_action(self, pending: Any | None, queue_empty: bool) -> bool:
        """
        Check if messaging should stop based on configured stop action.

        :param pending: Currently pending message being processed
        :param queue_empty: Whether the message queue is currently empty
        :return: True if messaging should stop, False otherwise
        :raises RuntimeError: When stop action is 'error' and stop event is set
        """
        shutdown_set = self.shutdown_event.is_set()

        if self.on_stop_action == "ignore":
            return shutdown_set and pending is None

        stop_set = any(event.is_set() for event in self.stop_events)

        if self.on_stop_action == "error":
            if stop_set:
                raise RuntimeError("Stop event set (on_stop_action='error').")
            return shutdown_set and pending is None

        return (
            (
                self.on_stop_action == "stop"
                or (self.on_stop_action == "stop_after_empty" and queue_empty)
            )
            and (shutdown_set or stop_set)
            and pending is None
        )

    def check_on_queue_empty_action(self, pending: Any | None) -> bool:
        """
        Check if messaging should stop based on empty queue action.

        :param pending: Currently pending message being processed
        :return: True if messaging should stop, False otherwise
        :raises RuntimeError: When empty action is 'error' and queue is empty
        """
        if self.on_empty_action == "ignore":
            return False

        if self.on_empty_action == "error":
            raise RuntimeError("Queue empty (on_empty_action='error').")

        return (
            self.shutdown_event.is_set()
            or any(event.is_set() for event in self.stop_events)
        ) and pending is None

    def check_on_queue_full_action(self, pending: Any | None) -> bool:
        """
        Check if messaging should stop based on full queue action.

        :param pending: Currently pending message being processed
        :return: True if messaging should stop, False otherwise
        :raises RuntimeError: When full action is 'error' and queue is full
        """
        if self.on_full_action == "ignore":
            return False

        if self.on_full_action == "error":
            raise RuntimeError("Queue full (on_full_action='error').")

        return (
            self.shutdown_event.is_set()
            or any(event.is_set() for event in self.stop_events)
        ) and pending is None


class InterProcessMessagingQueue(InterProcessMessaging[MessageT]):
    """
    Queue-based inter-process messaging implementation for scheduler coordination.

    Provides message passing using multiprocessing.Queue objects for communication
    between scheduler workers and main process. Handles message encoding, buffering,
    flow control, and coordinated shutdown with configurable queue behavior and
    error handling policies for distributed load testing operations.

    Example:
    ::
        from guidellm.utils.messaging import InterProcessMessagingQueue

        messaging = InterProcessMessagingQueue(
            serialization="pickle",
            max_send_size=100,
            on_stop_action="stop_after_empty"
        )

        # Create worker copy for distributed processing
        worker_messaging = messaging.create_worker_copy(worker_index=0)
    """

    def __init__(
        self,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias = None,
        max_send_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_receive_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        on_stop_action: Literal[
            "ignore", "stop", "stop_after_empty", "error"
        ] = "stop_after_empty",
        on_empty_action: Literal["ignore", "stop", "error"] = "ignore",
        on_full_action: Literal["ignore", "stop", "error"] = "ignore",
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        send_queue: multiprocessing.Queue | None = None,
        done_queue: multiprocessing.Queue | None = None,
    ):
        """
        Initialize queue-based messaging for inter-process communication.

        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_send_size: Maximum number of items in send queue before blocking
        :param max_buffer_send_size: Maximum number of items in buffer send queue
        :param max_receive_size: Maximum number of items in receive queue before
            blocking
        :param max_buffer_receive_size: Maximum number of items in buffer receive queue
        :param on_stop_action: Behavior when stop events are triggered
        :param on_empty_action: Behavior when message queues become empty
        :param on_full_action: Behavior when message queues become full
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param send_queue: Multiprocessing queue for sending messages
        :param done_queue: Multiprocessing queue for receiving completed messages
        """
        super().__init__(
            serialization=serialization,
            encoding=encoding,
            max_send_size=max_send_size,
            max_buffer_send_size=max_buffer_send_size,
            max_receive_size=max_receive_size,
            max_buffer_receive_size=max_buffer_receive_size,
            on_stop_action=on_stop_action,
            on_empty_action=on_empty_action,
            on_full_action=on_full_action,
            poll_interval=poll_interval,
            worker_index=worker_index,
        )
        self.send_queue = send_queue or multiprocessing.Queue(
            maxsize=max_send_size or 0
        )
        self.done_queue = done_queue or multiprocessing.Queue(
            maxsize=max_receive_size or 0
        )

    def create_worker_copy(
        self, worker_index: int
    ) -> InterProcessMessagingQueue[MessageT]:
        """
        Create worker-specific copy for distributed queue-based coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured queue messaging instance for the specified worker
        """
        return InterProcessMessagingQueue(
            serialization=self.serialization,
            encoding=self.encoding,
            max_send_size=self.max_send_size,
            max_buffer_send_size=self.max_buffer_send_size,
            max_receive_size=self.max_receive_size,
            max_buffer_receive_size=self.max_buffer_receive_size,
            on_stop_action=self.on_stop_action,
            on_empty_action=self.on_empty_action,
            on_full_action=self.on_full_action,
            poll_interval=self.poll_interval,
            worker_index=worker_index,
            send_queue=self.send_queue,
            done_queue=self.done_queue,
        )

    async def send_messages_task(self, send_items: Iterable[Any] | None):
        """
        Execute asynchronous queue-based message sending task.

        :param send_items: Optional collection of items to send via queues
        """
        canceled_event = ThreadingEvent()

        try:
            await asyncio.to_thread(
                self.send_messages_task_thread, send_items, canceled_event
            )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            self.stopped_event.set()

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await super().stop()
        self.send_queue.close()
        self.done_queue.close()
        self.send_queue = None
        self.done_queue = None

    async def receive_messages_task(
        self, receive_callback: Callable[[Any], None] | None
    ):
        """
        Execute asynchronous queue-based message receiving task.

        :param receive_callback: Optional callback to process received messages
        """
        canceled_event = ThreadingEvent()

        try:
            return await asyncio.to_thread(
                self.receive_messages_task_thread, receive_callback, canceled_event
            )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            self.stopped_event.set()

    def send_messages_task_thread(  # noqa: C901, PLR0912
        self, send_items: Iterable[Any] | None, canceled_event: ThreadingEvent
    ):
        send_items_iter = iter(send_items) if send_items is not None else None
        pending_item = None
        queue_empty_reported = False

        while not canceled_event.is_set():
            if self.check_on_stop_action(pending_item, queue_empty_reported):
                break

            queue_empty_reported = False

            if pending_item is None:
                try:
                    if send_items_iter is not None:
                        item = next(send_items_iter)
                    else:
                        item = self.buffer_send_queue.sync_get(
                            timeout=self.poll_interval
                        )
                    pending_item = self.message_encoding.encode(item)
                except (culsans.QueueEmpty, queue.Empty, StopIteration):
                    queue_empty_reported = True
                    if self.check_on_queue_empty_action(pending_item):
                        break

            if pending_item is not None:
                try:
                    if self.worker_index is None:
                        # Main publisher
                        self.send_queue.put(pending_item, timeout=self.poll_interval)
                    else:
                        # Worker
                        self.done_queue.put(pending_item, timeout=self.poll_interval)
                    if send_items_iter is None:
                        self.buffer_send_queue.task_done()
                    pending_item = None
                except (culsans.QueueFull, queue.Full):
                    if self.check_on_queue_full_action(pending_item):
                        break

    def receive_messages_task_thread(  # noqa: C901
        self,
        receive_callback: Callable[[Any], None] | None,
        canceled_event: ThreadingEvent,
    ):
        pending_item = None
        received_item = None
        queue_empty_reported = False

        while not canceled_event.is_set():
            if self.check_on_stop_action(pending_item, queue_empty_reported):
                break

            if pending_item is None:
                try:
                    if self.worker_index is None:
                        # Main publisher
                        item = self.done_queue.get(timeout=self.poll_interval)
                    else:
                        # Worker
                        item = self.send_queue.get(timeout=self.poll_interval)
                    pending_item = self.message_encoding.decode(item)
                except (culsans.QueueEmpty, queue.Empty):
                    queue_empty_reported = True
                    if self.check_on_queue_empty_action(pending_item):
                        break

            if pending_item is not None or received_item is not None:
                try:
                    if received_item is None:
                        received_item = (
                            pending_item
                            if not receive_callback
                            else receive_callback(pending_item)
                        )

                    self.buffer_receive_queue.sync_put(received_item)
                    pending_item = None
                    received_item = None
                except (culsans.QueueFull, queue.Full):
                    if self.check_on_queue_full_action(pending_item):
                        break


class InterProcessMessagingManagerQueue(InterProcessMessagingQueue[MessageT]):
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
        manager: BaseContext,
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias = None,
        max_send_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_receive_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        on_stop_action: Literal[
            "ignore", "stop", "stop_after_empty", "error"
        ] = "stop_after_empty",
        on_empty_action: Literal["ignore", "stop", "error"] = "ignore",
        on_full_action: Literal["ignore", "stop", "error"] = "ignore",
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        send_queue: multiprocessing.Queue | None = None,
        done_queue: multiprocessing.Queue | None = None,
    ):
        """
        Initialize manager-based queue messaging for inter-process communication.

        :param manager: Multiprocessing manager for shared queue creation
        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_send_size: Maximum number of items in send queue before blocking
        :param max_buffer_send_size: Maximum number of items in buffer send queue
        :param max_receive_size: Maximum number of items in receive queue before
            blocking
        :param max_buffer_receive_size: Maximum number of items in buffer receive queue
        :param on_stop_action: Behavior when stop events are triggered
        :param on_empty_action: Behavior when message queues become empty
        :param on_full_action: Behavior when message queues become full
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param send_queue: Managed multiprocessing queue for sending messages
        :param done_queue: Managed multiprocessing queue for receiving completed
            messages
        """
        super().__init__(
            serialization=serialization,
            encoding=encoding,
            max_send_size=max_send_size,
            max_buffer_send_size=max_buffer_send_size,
            max_receive_size=max_receive_size,
            max_buffer_receive_size=max_buffer_receive_size,
            on_stop_action=on_stop_action,
            on_empty_action=on_empty_action,
            on_full_action=on_full_action,
            poll_interval=poll_interval,
            worker_index=worker_index,
            send_queue=send_queue or manager.Queue(maxsize=max_send_size or 0),
            done_queue=done_queue or manager.Queue(maxsize=max_receive_size or 0),
        )

    def create_worker_copy(
        self, worker_index: int
    ) -> InterProcessMessagingManagerQueue[MessageT]:
        """
        Create worker-specific copy for managed queue-based coordination.

        :param worker_index: Index of the worker process for message routing
        :return: Configured manager queue messaging instance for the specified worker
        """
        return InterProcessMessagingManagerQueue(
            manager=None,
            serialization=self.serialization,
            encoding=self.encoding,
            max_send_size=self.max_send_size,
            max_buffer_send_size=self.max_buffer_send_size,
            max_receive_size=self.max_receive_size,
            max_buffer_receive_size=self.max_buffer_receive_size,
            on_stop_action=self.on_stop_action,
            on_empty_action=self.on_empty_action,
            on_full_action=self.on_full_action,
            poll_interval=self.poll_interval,
            worker_index=worker_index,
            send_queue=self.send_queue,
            done_queue=self.done_queue,
        )

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await InterProcessMessaging.stop(self)
        self.send_queue = None
        self.done_queue = None


class InterProcessMessagingPipe(InterProcessMessaging[MessageT]):
    """
    Pipe-based inter-process messaging implementation for scheduler coordination.

    Provides message passing using multiprocessing.Pipe objects for direct
    communication between scheduler workers and main process. Offers lower
    latency than queue-based messaging with duplex communication channels
    for high-performance distributed load testing operations.

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
        serialization: SerializationTypesAlias = "dict",
        encoding: EncodingTypesAlias = None,
        max_send_size: int | None = None,
        max_buffer_send_size: int | None = None,
        max_receive_size: int | None = None,
        max_buffer_receive_size: int | None = None,
        on_stop_action: Literal[
            "ignore", "stop", "stop_after_empty", "error"
        ] = "stop_after_empty",
        on_empty_action: Literal["ignore", "stop", "error"] = "ignore",
        on_full_action: Literal["ignore", "stop", "error"] = "ignore",
        poll_interval: float = 0.1,
        worker_index: int | None = None,
        pipe: ProcessingPipe | None = None,
    ):
        """
        Initialize pipe-based messaging for inter-process communication.

        :param num_workers: Number of worker processes requiring pipe connections
        :param serialization: Message serialization method for transport encoding
        :param encoding: Optional encoding scheme for serialized message data
        :param max_send_size: Maximum number of items in send queue before blocking
        :param max_buffer_send_size: Maximum number of items in buffer send queue
        :param max_receive_size: Maximum number of items in receive queue before
            blocking
        :param max_buffer_receive_size: Maximum number of items in buffer receive queue
        :param on_stop_action: Behavior when stop events are triggered
        :param on_empty_action: Behavior when message queues become empty
        :param on_full_action: Behavior when message queues become full
        :param poll_interval: Time interval for checking queue status and events
        :param worker_index: Index identifying this worker in the process group
        :param pipe: Existing pipe connection for worker-specific instances
        """
        super().__init__(
            serialization=serialization,
            encoding=encoding,
            max_send_size=max_send_size,
            max_buffer_send_size=max_buffer_send_size,
            max_receive_size=max_receive_size,
            max_buffer_receive_size=max_buffer_receive_size,
            on_stop_action=on_stop_action,
            on_empty_action=on_empty_action,
            on_full_action=on_full_action,
            poll_interval=poll_interval,
            worker_index=worker_index,
        )
        self.num_workers = num_workers

        if pipe is None:
            self.pipes: list[ProcessingPipe] = [
                ProcessingPipe(duplex=True) for _ in range(num_workers)
            ]
        else:
            self.pipes: list[ProcessingPipe] = [pipe]

    def create_worker_copy(
        self, worker_index: int
    ) -> InterProcessMessagingPipe[MessageT]:
        """
        Create worker-specific copy for pipe-based coordination.

        :param worker_index: Index of the worker process for pipe routing
        :return: Configured pipe messaging instance for the specified worker
        """
        return InterProcessMessagingPipe(
            num_workers=self.num_workers,
            serialization=self.serialization,
            encoding=self.encoding,
            max_send_size=self.max_send_size,
            max_receive_size=self.max_receive_size,
            on_stop_action=self.on_stop_action,
            on_empty_action=self.on_empty_action,
            on_full_action=self.on_full_action,
            poll_interval=self.poll_interval,
            worker_index=worker_index,
            pipe=self.pipes[worker_index],
        )

    async def stop(self):
        """
        Stop the messaging system and wait for all tasks to complete.
        """
        await super().stop()
        if self.worker_index is None:
            for main_con, worker_con in self.pipes:
                main_con.close()
                worker_con.close()

    async def send_messages_task(self, send_items: Iterable[Any] | None):
        """
        Execute asynchronous pipe-based message sending task.

        :param send_items: Optional collection of items to send via pipes
        """
        canceled_event = ThreadingEvent()

        try:
            if self.worker_index is None:
                # Create a separate task for each worker's pipe
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            self.send_messages_task_thread,
                            self.pipes[index],
                            send_items,
                            canceled_event,
                        )
                        for index in range(self.num_workers)
                    ]
                )
            else:
                await asyncio.to_thread(
                    self.send_messages_task_thread,
                    self.pipes[0],
                    send_items,
                    canceled_event,
                )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            self.stopped_event.set()

    async def receive_messages_task(
        self, receive_callback: Callable[[Any], None] | None
    ):
        """
        Execute asynchronous pipe-based message receiving task.

        :param receive_callback: Optional callback to process received messages
        """
        canceled_event = ThreadingEvent()

        try:
            if self.worker_index is None:
                # Create a separate task for each worker's pipe
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            self.receive_messages_task_thread,
                            self.pipes[index],
                            receive_callback,
                            canceled_event,
                        )
                        for index in range(self.num_workers)
                    ]
                )
            else:
                await asyncio.to_thread(
                    self.receive_messages_task_thread,
                    self.pipes[0],
                    receive_callback,
                    canceled_event,
                )
        except asyncio.CancelledError:
            canceled_event.set()
            raise
        finally:
            self.stopped_event.set()

    def send_messages_task_thread(  # noqa: C901, PLR0912
        self,
        pipe: ProcessingPipe,
        send_items: Iterable[Any] | None,
        canceled_event: ThreadingEvent,
    ):
        send_connection: Connection = pipe[0] if self.worker_index is None else pipe[1]
        send_items_iter = iter(send_items) if send_items is not None else None
        pending_item = None
        queue_empty_reported = False
        pipe_item = None
        pipe_lock = threading.Lock()

        def _background_pipe_recv():
            nonlocal pipe_item

            while (
                not canceled_event.is_set()
                and self.stopped_event is not None
                and not self.stopped_event.is_set()
            ):
                try:
                    with pipe_lock:
                        pending = pipe_item
                        pipe_item = None  # Clear after taking

                    if pending is not None:
                        # pending is already encoded, just send it directly
                        send_connection.send(pending)
                except (EOFError, ConnectionResetError):
                    break

        if send_items_iter is None:
            threading.Thread(target=_background_pipe_recv, daemon=True).start()

        while not canceled_event.is_set():
            if self.check_on_stop_action(pending_item, queue_empty_reported):
                break

            queue_empty_reported = False

            if pending_item is None:
                try:
                    if send_items_iter is not None:
                        item = next(send_items_iter)
                    else:
                        item = self.buffer_send_queue.sync_get(
                            timeout=self.poll_interval
                        )
                    pending_item = self.message_encoding.encode(item)
                except (culsans.QueueEmpty, queue.Empty, StopIteration):
                    queue_empty_reported = True
                    if self.check_on_queue_empty_action(pending_item):
                        break

            if pending_item is not None:
                try:
                    with pipe_lock:
                        if pipe_item is not None:
                            time.sleep(self.poll_interval / 100)
                            raise queue.Full
                        else:
                            pipe_item = pending_item
                    if send_items_iter is None:
                        self.buffer_send_queue.task_done()
                    pending_item = None
                except (culsans.QueueFull, queue.Full):
                    if self.check_on_queue_full_action(pending_item):
                        break

    def receive_messages_task_thread(  # noqa: C901
        self,
        pipe: ProcessingPipe,
        receive_callback: Callable[[Any], None] | None,
        canceled_event: ThreadingEvent,
    ):
        receive_connection: Connection = (
            pipe[0] if self.worker_index is not None else pipe[1]
        )
        pending_item = None
        received_item = None
        queue_empty_reported = False

        while not canceled_event.is_set():
            if self.check_on_stop_action(pending_item, queue_empty_reported):
                break

            if pending_item is None:
                try:
                    if receive_connection.poll(self.poll_interval):
                        item = receive_connection.recv()
                        pending_item = self.message_encoding.decode(item)
                    else:
                        raise queue.Empty
                except (culsans.QueueEmpty, queue.Empty):
                    queue_empty_reported = True
                    if self.check_on_queue_empty_action(pending_item):
                        break

            if pending_item is not None or received_item is not None:
                try:
                    if received_item is None:
                        received_item = (
                            pending_item
                            if not receive_callback
                            else receive_callback(pending_item)
                        )

                    self.buffer_receive_queue.sync_put(received_item)
                    pending_item = None
                    received_item = None
                except (culsans.QueueFull, queue.Full):
                    if self.check_on_queue_full_action(pending_item):
                        break
