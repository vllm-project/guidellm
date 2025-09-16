"""
Async utilities for waiting on synchronization objects.

This module provides async-compatible wrappers for threading and multiprocessing
synchronization primitives (Events and Barriers). These utilities enable async code
to wait for synchronization objects without blocking the event loop, essential for
coordinating between async and sync code or between processes in the guidellm system.
"""

from __future__ import annotations

import asyncio
import contextlib
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Barrier as ThreadingBarrier
from threading import Event as ThreadingEvent
from typing import Annotated, Union

from typing_extensions import TypeAlias

__all__ = [
    "SyncObjectTypesAlias",
    "wait_for_sync_barrier",
    "wait_for_sync_event",
    "wait_for_sync_objects",
]


SyncObjectTypesAlias: TypeAlias = Annotated[
    Union[ThreadingEvent, ProcessingEvent, ThreadingBarrier, ProcessingBarrier],
    "Type alias for threading and multiprocessing synchronization object types",
]


async def wait_for_sync_event(
    event: ThreadingEvent | ProcessingEvent,
    poll_interval: float,
) -> None:
    """
    Asynchronously wait for a threading or multiprocessing Event to be set.

    This function polls the event at regular intervals without blocking the async
    event loop, allowing other async tasks to continue executing while waiting.

    :param event: The Event object to wait for (threading or multiprocessing)
    :param poll_interval: Time in seconds between polling checks
    :raises asyncio.CancelledError: If the async task is cancelled
    """
    stop = ThreadingEvent()

    def _watch():
        try:
            while not stop.is_set():
                if event.wait(timeout=poll_interval):
                    return
        except Exception as err:  # noqa: BLE001
            if stop.is_set():
                return  # Ignore error if we should have stopped
            raise err

    try:
        await asyncio.to_thread(_watch)
    except asyncio.CancelledError:
        stop.set()
        raise


async def wait_for_sync_barrier(
    barrier: ThreadingBarrier | ProcessingBarrier,
    poll_interval: float,
) -> None:
    """
    Asynchronously wait for a threading or multiprocessing Barrier to be reached.

    This function polls the barrier at regular intervals without blocking the async
    event loop, allowing other async tasks to continue executing while waiting.

    :param barrier: The Barrier object to wait for (threading or multiprocessing)
    :param poll_interval: Time in seconds between polling checks
    :raises asyncio.CancelledError: If the async task is cancelled
    """
    stop = ThreadingEvent()
    barrier_broken = ThreadingEvent()

    def _wait_indefinite():
        try:
            # wait forever, count on barrier broken event to exit
            barrier.wait()
            barrier_broken.set()
        except Exception as err:
            if stop.is_set():
                return  # Ignore error if we should have stopped
            raise err

    def _watch():
        while not barrier_broken.is_set():
            if stop.is_set():
                with contextlib.suppress(Exception):
                    if not barrier.broken:
                        barrier.abort()
                break

    try:
        await asyncio.gather(
            asyncio.to_thread(_wait_indefinite),
            asyncio.to_thread(_watch),
        )
    except asyncio.CancelledError:
        stop.set()
        raise


async def wait_for_sync_objects(
    objects: SyncObjectTypesAlias
    | list[SyncObjectTypesAlias]
    | dict[str, SyncObjectTypesAlias],
    poll_interval: float = 0.1,
) -> int | str:
    """
    Asynchronously wait for the first synchronization object to complete.

    This function waits for the first Event to be set or Barrier to be reached
    from a collection of synchronization objects. It returns immediately when
    any object completes and cancels waiting on the remaining objects.

    :param objects: Single sync object, list of objects, or dict mapping names
        to objects
    :param poll_interval: Time in seconds between polling checks for each object
    :return: Index (for list/single) or key name (for dict) of the first
        completed object
    :raises asyncio.CancelledError: If the async task is cancelled
    """
    if isinstance(objects, dict):
        keys = list(objects.keys())
        objects = list(objects.values())
    elif isinstance(objects, list):
        keys = list(range(len(objects)))
    else:
        keys = [0]
        objects = [objects]

    tasks = [
        asyncio.create_task(
            wait_for_sync_barrier(obj, poll_interval)
            if isinstance(obj, (ThreadingBarrier, ProcessingBarrier))
            else wait_for_sync_event(obj, poll_interval)
        )
        for obj in objects
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel the remaining pending tasks
    for pend in pending:
        pend.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    return keys[tasks.index(list(done)[0])]
