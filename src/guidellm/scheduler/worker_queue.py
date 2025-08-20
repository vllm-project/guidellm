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
from typing import Any, Generic, TypeVar, Literal
from multiprocessing.synchronize import Event as ProcessingEvent

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


__all__ = [
    "WorkerQueueProxy",
]


MessageT = TypeVar("MessageT", bound=Any)


class WorkerQueueProxy(Generic[MessageT]):
    def __init__(
        self,
        mp_queue: Queue[MessageT],
        usage: Literal["producer", "consumer"],
        stopped_event: ThreadingEvent | ProcessingEvent | None = None,
        stop_events: list[ThreadingEvent | ProcessingEvent | None] = None,
        on_stop_event: Literal["continue", "stop", "error"] = "stop",
        on_queue_empty: Literal["continue", "stop", "stop_if_event", "error"] = "stop",
        on_queue_full: Literal["continue", "stop", "stop_if_event", "error"] = "stop",
        on_queue_shutdown: Literal[
            "continue", "stop", "stop_if_event", "error"
        ] = "stop",
        poll_interval: float = 0.1,
    ):
        self.mp_queue = mp_queue
        self.usage = usage
        self.stopped_event = stopped_event
        self.stop_events = stop_events
        self.on_stop_event = on_stop_event
        self.on_queue_empty = on_queue_empty
        self.on_queue_full = on_queue_full
        self.on_queue_shutdown = on_queue_shutdown
        self.poll_interval = poll_interval

        self.local_queue: culsans.Queue[MessageT] = culsans.Queue()
        self.running = False

    async def run(self):
        self.running = True
        func = (
            self._producer_generator
            if self.usage == "producer"
            else self._consumer_generator
        )
        await synchronous_to_exitable_async(synchronous=func(), poll_interval=0.0)
        self.running = False

    def sync_put(
        self, item: MessageT, block: bool = True, timeout: float | None = None
    ):
        if self.usage != "producer":
            raise ValueError("WorkerQueueProxy is not a producer")

        self.local_queue.sync_put(item, block=block, timeout=timeout)

    def sync_put_nowait(self, item: MessageT):
        if self.usage != "producer":
            raise ValueError("WorkerQueueProxy is not a producer")

        self.local_queue.put_nowait(item)

    async def async_put(self, item: MessageT, timeout: float | None = None):
        if self.usage != "producer":
            raise ValueError("WorkerQueueProxy is not a producer")

        await asyncio.wait_for(self.local_queue.async_put(item), timeout)

    def sync_get(self, block: bool = True, timeout: float | None = None) -> MessageT:
        if self.usage != "consumer":
            raise ValueError("WorkerQueueProxy is not a consumer")

        return self.local_queue.sync_get(block=block, timeout=timeout)

    def sync_get_nowait(self) -> MessageT:
        if self.usage != "consumer":
            raise ValueError("WorkerQueueProxy is not a consumer")

        return self.local_queue.get_nowait()

    async def async_get(self, timeout: float | None = None) -> MessageT:
        if self.usage != "consumer":
            raise ValueError("WorkerQueueProxy is not a consumer")

        return await asyncio.wait_for(self.local_queue.async_get(), timeout)

    def _producer_generator(self):
        last_yield_time = time.time()

        while True:
            stop_set = (
                any(event.is_set() for event in self.stop_events)
                if self.stop_events
                else False
            )

            if stop_set and self.on_stop_event == "stop":
                break

            if stop_set and self.on_stop_event == "error":
                raise RuntimeError(
                    "WorkerQueueProxy stop event set unexpectedly "
                    "(on_stop_event==error)"
                )

            if self.on_stop_event != "continue" and any(
                event.is_set() for event in self.stop_events
            ):
                if self.on_stop_event == "stop":
                    break
                if self.on_stop_event == "error":
                    raise RuntimeError(
                        "WorkerQueueProxy stop event set unexpectedly "
                        "(on_stop_event==error)"
                    )

    def _consumer_generator(self):
        pass
