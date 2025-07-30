"""
Helper module for importing the correct queue types.
"""

from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
from typing import Generic

from guidellm.request.types import RequestT, ResponseT
from guidellm.scheduler.result import WorkerProcessRequest, WorkerProcessResult

__all__ = [
    "MPQueues",
    "Queue",
    "QueueEmpty",
    "QueueFull",
]


@dataclass
class MPQueues(Generic[RequestT, ResponseT]):
    requests: Queue[WorkerProcessRequest[RequestT, ResponseT]]
    responses: Queue[WorkerProcessResult[RequestT, ResponseT]]
