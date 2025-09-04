from dataclasses import dataclass
from typing import (
    Generic,
    Literal,
    Optional,
)

from guidellm.objects import StandardBaseModel
from guidellm.request.types import RequestT, ResponseT
from guidellm.scheduler.strategy import SchedulingStrategy

__all__ = [
    "SchedulerRequestInfo",
    "SchedulerRequestResult",
    "SchedulerResult",
    "SchedulerRunInfo",
    "WorkerProcessRequest",
    "WorkerProcessResult",
]


class SchedulerRunInfo(StandardBaseModel):
    """
    Information about the current run of the scheduler.
    This class holds metadata about the scheduling run,
    including the start and end times, the number of processes,
    and the scheduling strategy used.
    It also tracks the number of requests created, queued, pending,
    and completed during the run.

    :param start_time: The start time of the scheduling run.
    :param end_time: The end time of the scheduling run;
        if None, then this will be math.inf.
    :param end_number: The maximum number of requests to be processed;
        if None, then this will be math.inf.
    :param processes: The number of processes used in the scheduling run.
    :param strategy: The scheduling strategy used in the run.
        This should be an instance of SchedulingStrategy.
    :param created_requests: The number of requests created during the run.
    :param queued_requests: The number of requests queued during the run.
    :param scheduled_requests: The number of requests scheduled during the run.
        (requests pending being sent to the worker but recieved by a process)
    :param processing_requests: The number of requests actively being run.
    :param completed_requests: The number of requests completed during the run.
    """

    start_time: float
    end_time: float
    end_number: float
    processes: int
    strategy: SchedulingStrategy

    created_requests: int = 0
    queued_requests: int = 0
    scheduled_requests: int = 0
    processing_requests: int = 0
    completed_requests: int = 0


class SchedulerRequestInfo(StandardBaseModel):
    """
    Information about a specific request run through the scheduler.
    This class holds metadata about the request, including
    the targeted start time, queued time, start time, end time,
    and the process ID that handled the request.

    :param targeted_start_time: The targeted start time for the request (time.time()).
    :param queued_time: The time the request was queued (time.time()).
    :param scheduled_time: The time the request was scheduled (time.time())
        (any sleep time before the request was sent to the worker).
    :param worker_start: The time the worker started processing request (time.time()).
    :param worker_end: The time the worker finished processing request. (time.time()).
    :param process_id: The ID of the underlying process that handled the request.
    """

    requested: bool = False
    completed: bool = False
    errored: bool = False
    canceled: bool = False

    targeted_start_time: float = -1
    queued_time: float = -1
    dequeued_time: float = -1
    scheduled_time: float = -1
    worker_start: float = -1
    request_start: float = -1
    request_end: float = -1
    worker_end: float = -1
    process_id: int = -1


class SchedulerResult(StandardBaseModel):
    """
    The yielded, iterative result for a scheduler run.
    These are triggered on the start and end of the run,
    as well as on the start and end of each request.
    Depending on the type, it will hold the request and response
    along with information and statistics about the request and general run.

    :param type_: The type of the result, which can be one of:
        - "run_start": Indicates the start of the run.
        - "run_complete": Indicates the completion of the run (teardown happens after).
        - "request_start": Indicates the start of a request.
        - "request_complete": Indicates the completion of a request.
    :param request: The request that was processed.
    :param response: The response from the worker for the request.
    :param request_info: Information about the request, including
        the targeted start time, queued time, start time, end time,
        and the process ID that handled the request.
    :param run_info: Information about the current run of the scheduler,
        including the start and end times, the number of processes,
        and the scheduling strategy used.
        It also tracks the number of requests created, queued, pending,
        and completed during the run.
    """

    pydantic_type: Literal["scheduler_result"] = "scheduler_result"
    type_: Literal[
        "run_start",
        "run_complete",
        "request_scheduled",
        "request_start",
        "request_complete",
    ]
    run_info: SchedulerRunInfo


class SchedulerRequestResult(
    SchedulerResult,
    Generic[RequestT, ResponseT],
):
    pydantic_type: Literal["scheduler_request_result"] = "scheduler_request_result"  # type: ignore[assignment]
    type_: Literal[
        "request_scheduled",
        "request_start",
        "request_complete",
    ]
    request: RequestT
    request_info: SchedulerRequestInfo
    response: Optional[ResponseT] = None


@dataclass
class WorkerProcessRequest(Generic[RequestT, ResponseT]):
    request: RequestT
    timeout_time: float
    queued_time: float


@dataclass
class WorkerProcessResult(Generic[RequestT, ResponseT]):
    type_: Literal["request_scheduled", "request_start", "request_complete"]
    request: RequestT
    response: Optional[ResponseT]
    info: SchedulerRequestInfo
