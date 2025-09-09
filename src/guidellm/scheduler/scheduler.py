import asyncio
import math
import time
from collections.abc import AsyncGenerator, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from threading import Event
from typing import (
    Any,
    Generic,
    Optional,
    Union,
)

from loguru import logger

from guidellm.config import settings
from guidellm.request.types import (
    RequestT,
    ResponseT,
)
from guidellm.scheduler.queues import MPQueues, Queue, QueueEmpty
from guidellm.scheduler.result import (
    SchedulerRequestResult,
    SchedulerResult,
    SchedulerRunInfo,
    WorkerProcessRequest,
    WorkerProcessResult,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.worker import (
    RequestsWorker,
)

__all__ = ["Scheduler"]


class Scheduler(Generic[RequestT, ResponseT]):
    """
    A class that handles the scheduling of requests to a worker.
    This class is responsible for managing the lifecycle of the requests,
    including their creation, queuing, and processing.
    It uses a multiprocessing approach to handle requests concurrently
    and efficiently, based on the specified scheduling strategy.
    The Scheduler class is designed to work with a RequestsWorker,
    which is an abstract base class that defines the interface for a worker
    that can resolve requests asynchronously or synchronously.
    The Scheduler class also supports different scheduling strategies,
    including synchronous, throughput, and concurrent strategies.

    :param worker: The worker that will process the requests.
        This should be an instance of RequestsWorker.
    :param request_loader: An iterable that generates requests.
        This can be a list, generator, or any other iterable.
        The requests will be processed by the worker.
    """

    def __init__(
        self,
        worker: RequestsWorker[RequestT, ResponseT],
        request_loader: Iterable[RequestT],
    ):
        if not isinstance(worker, RequestsWorker):
            raise ValueError(f"Invalid worker: {worker}")

        if not isinstance(request_loader, Iterable):
            raise ValueError(f"Invalid request_loader: {request_loader}")

        self.worker = worker
        self.request_loader = request_loader

    async def run(
        self,
        scheduling_strategy: SchedulingStrategy,
        max_number: Optional[int] = None,
        max_duration: Optional[float] = None,
    ) -> AsyncGenerator[
        Union[SchedulerResult, SchedulerRequestResult[RequestT, ResponseT]], None
    ]:
        """
        The main method that runs the scheduler.
        This method is a generator that yields SchedulerResult objects
        at the start and end of the run, as well as at the start and end
        of each request.
        It uses multiprocessing to handle requests concurrently
        and efficiently, based on the specified scheduling strategy.
        The method also handles the lifecycle of the requests,
        including their creation, queuing, and processing.
        The method is designed to be used as an asynchronous generator,
        allowing it to be used with asyncio and other asynchronous frameworks.

        :param scheduling_strategy: The scheduling strategy to use.
            Specifies the times at which requests will be sent as well how many
            worker processes are used and if requests are scheduled sync or async.
            This can be one of the following:
            - "synchronous": Requests are sent synchronously.
            - "throughput": Requests are sent at the maximum rate possible.
            - An instance of SchedulingStrategy.
        :param max_number: The maximum number of requests to process.
            If None, then no limit is set and either the iterator must be exhaustible
            or the max_duration must be set.
        :param max_duration: The maximum duration for the scheduling run.
            If None, then no limit is set and either the iterator must be exhaustible
            or the max_number must be set.
        :return: An asynchronous generator that yields SchedulerResult objects.
            Each SchedulerResult object contains information about the request,
            the response, and the run information.
        """
        if scheduling_strategy is None or not isinstance(
            scheduling_strategy, SchedulingStrategy
        ):
            raise ValueError(f"Invalid scheduling strategy: {scheduling_strategy}")

        if max_number is not None and max_number < 1:
            raise ValueError(f"Invalid max_number: {max_number}")

        if max_duration is not None and max_duration < 0:
            raise ValueError(f"Invalid max_duration: {max_duration}")

        with (
            Manager() as manager,
            ProcessPoolExecutor(
                max_workers=scheduling_strategy.processes_limit
            ) as executor,
        ):
            requests_iter: Optional[Iterator[Any]] = None
            scheduling_strategy.start_time = (
                time.time() + settings.scheduler_start_delay
            )  # Add a small delay to allow processes to start
            futures, queues, stop_event = await self._start_processes(
                manager, executor, scheduling_strategy
            )
            run_info, requests_iter, times_iter = self._run_setup(
                futures, scheduling_strategy, max_number, max_duration
            )

            # Add some initial requests to the queue
            requests_iter = self._add_requests(
                requests_iter,
                queues.requests,
                times_iter,
                run_info,
            )
            # Wait for the test to start
            await asyncio.sleep(time.time() - scheduling_strategy.start_time)
            yield SchedulerResult(
                type_="run_start",
                run_info=run_info,
            )

            try:
                while True:
                    # check errors and raise them
                    for future in futures:
                        if future.done() and (err := future.exception()) is not None:
                            raise err

                    if (
                        requests_iter is None
                        and run_info.processing_requests <= 0
                        and (  # Ensure we have met one of the end conditions
                            time.time() >= run_info.end_time
                            or run_info.completed_requests >= run_info.end_number
                        )
                    ):
                        # we've exhausted all requests we've wanted to run
                        # and yielded all responses
                        break

                    requests_iter = self._add_requests(
                        requests_iter,
                        queues.requests,
                        times_iter,
                        run_info,
                    )
                    await asyncio.sleep(0)  # enable requests to start

                    iter_result = self._check_result_ready(
                        queues.responses,
                        run_info,
                    )
                    if iter_result is not None:
                        yield iter_result

                    # yield control to the event loop
                    await asyncio.sleep(settings.default_async_loop_sleep)
            except Exception as err:
                raise RuntimeError(f"Scheduler run failed: {err}") from err

            yield SchedulerResult(
                type_="run_complete",
                run_info=run_info,
            )

            await self._stop_processes(futures, stop_event)

    async def _start_processes(
        self,
        manager,
        executor: ProcessPoolExecutor,
        scheduling_strategy: SchedulingStrategy,
    ) -> tuple[
        list[asyncio.Future],
        MPQueues[RequestT, ResponseT],
        Event,
    ]:
        await self.worker.prepare_multiprocessing()
        queues: MPQueues[RequestT, ResponseT] = MPQueues(
            requests=manager.Queue(
                maxsize=scheduling_strategy.processing_requests_limit
            ),
            responses=manager.Queue(),
        )
        stop_event = manager.Event()

        num_processes = min(
            scheduling_strategy.processes_limit,
            scheduling_strategy.processing_requests_limit,
        )
        requests_limit_split = (
            scheduling_strategy.processing_requests_limit
            // scheduling_strategy.processes_limit
        )
        requests_limit_remain = (
            scheduling_strategy.processing_requests_limit
            % scheduling_strategy.processes_limit
        )
        process_ids = (id_ for id_ in range(num_processes))
        process_requests_limits = (
            requests_limit_split + 1
            if i < requests_limit_remain
            else requests_limit_split
            for i in range(num_processes)
        )

        futures = []
        loop = asyncio.get_event_loop()
        for id_, requests_limit in zip(process_ids, process_requests_limits):
            futures.append(
                loop.run_in_executor(
                    executor,
                    self.worker.process_loop_asynchronous,
                    queues,
                    scheduling_strategy,
                    stop_event,
                    requests_limit,
                    id_,
                    num_processes,
                )
            )

        await asyncio.sleep(0.1)  # give time for processes to start

        return futures, queues, stop_event

    def _run_setup(
        self,
        processes: list[asyncio.Future],
        scheduling_strategy: SchedulingStrategy,
        max_number: Optional[int],
        max_duration: Optional[float],
    ) -> tuple[SchedulerRunInfo, Iterator[Any], Iterator[float]]:
        requests_iter = iter(self.request_loader)
        times_iter = iter(scheduling_strategy.request_times())
        end_time = scheduling_strategy.start_time + (max_duration or math.inf)
        end_number = max_number or math.inf

        try:
            # update end number if the request loader is finite and less than max
            iter_length = len(self.request_loader)  # type: ignore[arg-type]
            if 0 < iter_length < end_number:
                end_number = iter_length
        except Exception:  # noqa: BLE001, S110
            pass

        if end_number == math.inf and end_time is None:
            logger.warning(
                "No end number or end time set, "
                "scheduler will run indefinitely until the request loader is exhausted."
            )

        info = SchedulerRunInfo(
            start_time=scheduling_strategy.start_time,
            end_time=end_time,
            end_number=end_number,
            processes=len(processes),
            strategy=scheduling_strategy,
        )

        return info, requests_iter, times_iter

    def _add_requests(
        self,
        requests_iter: Optional[Iterator[Any]],
        requests_queue: Queue[WorkerProcessRequest[RequestT, ResponseT]],
        times_iter: Iterator[float],
        run_info: SchedulerRunInfo,
    ) -> Optional[Iterator[Any]]:
        if requests_iter is not None:
            try:
                added_count = 0

                while not requests_queue.full() and added_count < (
                    run_info.strategy.queued_requests_limit
                    or settings.min_queued_requests
                ):
                    if run_info.created_requests >= run_info.end_number:
                        raise StopIteration

                    if (
                        next(times_iter) >= run_info.end_time
                        or time.time() >= run_info.end_time
                    ):
                        raise StopIteration

                    work_req = WorkerProcessRequest[RequestT, ResponseT](
                        request=next(requests_iter),
                        timeout_time=run_info.end_time,
                        queued_time=time.time(),
                    )
                    requests_queue.put(work_req)

                    run_info.created_requests += 1
                    run_info.queued_requests += 1
                    added_count += 1
            except StopIteration:
                # we've reached the limit number, limit time, or exhausted the requests
                # set to None to stop adding more and tell the loop no more requests
                requests_iter = None

        return requests_iter

    def _check_result_ready(
        self,
        responses_queue: Queue[WorkerProcessResult[RequestT, ResponseT]],
        run_info: SchedulerRunInfo,
    ) -> Optional[SchedulerRequestResult[RequestT, ResponseT]]:
        try:
            process_response: WorkerProcessResult[RequestT, ResponseT] = (
                responses_queue.get_nowait()
            )
        except QueueEmpty:
            return None

        if process_response.type_ == "request_scheduled":
            run_info.queued_requests -= 1
            run_info.scheduled_requests += 1

            return SchedulerRequestResult(
                type_="request_scheduled",
                run_info=run_info,
                request=process_response.request,
                request_info=process_response.info,
                response=None,
            )

        if process_response.type_ == "request_start":
            run_info.scheduled_requests -= 1
            run_info.processing_requests += 1

            return SchedulerRequestResult(
                type_="request_start",
                run_info=run_info,
                request=process_response.request,
                request_info=process_response.info,
                response=None,
            )

        if process_response.type_ == "request_complete":
            run_info.processing_requests -= 1
            run_info.completed_requests += 1

            return SchedulerRequestResult(
                type_="request_complete",
                run_info=run_info,
                request=process_response.request,
                request_info=process_response.info,
                response=process_response.response,
            )
        raise ValueError(f"Invalid process response type: {process_response}")

    async def _stop_processes(
        self,
        futures: list[asyncio.Future],
        stop_event: Event,
    ):
        # stop all processes
        stop_event.set()

        await asyncio.gather(*futures)
