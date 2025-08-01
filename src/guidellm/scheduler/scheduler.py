import asyncio
import math
import multiprocessing
import multiprocessing.queues
import time
from collections.abc import AsyncGenerator, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.synchronize import Event as MultiprocessingEvent
from typing import (
    Any,
    Generic,
    Optional,
    Union,
)

from loguru import logger

from guidellm.config import settings
from guidellm.scheduler.result import (
    SchedulerRequestResult,
    SchedulerResult,
    SchedulerRunInfo,
)
from guidellm.scheduler.strategy import SchedulingStrategy
from guidellm.scheduler.types import RequestT, ResponseT
from guidellm.scheduler.worker import (
    RequestsWorker,
    WorkerProcessRequest,
    WorkerProcessResult,
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
        self.error_rate: Optional[float] = None

    async def run(
        self,
        scheduling_strategy: SchedulingStrategy,
        max_number: Optional[int] = None,
        max_duration: Optional[float] = None,
        max_error: Optional[float] = None,
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
        :param max_error: The maximum error rate or const
            after which the scheduler shuts down.
            Only applicable in benchmarks with finite deterministic number of requests.
            If None or not applicable then scheduler will continue regardless of errors.
        :return: An asynchronous generator that yields SchedulerResult objects.
            Each SchedulerResult object contains information about the request,
            the response, and the run information.
        """
        self._validate_scheduler_params(
            scheduling_strategy, max_number, max_duration, max_error
        )

        with (
            multiprocessing.Manager() as manager,
            ProcessPoolExecutor(
                max_workers=scheduling_strategy.processes_limit
            ) as executor,
        ):
            requests_iter: Optional[Iterator[Any]] = None
            (
                futures,
                requests_queue,
                responses_queue,
                shutdown_event,
            ) = await self._start_processes(manager, executor, scheduling_strategy)
            if shutdown_event.is_set():
                raise RuntimeError("shutdown_event is set before starting scheduling")

            run_info, requests_iter, times_iter = self._run_setup(
                futures, scheduling_strategy, max_number, max_duration, max_error
            )
            yield SchedulerResult(
                type_="run_start",
                run_info=run_info,
            )

            try:
                while not self._should_stop(
                    run_info=run_info,
                    requests_iter=requests_iter,
                ):
                    # check errors and raise them
                    for future in futures:
                        if future.done() and (err := future.exception()) is not None:
                            raise err

                    requests_iter = self._add_requests(
                        requests_iter,
                        times_iter,
                        requests_queue,
                        run_info,
                    )
                    await asyncio.sleep(0)  # enable requests to start

                    iter_result = self._check_result_ready(
                        responses_queue,
                        run_info,
                    )
                    if iter_result is not None:
                        yield iter_result

                    # yield control to the event loop
                    await asyncio.sleep(settings.default_async_loop_sleep)
            except Exception as err:
                raise RuntimeError(f"Scheduler run failed: {err}") from err

            shutdown_event.set()
            yield SchedulerResult(
                type_="run_complete",
                run_info=run_info,
            )

            logger.debug("Waiting for futures to shut down")
            await asyncio.gather(*futures)

    def _should_stop(
        self, run_info: SchedulerRunInfo, requests_iter: Optional[Iterator[Any]]
    ):
        # we've exhausted all requests we've wanted to run
        # and yielded all responses
        is_complete = (
            requests_iter is None
            and run_info.completed_requests >= run_info.created_requests
        )

        max_error_reached = self._handle_max_error(run_info)

        return is_complete or max_error_reached

    def _handle_max_error(
        self,
        run_info: SchedulerRunInfo,
    ) -> bool:
        max_error_reached = self._is_max_error_reached(run_info)

        if max_error_reached:
            run_info.termination_reason = "max_error_reached"
            logger.info(
                f"Max error rate of "
                f"({run_info.max_error}) "
                f"reached, sending shutdown signal"
            )
        return max_error_reached

    def _validate_scheduler_params(
        self,
        scheduling_strategy: SchedulingStrategy,
        max_number: Optional[int],
        max_duration: Optional[float],
        max_error: Optional[float],
    ) -> None:
        if scheduling_strategy is None or not isinstance(
            scheduling_strategy, SchedulingStrategy
        ):
            raise ValueError(f"Invalid scheduling strategy: {scheduling_strategy}")
        if max_number is not None and max_number < 1:
            raise ValueError(f"Invalid max_number: {max_number}")
        if max_duration is not None and max_duration < 0:
            raise ValueError(f"Invalid max_duration: {max_duration}")
        if max_error is not None and (max_error < 0):
            raise ValueError(f"Invalid max_error: {max_error}")

    def _is_max_error_reached(self, run_info: SchedulerRunInfo) -> bool:
        if settings.error_check_window_size > run_info.current_window:
            return False
        max_error = run_info.max_error
        if max_error is None:
            max_error = math.inf

        if max_error >= 1:
            # Absolute error count, i.e not a ratio
            logger.debug(
                f"Window error count: "
                f"{run_info.errors_in_window} / {max_error} (max error)"
            )
            max_error_reached = max_error < run_info.errors_in_window
        else:
            window_error_ratio = run_info.errors_in_window / run_info.current_window
            logger.debug(
                f"Window error rate: {window_error_ratio} "
                f"i.e errors_in_window / current_window"
            )
            max_error_reached = max_error < window_error_ratio

        if not max_error_reached:
            run_info.current_window = 0
            run_info.errors_in_window = 0

        return max_error_reached

    async def _start_processes(
        self,
        manager,
        executor: ProcessPoolExecutor,
        scheduling_strategy: SchedulingStrategy,
    ) -> tuple[
        list[asyncio.Future],
        multiprocessing.Queue,
        multiprocessing.Queue,
        MultiprocessingEvent,
    ]:
        await self.worker.prepare_multiprocessing()
        shutdown_event = manager.Event()
        requests_queue = manager.Queue(
            maxsize=scheduling_strategy.queued_requests_limit
        )
        responses_queue = manager.Queue()

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
            if scheduling_strategy.processing_mode in ["sync", "async"]:
                futures.append(
                    loop.run_in_executor(
                        executor,
                        self.worker.run_process,
                        scheduling_strategy.processing_mode,
                        requests_queue,
                        responses_queue,
                        shutdown_event,
                        settings.shutdown_poll_interval_seconds,
                        id_,
                        requests_limit,
                    )
                )
            else:
                raise ValueError(
                    f"Invalid processing mode: {scheduling_strategy.processing_mode} "
                    f"for strategy: {scheduling_strategy}"
                )

        await asyncio.sleep(0.1)  # give time for processes to start

        return futures, requests_queue, responses_queue, shutdown_event

    def _run_setup(
        self,
        processes: list[asyncio.Future],
        scheduling_strategy: SchedulingStrategy,
        max_number: Optional[int],
        max_duration: Optional[float],
        max_error: Optional[float],
    ) -> tuple[SchedulerRunInfo, Iterator[Any], Iterator[float]]:
        requests_iter = iter(self.request_loader)
        start_time = time.time()
        times_iter = iter(scheduling_strategy.request_times())
        end_time = time.time() + (max_duration or math.inf)
        end_number = self._determine_total_requests_count(max_number)

        if end_number == math.inf and end_time is None:
            logger.warning(
                "No end number or end time set, "
                "scheduler will run indefinitely until the request loader is exhausted."
            )

        info = SchedulerRunInfo(
            start_time=start_time,
            end_time=end_time,
            end_number=end_number,
            processes=len(processes),
            strategy=scheduling_strategy,
            max_error=max_error,
        )

        return info, requests_iter, times_iter

    def _determine_total_requests_count(
        self,
        max_number: Optional[int],
    ) -> Union[int, float]:
        end_number = max_number or math.inf
        try:
            # update end_number if the request_loader is finite and less than max_number
            iter_length = len(self.request_loader)  # type: ignore[arg-type]
            if 0 < iter_length < end_number:
                end_number = iter_length
        except Exception:  # noqa: BLE001, S110
            pass
        return end_number

    def _add_requests(
        self,
        requests_iter: Optional[Iterator[Any]],
        times_iter: Iterator[float],
        requests_queue: multiprocessing.Queue,
        run_info: SchedulerRunInfo,
    ) -> Optional[Iterator[Any]]:
        if requests_iter is not None:
            try:
                added_count = 0

                while (
                    not requests_queue.full()
                    and added_count < settings.max_add_requests_per_loop
                ):
                    if run_info.created_requests >= run_info.end_number:
                        # When `--max-seconds` is set and the dataset is finite -
                        # almost always the benchmark will end slightly before the
                        # actual `end_time`.
                        # Since the units are seconds, rounding up 1s is reasonable.
                        if time.time() >= run_info.end_time - 1:
                            run_info.termination_reason = "max_seconds_reached"
                        else:
                            run_info.termination_reason = "max_requests_reached"
                        raise StopIteration

                    if (
                        request_time := next(times_iter)
                    ) >= run_info.end_time or time.time() >= run_info.end_time:
                        run_info.termination_reason = "max_seconds_reached"
                        raise StopIteration

                    request = next(requests_iter)
                    work_req: WorkerProcessRequest[RequestT] = WorkerProcessRequest(
                        request=request,
                        start_time=request_time,
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
        responses_queue: multiprocessing.Queue,
        run_info: SchedulerRunInfo,
    ) -> Optional[SchedulerRequestResult[RequestT, ResponseT]]:
        try:
            process_response: WorkerProcessResult[RequestT, ResponseT] = (
                responses_queue.get_nowait()
            )
        except multiprocessing.queues.Empty:  # type: ignore[attr-defined]
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
            run_info.current_window += 1

            if process_response.info.errored:
                run_info.errored_requests += 1
                run_info.errors_in_window += 1

            return SchedulerRequestResult(
                type_="request_complete",
                run_info=run_info,
                request=process_response.request,
                request_info=process_response.info,
                response=process_response.response,
            )
        raise ValueError(f"Invalid process response type: {process_response}")
