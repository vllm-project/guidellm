import asyncio
import contextlib
import functools
import time
from collections.abc import Generator, Iterable, Iterator
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from threading import Barrier as ThreadingBarrier
from threading import BrokenBarrierError, Thread
from threading import Event as ThreadingEvent
from typing import Any, Callable, Literal, Optional, Union

__all__ = ["synchronous_to_exitable_async"]


def _start_barrier_monitor_thread(
    barrier: Optional[Union[ThreadingBarrier, ProcessingBarrier]],
    barrier_event: ThreadingEvent,
):
    if barrier is None:
        return

    def _watch() -> None:
        try:
            barrier.wait()
        except BrokenBarrierError:
            pass
        finally:
            barrier_event.set()

    Thread(target=_watch, daemon=True).start()


def _check_event_set(
    events: list[tuple[str, Union[ThreadingEvent, ProcessingEvent]]],
) -> Optional[str]:
    for name, event in events:
        if event.is_set():
            return name
    return None


def _run_worker(
    events_list: list[tuple[str, Union[ThreadingEvent, ProcessingEvent]]],
    exit_barrier: Optional[Union[ThreadingBarrier, ProcessingBarrier]],
    synchronous: Optional[Union[Iterator, Iterable, Generator, Callable]],
    poll_interval: float,
    args: tuple,
    kwargs: dict,
) -> tuple[str, Any]:
    finish_reason: str = "completed"
    last_val: Any = None

    try:
        barrier_event = list(filter(lambda x: x[0] == "barrier", events_list))[0][1]
        _start_barrier_monitor_thread(exit_barrier, barrier_event)

        if isinstance(synchronous, Iterable):
            synchronous = iter(synchronous)

        while True:
            if (check_event := _check_event_set(events_list)) is not None:
                finish_reason = check_event
                break

            if isinstance(synchronous, (Iterator, Generator)):
                try:
                    last_val = next(synchronous)
                except StopIteration:
                    break
            elif isinstance(synchronous, Callable):
                last_val = synchronous(*args, **kwargs)
                break

            time.sleep(poll_interval)

        if (
            finish_reason == "completed"
            and (check_event := _check_event_set(events_list)) is not None
        ):
            # Final check for any exit signals
            finish_reason = check_event
    except Exception as err:  # noqa: BLE001
        finish_reason = "internal_error"
        last_val = err
    finally:
        if exit_barrier is not None:
            with contextlib.suppress(BrokenBarrierError, RuntimeError):
                exit_barrier.abort()

    return finish_reason, last_val


async def synchronous_to_exitable_async(
    synchronous: Optional[Union[Iterator, Iterable, Generator, Callable]],
    exit_events: Optional[dict[str, Union[ThreadingEvent, ProcessingEvent]]] = None,
    exit_barrier: Optional[Union[ThreadingBarrier, ProcessingBarrier]] = None,
    poll_interval: float = 0.1,
    *args,
    **kwargs,
) -> tuple[Union[Literal["completed", "canceled", "barrier"], str], Any]:
    """
    Run a sync callable or iterable inside an async context with exit controls.
    Supports cooperative termination via exit events and an optional barrier.

    :param synchronous: Callable (invoked once) or iterable/iterator (next()). If
        None, only watch exit events (poll mode).
    :param exit_events: Optional mapping of name -> Event objects to signal exit.
        'canceled', 'barrier', and 'internal_error' are reserved keywords.
    :param exit_barrier: Optional barrier to coordinate shutdown; when it trips or is
        aborted, the worker exits with reason "barrier". On exit, this function aborts
        the barrier to release any waiters.
    :param poll_interval: Sleep duration (seconds) used only in poll mode.
    :param args: Positional arguments passed to the callable (if provided).
    :param kwargs: Keyword arguments passed to the callable (if provided).
    :return: (exit_reason, last_item). exit_reason is "completed", "canceled",
        "barrier", or a key from exit_events. last_item is the last yielded value for
        an iterator or the return value for a callable.
    :raises asyncio.CancelledError: If the async task is canceled.
    """
    events_map = exit_events or {}

    canceled_event = ThreadingEvent()
    barrier_event = ThreadingEvent()
    events_list = [
        ("canceled", canceled_event),
        ("barrier", barrier_event),
        *list(events_map.items()),
    ]
    worker = functools.partial(
        _run_worker,
        events_list,
        exit_barrier,
        synchronous,
        poll_interval,
        args,
        kwargs,
    )

    try:
        return await asyncio.to_thread(worker)
    except asyncio.CancelledError:
        if exit_barrier is not None:
            with contextlib.suppress(BrokenBarrierError, RuntimeError):
                exit_barrier.abort()
        canceled_event.set()
        raise
    except Exception as err:  # noqa: BLE001
        print(f"******EXCEPTION in synchronous_to_exitable_async: {err}")
