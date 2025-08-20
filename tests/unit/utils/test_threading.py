import asyncio
import threading
from collections.abc import Iterator

import pytest

from guidellm.utils.threading import synchronous_to_exitable_async


def _infinite_counter() -> Iterator[int]:
    i = 0
    while True:
        i += 1
        yield i


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_callable_completed_returns_value():
    async def run():
        def add(a: int, b: int) -> int:
            return a + b

        reason, value = await synchronous_to_exitable_async(add, None, None, 0.01, 2, 3)
        return reason, value

    reason, value = await run()
    assert reason == "completed"
    assert value == 5


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_iterable_completed_returns_last_item():
    items = ["a", "b", "c"]
    reason, value = await synchronous_to_exitable_async(items, None, None, 0.005)
    assert reason == "completed"
    assert value == "c"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_iterator_exits_on_custom_event():
    stop_event = threading.Event()

    async def trigger_event():
        await asyncio.sleep(0.02)
        stop_event.set()

    task = asyncio.create_task(
        synchronous_to_exitable_async(
            _infinite_counter(),
            exit_events={"stop": stop_event},
            exit_barrier=None,
            poll_interval=0.005,
        )
    )
    trigger = asyncio.create_task(trigger_event())
    reason, value = await task
    await trigger

    assert reason == "stop"
    assert isinstance(value, int)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_barrier_triggers_exit():
    barrier = threading.Barrier(2)

    waiter = threading.Thread(target=barrier.wait, daemon=True)
    waiter.start()

    reason, _ = await synchronous_to_exitable_async(
        _infinite_counter(),
        exit_events=None,
        exit_barrier=barrier,
        poll_interval=0.005,
    )

    assert reason == "barrier"


@pytest.mark.sanity
@pytest.mark.asyncio
async def test_cancellation_sets_canceled_and_aborts_barrier():
    barrier = threading.Barrier(2)

    async def runner():
        return await synchronous_to_exitable_async(
            _infinite_counter(),
            exit_events=None,
            exit_barrier=barrier,
            poll_interval=0.01,
        )

    task = asyncio.create_task(runner())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(50):
        if barrier.broken:
            break
        await asyncio.sleep(0.01)
    assert barrier.broken is True


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_callable_internal_error_propagates_in_tuple():
    def boom():
        raise ValueError("boom!")

    reason, err = await synchronous_to_exitable_async(boom, None, None, 0.001)
    assert reason == "internal_error"
    assert isinstance(err, ValueError)
    assert str(err) == "boom!"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_poll_mode_only_exits_on_custom_event():
    stop_event = threading.Event()

    async def trigger():
        await asyncio.sleep(0.02)
        stop_event.set()

    trigger_task = asyncio.create_task(trigger())
    reason, last = await synchronous_to_exitable_async(
        None,
        exit_events={"stop": stop_event},
        exit_barrier=None,
        poll_interval=0.005,
    )
    await trigger_task

    assert reason == "stop"
    assert last is None
