from __future__ import annotations

import asyncio
import multiprocessing
import threading
from multiprocessing.synchronize import Barrier as ProcessingBarrier
from multiprocessing.synchronize import Event as ProcessingEvent
from typing import get_args

import pytest

from guidellm.utils.synchronous import (
    SyncObjectTypesAlias,
    wait_for_sync_barrier,
    wait_for_sync_event,
    wait_for_sync_objects,
)
from tests.unit.testing_utils import async_timeout


def test_sync_object_types_alias():
    """
    Test that SyncObjectTypesAlias is defined correctly as a type alias.

    ## WRITTEN BY AI ##
    """
    # Get the actual types from the union alias
    actual_types = get_args(SyncObjectTypesAlias)

    # Define the set of expected types
    expected_types = {
        threading.Event,
        ProcessingEvent,
        threading.Barrier,
        ProcessingBarrier,
    }

    # Assert that the set of actual types matches the expected set.
    # Using a set comparison is robust as it ignores the order.
    assert set(actual_types) == expected_types


class TestWaitForSyncEvent:
    """Test suite for wait_for_sync_event function."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type",
        [threading.Event, multiprocessing.Event],
        ids=["threading", "multiprocessing"],
    )
    @async_timeout(2.0)
    async def test_invocation(self, event_type):
        """Test wait_for_sync_event with valid events that get set."""
        event: threading.Event | ProcessingEvent = event_type()

        async def set_event():
            await asyncio.sleep(0.01)
            event.set()

        asyncio.create_task(set_event())
        await wait_for_sync_event(event, poll_interval=0.001)
        assert event.is_set()

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type",
        [threading.Event, multiprocessing.Event],
        ids=["threading", "multiprocessing"],
    )
    @async_timeout(2.0)
    async def test_cancellation_stops_waiting(self, event_type):
        """Test that cancelling the task stops waiting for the event."""
        event: threading.Event | ProcessingEvent = event_type()

        async def waiter():
            await wait_for_sync_event(event, poll_interval=0.001)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestWaitForSyncBarrier:
    """Test suite for wait_for_sync_barrier function."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "barrier_type",
        [threading.Barrier, multiprocessing.Barrier],
        ids=["threading", "multiprocessing"],
    )
    @async_timeout(5.0)
    async def test_invocation(self, barrier_type):
        """Test wait_for_sync_barrier with barrier that gets reached."""
        barrier: threading.Barrier | ProcessingBarrier = barrier_type(2)

        async def reach_barrier():
            await asyncio.sleep(0.01)
            await asyncio.to_thread(barrier.wait)

        task = asyncio.create_task(reach_barrier())
        await wait_for_sync_barrier(barrier, poll_interval=0.01)
        await task

    @pytest.mark.sanity
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "barrier_type",
        [threading.Barrier, multiprocessing.Barrier],
        ids=["threading", "multiprocessing"],
    )
    @async_timeout(2.0)
    async def test_cancellation_stops_waiting(self, barrier_type):
        """Test that cancelling the task stops waiting for the barrier."""
        barrier: threading.Barrier | ProcessingBarrier = barrier_type(2)

        async def waiter():
            await wait_for_sync_barrier(barrier, 0.01)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestWaitForSyncObjects:
    """Test suite for wait_for_sync_objects function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("objects_types", "expected_result"),
        [
            (threading.Event, 0),
            (multiprocessing.Event, 0),
            (threading.Barrier, 0),
            (multiprocessing.Barrier, 0),
            ([threading.Event, multiprocessing.Barrier], 1),
            ([multiprocessing.Event, threading.Barrier], 0),
            (
                [
                    threading.Event,
                    multiprocessing.Event,
                    threading.Barrier,
                    multiprocessing.Barrier,
                ],
                2,
            ),
            (
                {
                    "multiprocessing.Event": multiprocessing.Event,
                    "threading.Barrier": threading.Barrier,
                },
                "threading.Barrier",
            ),
            (
                {
                    "threading.Event": threading.Event,
                    "multiprocessing.Barrier": multiprocessing.Barrier,
                },
                "threading.Event",
            ),
            (
                {
                    "multiprocessing.Event": multiprocessing.Event,
                    "threading.Event": threading.Event,
                    "multiprocessing.Barrier": multiprocessing.Barrier,
                    "threading.Barrier": threading.Barrier,
                },
                "threading.Event",
            ),
        ],
        ids=[
            "threading_event",
            "multiprocessing_event",
            "threading_barrier",
            "multiprocessing_barrier",
            "mixed_list_event_barrier_1",
            "mixed_list_event_barrier_2",
            "mixed_list_all",
            "mixed_dict_event_barrier_1",
            "mixed_dict_event_barrier_2",
            "mixed_dict_all",
        ],
    )
    @pytest.mark.asyncio
    @async_timeout(2.0)
    async def test_invocation(self, objects_types, expected_result):
        """Test wait_for_sync_objects with various object configurations."""
        if isinstance(objects_types, list):
            objects = [
                obj()
                if obj not in (threading.Barrier, multiprocessing.Barrier)
                else obj(2)
                for obj in objects_types
            ]
        elif isinstance(objects_types, dict):
            objects = {
                key: (
                    obj()
                    if obj not in (threading.Barrier, multiprocessing.Barrier)
                    else obj(2)
                )
                for key, obj in objects_types.items()
            }
        else:
            objects = [
                objects_types()
                if objects_types not in (threading.Barrier, multiprocessing.Barrier)
                else objects_types(2)
            ]

        async def set_target():
            await asyncio.sleep(0.01)
            obj = objects[expected_result]
            if isinstance(obj, threading.Event | ProcessingEvent):
                obj.set()
            else:
                await asyncio.to_thread(obj.wait)

        task = asyncio.create_task(set_target())
        result = await wait_for_sync_objects(objects, poll_interval=0.001)
        await task

        assert result == expected_result
