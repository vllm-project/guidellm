from __future__ import annotations

import asyncio
import multiprocessing
import threading
from typing import Any, TypeVar

import culsans
import pytest
from pydantic import BaseModel

from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)
from guidellm.schemas.request import GenerationRequestArguments
from guidellm.utils import (
    InterProcessMessaging,
    InterProcessMessagingManagerQueue,
    InterProcessMessagingPipe,
    InterProcessMessagingQueue,
)
from guidellm.utils.messaging import ReceiveMessageT, SendMessageT
from tests.unit.testing_utils import async_timeout


class MockMessage(BaseModel):
    content: str
    num: int


class MockProcessTarget:
    """Mock process target for testing."""

    def __init__(
        self,
        messaging: InterProcessMessaging,
        num_messages: int,
        worker_index: int = 0,
    ):
        self.messaging = messaging
        self.num_messages = num_messages
        self.worker_index = worker_index

    def run(self):
        loop = asyncio.new_event_loop()

        try:
            asyncio.set_event_loop(loop)
            asyncio.run(asyncio.wait_for(self._async_runner(), timeout=10.0))
        except RuntimeError:
            pass
        finally:
            loop.close()

    async def _async_runner(self):
        await self.messaging.start(
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )

        try:
            for _ in range(self.num_messages):
                obj = await self.messaging.get(timeout=2.0)
                await self.messaging.put(obj, timeout=2.0)
        finally:
            await self.messaging.stop()


@pytest.fixture(
    params=[
        {"ctx_name": "fork"},
        {"ctx_name": "spawn"},
    ],
    ids=["fork_ctx", "spawn_ctx"],
)
def multiprocessing_contexts(request):
    context = multiprocessing.get_context(request.param["ctx_name"])
    manager = context.Manager()
    try:
        yield manager, context
    finally:
        manager.shutdown()


def test_send_message_type():
    """Test that SendMessageT is filled out correctly as a TypeVar."""
    assert isinstance(SendMessageT, type(TypeVar("test")))
    assert SendMessageT.__name__ == "SendMessageT"
    assert SendMessageT.__bound__ is Any
    assert SendMessageT.__constraints__ == ()


def test_receive_message_type():
    """Test that ReceiveMessageT is filled out correctly as a TypeVar."""
    assert isinstance(ReceiveMessageT, type(TypeVar("test")))
    assert ReceiveMessageT.__name__ == "ReceiveMessageT"
    assert ReceiveMessageT.__bound__ is Any
    assert ReceiveMessageT.__constraints__ == ()


class TestInterProcessMessaging:
    """Test suite for InterProcessMessaging abstract base class."""

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test InterProcessMessaging abstract class signatures."""
        assert hasattr(InterProcessMessaging, "__init__")
        assert hasattr(InterProcessMessaging, "create_worker_copy")
        assert hasattr(InterProcessMessaging, "create_send_messages_threads")
        assert hasattr(InterProcessMessaging, "create_receive_messages_threads")
        assert hasattr(InterProcessMessaging, "start")
        assert hasattr(InterProcessMessaging, "stop")
        assert hasattr(InterProcessMessaging, "get")
        assert hasattr(InterProcessMessaging, "put")

        # Check abstract methods
        assert getattr(
            InterProcessMessaging.create_worker_copy, "__isabstractmethod__", False
        )
        assert getattr(
            InterProcessMessaging.create_send_messages_threads,
            "__isabstractmethod__",
            False,
        )
        assert getattr(
            InterProcessMessaging.create_receive_messages_threads,
            "__isabstractmethod__",
            False,
        )

    @pytest.mark.smoke
    def test_cannot_instantiate_directly(self):
        """Test InterProcessMessaging cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InterProcessMessaging()


class TestInterProcessMessagingQueue:
    """Test suite for InterProcessMessagingQueue."""

    @pytest.fixture(
        params=[
            {
                "serialization": "dict",
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
            {
                "serialization": "sequence",
                "encoding": None,
                "max_pending_size": 10,
                "max_buffer_send_size": 2,
                "max_done_size": 5,
                "max_buffer_receive_size": 3,
                "worker_index": None,
            },
            {
                "serialization": None,
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
        ],
    )
    def valid_instances(self, multiprocessing_contexts, request):
        """Fixture providing test data for InterProcessMessagingQueue."""
        constructor_args = request.param
        manager, context = multiprocessing_contexts
        instance = InterProcessMessagingQueue(
            **constructor_args, poll_interval=0.01, mp_context=context
        )

        return instance, constructor_args, manager, context

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test InterProcessMessagingQueue inheritance and signatures."""
        assert issubclass(InterProcessMessagingQueue, InterProcessMessaging)
        assert hasattr(InterProcessMessagingQueue, "__init__")
        assert hasattr(InterProcessMessagingQueue, "create_worker_copy")
        assert hasattr(InterProcessMessagingQueue, "create_send_messages_threads")
        assert hasattr(InterProcessMessagingQueue, "create_receive_messages_threads")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test InterProcessMessagingQueue initialization."""
        instance, constructor_args, _, _ = valid_instances

        assert isinstance(instance, InterProcessMessagingQueue)
        assert instance.worker_index == constructor_args["worker_index"]
        assert instance.max_pending_size == constructor_args["max_pending_size"]
        assert instance.max_done_size == constructor_args["max_done_size"]
        assert hasattr(instance, "pending_queue")
        assert hasattr(instance, "done_queue")
        assert instance.running is False

    @pytest.mark.smoke
    def test_create_worker_copy(self, valid_instances):
        """Test InterProcessMessagingQueue.create_worker_copy."""
        instance, _, _, _ = valid_instances
        worker_index = 42

        worker_copy = instance.create_worker_copy(worker_index)

        assert isinstance(worker_copy, InterProcessMessagingQueue)
        assert worker_copy.worker_index == worker_index
        assert worker_copy.pending_queue is instance.pending_queue
        assert worker_copy.done_queue is instance.done_queue
        assert worker_copy.max_pending_size == instance.max_pending_size
        assert worker_copy.max_done_size == instance.max_done_size

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_events_lambda",
        [
            list,
            lambda: [threading.Event()],
            lambda: [multiprocessing.Event()],
            lambda: [threading.Event(), multiprocessing.Event()],
        ],
    )
    @async_timeout(5.0)
    async def test_start_stop_lifecycle(self, valid_instances, stop_events_lambda):
        """Test InterProcessMessagingQueue start/stop lifecycle."""
        instance, _, _, _ = valid_instances
        stop_events = stop_events_lambda()

        # Initially not running
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

        # Start should work
        await instance.start(
            send_stop_criteria=stop_events, receive_stop_criteria=stop_events
        )
        assert instance.running is True
        assert instance.send_stopped_event is not None
        assert isinstance(instance.send_stopped_event, threading.Event)
        assert instance.receive_stopped_event is not None
        assert isinstance(instance.receive_stopped_event, threading.Event)
        assert instance.shutdown_event is not None
        assert isinstance(instance.shutdown_event, threading.Event)
        assert instance.buffer_send_queue is not None
        assert isinstance(instance.buffer_send_queue, culsans.Queue)
        assert instance.buffer_receive_queue is not None
        assert isinstance(instance.buffer_receive_queue, culsans.Queue)
        assert instance.send_task is not None
        assert isinstance(instance.send_task, asyncio.Task)
        assert instance.receive_task is not None
        assert isinstance(instance.receive_task, asyncio.Task)

        # Stop should work
        if stop_events:
            for event in stop_events:
                event.set()

            await asyncio.sleep(0.1)
            assert instance.send_stopped_event.is_set()
            assert instance.receive_stopped_event.is_set()
            assert instance.send_task.done()
            assert instance.receive_task.done()

        await instance.stop()
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_obj",
        [
            123451,
            "asdfghjkl",
            [None, 123, 45.67, "string", {"key": "value"}, [1, 2, 3]],
            {"key": "value", "another_key": 123.456, "yet_another_key": [1, 2, 3]},
            MockMessage(content="hello", num=42),
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
            (
                GenerationResponse(
                    request_id="",
                    request_args=None,
                    text="test response",
                ),
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
        ],
    )
    @async_timeout(10.0)
    async def test_lifecycle_put_get(self, valid_instances, test_obj):
        instance, constructor_args, manager, context = valid_instances

        if (
            (
                isinstance(test_obj, RequestInfo)
                or (
                    isinstance(test_obj, tuple)
                    and any(isinstance(item, RequestInfo) for item in test_obj)
                )
            )
            and constructor_args["serialization"] is None
            and constructor_args["encoding"] is None
        ):
            # Handle case where RequestInfo is not pickleable
            pytest.skip("RequestInfo is not pickleable")

        # Worker setup
        process_target = MockProcessTarget(
            instance.create_worker_copy(0), num_messages=5
        )
        process = context.Process(target=process_target.run)
        process.start()

        # Local startup and wait
        await instance.start(
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )
        await asyncio.sleep(0.1)

        try:
            for _ in range(5):
                await instance.put(test_obj, timeout=2.0)

            for _ in range(5):
                val = await instance.get(timeout=2.0)
                if not isinstance(test_obj, tuple):
                    assert val == test_obj
                else:
                    assert list(val) == list(test_obj)
        finally:
            # Clean up
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            await instance.stop()

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_obj",
        [
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
            (
                GenerationResponse(
                    request_id="",
                    request_args=None,
                    text="test response",
                ),
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
        ],
    )
    @async_timeout(10.0)
    async def test_lifecycle_put_get_iter(self, valid_instances, test_obj):
        instance, constructor_args, manager, context = valid_instances

        if (
            (
                isinstance(test_obj, RequestInfo)
                or (
                    isinstance(test_obj, tuple)
                    and any(isinstance(item, RequestInfo) for item in test_obj)
                )
            )
            and constructor_args["serialization"] is None
            and constructor_args["encoding"] is None
        ):
            # Handle case where RequestInfo is not pickleable
            pytest.skip("RequestInfo is not pickleable")

        # Worker setup
        process_target = MockProcessTarget(
            instance.create_worker_copy(0), num_messages=5
        )
        process = context.Process(target=process_target.run)
        process.start()

        def _received_callback(msg):
            if not isinstance(test_obj, tuple):
                assert msg == test_obj
            else:
                assert list(msg) == list(test_obj)
            return "changed_obj"

        # Local startup and wait
        await instance.start(
            send_items=[test_obj for _ in range(5)],
            receive_callback=_received_callback,
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )
        await asyncio.sleep(0.1)

        try:
            for _ in range(5):
                val = await instance.get(timeout=2.0)
                assert val == "changed_obj"
        finally:
            # Clean up
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            await instance.stop()


class TestInterProcessMessagingManagerQueue:
    """Test suite for InterProcessMessagingManagerQueue."""

    @pytest.fixture(
        params=[
            {
                "serialization": "dict",
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
            {
                "serialization": "sequence",
                "encoding": None,
                "max_pending_size": 10,
                "max_buffer_send_size": 2,
                "max_done_size": 5,
                "max_buffer_receive_size": 3,
                "worker_index": None,
            },
            {
                "serialization": None,
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
        ],
    )
    def valid_instances(self, multiprocessing_contexts, request):
        """Fixture providing test data for InterProcessMessagingManagerQueue."""
        constructor_args = request.param
        manager, context = multiprocessing_contexts
        instance = InterProcessMessagingManagerQueue(
            **constructor_args, manager=manager, poll_interval=0.01
        )
        return instance, constructor_args, manager, context

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test InterProcessMessagingManagerQueue inheritance and signatures."""
        assert issubclass(InterProcessMessagingManagerQueue, InterProcessMessaging)
        assert issubclass(InterProcessMessagingManagerQueue, InterProcessMessagingQueue)
        assert hasattr(InterProcessMessagingManagerQueue, "__init__")
        assert hasattr(InterProcessMessagingManagerQueue, "create_worker_copy")
        assert hasattr(InterProcessMessagingManagerQueue, "_send_messages_task_thread")
        assert hasattr(
            InterProcessMessagingManagerQueue, "_receive_messages_task_thread"
        )

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test InterProcessMessagingManagerQueue initialization."""
        instance, constructor_args, _, _ = valid_instances

        assert isinstance(instance, InterProcessMessagingManagerQueue)
        assert instance.worker_index == constructor_args["worker_index"]
        assert instance.max_pending_size == constructor_args["max_pending_size"]
        assert instance.max_done_size == constructor_args["max_done_size"]
        assert hasattr(instance, "pending_queue")
        assert hasattr(instance, "done_queue")
        assert instance.running is False

    @pytest.mark.smoke
    def test_create_worker_copy(self, valid_instances):
        """Test InterProcessMessagingQueue.create_worker_copy."""
        instance, _, _, _ = valid_instances
        worker_index = 42

        worker_copy = instance.create_worker_copy(worker_index)

        assert isinstance(worker_copy, InterProcessMessagingManagerQueue)
        assert worker_copy.worker_index == worker_index
        assert worker_copy.pending_queue is instance.pending_queue
        assert worker_copy.done_queue is instance.done_queue
        assert worker_copy.max_pending_size == instance.max_pending_size
        assert worker_copy.max_done_size == instance.max_done_size

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_events_lambda",
        [
            list,
            lambda: [threading.Event()],
            lambda: [multiprocessing.Event()],
            lambda: [threading.Event(), multiprocessing.Event()],
        ],
    )
    @async_timeout(5.0)
    async def test_start_stop_lifecycle(self, valid_instances, stop_events_lambda):
        """Test InterProcessMessagingQueue start/stop lifecycle."""
        instance, _, _, _ = valid_instances
        stop_events = stop_events_lambda()

        # Initially not running
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

        # Start should work
        await instance.start(
            send_stop_criteria=stop_events, receive_stop_criteria=stop_events
        )
        assert instance.running is True
        assert instance.send_stopped_event is not None
        assert isinstance(instance.send_stopped_event, threading.Event)
        assert instance.receive_stopped_event is not None
        assert isinstance(instance.receive_stopped_event, threading.Event)
        assert instance.shutdown_event is not None
        assert isinstance(instance.shutdown_event, threading.Event)
        assert instance.buffer_send_queue is not None
        assert isinstance(instance.buffer_send_queue, culsans.Queue)
        assert instance.buffer_receive_queue is not None
        assert isinstance(instance.buffer_receive_queue, culsans.Queue)
        assert instance.send_task is not None
        assert isinstance(instance.send_task, asyncio.Task)
        assert instance.receive_task is not None
        assert isinstance(instance.receive_task, asyncio.Task)

        # Stop should work
        if stop_events:
            for event in stop_events:
                event.set()

            await asyncio.sleep(0.1)
            assert instance.send_stopped_event.is_set()
            assert instance.receive_stopped_event.is_set()
            assert instance.send_task.done()
            assert instance.receive_task.done()

        await instance.stop()
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_obj",
        [
            123451,
            "asdfghjkl",
            [None, 123, 45.67, "string", {"key": "value"}, [1, 2, 3]],
            {"key": "value", "another_key": 123.456, "yet_another_key": [1, 2, 3]},
            MockMessage(content="hello", num=42),
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
        ],
    )
    @async_timeout(10.0)
    async def test_lifecycle_put_get(self, valid_instances, test_obj):
        instance, constructor_args, _, context = valid_instances

        if (
            (
                isinstance(test_obj, RequestInfo)
                or (
                    isinstance(test_obj, tuple)
                    and any(isinstance(item, RequestInfo) for item in test_obj)
                )
            )
            and constructor_args["serialization"] is None
            and constructor_args["encoding"] is None
        ):
            # Handle case where RequestInfo is not pickleable
            pytest.skip("RequestInfo is not pickleable")

        # Worker setup
        process_target = MockProcessTarget(
            instance.create_worker_copy(0), num_messages=5
        )
        process = context.Process(target=process_target.run)
        process.start()

        # Local startup and wait
        await instance.start(
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )
        await asyncio.sleep(0.1)

        try:
            for _ in range(5):
                await instance.put(test_obj, timeout=2.0)

            for _ in range(5):
                val = await instance.get(timeout=2.0)
                if not isinstance(test_obj, tuple):
                    assert val == test_obj
                else:
                    assert list(val) == list(test_obj)
        finally:
            # Clean up
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            await instance.stop()

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_obj",
        [
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
            (
                GenerationResponse(
                    request_id="",
                    request_args=None,
                    text="test response",
                ),
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
        ],
    )
    @async_timeout(10.0)
    async def test_lifecycle_put_get_iter(self, valid_instances, test_obj):
        instance, constructor_args, _, context = valid_instances

        if (
            (
                isinstance(test_obj, RequestInfo)
                or (
                    isinstance(test_obj, tuple)
                    and any(isinstance(item, RequestInfo) for item in test_obj)
                )
            )
            and constructor_args["serialization"] is None
            and constructor_args["encoding"] is None
        ):
            # Handle case where RequestInfo is not pickleable
            pytest.skip("RequestInfo is not pickleable")

        # Worker setup
        process_target = MockProcessTarget(
            instance.create_worker_copy(0), num_messages=5
        )
        process = context.Process(target=process_target.run)
        process.start()

        def _received_callback(msg):
            if not isinstance(test_obj, tuple):
                assert msg == test_obj
            else:
                assert list(msg) == list(test_obj)
            return "changed_obj"

        # Local startup and wait
        await instance.start(
            send_items=[test_obj for _ in range(5)],
            receive_callback=_received_callback,
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )
        await asyncio.sleep(0.1)

        try:
            for _ in range(5):
                val = await instance.get(timeout=2.0)
                assert val == "changed_obj"
        finally:
            # Clean up
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            await instance.stop()


class TestInterProcessMessagingPipe:
    """Test suite for InterProcessMessagingPipe."""

    @pytest.fixture(
        params=[
            {
                "num_workers": 2,
                "serialization": "dict",
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
            {
                "num_workers": 1,
                "serialization": "sequence",
                "encoding": None,
                "max_pending_size": 10,
                "max_buffer_send_size": 2,
                "max_done_size": 5,
                "max_buffer_receive_size": 3,
                "worker_index": None,
            },
            {
                "num_workers": 1,
                "serialization": None,
                "encoding": None,
                "max_pending_size": None,
                "max_done_size": None,
                "worker_index": None,
            },
        ],
    )
    def valid_instances(self, multiprocessing_contexts, request):
        """Fixture providing test data for InterProcessMessagingPipe."""
        constructor_args = request.param
        manager, context = multiprocessing_contexts
        instance = InterProcessMessagingPipe(**constructor_args, poll_interval=0.01)
        return instance, constructor_args, manager, context

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test InterProcessMessagingPipe inheritance and signatures."""
        assert issubclass(InterProcessMessagingPipe, InterProcessMessaging)
        assert hasattr(InterProcessMessagingPipe, "__init__")
        assert hasattr(InterProcessMessagingPipe, "create_worker_copy")
        assert hasattr(InterProcessMessagingPipe, "_send_messages_task_thread")
        assert hasattr(InterProcessMessagingPipe, "_receive_messages_task_thread")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test InterProcessMessagingPipe initialization."""
        instance, constructor_args, _, _ = valid_instances

        assert isinstance(instance, InterProcessMessagingPipe)
        assert instance.worker_index == constructor_args["worker_index"]
        assert instance.max_pending_size == constructor_args["max_pending_size"]
        assert instance.max_done_size == constructor_args["max_done_size"]
        assert instance.num_workers == constructor_args["num_workers"]
        assert hasattr(instance, "pipes")
        assert len(instance.pipes) == constructor_args["num_workers"]
        assert len(instance.pipes) == constructor_args["num_workers"]
        assert instance.running is False

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("kwargs", "expected_error"),
        [
            ({"invalid_param": "value"}, TypeError),
            ({"num_workers": 1, "unknown_arg": "test"}, TypeError),
        ],
    )
    def test_invalid_initialization_values(self, kwargs, expected_error):
        """Test InterProcessMessagingPipe with invalid field values."""
        with pytest.raises(expected_error):
            InterProcessMessagingPipe(**kwargs)

    @pytest.mark.sanity
    def test_invalid_initialization_missing(self):
        """Test InterProcessMessagingPipe initialization without required field."""
        with pytest.raises(TypeError):
            InterProcessMessagingPipe()

    @pytest.mark.smoke
    def test_create_worker_copy(self, valid_instances):
        """Test InterProcessMessagingPipe.create_worker_copy."""
        instance, _, _, _ = valid_instances
        worker_index = 0

        worker_copy = instance.create_worker_copy(worker_index)

        assert isinstance(worker_copy, InterProcessMessagingPipe)
        assert worker_copy.worker_index == worker_index
        assert worker_copy.pipes[0] is instance.pipes[worker_index]
        assert worker_copy.max_pending_size == instance.max_pending_size
        assert worker_copy.max_done_size == instance.max_done_size
        assert worker_copy.num_workers == instance.num_workers

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(5.0)
    async def test_start_stop_lifecycle(self, valid_instances):
        """Test InterProcessMessagingPipe start/stop lifecycle."""
        instance, _, _, _ = valid_instances
        stop_events = []

        # Initially not running
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

        # Start should work
        await instance.start(
            send_stop_criteria=stop_events, receive_stop_criteria=stop_events
        )
        assert instance.running is True
        assert instance.send_stopped_event is not None
        assert isinstance(instance.send_stopped_event, threading.Event)
        assert instance.receive_stopped_event is not None
        assert isinstance(instance.receive_stopped_event, threading.Event)
        assert instance.shutdown_event is not None
        assert isinstance(instance.shutdown_event, threading.Event)
        assert instance.buffer_send_queue is not None
        assert isinstance(instance.buffer_send_queue, culsans.Queue)
        assert instance.buffer_receive_queue is not None
        assert isinstance(instance.buffer_receive_queue, culsans.Queue)
        assert instance.send_task is not None
        assert isinstance(instance.send_task, asyncio.Task)
        assert instance.receive_task is not None
        assert isinstance(instance.receive_task, asyncio.Task)

        # Stop should work
        await instance.stop()
        assert instance.running is False
        assert instance.send_stopped_event is None
        assert instance.receive_stopped_event is None
        assert instance.shutdown_event is None
        assert instance.buffer_send_queue is None
        assert instance.buffer_receive_queue is None
        assert instance.send_task is None
        assert instance.receive_task is None

    @pytest.mark.xfail(reason="old and broken", run=False)
    @pytest.mark.smoke
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_obj",
        [
            123451,
            "asdfghjkl",
            [None, 123, 45.67, "string", {"key": "value"}, [1, 2, 3]],
            {"key": "value", "another_key": 123.456, "yet_another_key": [1, 2, 3]},
            MockMessage(content="hello", num=42),
            (
                None,
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
            (
                GenerationResponse(
                    request_id="",
                    request_args=None,
                    text="test response",
                ),
                GenerationRequest(
                    request_type="text_completions",
                    arguments=GenerationRequestArguments(),
                ),
                RequestInfo(),
            ),
        ],
    )
    @async_timeout(10.0)
    async def test_lifecycle_put_get(self, valid_instances, test_obj):
        instance, constructor_args, manager, context = valid_instances

        if (
            (
                isinstance(test_obj, RequestInfo)
                or (
                    isinstance(test_obj, tuple)
                    and any(isinstance(item, RequestInfo) for item in test_obj)
                )
            )
            and constructor_args["serialization"] is None
            and constructor_args["encoding"] is None
        ):
            pytest.skip("RequestInfo is not pickleable")

        # Worker setup
        processes = []
        for index in range(constructor_args["num_workers"]):
            process_target = MockProcessTarget(
                instance.create_worker_copy(index), num_messages=5
            )
            process = context.Process(target=process_target.run)
            processes.append(process)
            process.start()

        # Local startup and wait
        await instance.start(
            pydantic_models=[
                MockMessage,
                GenerationRequest,
                GenerationResponse,
                RequestInfo,
            ],
        )
        await asyncio.sleep(0.1)

        try:
            for _ in range(5 * constructor_args["num_workers"]):
                await instance.put(test_obj, timeout=2.0)

            for _ in range(5 * constructor_args["num_workers"]):
                val = await instance.get(timeout=2.0)
                if not isinstance(test_obj, tuple):
                    assert val == test_obj
                else:
                    assert list(val) == list(test_obj)
        finally:
            # Clean up
            for process in processes:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)

            await instance.stop()
