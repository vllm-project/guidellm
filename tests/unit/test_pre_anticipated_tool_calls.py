"""
Tests for pre-anticipated tool call design: tool-call turns are pre-planned at
data generation time rather than dynamically created in the worker loop.

Covers config validation, synthetic data generation, finalizer flag setting,
request handler tool_choice overrides, and worker missing-tool-call behavior.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import time
from multiprocessing import Barrier, Event
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from guidellm.backends.openai.request_handlers import (
    ChatCompletionsRequestHandler,
)
from guidellm.data.finalizers import GenerativeRequestFinalizer
from guidellm.data.schemas import SyntheticTextDatasetConfig
from guidellm.scheduler import SynchronousStrategy, WorkerProcess
from guidellm.schemas import GenerationRequest, RequestInfo, UsageMetrics
from guidellm.utils.imports import json
from guidellm.utils.messaging import InterProcessMessagingQueue
from tests.unit.testing_utils import async_timeout

# ---------------------------------------------------------------------------
# SyntheticTextDatasetConfig validation
# ---------------------------------------------------------------------------


class TestSyntheticTextDatasetConfigToolCallFields:
    """Validate tool_call_turns and tools fields on SyntheticTextDatasetConfig.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_defaults_no_tool_calling(self):
        """Default config has no tool calling enabled.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDatasetConfig(prompt_tokens=50, output_tokens=50)
        assert config.tool_call_turns == 0
        assert config.tools is None

    @pytest.mark.smoke
    def test_tool_call_turns_less_than_turns(self):
        """tool_call_turns must be strictly less than turns.

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=2
        )
        assert config.tool_call_turns == 2

    @pytest.mark.sanity
    def test_tool_call_turns_equal_to_turns_accepted(self):
        """
        tool_call_turns == turns is valid (all turns are tool-call turns,
        no final plain-text response).

        ## WRITTEN BY AI ##
        """
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50, output_tokens=50, turns=3, tool_call_turns=3
        )
        assert config.tool_call_turns == 3

    @pytest.mark.sanity
    def test_tool_call_turns_greater_than_turns_rejected(self):
        """tool_call_turns > turns is invalid.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="tool_call_turns"):
            SyntheticTextDatasetConfig(
                prompt_tokens=50, output_tokens=50, turns=2, tool_call_turns=5
            )

    @pytest.mark.sanity
    def test_tools_without_tool_call_turns_rejected(self):
        """Providing tools but tool_call_turns=0 is invalid.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValueError, match="tool_call_turns is 0"):
            SyntheticTextDatasetConfig(
                prompt_tokens=50,
                output_tokens=50,
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

    @pytest.mark.sanity
    def test_custom_tools_accepted(self):
        """Custom tools with valid tool_call_turns are accepted.

        ## WRITTEN BY AI ##
        """
        custom_tools = [{"type": "function", "function": {"name": "my_func"}}]
        config = SyntheticTextDatasetConfig(
            prompt_tokens=50,
            output_tokens=50,
            turns=3,
            tool_call_turns=1,
            tools=custom_tools,
        )
        assert config.tools == custom_tools
        assert config.tool_call_turns == 1


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestSyntheticDataToolColumns:
    """Verify synthetic data emits tools_{turn} columns for tool_call_turns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def processor(self):
        """Minimal mock processor for token encoding/decoding.

        ## WRITTEN BY AI ##
        """
        proc = MagicMock()
        proc.encode.return_value = list(range(100))
        proc.decode.return_value = "mock text"
        return proc

    @pytest.mark.smoke
    def test_no_tools_columns_when_tool_call_turns_zero(self, processor):
        """With tool_call_turns=0, no tools columns are emitted.

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            _SyntheticTextExamplesIterable,
        )

        config = SyntheticTextDatasetConfig(
            prompt_tokens=10, output_tokens=10, turns=3
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" not in row
        assert "tools_1" not in row
        assert "tools_2" not in row

    @pytest.mark.smoke
    def test_tools_columns_emitted_for_tool_call_turns(self, processor):
        """With tool_call_turns=2 and turns=3, tools_0 and tools_1 are emitted.

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            DEFAULT_SYNTHETIC_TOOLS,
            _SyntheticTextExamplesIterable,
        )

        config = SyntheticTextDatasetConfig(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        assert "tools_0" in row
        assert "tools_1" in row
        assert "tools_2" not in row

        # Values are JSON-serialized lists
        tools_0 = json.loads(row["tools_0"])
        assert tools_0 == DEFAULT_SYNTHETIC_TOOLS

    @pytest.mark.sanity
    def test_custom_tools_used_in_synthetic_data(self, processor):
        """User-provided tools are used instead of the default placeholder.

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            _SyntheticTextExamplesIterable,
        )

        custom_tools = [{"type": "function", "function": {"name": "custom_fn"}}]
        config = SyntheticTextDatasetConfig(
            prompt_tokens=10,
            output_tokens=10,
            turns=2,
            tool_call_turns=1,
            tools=custom_tools,
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        _, row = next(iter(iterable))

        tools_0 = json.loads(row["tools_0"])
        assert tools_0 == custom_tools

    @pytest.mark.sanity
    def test_features_include_tools_columns(self, processor):
        """Features property includes tools_{i} entries for tool_call_turns.

        ## WRITTEN BY AI ##
        """
        from guidellm.data.deserializers.synthetic import (
            _SyntheticTextExamplesIterable,
        )

        config = SyntheticTextDatasetConfig(
            prompt_tokens=10, output_tokens=10, turns=3, tool_call_turns=2
        )
        iterable = _SyntheticTextExamplesIterable(config, processor, random_seed=42)
        features = iterable.features

        assert "tools_0" in features
        assert "tools_1" in features
        assert "tools_2" not in features


# ---------------------------------------------------------------------------
# Finalizer: expects_tool_call flag
# ---------------------------------------------------------------------------


class TestFinalizerExpectsToolCall:
    """Verify GenerativeRequestFinalizer sets expects_tool_call correctly.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def finalizer(self):
        """
        ## WRITTEN BY AI ##
        """
        return GenerativeRequestFinalizer()

    @pytest.mark.smoke
    def test_expects_tool_call_matches_tools_column_presence(self, finalizer):
        """expects_tool_call is True only on turns that have tools_column.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"], "tools_column": ['[{"type": "function"}]']},
            {"text_column": ["world"]},
        ]
        results = finalizer(items)

        assert results[0].expects_tool_call is True
        assert results[1].expects_tool_call is False

    @pytest.mark.smoke
    def test_all_turns_with_tools_all_expect_tool_call(self, finalizer):
        """When every turn has tools_column, every turn expects a tool call.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"], "tools_column": ['[{"type": "function"}]']},
            {"text_column": ["world"], "tools_column": ['[{"type": "function"}]']},
        ]
        results = finalizer(items)

        assert results[0].expects_tool_call is True
        assert results[1].expects_tool_call is True

    @pytest.mark.sanity
    def test_expects_tool_call_false_without_tools(self, finalizer):
        """Turns without tools_column have expects_tool_call=False.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"]},
            {"text_column": ["world"]},
        ]
        results = finalizer(items)

        assert results[0].expects_tool_call is False
        assert results[1].expects_tool_call is False

    @pytest.mark.sanity
    def test_single_turn_with_tools_expects_tool_call(self, finalizer):
        """A single-turn conversation with tools has expects_tool_call=True.

        ## WRITTEN BY AI ##
        """
        items = [
            {"text_column": ["hello"], "tools_column": ['[{"type": "function"}]']},
        ]
        results = finalizer(items)
        assert results[0].expects_tool_call is True


# ---------------------------------------------------------------------------
# Request handler: tool_choice override
# ---------------------------------------------------------------------------


class TestChatCompletionsToolChoiceOverride:
    """Verify tool_choice is overridden to 'none' on non-tool-call turns.

    ## WRITTEN BY AI ##
    """

    @pytest.fixture
    def handler(self):
        """
        ## WRITTEN BY AI ##
        """
        return ChatCompletionsRequestHandler()

    @pytest.mark.smoke
    def test_tool_choice_none_when_expects_false(self, handler):
        """When expects_tool_call=False and tools come from dataset, tool_choice='none'.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=False,
        )
        extras = {"body": {"tool_choice": "required"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "none"

    @pytest.mark.smoke
    def test_tool_choice_preserved_when_expects_true(self, handler):
        """When expects_tool_call=True, the configured tool_choice is kept.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=True,
        )
        extras = {"body": {"tool_choice": "required"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "required"

    @pytest.mark.sanity
    def test_auto_tool_choice_preserved_when_expects_true(self, handler):
        """When expects_tool_call=True with auto mode, tool_choice stays 'auto'.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=True,
        )
        extras = {"body": {"tool_choice": "auto"}}
        result = handler.format(data, extras=extras)

        assert result.body["tool_choice"] == "auto"

    @pytest.mark.sanity
    def test_no_override_without_tools(self, handler):
        """Without tools in body, no tool_choice override happens.

        ## WRITTEN BY AI ##
        """
        data = GenerationRequest(
            columns={"text_column": ["test"]},
            expects_tool_call=False,
        )
        result = handler.format(data)

        assert "tool_choice" not in result.body

    @pytest.mark.sanity
    def test_per_request_tools_deserialized_from_json(self, handler):
        """JSON-serialized tools from synthetic data are deserialized.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "get_data"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=True,
        )
        result = handler.format(data)

        assert result.body["tools"] == tools

    @pytest.mark.smoke
    def test_max_completion_tokens_stripped_on_tool_call_turn(self, handler):
        """On tool-call turns, max_completion_tokens is removed so the model
        can finish producing valid tool call JSON without truncation.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=True,
            output_metrics=UsageMetrics(text_tokens=100),
        )
        result = handler.format(data)

        assert "max_completion_tokens" not in result.body
        assert "max_tokens" not in result.body

    @pytest.mark.smoke
    def test_max_completion_tokens_kept_on_plain_text_turn(self, handler):
        """On the final plain-text turn, max_completion_tokens is preserved
        to control output length.

        ## WRITTEN BY AI ##
        """
        tools = [{"type": "function", "function": {"name": "fn"}}]
        data = GenerationRequest(
            columns={
                "text_column": ["test"],
                "tools_column": [json.dumps(tools)],
            },
            expects_tool_call=False,
            output_metrics=UsageMetrics(text_tokens=100),
        )
        result = handler.format(data)

        assert result.body["max_completion_tokens"] == 100


# ---------------------------------------------------------------------------
# Worker: missing tool call handling
# ---------------------------------------------------------------------------


class _MockToolBackend:
    """Mock backend that returns configurable tool_calls on responses."""

    def __init__(
        self,
        has_tool_calls: bool = True,
        tool_call_missing_behavior: str = "error_stop",
    ):
        self.has_tool_calls = has_tool_calls
        self.tool_call_missing_behavior = tool_call_missing_behavior
        self.process_startup_called = False
        self.validate_called = False
        self.process_shutdown_called = False

    @property
    def processes_limit(self):
        return None

    @property
    def requests_limit(self):
        return None

    @property
    def info(self):
        return {"type": "mock_tool"}

    async def process_startup(self):
        self.process_startup_called = True

    async def validate(self):
        self.validate_called = True

    async def process_shutdown(self):
        self.process_shutdown_called = True

    async def resolve(self, request, request_info, history=None):
        response = MagicMock()
        response.tool_calls = (
            [{"id": "call_1", "type": "function", "function": {"name": "fn"}}]
            if self.has_tool_calls
            else None
        )
        yield response, request_info


def _make_conversation(
    num_turns: int, tool_call_turns: int
) -> list[tuple[Any, RequestInfo]]:
    """Build a pre-planned conversation list for testing.

    ## WRITTEN BY AI ##
    """
    conv = []
    for i in range(num_turns):
        req = GenerationRequest(
            columns={"text_column": [f"turn_{i}"]},
            expects_tool_call=(i < tool_call_turns),
        )
        info = RequestInfo(
            request_id=req.request_id,
            conversation_id="conv_1",
            turn_index=i,
            status="queued",
        )
        conv.append((req, info))
    return conv


class TestWorkerMissingToolCallBehavior:
    """Test worker handling of missing tool calls for all 3 behaviors.

    ## WRITTEN BY AI ##
    """

    @pytest_asyncio.fixture
    async def make_worker(self):
        """Factory fixture that creates a worker with the given backend.

        ## WRITTEN BY AI ##
        """
        workers = []

        async def _factory(backend):
            messaging = InterProcessMessagingQueue(
                serialization="dict",
                encoding=None,
                max_buffer_receive_size=10,
                poll_interval=0.01,
            )
            await messaging.start(pydantic_models=[])

            worker = WorkerProcess(
                worker_index=0,
                messaging=messaging.create_worker_copy(0),
                backend=backend,
                strategy=SynchronousStrategy(),
                async_limit=1,
                fut_scheduling_time_limit=10.0,
                startup_barrier=Barrier(1),
                requests_generated_event=Event(),
                constraint_reached_event=Event(),
                shutdown_event=Event(),
                error_event=Event(),
            )
            workers.append((worker, messaging))
            return worker, messaging

        yield _factory

        for _, msg in workers:
            await msg.stop()

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_ignore_continue_keeps_remaining_turns(self, make_worker):
        """ignore_continue: conversation continues to next turn normally.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_continue",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        # ignore_continue: remaining turns are preserved
        assert len(remaining_conv) == 2
        assert remaining_conv[0][0].expects_tool_call is True
        assert remaining_conv[1][0].expects_tool_call is False

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_ignore_stop_cancels_remaining_turns(self, make_worker):
        """ignore_stop: all remaining turns are cancelled.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        # Collect status updates
        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        # ignore_stop: conversation should be empty
        assert len(remaining_conv) == 0

        # Should have cancelled the remaining 2 turns
        cancelled_updates = [u for u in updates if u[0] == "cancelled"]
        assert len(cancelled_updates) == 2

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_error_stop_errors_and_cancels(self, make_worker):
        """error_stop: current turn errored, remaining cancelled.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        # error_stop: conversation should be empty
        assert len(remaining_conv) == 0

        # Should have errored the current turn
        errored_updates = [u for u in updates if u[0] == "errored"]
        assert len(errored_updates) == 1

        # Should have cancelled the remaining 2 turns
        cancelled_updates = [u for u in updates if u[0] == "cancelled"]
        assert len(cancelled_updates) == 2

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_tool_call_present_continues_normally(self, make_worker):
        """When tool call IS present, conversation continues normally.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=True,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        # Tool call present: conversation continues with remaining turns
        assert len(remaining_conv) == 2

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_non_tool_turn_ignores_behavior(self, make_worker):
        """Non-tool turns don't trigger missing-tool-call logic at all.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="error_stop",
        )
        worker, messaging = await make_worker(backend)

        # Conversation with no tool call turns
        conv = _make_conversation(num_turns=2, tool_call_turns=0)
        await messaging.put(conv)

        await worker._processing_startup()

        updates = []
        original_send = worker._send_update

        def capture_send(status, response, request, request_info):
            updates.append((status, request_info.request_id))
            original_send(status, response, request, request_info)

        worker._send_update = capture_send

        history, remaining_conv, info = await worker._process_next_request(
            target_start=time.time()
        )

        # No error or cancellation -- normal flow
        assert len(remaining_conv) == 1
        errored = [u for u in updates if u[0] == "errored"]
        cancelled = [u for u in updates if u[0] == "cancelled"]
        assert len(errored) == 0
        assert len(cancelled) == 0

    @async_timeout(5.0)
    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_ignore_continue_per_turn_independence(self, make_worker):
        """ignore_continue: each tool turn succeeds or fails independently.

        Process two turns sequentially. First turn misses tool call but
        conversation continues. Second tool turn gets its own chance.

        ## WRITTEN BY AI ##
        """
        backend = _MockToolBackend(
            has_tool_calls=False,
            tool_call_missing_behavior="ignore_continue",
        )
        worker, messaging = await make_worker(backend)

        conv = _make_conversation(num_turns=3, tool_call_turns=2)
        await messaging.put(conv)

        await worker._processing_startup()

        # Process first turn (tool turn, no tool call produced)
        history, remaining_conv, _ = await worker._process_next_request(
            target_start=time.time()
        )

        # Still have 2 remaining turns
        assert len(remaining_conv) == 2

        # The next turn is also a tool turn
        assert remaining_conv[0][0].expects_tool_call is True


# ---------------------------------------------------------------------------
# Backend: tool_call_missing_behavior validation
# ---------------------------------------------------------------------------


class TestOpenAIBackendToolCallMissingBehavior:
    """Validate tool_call_missing_behavior field on the backend.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_default_is_error_stop(self):
        """Default tool_call_missing_behavior is error_stop.

        ## WRITTEN BY AI ##
        """
        from guidellm.backends.openai.http import OpenAIHTTPBackend

        backend = OpenAIHTTPBackend(target="http://localhost:8000")
        assert backend.tool_call_missing_behavior == "error_stop"

    @pytest.mark.sanity
    def test_valid_behaviors_accepted(self):
        """All valid tool_call_missing_behavior values are accepted.

        ## WRITTEN BY AI ##
        """
        from guidellm.backends.openai.http import OpenAIHTTPBackend

        for behavior in ("ignore_continue", "ignore_stop", "error_stop"):
            backend = OpenAIHTTPBackend(
                target="http://localhost:8000",
                tool_call_missing_behavior=behavior,
            )
            assert backend.tool_call_missing_behavior == behavior

    @pytest.mark.sanity
    def test_invalid_behavior_rejected(self):
        """Invalid tool_call_missing_behavior is rejected by the args validator.

        ## WRITTEN BY AI ##
        """
        from guidellm.backends.openai.http import OpenAIHttpBackendArgs

        with pytest.raises(ValueError, match="Invalid tool_call_missing_behavior"):
            OpenAIHttpBackendArgs(
                target="http://localhost:8000",
                tool_call_missing_behavior="invalid_mode",
            )
