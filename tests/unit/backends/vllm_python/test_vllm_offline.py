"""
Unit tests for VLLM Offline (batch) backend.

Tests engine laziness, batch processing, deferred flush, generate-lock
serialization, shutdown drain, shutting-down guard, and metrics wiring.
"""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from guidellm.backends.vllm_python.offline import (
    VLLMOfflineBackend,
    VLLMOfflineBackendArgs,
    _BatchedRequest,
    _OfflineResolvedRequest,
)
from guidellm.schemas import GenerationRequest, RequestInfo
from tests.unit.testing_utils import async_timeout


def _fake_sampling_params(**kwargs):
    return SimpleNamespace(**kwargs)


def _make_offline_backend(**kwargs) -> VLLMOfflineBackend:
    args = VLLMOfflineBackendArgs(**kwargs)
    return VLLMOfflineBackend(args)


def _mock_request_output(
    text="hello",
    token_ids=None,
    prompt_token_ids=None,
    request_id="r1",
    metrics=None,
):
    """Build a mock vLLM RequestOutput."""
    if token_ids is None:
        token_ids = [1, 2, 3]
    if prompt_token_ids is None:
        prompt_token_ids = [10, 20, 30]
    out = Mock()
    out.outputs = [Mock(text=text, token_ids=token_ids)]
    out.prompt_token_ids = prompt_token_ids
    out.request_id = request_id
    out.metrics = metrics
    return out


@pytest.fixture
def offline_backend():
    """VLLMOfflineBackend instance without requiring vllm installed."""
    mock_vllm = MagicMock()
    mock_vllm.SamplingParams = _fake_sampling_params
    with (
        patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
        patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
    ):
        yield _make_offline_backend(model="test-model", batch_size=4)


@pytest.fixture
def started_backend(offline_backend):
    """Offline backend with _in_process=True and a mock LLM engine."""
    offline_backend._in_process = True
    mock_llm = Mock()
    mock_llm.generate.return_value = []
    mock_llm.get_tokenizer.return_value = Mock()
    offline_backend._llm = mock_llm
    return offline_backend


# ------------------------------------------------------------------
# Engine laziness
# ------------------------------------------------------------------


class TestEngineLaziness:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_startup_does_not_create_engine(self):
        """Engine is None after startup. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
        assert backend._llm is None
        assert backend._in_process is True

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_ensure_engine_creates_on_first_call(self):
        """First _ensure_engine call creates the LLM. ## WRITTEN BY AI ##"""
        mock_llm = Mock()
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        mock_vllm.EngineArgs.return_value = Mock()
        mock_vllm.LLM.from_engine_args.return_value = mock_llm
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.offline.reset_cpu_affinity"),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            result = await backend._ensure_engine()
        assert result is mock_llm
        assert backend._llm is mock_llm
        mock_vllm.LLM.from_engine_args.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_ensure_engine_idempotent(self):
        """Repeated _ensure_engine calls do not recreate. ## WRITTEN BY AI ##"""
        mock_llm = Mock()
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        mock_vllm.EngineArgs.return_value = Mock()
        mock_vllm.LLM.from_engine_args.return_value = mock_llm
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.offline.reset_cpu_affinity"),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            await backend._ensure_engine()
            await backend._ensure_engine()
        mock_vllm.LLM.from_engine_args.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_ensure_engine_concurrent_creates_once(self):
        """Concurrent _ensure_engine calls create one engine. ## WRITTEN BY AI ##"""
        mock_llm = Mock()
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        mock_vllm.EngineArgs.return_value = Mock()
        mock_vllm.LLM.from_engine_args.return_value = mock_llm
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.offline.reset_cpu_affinity"),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            results = await asyncio.gather(
                backend._ensure_engine(),
                backend._ensure_engine(),
                backend._ensure_engine(),
            )
        assert all(r is mock_llm for r in results)
        mock_vllm.LLM.from_engine_args.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_startup_raises_if_already_started(self):
        """Double startup raises RuntimeError. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            with pytest.raises(RuntimeError, match="already started"):
                await backend.process_startup()


# ------------------------------------------------------------------
# Lifecycle / shutdown
# ------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_shutdown_resets_state(self, started_backend):
        """Shutdown clears state flags.

        ## WRITTEN BY AI ##
        """
        await started_backend.process_shutdown()
        assert started_backend._in_process is False
        assert started_backend._llm is None
        assert started_backend._shutting_down is True

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_calls_llm_shutdown(self, started_backend):
        """Shutdown calls llm.shutdown() when available. ## WRITTEN BY AI ##"""
        shutdown_mock = Mock()
        started_backend._llm.shutdown = shutdown_mock
        await started_backend.process_shutdown()
        shutdown_mock.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_tolerates_missing_shutdown_method(self, started_backend):
        """Shutdown succeeds when llm has no shutdown method. ## WRITTEN BY AI ##"""
        del started_backend._llm.shutdown
        await started_backend.process_shutdown()
        assert started_backend._llm is None

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_not_started_raises(self, offline_backend):
        """Shutdown before startup raises RuntimeError. ## WRITTEN BY AI ##"""
        with pytest.raises(RuntimeError, match="not started"):
            await offline_backend.process_shutdown()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_drains_pending_batch(self, started_backend):
        """Shutdown processes pending batch before teardown. ## WRITTEN BY AI ##"""
        req = _BatchedRequest(
            resolved_prompt="hello",
            multi_modal_data=None,
            max_tokens=10,
        )
        started_backend._pending_batch.append(req)
        mock_llm = started_backend._llm
        mock_llm.generate.return_value = [_mock_request_output()]
        await started_backend.process_shutdown()
        mock_llm.generate.assert_called_once()
        assert req.ready.is_set()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(10.0)
    async def test_shutdown_waits_for_inflight_generate(self, started_backend):
        """Shutdown blocks until in-flight generate completes. ## WRITTEN BY AI ##"""
        generate_entered = threading.Event()
        generate_proceed = threading.Event()

        def slow_generate(*args, **kwargs):
            generate_entered.set()
            generate_proceed.wait(timeout=5.0)
            return [_mock_request_output() for _ in args[0]]

        started_backend._llm.generate.side_effect = slow_generate

        req = _BatchedRequest(resolved_prompt="p", multi_modal_data=None, max_tokens=10)
        started_backend._pending_batch.append(req)
        await started_backend._schedule_deferred_flush()

        # Wait for generate to start in the executor thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: generate_entered.wait(timeout=5.0))

        # Start shutdown concurrently
        shutdown_task = asyncio.ensure_future(started_backend.process_shutdown())
        await asyncio.sleep(0.05)

        assert not shutdown_task.done(), "shutdown completed before generate finished"

        generate_proceed.set()
        await shutdown_task
        assert started_backend._llm is None

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_validate_passes_after_startup(self, started_backend):
        """Validate succeeds after startup. ## WRITTEN BY AI ##"""
        await started_backend.validate()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_validate_fails_before_startup(self, offline_backend):
        """Validate before startup raises RuntimeError. ## WRITTEN BY AI ##"""
        with pytest.raises(RuntimeError, match="not started"):
            await offline_backend.validate()


# ------------------------------------------------------------------
# process_startup lock/state reset
# ------------------------------------------------------------------


class TestProcessStartupReset:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_startup_recreates_locks(self):
        """Locks are fresh objects after shutdown+startup cycle. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()

            old_batch_lock = backend._batch_lock
            old_generate_lock = backend._generate_lock
            old_engine_lock = backend._engine_lock

            backend._llm = Mock()
            await backend.process_shutdown()
            await backend.process_startup()

        assert backend._batch_lock is not old_batch_lock
        assert backend._generate_lock is not old_generate_lock
        assert backend._engine_lock is not old_engine_lock

        assert backend._pending_batch == []
        assert backend._processing_task is None

        async with backend._batch_lock:
            pass
        async with backend._generate_lock:
            pass
        async with backend._engine_lock:
            pass

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_startup_clears_llm(self):
        """process_startup sets _llm to None. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
            backend._llm = Mock()
            backend._in_process = True
            await backend.process_shutdown()
            await backend.process_startup()
        assert backend._llm is None


# ------------------------------------------------------------------
# PID-based engine preload
# ------------------------------------------------------------------


class TestPidBasedPreload:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_validate_skips_engine_in_parent(self):
        """validate() in parent process does not create engine. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            await backend.validate()
        assert backend._llm is None

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_validate_preloads_engine_in_worker(self):
        """validate() in forked worker preloads the engine. ## WRITTEN BY AI ##"""
        mock_llm = Mock()
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        mock_vllm.EngineArgs.return_value = Mock()
        mock_vllm.LLM.from_engine_args.return_value = mock_llm
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.offline.reset_cpu_affinity"),
        ):
            backend = _make_offline_backend(model="test-model")
            await backend.process_startup()
            # Simulate a forked worker by changing _creator_pid
            backend._creator_pid = os.getpid() + 1
            await backend.validate()
        assert backend._llm is mock_llm

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_creator_pid_set_in_init(self):
        """_creator_pid is set to current PID in __init__. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
        assert backend._creator_pid == os.getpid()


# ------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------


class TestBatchProcessing:
    @pytest.mark.sanity
    def test_take_pending_batch_clears_queue(self, started_backend):
        """_take_pending_batch snapshots and clears the queue. ## WRITTEN BY AI ##"""
        req1 = _BatchedRequest(
            resolved_prompt="a", multi_modal_data=None, max_tokens=10
        )
        req2 = _BatchedRequest(
            resolved_prompt="b", multi_modal_data=None, max_tokens=10
        )
        started_backend._pending_batch = [req1, req2]
        batch = started_backend._take_pending_batch()
        assert batch == [req1, req2]
        assert started_backend._pending_batch == []

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_run_generate_empty_batch_noop(self, started_backend):
        """Empty batch skips generate call. ## WRITTEN BY AI ##"""
        await started_backend._run_generate([])
        started_backend._llm.generate.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_run_generate_distributes_results(self, started_backend):
        """Results are distributed to matching batch requests. ## WRITTEN BY AI ##"""
        reqs = [
            _BatchedRequest(
                resolved_prompt=f"prompt-{i}",
                multi_modal_data=None,
                max_tokens=10,
            )
            for i in range(3)
        ]
        outputs = [_mock_request_output(text=f"out-{i}") for i in range(3)]
        started_backend._llm.generate.return_value = outputs
        await started_backend._run_generate(reqs)
        for req, out in zip(reqs, outputs, strict=True):
            assert req.result is out
            assert req.ready.is_set()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_run_generate_error_signals_all_waiters(self, started_backend):
        """Generate errors signal all waiters with the exception. ## WRITTEN BY AI ##"""
        reqs = [
            _BatchedRequest(resolved_prompt="p", multi_modal_data=None, max_tokens=10)
            for _ in range(2)
        ]
        started_backend._llm.generate.side_effect = RuntimeError("boom")
        await started_backend._run_generate(reqs)
        for req in reqs:
            assert isinstance(req.result, RuntimeError)
            assert req.ready.is_set()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_run_generate_multimodal_prompt_format(self, started_backend):
        """Multimodal data is passed as dict prompt. ## WRITTEN BY AI ##"""
        mm_data = {"image": Mock()}
        req = _BatchedRequest(
            resolved_prompt="describe image",
            multi_modal_data=mm_data,
            max_tokens=10,
        )
        started_backend._llm.generate.return_value = [_mock_request_output()]
        await started_backend._run_generate([req])
        call_args = started_backend._llm.generate.call_args
        prompts = call_args[0][0]
        assert isinstance(prompts[0], dict)
        assert prompts[0]["prompt"] == "describe image"
        assert prompts[0]["multi_modal_data"] is mm_data

    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_maybe_process_batch_triggers_at_capacity(self, started_backend):
        """Full batch triggers immediate generate. ## WRITTEN BY AI ##"""
        reqs = [
            _BatchedRequest(
                resolved_prompt=f"p-{i}",
                multi_modal_data=None,
                max_tokens=10,
            )
            for i in range(4)  # batch_size=4 from fixture
        ]
        started_backend._pending_batch = list(reqs)
        outputs = [_mock_request_output() for _ in range(4)]
        started_backend._llm.generate.return_value = outputs
        await started_backend._maybe_process_batch()
        started_backend._llm.generate.assert_called_once()
        assert started_backend._pending_batch == []

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_maybe_process_batch_skips_under_capacity(self, started_backend):
        """Under-capacity batch is not processed immediately. ## WRITTEN BY AI ##"""
        started_backend._pending_batch = [
            _BatchedRequest(
                resolved_prompt="p",
                multi_modal_data=None,
                max_tokens=10,
            )
        ]
        await started_backend._maybe_process_batch()
        started_backend._llm.generate.assert_not_called()


# ------------------------------------------------------------------
# Deferred flush
# ------------------------------------------------------------------


class TestDeferredFlush:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_deferred_flush_processes_partial_batch(self, started_backend):
        """Partial batch is flushed after batch_timeout. ## WRITTEN BY AI ##"""
        req = _BatchedRequest(
            resolved_prompt="partial",
            multi_modal_data=None,
            max_tokens=10,
        )
        started_backend._pending_batch.append(req)
        started_backend._llm.generate.return_value = [_mock_request_output()]
        await started_backend._schedule_deferred_flush()
        if started_backend._processing_task:
            await started_backend._processing_task
        assert req.ready.is_set()
        started_backend._llm.generate.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_deferred_flush_idempotent(self, started_backend):
        """Repeated schedule calls reuse the same task. ## WRITTEN BY AI ##"""
        req = _BatchedRequest(
            resolved_prompt="p",
            multi_modal_data=None,
            max_tokens=10,
        )
        started_backend._pending_batch.append(req)
        started_backend._llm.generate.return_value = [_mock_request_output()]
        await started_backend._schedule_deferred_flush()
        task1 = started_backend._processing_task
        await started_backend._schedule_deferred_flush()
        task2 = started_backend._processing_task
        assert task1 is task2
        if started_backend._processing_task:
            await started_backend._processing_task


# ------------------------------------------------------------------
# Generate lock serialization
# ------------------------------------------------------------------


class TestGenerateLock:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_generate_lock_serializes_calls(self, started_backend):
        """Concurrent _run_generate calls are serialized. ## WRITTEN BY AI ##"""
        call_order = []

        def tracking_generate(*args, **kwargs):
            call_order.append("enter")
            result = [_mock_request_output() for _ in args[0]]
            call_order.append("exit")
            return result

        started_backend._llm.generate.side_effect = tracking_generate

        batch1 = [
            _BatchedRequest(
                resolved_prompt="a",
                multi_modal_data=None,
                max_tokens=10,
            )
        ]
        batch2 = [
            _BatchedRequest(
                resolved_prompt="b",
                multi_modal_data=None,
                max_tokens=10,
            )
        ]

        await asyncio.gather(
            started_backend._run_generate(batch1),
            started_backend._run_generate(batch2),
        )
        assert call_order == ["enter", "exit", "enter", "exit"]


# ------------------------------------------------------------------
# Shutting-down guard
# ------------------------------------------------------------------


class TestShuttingDownGuard:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_resolve_rejects_during_shutdown(self, started_backend):
        """resolve() raises when backend is shutting down. ## WRITTEN BY AI ##"""
        started_backend._shutting_down = True
        request = GenerationRequest(columns={"text_column": ["test"]})
        request_info = RequestInfo()
        fake_resolved = _OfflineResolvedRequest(prompt="hello")
        with (
            patch.object(
                started_backend, "_resolve_request", return_value=fake_resolved
            ),
            pytest.raises(RuntimeError, match="shutting down"),
        ):
            async for _ in started_backend.resolve(request, request_info):
                pass

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_shutdown_flag_set_under_lock(self, started_backend):
        """_shutting_down is True after shutdown. ## WRITTEN BY AI ##"""
        await started_backend.process_shutdown()
        assert started_backend._shutting_down is True


# ------------------------------------------------------------------
# Batch timeout configuration
# ------------------------------------------------------------------


class TestBatchTimeout:
    @pytest.mark.smoke
    def test_batch_timeout_default(self):
        """Default batch_timeout is 0.01 seconds. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model")
        assert backend._args.batch_timeout == 0.01

    @pytest.mark.sanity
    def test_batch_timeout_custom(self):
        """Custom batch_timeout is accepted. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
        ):
            backend = _make_offline_backend(model="test-model", batch_timeout=0.05)
        assert backend._args.batch_timeout == 0.05

    @pytest.mark.sanity
    def test_batch_timeout_rejects_zero(self):
        """batch_timeout=0 is rejected by validation. ## WRITTEN BY AI ##"""
        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = _fake_sampling_params
        with (
            patch("guidellm.backends.vllm_python.offline.vllm", mock_vllm),
            patch("guidellm.backends.vllm_python.vllm.vllm", mock_vllm),
            pytest.raises(ValueError),
        ):
            _make_offline_backend(model="test-model", batch_timeout=0)


# ------------------------------------------------------------------
# Metrics wiring
# ------------------------------------------------------------------


class TestWireVllmMetrics:
    @pytest.mark.smoke
    def test_token_count_from_num_generation_tokens(self):
        """Token count uses num_generation_tokens when > 0. ## WRITTEN BY AI ##"""
        request_info = RequestInfo()
        metrics = SimpleNamespace(num_generation_tokens=42)
        output = _mock_request_output(metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 42
        assert request_info.timings.request_iterations == 1

    @pytest.mark.sanity
    def test_token_count_fallback_to_token_ids(self):
        """Token count falls back to len(token_ids) when 0. ## WRITTEN BY AI ##"""
        request_info = RequestInfo()
        metrics = SimpleNamespace(num_generation_tokens=0)
        output = _mock_request_output(token_ids=[1, 2, 3, 4, 5], metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 5
        assert request_info.timings.request_iterations == 1

    @pytest.mark.sanity
    def test_no_metrics_no_crash(self):
        """No metrics attribute does not crash. ## WRITTEN BY AI ##"""
        request_info = RequestInfo()
        output = _mock_request_output(metrics=None)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 3  # len(token_ids)

    @pytest.mark.smoke
    def test_timing_from_request_state_stats(self):
        """Wall-clock timings derived from RequestStateStats. ## WRITTEN BY AI ##"""
        request_info = RequestInfo()
        metrics = SimpleNamespace(
            num_generation_tokens=10,
            arrival_time=1000.0,
            scheduled_ts=100.0,
            queued_ts=99.0,
            first_token_ts=102.0,
            last_token_ts=110.0,
        )
        output = _mock_request_output(metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.first_request_iteration == 1000.0
        assert request_info.timings.first_token_iteration == pytest.approx(
            1000.0 + (102.0 - 100.0)
        )
        assert request_info.timings.last_token_iteration == pytest.approx(
            1000.0 + (110.0 - 100.0)
        )
        assert (
            request_info.timings.last_request_iteration
            == request_info.timings.last_token_iteration
        )

    @pytest.mark.sanity
    def test_timing_skipped_when_first_token_ts_missing(self):
        """Timing is skipped when first_token_ts is zero. ## WRITTEN BY AI ##"""
        request_info = RequestInfo()
        metrics = SimpleNamespace(
            num_generation_tokens=5,
            arrival_time=1000.0,
            scheduled_ts=100.0,
            queued_ts=99.0,
            first_token_ts=0.0,
            last_token_ts=0.0,
        )
        output = _mock_request_output(metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.first_token_iteration is None
        assert request_info.timings.last_token_iteration is None

    @pytest.mark.sanity
    def test_timing_uses_queued_ts_fallback(self):
        """queued_ts used as mono base fallback.

        ## WRITTEN BY AI ##
        """
        request_info = RequestInfo()
        metrics = SimpleNamespace(
            num_generation_tokens=5,
            arrival_time=1000.0,
            scheduled_ts=0.0,
            queued_ts=50.0,
            first_token_ts=52.0,
            last_token_ts=60.0,
        )
        output = _mock_request_output(metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.first_token_iteration == pytest.approx(
            1000.0 + (52.0 - 50.0)
        )
