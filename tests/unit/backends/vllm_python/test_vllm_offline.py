"""
Unit tests for VLLM Offline (batch) backend.

Tests engine laziness, batch processing, deferred flush, generate-lock
serialization, shutdown drain, shutting-down guard, and metrics wiring.
"""

from __future__ import annotations

import asyncio
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
        await started_backend.process_shutdown()
        assert started_backend._in_process is False
        assert started_backend._llm is None
        assert started_backend._shutting_down is True

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_calls_llm_shutdown(self, started_backend):
        shutdown_mock = Mock()
        started_backend._llm.shutdown = shutdown_mock
        await started_backend.process_shutdown()
        shutdown_mock.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_tolerates_missing_shutdown_method(self, started_backend):
        del started_backend._llm.shutdown
        await started_backend.process_shutdown()
        assert started_backend._llm is None

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_not_started_raises(self, offline_backend):
        with pytest.raises(RuntimeError, match="not started"):
            await offline_backend.process_shutdown()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_shutdown_drains_pending_batch(self, started_backend):
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
    async def test_shutdown_cancels_processing_task(self, started_backend):
        task = asyncio.ensure_future(asyncio.sleep(100))
        started_backend._processing_task = task
        await started_backend.process_shutdown()
        assert task.cancelling() or task.cancelled()
        assert started_backend._processing_task is None

    @pytest.mark.asyncio
    @pytest.mark.smoke
    async def test_validate_passes_after_startup(self, started_backend):
        await started_backend.validate()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    async def test_validate_fails_before_startup(self, offline_backend):
        with pytest.raises(RuntimeError, match="not started"):
            await offline_backend.validate()


# ------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------


class TestBatchProcessing:
    @pytest.mark.sanity
    def test_take_pending_batch_clears_queue(self, started_backend):
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
        await started_backend._run_generate([])
        started_backend._llm.generate.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_run_generate_distributes_results(self, started_backend):
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
        req = _BatchedRequest(
            resolved_prompt="partial",
            multi_modal_data=None,
            max_tokens=10,
        )
        started_backend._pending_batch.append(req)
        started_backend._llm.generate.return_value = [_mock_request_output()]
        await started_backend._schedule_deferred_flush()
        # Let the flush task run
        await asyncio.sleep(0.01)
        assert req.ready.is_set()
        started_backend._llm.generate.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.sanity
    @async_timeout(5.0)
    async def test_deferred_flush_idempotent(self, started_backend):
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
        await asyncio.sleep(0.01)


# ------------------------------------------------------------------
# Generate lock serialization
# ------------------------------------------------------------------


class TestGenerateLock:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_generate_lock_serializes_calls(self, started_backend):
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
        # Should be serialized: enter/exit/enter/exit, not interleaved
        assert call_order == ["enter", "exit", "enter", "exit"]


# ------------------------------------------------------------------
# Shutting-down guard
# ------------------------------------------------------------------


class TestShuttingDownGuard:
    @pytest.mark.asyncio
    @pytest.mark.smoke
    @async_timeout(5.0)
    async def test_resolve_rejects_during_shutdown(self, started_backend):
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
        await started_backend.process_shutdown()
        assert started_backend._shutting_down is True


# ------------------------------------------------------------------
# Metrics wiring
# ------------------------------------------------------------------


class TestWireVllmMetrics:
    @pytest.mark.smoke
    def test_token_count_from_num_generation_tokens(self):
        request_info = RequestInfo()
        metrics = SimpleNamespace(num_generation_tokens=42)
        output = _mock_request_output(metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 42
        assert request_info.timings.request_iterations == 1

    @pytest.mark.sanity
    def test_token_count_fallback_to_token_ids(self):
        request_info = RequestInfo()
        metrics = SimpleNamespace(num_generation_tokens=0)
        output = _mock_request_output(token_ids=[1, 2, 3, 4, 5], metrics=metrics)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 5
        assert request_info.timings.request_iterations == 1

    @pytest.mark.sanity
    def test_no_metrics_no_crash(self):
        request_info = RequestInfo()
        output = _mock_request_output(metrics=None)
        VLLMOfflineBackend._wire_vllm_metrics(request_info, output)
        assert request_info.timings.token_iterations == 3  # len(token_ids)

    @pytest.mark.smoke
    def test_timing_from_request_state_stats(self):
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
