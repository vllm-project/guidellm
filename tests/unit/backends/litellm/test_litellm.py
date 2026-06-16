"""
Unit tests for LiteLLMBackend implementation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from guidellm.backends.litellm.litellm import LiteLLMBackend, LiteLLMBackendArgs
from guidellm.schemas import GenerationRequest, GenerationResponse, RequestInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> LiteLLMBackendArgs:
    defaults = {"model": "anthropic/claude-haiku-4-5"}
    return LiteLLMBackendArgs(**{**defaults, **kwargs})


def _make_backend(**kwargs) -> LiteLLMBackend:
    return LiteLLMBackend(_make_args(**kwargs))


def _make_chunk(
    content: str | None = None,
    usage=None,
    chunk_id: str = "cid-1",
):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage, id=chunk_id)


def _make_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _stream_chunks(chunks):
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# BackendArgs
# ---------------------------------------------------------------------------


class TestLiteLLMBackendArgs:
    def test_default_kind(self):
        args = _make_args()
        assert args.kind == "litellm"

    def test_model_stored(self):
        args = _make_args(model="gemini/gemini-1.5-flash")
        assert args.model == "gemini/gemini-1.5-flash"

    def test_api_key_secret(self):
        args = _make_args(api_key="sk-test")
        assert args.api_key is not None
        assert args.api_key.get_secret_value() == "sk-test"

    def test_optional_fields_default_none(self):
        args = _make_args()
        assert args.api_key is None
        assert args.api_base is None
        assert args.max_tokens is None
        assert args.extras is None

    def test_registered_in_backend_args(self):
        from guidellm.backends.backend import BackendArgs

        assert "litellm" in BackendArgs.registry


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------


class TestLiteLLMBackendRegistration:
    def test_registered_in_backend(self):
        from guidellm.backends.backend import Backend

        assert "litellm" in Backend.registry

    def test_create_via_backend_factory(self):
        from guidellm.backends.backend import Backend

        args = _make_args()
        backend = Backend.create(args)
        assert isinstance(backend, LiteLLMBackend)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLiteLLMBackendLifecycle:
    @pytest.mark.asyncio
    async def test_process_startup_sets_in_process(self):
        backend = _make_backend()
        await backend.process_startup()
        assert backend._in_process is True
        await backend.process_shutdown()

    @pytest.mark.asyncio
    async def test_double_startup_raises(self):
        backend = _make_backend()
        await backend.process_startup()
        with pytest.raises(RuntimeError, match="already started"):
            await backend.process_startup()
        await backend.process_shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_without_startup_raises(self):
        backend = _make_backend()
        with pytest.raises(RuntimeError, match="not started"):
            await backend.process_shutdown()

    @pytest.mark.asyncio
    async def test_default_model(self):
        backend = _make_backend(model="groq/llama3-8b-8192")
        assert await backend.default_model() == "groq/llama3-8b-8192"

    @pytest.mark.asyncio
    async def test_available_models(self):
        backend = _make_backend(model="gpt-4o")
        models = await backend.available_models()
        assert models == ["gpt-4o"]


# ---------------------------------------------------------------------------
# resolve() — streaming dispatch
# ---------------------------------------------------------------------------


class TestLiteLLMBackendResolve:
    def _request(self) -> GenerationRequest:
        req = GenerationRequest()
        req.columns["text_column"] = ["What is 2+2?"]
        return req

    def _info(self) -> RequestInfo:
        return RequestInfo()

    @pytest.mark.asyncio
    async def test_dispatches_to_acompletion(self):
        backend = _make_backend()
        chunks = [
            _make_chunk("Four"),
            _make_chunk(".", usage=_make_usage()),
        ]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            results = []
            async for item in backend.resolve(
                self._request(), self._info(),
            ):
                results.append(item)

        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-haiku-4-5"
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_drop_params_always_set(self):
        backend = _make_backend()
        chunks = [_make_chunk("ok", usage=_make_usage())]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            async for _ in backend.resolve(
                self._request(), self._info(),
            ):
                pass

        assert m.call_args.kwargs.get("drop_params") is True

    @pytest.mark.asyncio
    async def test_api_key_forwarded(self):
        backend = _make_backend(api_key="sk-abc-123")
        chunks = [_make_chunk("ok", usage=_make_usage())]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            async for _ in backend.resolve(
                self._request(), self._info(),
            ):
                pass

        assert m.call_args.kwargs.get("api_key") == "sk-abc-123"

    @pytest.mark.asyncio
    async def test_api_base_forwarded(self):
        backend = _make_backend(
            api_base="https://proxy.example.com/v1",
        )
        chunks = [_make_chunk("ok", usage=_make_usage())]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            async for _ in backend.resolve(
                self._request(), self._info(),
            ):
                pass

        assert m.call_args.kwargs.get("api_base") == (
            "https://proxy.example.com/v1"
        )

    @pytest.mark.asyncio
    async def test_no_api_key_when_not_set(self):
        backend = _make_backend()
        chunks = [_make_chunk("ok", usage=_make_usage())]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            async for _ in backend.resolve(
                self._request(), self._info(),
            ):
                pass

        assert "api_key" not in m.call_args.kwargs

    @pytest.mark.asyncio
    async def test_final_response_has_text(self):
        backend = _make_backend()
        chunks = [
            _make_chunk("Hello"),
            _make_chunk(
                " world",
                usage=_make_usage(
                    prompt_tokens=5, completion_tokens=2,
                ),
            ),
        ]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            results = []
            async for response, _ in backend.resolve(
                self._request(), self._info(),
            ):
                if response is not None:
                    results.append(response)

        assert len(results) == 1
        assert results[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_token_usage_captured(self):
        backend = _make_backend()
        chunks = [
            _make_chunk(
                "Hi",
                usage=_make_usage(
                    prompt_tokens=7, completion_tokens=3,
                ),
            ),
        ]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            final = None
            async for response, _ in backend.resolve(
                self._request(), self._info(),
            ):
                if response is not None:
                    final = response

        assert final is not None
        assert final.input_metrics.text_tokens == 7
        assert final.output_metrics.text_tokens == 3

    @pytest.mark.asyncio
    async def test_ttft_none_yielded_on_first_token(self):
        backend = _make_backend()
        chunks = [
            _make_chunk("Hello"),
            _make_chunk(None, usage=_make_usage()),
        ]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            yielded = []
            async for response, _ in backend.resolve(
                self._request(), self._info(),
            ):
                yielded.append(response)

        assert yielded[0] is None
        assert isinstance(yielded[1], GenerationResponse)

    @pytest.mark.asyncio
    async def test_timing_fields_populated(self):
        backend = _make_backend()
        chunks = [
            _make_chunk("Hi"),
            _make_chunk(None, usage=_make_usage()),
        ]

        with patch(
            "guidellm.extras.litellm.acompletion",
            new_callable=AsyncMock,
        ) as m:
            m.return_value = _stream_chunks(chunks)
            info = self._info()
            async for _ in backend.resolve(self._request(), info):
                pass

        assert info.timings.request_start is not None
        assert info.timings.request_end is not None
        assert info.timings.first_token_iteration is not None
        assert info.timings.first_request_iteration is not None
