from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guidellm.benchmark.entrypoints import resolve_backend, resolve_output_formats
from guidellm.benchmark.outputs import GenerativeBenchmarkerOutput
from guidellm.schemas.backends import (
    OpenAIHTTPBackendArgs,
    VLLMPythonAsyncBackendArgs,
)
from guidellm.schemas.benchmark import JSONBenchmarkOutputArgs


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_resolve_output_formats_preserves_duplicate_kinds(tmp_path: Path):
    """
    resolve_output_formats returns one resolved output per arg, in order,
    without collapsing repeated kinds into a single entry.

    ## WRITTEN BY AI ##
    """
    outputs = [
        JSONBenchmarkOutputArgs(path=tmp_path / "first.json"),
        JSONBenchmarkOutputArgs(path=tmp_path / "second.json"),
    ]

    resolved = await resolve_output_formats(outputs)

    assert isinstance(resolved, list)
    assert len(resolved) == 2
    assert all(isinstance(o, GenerativeBenchmarkerOutput) for o in resolved)
    assert resolved[0] is not resolved[1]
    assert [o.output_path for o in resolved] == [
        tmp_path / "first.json",
        tmp_path / "second.json",
    ]


@pytest.mark.asyncio
@pytest.mark.regression
async def test_resolve_backend_shuts_down_after_validation_error():
    """
    resolve_backend shuts down a started backend when validation fails.

    ## WRITTEN BY AI ##
    """
    backend = MagicMock()
    backend.process_startup = AsyncMock()
    backend.validate = AsyncMock(side_effect=RuntimeError("validation failed"))
    backend.default_model = AsyncMock()
    backend.process_shutdown = AsyncMock()
    args = OpenAIHTTPBackendArgs(target="http://localhost:8000")

    with (
        patch("guidellm.benchmark.entrypoints.Backend.create", return_value=backend),
        pytest.raises(RuntimeError, match="validation failed"),
    ):
        await resolve_backend(args)

    backend.process_startup.assert_awaited_once_with()
    backend.validate.assert_awaited_once_with()
    backend.default_model.assert_not_awaited()
    backend.process_shutdown.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.regression
async def test_resolve_backend_skips_startup_for_in_process_backend():
    """
    resolve_backend does not start up or validate a backend that reports
    requires_startup_for_resolution is False; it resolves the model directly so
    the backend is only initialized in the worker process.

    ## WRITTEN BY AI ##
    """
    backend = MagicMock()
    backend.requires_startup_for_resolution = False
    backend.process_startup = AsyncMock()
    backend.validate = AsyncMock()
    backend.default_model = AsyncMock(return_value="Qwen/Qwen3-0.6B")
    backend.process_shutdown = AsyncMock()
    args = VLLMPythonAsyncBackendArgs(model="Qwen/Qwen3-0.6B")

    with patch("guidellm.benchmark.entrypoints.Backend.create", return_value=backend):
        resolved_backend, model = await resolve_backend(args)

    assert resolved_backend is backend
    assert model == "Qwen/Qwen3-0.6B"
    backend.process_startup.assert_not_awaited()
    backend.validate.assert_not_awaited()
    backend.process_shutdown.assert_not_awaited()
    backend.default_model.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.regression
async def test_resolve_backend_starts_up_for_remote_backend():
    """
    resolve_backend starts up, validates, resolves the model, and shuts down a
    backend that reports requires_startup_for_resolution is True.

    ## WRITTEN BY AI ##
    """
    backend = MagicMock()
    backend.requires_startup_for_resolution = True
    backend.process_startup = AsyncMock()
    backend.validate = AsyncMock()
    backend.default_model = AsyncMock(return_value="test-model")
    backend.process_shutdown = AsyncMock()
    args = OpenAIHTTPBackendArgs(target="http://localhost:8000")

    with patch("guidellm.benchmark.entrypoints.Backend.create", return_value=backend):
        resolved_backend, model = await resolve_backend(args)

    assert resolved_backend is backend
    assert model == "test-model"
    backend.process_startup.assert_awaited_once_with()
    backend.validate.assert_awaited_once_with()
    backend.default_model.assert_awaited_once_with()
    backend.process_shutdown.assert_awaited_once_with()
