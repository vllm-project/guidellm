from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guidellm.benchmark import entrypoints as entrypoints_module
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.entrypoints import resolve_backend, resolve_output_formats
from guidellm.benchmark.outputs import GenerativeBenchmarkerOutput
from guidellm.benchmark.profiles import ProfileFactory
from guidellm.schemas.backends import OpenAIHTTPBackendArgs
from guidellm.schemas.benchmark import (
    BenchmarkArgs,
    BenchmarkScenario,
    GoodputSLO,
    JSONBenchmarkOutputArgs,
    SynchronousProfileArgs,
    TransientPhaseConfig,
)


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


class _RecordingAccumulator:
    """Accumulator stub that records the config the benchmarker builds."""

    configs: list = []

    def __init__(self, config):
        type(self).configs.append(config)

    def update_estimate(self, *args, **kwargs):
        """Ignore request updates; only the config matters here."""


class _StubBenchmark:
    """Benchmark stub standing in for a compiled result."""

    @classmethod
    def compile(cls, accumulator, scheduler_state):
        """Return a sentinel instead of compiling real metrics."""
        _ = (accumulator, scheduler_state)
        return "compiled"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_benchmarker_forwards_objectives_into_benchmark_config():
    """
    Carry latency objectives from Benchmarker.run onto each BenchmarkConfig.

    This is the only link between the configured objectives and the accumulator
    that compiles goodput. Severing it leaves every goodput metric None while
    the run still reports success, so nothing else in the suite would fail.

    ## WRITTEN BY AI ##
    """

    class _StubScheduler:
        async def run(self, **kwargs):
            _ = kwargs
            yield (None, None, None, MagicMock())

    def _info_stub():
        stub = MagicMock()
        stub.info = {}
        return stub

    _RecordingAccumulator.configs = []
    slo = GoodputSLO(ttft_ms=1234)
    profile = ProfileFactory.create(SynchronousProfileArgs(), 42, {})

    with patch("guidellm.benchmark.benchmarker.Scheduler", _StubScheduler):
        results = [
            benchmark
            async for benchmark in Benchmarker().run(
                accumulator_class=_RecordingAccumulator,
                benchmark_class=_StubBenchmark,
                requests=_info_stub(),
                backend=_info_stub(),
                profile=profile,
                environment=_info_stub(),
                warmup=TransientPhaseConfig(),
                cooldown=TransientPhaseConfig(),
                slo=slo,
            )
        ]

    assert results == ["compiled"]
    assert _RecordingAccumulator.configs
    assert all(config.slo == slo for config in _RecordingAccumulator.configs)


@pytest.mark.regression
@pytest.mark.asyncio
async def test_entrypoint_passes_configured_objectives_to_benchmarker():
    """
    Forward the objectives from the metrics arguments into the benchmarker.

    ## WRITTEN BY AI ##
    """
    captured: dict = {}

    async def _fake_run(self, **kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover - makes this an async generator

    slo = GoodputSLO(ttft_ms=4321)
    args = BenchmarkScenario(
        spec=BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "data": [
                    {"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}
                ],
                "profile": {"kind": "synchronous"},
                "metrics": {"kind": "generative", "slo": slo.model_dump()},
                "outputs": [],
            }
        )
    )

    with (
        patch.object(entrypoints_module.Benchmarker, "run", _fake_run),
        patch.object(
            entrypoints_module,
            "resolve_backend",
            AsyncMock(return_value=(MagicMock(), "model")),
        ),
        patch.object(
            entrypoints_module, "resolve_tokenizer", AsyncMock(return_value=None)
        ),
        patch.object(
            entrypoints_module,
            "create_data_loader",
            AsyncMock(return_value=MagicMock()),
        ),
        patch.object(
            entrypoints_module, "resolve_profile", AsyncMock(return_value=MagicMock())
        ),
        patch.object(
            entrypoints_module, "resolve_output_formats", AsyncMock(return_value=[])
        ),
    ):
        await entrypoints_module.benchmark_generative_text(args=args)

    assert captured.get("slo") == slo
