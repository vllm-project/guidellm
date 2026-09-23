from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guidellm.benchmark import entrypoints as entrypoints_module
from guidellm.benchmark.analysis import (
    KneeAnalysis,
    KneeResult,
    SaturationAssessment,
    SaturationBoundary,
)
from guidellm.benchmark.benchmarker import Benchmarker
from guidellm.benchmark.entrypoints import resolve_backend, resolve_output_formats
from guidellm.benchmark.outputs import GenerativeBenchmarkerOutput
from guidellm.benchmark.profiles import ProfileFactory
from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.scheduler import ConcurrentStrategy, SchedulerState, SchedulerUpdateAction
from guidellm.schemas.backends import (
    OpenAIHTTPBackendArgs,
    VLLMPythonAsyncBackendArgs,
)
from guidellm.schemas.benchmark import (
    BenchmarkArgs,
    BenchmarkScenario,
    GoodputSLO,
    JSONBenchmarkOutputArgs,
    KneeDetectionArgs,
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
    backend.backend_defines_model = False
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
    resolve_backend does not start up or validate a backend that defines its own
    model (backend_defines_model is True); it resolves the model directly so the
    backend is only initialized in the worker process.

    ## WRITTEN BY AI ##
    """
    backend = MagicMock()
    backend.backend_defines_model = True
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
    backend that does not define its own model (backend_defines_model is False).

    ## WRITTEN BY AI ##
    """
    backend = MagicMock()
    backend.backend_defines_model = False
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


@pytest.mark.regression
def test_goodput_profile_requires_objectives():
    """
    Reject a goodput search configured without latency objectives.

    The search has nothing to search against, and without this check the run
    fails only after its first probe has already spent a full probe duration.

    ## WRITTEN BY AI ##
    """
    base = {
        "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
        "data": [{"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}],
    }

    with pytest.raises(ValueError, match="goodput profile"):
        BenchmarkArgs.model_validate({**base, "profile": {"kind": "goodput"}})

    configured = BenchmarkArgs.model_validate(
        {
            **base,
            "profile": {"kind": "goodput"},
            "metrics": {"kind": "generative", "slo": {"ttft_ms": 2000}},
        }
    )
    assert configured.metrics.slo == GoodputSLO(ttft_ms=2000)


@pytest.mark.regression
def test_other_profiles_do_not_require_objectives():
    """
    Leave profiles other than goodput unaffected by the objectives check.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkArgs.model_validate(
        {
            "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
            "data": [
                {"kind": "synthetic_text", "prompt_tokens": 8, "output_tokens": 8}
            ],
            "profile": {"kind": "sweep"},
        }
    )

    assert args.profile.kind == "sweep"
    assert args.metrics.slo is None


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


def _knee_analysis(center: float) -> KneeAnalysis:
    return KneeAnalysis(
        throughput=KneeResult(status="ok", reason="test", knee=center),
        saturation=SaturationBoundary(
            status="not_detected",
            reason="test",
            points=(),
        ),
        assessment=SaturationAssessment(
            status="throughput_only",
            reason="test",
            selection_center=center,
        ),
    )


def _concurrent_benchmark(streams: int) -> GenerativeBenchmark:
    return cast(
        "GenerativeBenchmark",
        SimpleNamespace(
            config=SimpleNamespace(strategy=ConcurrentStrategy(streams=streams)),
            scheduler_state=SchedulerState(),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.regression
@pytest.mark.parametrize("progress_enabled", [True, False])
@pytest.mark.parametrize(
    ("knee_config", "stop_group", "stop_scope", "expected_runs"),
    [
        (None, None, "current", 1),
        ({"enabled": False}, None, "current", 1),
        ({"enabled": True}, None, "current", 1),
        ({"enabled": True, "adaptive": True}, None, "current", 2),
        ({"enabled": True, "adaptive": True}, "end_queuing_constraints", "all", 1),
        ({"enabled": True, "adaptive": True}, "end_processing_constraints", "all", 1),
        ({"enabled": True, "adaptive": True}, "end_queuing_constraints", "current", 2),
        ({"enabled": True, "adaptive": True}, "scheduler_constraints", "all", 2),
    ],
)
async def test_entrypoint_runs_adaptive_concurrencies_and_reports_analysis(
    knee_config, stop_group, stop_scope, expected_runs, progress_enabled
):
    """Preserve execution, stopping constraints, and progress during refinement.

    ## WRITTEN BY AI ##
    """
    initial_benchmarks = [
        _concurrent_benchmark(streams) for streams in [10, 20, 30, 40, 50]
    ]
    adaptive_benchmarks = [_concurrent_benchmark(streams) for streams in [25, 35]]
    run_calls: list[dict] = []
    logging_tracker = AsyncMock(
        spec=entrypoints_module.GenerativeLoggingBenchmarkerProgress
    )
    console_tracker = AsyncMock(
        spec=entrypoints_module.GenerativeConsoleBenchmarkerProgress
    )
    if stop_group is not None:
        action = SchedulerUpdateAction(stopping_scope=stop_scope)
        initial_benchmarks[-1].scheduler_state = SchedulerState.model_validate(
            {stop_group: {"test_constraint": action}}
        )

    async def _fake_run(self, **kwargs):
        _ = self
        run_calls.append(kwargs)
        await kwargs["progress"].on_initialize(kwargs["profile"])
        benchmarks = initial_benchmarks if len(run_calls) == 1 else adaptive_benchmarks
        for benchmark in benchmarks:
            yield benchmark
        await kwargs["progress"].on_finalize()

    initial_profile = MagicMock()
    initial_profile.conclusion = {"kind": "existing_profile_conclusion"}
    adaptive_profile = MagicMock()
    adaptive_profile.conclusion = None
    resolve_profile_mock = AsyncMock(side_effect=[initial_profile, adaptive_profile])
    initial_analysis = _knee_analysis(30)
    final_analysis = _knee_analysis(30)
    analyze_mock = MagicMock(side_effect=[initial_analysis, final_analysis])
    output = MagicMock()
    output.finalize = AsyncMock(return_value="saved-report")
    backend = MagicMock()
    loader = MagicMock()
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
                "profile": {
                    "kind": "concurrent",
                    "streams": [10, 20, 30, 40, 50],
                },
                "outputs": [{"kind": "json"}],
            }
        ),
    )
    if knee_config is not None:
        args.knee_detection = KneeDetectionArgs(
            **knee_config, points_each_side=1, max_step=5
        )

    with (
        patch.object(entrypoints_module.Benchmarker, "run", _fake_run),
        patch.object(
            entrypoints_module,
            "resolve_backend",
            AsyncMock(return_value=(backend, "model")),
        ),
        patch.object(
            entrypoints_module, "resolve_tokenizer", AsyncMock(return_value=None)
        ),
        patch.object(
            entrypoints_module,
            "create_data_loader",
            AsyncMock(return_value=loader),
        ),
        patch.object(entrypoints_module, "resolve_profile", resolve_profile_mock),
        patch.object(
            entrypoints_module,
            "resolve_output_formats",
            AsyncMock(return_value=[output]),
        ),
        patch.object(entrypoints_module, "analyze_knee", analyze_mock),
        patch.object(
            entrypoints_module,
            "GenerativeLoggingBenchmarkerProgress",
            return_value=logging_tracker,
        ) as logging_factory,
        patch.object(
            entrypoints_module,
            "GenerativeConsoleBenchmarkerProgress",
            return_value=console_tracker,
        ) as console_factory,
    ):
        report, outputs = await entrypoints_module.benchmark_generative_text(
            args=args, progress=progress_enabled
        )

    assert len(run_calls) == expected_runs
    assert resolve_profile_mock.await_count == expected_runs
    logging_factory.assert_called_once_with()
    assert logging_tracker.on_initialize.await_count == expected_runs
    assert logging_tracker.on_finalize.await_count == expected_runs
    assert [call.args[0] for call in logging_tracker.on_initialize.await_args_list] == [
        initial_profile,
        adaptive_profile,
    ][:expected_runs]
    if progress_enabled:
        console_factory.assert_called_once_with()
        assert console_tracker.on_initialize.await_count == expected_runs
        assert console_tracker.on_finalize.await_count == expected_runs
        assert [
            call.args[0] for call in console_tracker.on_initialize.await_args_list
        ] == [initial_profile, adaptive_profile][:expected_runs]
    else:
        console_factory.assert_not_called()
        console_tracker.on_initialize.assert_not_awaited()
        console_tracker.on_finalize.assert_not_awaited()
    assert report.conclusions[0] == initial_profile.conclusion
    assert report.benchmarks[:5] == initial_benchmarks
    output.finalize.assert_awaited_once_with(report)
    assert outputs == [("json", "saved-report")]
    for call in run_calls:
        assert call["backend"] is backend
        assert call["requests"] is loader
        assert call["warmup"] == args.spec.profile.warmup
        assert call["cooldown"] == args.spec.profile.cooldown

    if not args.knee_detection.enabled:
        analyze_mock.assert_not_called()
        assert report.benchmarks == initial_benchmarks
        assert report.conclusions == [initial_profile.conclusion]
    else:
        assert analyze_mock.call_args_list[0].args[0] == initial_benchmarks
        assert analyze_mock.call_args_list[1].args[0] == report.benchmarks
        knee_conclusion = report.conclusions[-1]
        assert knee_conclusion["kind"] == "knee_detection"
        if expected_runs == 2:
            adaptive_args = resolve_profile_mock.await_args_list[1].kwargs["profile"]
            assert adaptive_args.streams == [25, 35]
            assert report.benchmarks == initial_benchmarks + adaptive_benchmarks
            assert knee_conclusion["adaptive_plan"]["concurrencies"] == [25, 35]
        else:
            assert report.benchmarks == initial_benchmarks
            assert knee_conclusion["adaptive_plan"]["status"] == "skipped"
            if stop_group is not None:
                assert "stopping_scope" in knee_conclusion["adaptive_plan"]["reason"]


@pytest.mark.asyncio
@pytest.mark.sanity
async def test_entrypoint_rejects_knee_detection_for_nonconcurrent_profile():
    """Reject knee detection before backend setup for another profile kind.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkScenario(
        knee_detection={"enabled": True},
        spec=BenchmarkArgs.model_validate(
            {
                "backend": {
                    "kind": "openai_http",
                    "target": "http://localhost:8000",
                },
                "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
                "profile": {"kind": "synchronous"},
                "outputs": [],
            }
        ),
    )

    with pytest.raises(ValueError, match="requires a concurrent profile"):
        await entrypoints_module.benchmark_generative_text(args=args)


@pytest.mark.asyncio
@pytest.mark.regression
async def test_knee_rejects_repeated_streams_before_starting_backend():
    """Reject ambiguous detector measurements before spending time on benchmarks.

    ## WRITTEN BY AI ##
    """
    args = BenchmarkScenario.create(
        scenario=None,
        knee_detection={"enabled": True},
        spec={
            "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
            "profile": {"kind": "concurrent", "streams": [1, 5, 5, 10, 20, 40]},
            "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
        },
    )
    with patch.object(entrypoints_module, "resolve_backend", AsyncMock()) as backend:
        with pytest.raises(ValueError, match="distinct concurrency points"):
            await entrypoints_module.benchmark_generative_text(args=args)
        backend.assert_not_awaited()
