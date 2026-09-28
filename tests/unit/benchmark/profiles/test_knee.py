"""Tests for knee profile scheduling and its normal benchmark lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from guidellm.benchmark import entrypoints as entrypoints_module
from guidellm.benchmark.analysis import analyze_knee
from guidellm.benchmark.profiles import KneeProfile, ProfileFactory
from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmarksReport,
)
from guidellm.scheduler import (
    ConcurrentStrategy,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import (
    GenerativeRequestStats,
    RequestInfo,
    RequestTimings,
    UsageMetrics,
)
from guidellm.schemas.benchmark import BenchmarkScenario, KneeProfileArgs

INITIAL = [1, 5, 10, 20, 40, 80, 160]
ADAPTIVE = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]


def _benchmark(streams, throughput=None, saturated=None):
    """Compile a real benchmark with a controlled output-token throughput."""
    tokens = throughput if throughput is not None else min(streams, 20) * 100
    timings = RequestTimings(
        resolve_start=1000.0,
        resolve_end=1001.0,
        request_start=1000.0,
        request_end=1001.0,
    )
    accumulator = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="knee-test",
            run_index=0,
            strategy=ConcurrentStrategy(streams=streams),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
        )
    )
    accumulator.timings.measure_start = 1000.0
    accumulator.timings.measure_end = 1001.0
    accumulator.completed.requests_stats = [
        GenerativeRequestStats(
            request_id="request",
            info=RequestInfo(request_id="request", status="completed", timings=timings),
            input_metrics=UsageMetrics(text_tokens=8),
            output_metrics=UsageMetrics(text_tokens=tokens),
        )
    ]
    state = SchedulerState()
    if saturated is not None:
        state.scheduler_constraints["over_saturation"] = SchedulerUpdateAction(
            metadata={"is_over_saturated": saturated}
        )
    return GenerativeBenchmark.compile(accumulator=accumulator, scheduler_state=state)


def _profile(**kwargs):
    return KneeProfile(KneeProfileArgs(streams=INITIAL, max_step=3, **kwargs), 42, {})


def _run_profile(profile, make_benchmark=_benchmark):
    """Exercise the same result feedback protocol used by Benchmarker.run."""
    generator = profile.strategies_generator()
    strategy, _ = next(generator)
    benchmarks = []
    while True:
        benchmark = make_benchmark(strategy.streams)
        benchmarks.append(benchmark)
        try:
            strategy, _ = generator.send(benchmark)
        except StopIteration:
            return benchmarks


@pytest.mark.smoke
def test_knee_profile_is_registered():
    """Create knee scheduling through the standard factory.

    ## WRITTEN BY AI ##
    """
    profile = ProfileFactory.create(KneeProfileArgs(streams=INITIAL), 42, {})
    assert isinstance(profile, KneeProfile)
    assert profile.conclusion is None


@pytest.mark.regression
@pytest.mark.parametrize("adaptive", [False, True])
def test_knee_profile_refines_once_and_includes_last_result(adaptive):
    """Run a complete sweep and optional grid through one profile generator.

    ## WRITTEN BY AI ##
    """
    profile = _profile(adaptive=adaptive)
    budget = len(profile.strategy_types)
    benchmarks = _run_profile(profile)
    streams = [benchmark.config.strategy.streams for benchmark in benchmarks]
    assert streams == INITIAL + (ADAPTIVE if adaptive else [])
    assert len(streams) == len(set(streams)) == budget
    conclusion = profile.conclusion
    assert conclusion["kind"] == "knee_detection"
    assert conclusion["initial"] == analyze_knee(benchmarks[:7]).model_dump(mode="json")
    assert conclusion["final"] == analyze_knee(benchmarks).model_dump(mode="json")
    assert conclusion["final"]["throughput"]["knee"] == pytest.approx(20)
    assert len(conclusion["final"]["saturation"]["points"]) == len(benchmarks)
    assert profile.info["streams"] == INITIAL
    assert profile.info["kind"] == "knee"
    assert profile.next_strategy(None, None) is None


@pytest.mark.sanity
@pytest.mark.parametrize(
    "case", ["linear", "flat", "insufficient", "disagreement", "measured"]
)
def test_knee_profile_skips_unhelpful_refinement(case):
    """Avoid extra runs when evidence or remaining grid points do not support them.

    ## WRITTEN BY AI ##
    """
    streams = [1, 2, 3, 4] if case == "insufficient" else INITIAL
    if case == "measured":
        streams = [1, 2, 3, 4, 5]
    profile = KneeProfile(
        KneeProfileArgs(streams=streams, adaptive=True, points_each_side=2, max_step=1),
        42,
        {},
    )

    def result(concurrency):
        throughput = None
        if case == "linear":
            throughput = concurrency * 100
        elif case == "flat":
            throughput = 100
        elif case == "measured":
            throughput = min(concurrency, 3) * 100
        saturated = concurrency >= 5 if case == "disagreement" else None
        return _benchmark(concurrency, throughput, saturated)

    benchmarks = _run_profile(profile, result)
    assert len(benchmarks) == len(streams)
    assert profile.conclusion["adaptive_plan"]["status"] == "skipped"
    assert profile.conclusion["final"] == profile.conclusion["initial"]


@pytest.mark.regression
def test_knee_profile_can_refine_an_oversaturation_boundary_without_a_fit():
    """Use temporal evidence when the initial sweep has too few points to fit.

    ## WRITTEN BY AI ##
    """
    initial = [1, 5, 10, 20]
    profile = KneeProfile(
        KneeProfileArgs(streams=initial, adaptive=True, points_each_side=1, max_step=1),
        42,
        {},
    )
    benchmarks = _run_profile(
        profile, lambda streams: _benchmark(streams, streams * 100, streams >= 10)
    )
    assert [benchmark.config.strategy.streams for benchmark in benchmarks] == (
        initial + [7, 8, 9]
    )
    conclusion = profile.conclusion
    assert conclusion["initial"]["throughput"]["status"] == "no_knee"
    assert conclusion["adaptive_plan"]["selection_center"] == 7.5
    assert conclusion["final"] == analyze_knee(benchmarks).model_dump(mode="json")


@pytest.mark.regression
@pytest.mark.parametrize("group", ["end_queuing", "end_processing"])
@pytest.mark.parametrize("stop_index", [0, 6, 7])
@pytest.mark.parametrize("scope", ["all", "current"])
def test_knee_profile_respects_stop_scope_in_both_phases(group, stop_index, scope):
    """Global stops halt both phases; per-point limits permit the next point.

    ## WRITTEN BY AI ##
    """
    profile = _profile(adaptive=True)
    index = 0

    def result(streams):
        nonlocal index
        benchmark = _benchmark(streams)
        if index == stop_index:
            action = SchedulerUpdateAction(
                request_queuing="stop",
                request_processing="stop_local",
                stopping_scope=scope,
            )
            if group == "end_queuing":
                benchmark.scheduler_state.end_queuing_constraints["limit"] = action
            else:
                benchmark.scheduler_state.end_processing_constraints["limit"] = action
        index += 1
        return benchmark

    benchmarks = _run_profile(profile, result)
    assert len(benchmarks) == (stop_index + 1 if scope == "all" else 18)
    assert profile.conclusion["final"] == analyze_knee(benchmarks).model_dump(
        mode="json"
    )
    if scope == "all" and stop_index < len(INITIAL):
        assert profile.conclusion["adaptive_plan"]["status"] == "skipped"
        assert "stopping_scope" in profile.conclusion["adaptive_plan"]["reason"]


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("progress_enabled", [False, True])
@pytest.mark.parametrize(
    ("kind", "adaptive"), [("concurrent", False), ("knee", False), ("knee", True)]
)
async def test_profile_uses_one_benchmark_lifecycle_and_normal_report(
    tmp_path, kind, adaptive, progress_enabled
):
    """Keep both phases in one run with shared timing, constraints and progress.

    ## WRITTEN BY AI ##
    """
    profile_args = {
        "kind": kind,
        "streams": INITIAL,
        "rampup_duration": 0.5,
        "warmup": 1.0,
        "cooldown": 1.0,
    }
    if kind == "knee":
        profile_args.update(adaptive=adaptive, max_step=3)
    output_path = tmp_path / "benchmarks.json"
    args = BenchmarkScenario.create(
        scenario=None,
        spec={
            "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
            "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
            "profile": profile_args,
            "constraints": [{"kind": "max_duration", "seconds": 5}],
            "outputs": [{"kind": "json", "path": str(output_path)}],
        },
    )
    results = {streams: _benchmark(streams) for streams in INITIAL + ADAPTIVE}
    configs = []
    scheduled = []

    async def schedule(self, **kwargs):
        scheduled.append(kwargs)
        return
        yield  # pragma: no cover

    def compile_result(accumulator, scheduler_state):
        config = accumulator.config
        configs.append(config)
        return results[config.strategy.streams].model_copy(update={"config": config})

    backend = MagicMock(info={})
    loader = MagicMock(info={})
    logging_tracker = AsyncMock()
    with (
        patch.object(
            entrypoints_module,
            "resolve_backend",
            AsyncMock(return_value=(backend, "model")),
        ),
        patch.object(entrypoints_module, "resolve_tokenizer", AsyncMock()),
        patch.object(
            entrypoints_module, "create_data_loader", AsyncMock(return_value=loader)
        ),
        patch("guidellm.benchmark.benchmarker.Scheduler.run", schedule),
        patch.object(GenerativeBenchmark, "compile", side_effect=compile_result),
        patch.object(
            entrypoints_module,
            "GenerativeLoggingBenchmarkerProgress",
            return_value=logging_tracker,
        ),
    ):
        report, _ = await entrypoints_module.benchmark_generative_text(
            args, progress=progress_enabled
        )

    expected = INITIAL + (ADAPTIVE if adaptive else [])
    assert [config.strategy.streams for config in configs] == expected
    assert len({config.run_id for config in configs}) == 1
    assert [config.run_index for config in configs] == list(range(len(expected)))
    assert all(config.profile == args.spec.profile.model_dump() for config in configs)
    assert all(config.strategy.rampup_duration == 0.5 for config in configs)
    assert all(config.warmup == args.spec.profile.warmup for config in configs)
    assert all(config.cooldown == args.spec.profile.cooldown for config in configs)
    assert all("max_duration" in config.constraints for config in configs)
    assert all(
        call["backend"] is backend and call["requests"] is loader for call in scheduled
    )
    logging_tracker.on_initialize.assert_awaited_once()
    logging_tracker.on_finalize.assert_awaited_once()
    assert logging_tracker.on_benchmark_complete.await_count == len(expected)
    restored = GenerativeBenchmarksReport.model_validate_json(output_path.read_text())
    assert len(restored.benchmarks) == len(expected)
    assert restored.config.spec.profile == args.spec.profile
    assert restored.conclusions == report.conclusions
    if kind == "knee":
        assert len(restored.conclusions) == 1
        assert restored.conclusions[0]["final"] == analyze_knee(
            report.benchmarks
        ).model_dump(mode="json")
    else:
        assert restored.conclusions == []
