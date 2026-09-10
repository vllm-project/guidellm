"""Unit tests for the goodput search profile."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.profiles import GoodputProfile, ProfileFactory
from guidellm.benchmark.profiles.goodput import wilson_interval
from guidellm.benchmark.schemas import (
    BenchmarkConfig,
    GenerativeBenchmark,
    GenerativeBenchmarkAccumulator,
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
from guidellm.schemas.benchmark import GoodputProfileArgs, GoodputSLO

TARGET = 0.95
# Non-zero epoch base; a measurement window starting at 0.0 reads as unset.
BASE_TIME = 1000.0


def _make_profile(**kwargs) -> GoodputProfile:
    """Build a goodput profile with the given argument overrides."""
    return GoodputProfile(
        GoodputProfileArgs(**kwargs), random_seed=42, constraints=None
    )


def _make_request(request_id: str, latency_seconds: float) -> GenerativeRequestStats:
    """
    Build a completed request with the given end-to-end latency.

    ## WRITTEN BY AI ##
    """
    timings = RequestTimings(
        resolve_start=BASE_TIME,
        resolve_end=BASE_TIME + latency_seconds,
        request_start=BASE_TIME,
        request_end=BASE_TIME + latency_seconds,
    )
    return GenerativeRequestStats(
        request_id=request_id,
        info=RequestInfo(request_id=request_id, status="completed", timings=timings),
        input_metrics=UsageMetrics(text_tokens=8),
        output_metrics=UsageMetrics(text_tokens=8),
    )


def _make_benchmark_with_slo(
    slo: GoodputSLO, determined: int, conforming: int = 0
) -> GenerativeBenchmark:
    """
    Compile a benchmark of fixed-latency requests against the given objective.

    ## WRITTEN BY AI ##
    """
    requests = [_make_request(f"ok-{i}", 0.5) for i in range(conforming)]
    requests += [
        _make_request(f"slow-{i}", 2.0) for i in range(determined - conforming)
    ]

    accumulator = GenerativeBenchmarkAccumulator(
        config=BenchmarkConfig(
            run_id="goodput-search",
            run_index=0,
            strategy=ConcurrentStrategy(streams=1),
            constraints={},
            profile={},
            requests={},
            backend={},
            environment={},
            slo=slo,
        )
    )
    accumulator.timings.measure_start = BASE_TIME
    accumulator.timings.measure_end = BASE_TIME + 10.0
    accumulator.completed.requests_stats = requests

    return GenerativeBenchmark.compile(
        accumulator=accumulator, scheduler_state=SchedulerState()
    )


def _make_benchmark(attainment: float, determined: int = 2000) -> GenerativeBenchmark:
    """
    Build a real compiled benchmark whose attainment is the requested fraction.

    Goes through GenerativeMetrics.compile rather than stubbing the accessors,
    so the adapter between compiled metrics and the search is exercised.

    ## WRITTEN BY AI ##
    """
    # A 1s objective; conforming requests finish in 0.5s, the rest in 2s.
    return _make_benchmark_with_slo(
        GoodputSLO(e2el_ms=1000),
        determined=determined,
        conforming=round(attainment * determined),
    )


def _attainment_at(streams: int, knee: int) -> float:
    """Hold attainment high up to the knee, then fall off linearly past it."""
    if streams <= knee:
        return 0.99

    return max(0.0, 0.99 - 0.35 * (streams - knee) / max(knee, 1))


def _true_boundary(knee: int, ceiling: int = 4096) -> int | None:
    """Highest concurrency whose modelled attainment still meets the target."""
    passing = [s for s in range(1, ceiling + 1) if _attainment_at(s, knee) >= TARGET]

    return passing[-1] if passing else None


def _drive_search(knee: int, determined: int = 2000, **kwargs):
    """Run a full search against the modelled server and return the trace."""
    profile = _make_profile(**kwargs)
    prev_strategy = None
    prev_benchmark = None
    probed: list[int] = []

    while (
        strategy := profile.next_strategy(prev_strategy, prev_benchmark)
    ) is not None:
        probed.append(strategy.streams)
        prev_benchmark = _make_benchmark(
            _attainment_at(strategy.streams, knee), determined
        )
        prev_strategy = strategy
        profile.completed_strategies.append(strategy)

    return probed, profile.search


class TestGoodputProfileArgs:
    @pytest.mark.smoke
    def test_registered_in_profile_factory(self):
        """
        Expose the goodput profile through the profile factory.

        ## WRITTEN BY AI ##
        """
        assert "goodput" in ProfileFactory.registered_names()
        profile = ProfileFactory.create(GoodputProfileArgs(), 42, {})
        assert isinstance(profile, GoodputProfile)

    @pytest.mark.sanity
    def test_rejects_initial_above_max_streams(self):
        """
        Reject a search range whose start already exceeds its ceiling.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match="must not exceed"):
            GoodputProfileArgs(initial_streams=64, max_streams=32)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target_attainment", 0.0),
            ("target_attainment", 1.5),
            ("tolerance", 0.0),
            ("confidence", 1.0),
        ],
    )
    def test_rejects_out_of_range_fractions(self, field, value):
        """
        Reject fraction arguments outside their valid interval.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError, match=field):
            GoodputProfileArgs(**{field: value})


class TestWilsonInterval:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("successes", "trials", "confidence", "expected"),
        [
            # Closed-form Wilson bounds computed with an exact normal
            # quantile (statistics.NormalDist.inv_cdf), independent of the
            # approximation this module uses.
            (95, 100, 0.95, (0.888250, 0.978456)),
            (50, 100, 0.95, (0.403832, 0.596168)),
            (95, 100, 0.99, (0.860850, 0.983152)),
            (990, 1000, 0.95, (0.981691, 0.994559)),
        ],
    )
    def test_matches_the_closed_form_interval(
        self, successes, trials, confidence, expected
    ):
        """
        Reproduce known Wilson score bounds rather than only bracketing.

        Pins the z lookup and both denominator terms: a qualitative check
        passes even when the formula silently computes a different confidence
        level or an interval several times too wide.

        ## WRITTEN BY AI ##
        """
        lower, upper = wilson_interval(successes, trials, confidence)

        # Tolerance covers the documented sub-1e-3 error of the quantile
        # approximation while staying far tighter than any formula mutation.
        assert lower == pytest.approx(expected[0], abs=1e-4)
        assert upper == pytest.approx(expected[1], abs=1e-4)

    @pytest.mark.sanity
    def test_widens_with_confidence(self):
        """
        Return a wider interval at a higher confidence level.

        ## WRITTEN BY AI ##
        """
        narrow = wilson_interval(950, 1000, 0.90)
        wide = wilson_interval(950, 1000, 0.99)

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    @pytest.mark.regression
    @pytest.mark.parametrize(("successes", "trials"), [(10, 5), (-3, 100), (150, 100)])
    def test_clamps_counts_outside_the_valid_range(self, successes, trials):
        """
        Return an ordered interval inside [0, 1] for out-of-range counts.

        This is public API, and slo_attainment is an unbounded float field on a
        deserialized report, so a caller can reach it with successes > trials.

        ## WRITTEN BY AI ##
        """
        lower, upper = wilson_interval(successes, trials)

        assert 0.0 <= lower <= upper <= 1.0

    @pytest.mark.sanity
    def test_narrows_as_sample_grows(self):
        """
        Return a tighter interval for the same proportion at a larger sample.

        ## WRITTEN BY AI ##
        """
        small = wilson_interval(successes=95, trials=100)
        large = wilson_interval(successes=9500, trials=10000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    @pytest.mark.sanity
    def test_stays_within_unit_interval_at_the_boundary(self):
        """
        Clamp the interval to [0, 1] when every request conforms.

        The normal approximation extends above 1.0 here, which is the reason
        this uses a Wilson score interval instead.

        ## WRITTEN BY AI ##
        """
        lower, upper = wilson_interval(successes=40, trials=40)
        assert upper == 1.0
        assert lower == pytest.approx(0.912378, abs=1e-4)

    @pytest.mark.sanity
    def test_no_trials_is_maximally_uncertain(self):
        """
        Return the full unit interval when nothing was measured.

        ## WRITTEN BY AI ##
        """
        assert wilson_interval(successes=0, trials=0) == (0.0, 1.0)


class TestGoodputSearch:
    @pytest.mark.smoke
    def test_first_probe_uses_initial_streams(self):
        """
        Begin the search at the configured starting concurrency.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile(initial_streams=7)
        strategy = profile.next_strategy(None, None)
        assert strategy is not None
        assert strategy.streams == 7

    @pytest.mark.sanity
    @pytest.mark.parametrize("knee", [1, 3, 4, 7, 16, 17, 40, 100])
    def test_converges_within_tolerance_of_the_knee(self, knee):
        """
        Converge on the highest concurrency meeting the target attainment.

        ## WRITTEN BY AI ##
        """
        _, search = _drive_search(knee, max_probes=20)
        truth = _true_boundary(knee)

        assert search.stop_reason == "converged"
        assert search.best_passing_streams is not None
        assert abs(search.best_passing_streams - truth) <= max(1, 0.1 * truth)

    @pytest.mark.sanity
    def test_brackets_by_doubling_before_bisecting(self):
        """
        Double concurrency until a level fails, then bisect the bracket.

        ## WRITTEN BY AI ##
        """
        probed, _ = _drive_search(knee=16, max_probes=20, initial_streams=4)

        assert probed[:4] == [4, 8, 16, 32]
        assert max(probed) == 32
        assert all(16 < value < 32 for value in probed[4:])

    @pytest.mark.sanity
    def test_reports_objectives_unmet_at_minimum(self):
        """
        Stop and report when even a single stream misses the objectives.

        ## WRITTEN BY AI ##
        """
        _, search = _drive_search(knee=0)

        assert search.best_passing_streams is None
        assert search.stop_reason == "objectives_unmet_at_minimum"

    @pytest.mark.sanity
    def test_stops_at_the_stream_ceiling(self):
        """
        Stop at max_streams when no tested level fails its objectives.

        ## WRITTEN BY AI ##
        """
        probed, search = _drive_search(knee=100_000, max_streams=64)

        assert max(probed) == 64
        assert search.best_passing_streams == 64
        assert search.stop_reason == "max_streams_reached"

    @pytest.mark.sanity
    def test_honours_the_probe_budget(self):
        """
        Stop after max_probes runs and report the best level found so far.

        ## WRITTEN BY AI ##
        """
        probed, search = _drive_search(knee=500, max_probes=5)

        assert len(probed) == 5
        assert search.stop_reason == "max_probes_exhausted"

    @pytest.mark.regression
    def test_tolerance_bounds_the_bisection_cost(self):
        """
        Spend fewer probes at a loose tolerance than at a tight one.

        Bisecting to an exact integer costs one probe per halving regardless of
        scale, which exhausts the budget on high-capacity servers. The relative
        tolerance is what keeps the probe count independent of the knee's
        magnitude.

        ## WRITTEN BY AI ##
        """
        loose, _ = _drive_search(knee=400, max_probes=30, tolerance=0.2)
        tight, _ = _drive_search(knee=400, max_probes=30, tolerance=0.01)

        assert len(loose) < len(tight)

    @pytest.mark.regression
    def test_flags_probes_the_run_was_too_short_to_resolve(self):
        """
        Mark a probe unresolved when its confidence interval straddles the
        target attainment, and resolved when the sample is large enough.

        ## WRITTEN BY AI ##
        """
        _, short_run = _drive_search(knee=16, determined=100, max_probes=20)
        _, long_run = _drive_search(knee=16, determined=20000, max_probes=20)

        short_first = short_run.probes[0]
        long_first = long_run.probes[0]

        assert short_first.attainment == pytest.approx(long_first.attainment)
        assert short_first.resolved is False
        assert long_first.resolved is True
        # The short probe is unresolved because its interval is wider, not
        # because its attainment differs.
        short_width = short_first.attainment_upper - short_first.attainment_lower
        long_width = long_first.attainment_upper - long_first.attainment_lower
        assert short_width > long_width

    @pytest.mark.regression
    def test_rejects_workloads_with_no_measurable_objective(self):
        """
        Fail loudly when no request produced a determined verdict, rather than
        searching against an attainment that cannot be computed.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile()
        strategy = profile.next_strategy(None, None)
        # A time-to-first-token objective against requests with no token
        # timings leaves every request undetermined.
        undetermined = _make_benchmark_with_slo(GoodputSLO(ttft_ms=500), determined=8)
        assert undetermined.metrics.slo_attainment is None

        with pytest.raises(RuntimeError, match="latency objectives"):
            profile.next_strategy(strategy, undetermined)

    @pytest.mark.regression
    def test_target_attainment_boundary_is_inclusive(self):
        """
        Treat attainment exactly at the target as passing, not failing.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile(target_attainment=0.95)
        strategy = profile.next_strategy(None, None)
        profile.completed_strategies.append(strategy)
        profile.next_strategy(strategy, _make_benchmark(0.95, determined=2000))

        assert profile.search.probes[0].passed is True
        assert profile.search.best_passing_streams == strategy.streams

    @pytest.mark.regression
    def test_probe_just_below_target_fails(self):
        """
        Treat attainment one request below the target as failing.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile(target_attainment=0.95)
        strategy = profile.next_strategy(None, None)
        profile.completed_strategies.append(strategy)
        profile.next_strategy(strategy, _make_benchmark(0.9495, determined=2000))

        assert profile.search.probes[0].passed is False
        assert profile.search.best_passing_streams is None

    @pytest.mark.regression
    def test_configured_confidence_reaches_the_interval(self):
        """
        Widen each probe's interval when a higher confidence is configured.

        ## WRITTEN BY AI ##
        """
        widths = []
        for confidence in (0.90, 0.99):
            profile = _make_profile(confidence=confidence)
            strategy = profile.next_strategy(None, None)
            profile.completed_strategies.append(strategy)
            profile.next_strategy(strategy, _make_benchmark(0.97, determined=1000))
            probe = profile.search.probes[0]
            widths.append(probe.attainment_upper - probe.attainment_lower)

        assert widths[1] > widths[0]

    @pytest.mark.regression
    def test_stops_escalating_when_a_constraint_halts_the_run(self):
        """
        Halt the search when a stopping_scope='all' constraint fired, matching
        the concurrent, async and sweep profiles.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile()
        strategy = profile.next_strategy(None, None)
        profile.completed_strategies.append(strategy)
        benchmark = _make_benchmark(0.99, determined=2000)
        benchmark.scheduler_state.end_queuing_constraints = {
            "max_errors": SchedulerUpdateAction(
                request_queuing="stop",
                request_processing="stop_all",
                stopping_scope="all",
            )
        }

        assert profile.next_strategy(strategy, benchmark) is None
        assert profile.search.stop_reason == "constraint_stopped_escalation"

    @pytest.mark.regression
    def test_constraint_aborted_probe_is_not_a_passing_bound(self):
        """
        Never report a concurrency whose probe a constraint aborted as the
        highest passing level.

        Enforced over-saturation cancels active requests, and cancelled
        requests are excluded from attainment, so the completed remainder can
        look conforming. Accepting that as a bound would report an unsafe
        concurrency as supported.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile()
        strategy = profile.next_strategy(None, None)
        profile.completed_strategies.append(strategy)
        benchmark = _make_benchmark(1.0, determined=2000)
        benchmark.scheduler_state.end_queuing_constraints = {
            "over_saturation": SchedulerUpdateAction(
                request_queuing="stop",
                request_processing="stop_all",
                stopping_scope="all",
            )
        }

        assert profile.next_strategy(strategy, benchmark) is None
        assert profile.search.probes[0].passed is True
        assert profile.search.probes[0].aborted is True
        assert profile.search.best_passing_streams is None
        assert profile.search.stop_reason == "constraint_stopped_escalation"

    @pytest.mark.regression
    def test_unresolved_descent_is_reported_as_indeterminate(self):
        """
        Distinguish "the objectives cannot be met" from "the probes were too
        short to tell" when every level tested failed.

        Each failure is decided on a point estimate. When those intervals
        straddle the target, descending to a single stream is not evidence
        that no concurrency meets the objectives.

        ## WRITTEN BY AI ##
        """

        def descend(determined: int):
            """Every level misses the target by the same small margin."""
            profile = _make_profile()
            prev_strategy = None
            prev_benchmark = None
            while (
                strategy := profile.next_strategy(prev_strategy, prev_benchmark)
            ) is not None:
                prev_benchmark = _make_benchmark(0.90, determined)
                prev_strategy = strategy
                profile.completed_strategies.append(strategy)
            return profile.search

        # 18/20 against a 0.95 target fails on the point estimate while its
        # interval still contains the target.
        unresolved_run = descend(20)
        # The same margin with enough requests per probe to decide.
        resolved_run = descend(20000)

        assert unresolved_run.best_passing_streams is None
        assert unresolved_run.stop_reason == "indeterminate_at_minimum"
        assert resolved_run.best_passing_streams is None
        assert resolved_run.stop_reason == "objectives_unmet_at_minimum"

    @pytest.mark.regression
    def test_declares_the_probe_budget_before_running(self):
        """
        Report the planned probe count up front so the progress display can
        size its task list, rather than reporting probes already completed.

        ## WRITTEN BY AI ##
        """
        profile = _make_profile(max_probes=9)

        assert profile.strategy_types == ["concurrent"] * 9

    @pytest.mark.regression
    def test_clamps_doubling_to_the_stream_ceiling(self):
        """
        Stop at max_streams when doubling would overshoot it.

        ## WRITTEN BY AI ##
        """
        probed, search = _drive_search(knee=100_000, initial_streams=4, max_streams=50)

        assert max(probed) == 50
        assert search.best_passing_streams == 50

    @pytest.mark.regression
    def test_result_carries_the_final_probe_and_stop_reason(self):
        """
        Expose the whole search, including the last probe, through the profile
        result rather than only through the per-benchmark config.

        Each benchmark's config is captured before that benchmark runs, so it
        can never contain the final probe. The result is read after the search
        ends, which is the only point at which the answer exists.

        ## WRITTEN BY AI ##
        """
        probed, search = _drive_search(knee=16, max_probes=20)
        profile = _make_profile()
        prev_strategy = None
        prev_benchmark = None
        while (
            strategy := profile.next_strategy(prev_strategy, prev_benchmark)
        ) is not None:
            prev_benchmark = _make_benchmark(_attainment_at(strategy.streams, 16))
            prev_strategy = strategy
            profile.completed_strategies.append(strategy)

        result = profile.result

        assert result is not None
        assert [probe["streams"] for probe in result["probes"]] == probed
        assert result["best_passing_streams"] == search.best_passing_streams
        assert result["stop_reason"] == "converged"

    @pytest.mark.sanity
    def test_records_each_probe_in_the_search_trace(self):
        """
        Record concurrency, attainment, and interval for every probe executed.

        ## WRITTEN BY AI ##
        """
        probed, search = _drive_search(knee=16, max_probes=20)

        assert [probe.streams for probe in search.probes] == probed
        for probe in search.probes:
            assert probe.determined_requests == 2000
            assert probe.goodput is not None
            assert probe.attainment_lower <= probe.attainment <= probe.attainment_upper
