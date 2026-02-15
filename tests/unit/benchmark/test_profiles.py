"""
Tests for cross-rate early exit behavior in AsyncProfile, ConcurrentProfile,
and SweepProfile.

Validates that:
- AsyncProfile and ConcurrentProfile sort rates/streams ascending
- Failure constraints (stop_all) stop rate escalation
- Normal completions (stop_local) do not stop rate escalation
- SweepProfile stops escalation during the async phase but not during throughput
"""

from types import SimpleNamespace

import pytest

from guidellm.benchmark.profiles import (
    AsyncProfile,
    ConcurrentProfile,
    Profile,
    SweepProfile,
)
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    SchedulerState,
    SchedulerUpdateAction,
    SynchronousStrategy,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_benchmark(
    end_queuing_constraints: dict[str, SchedulerUpdateAction] | None = None,
    request_throughput_mean: float = 10.0,
):
    """
    Create a lightweight mock benchmark with a scheduler_state.

    Uses SimpleNamespace to avoid constructing a full GenerativeBenchmark,
    which requires many nested fields.
    """
    state = SchedulerState()
    if end_queuing_constraints:
        state.end_queuing_constraints = end_queuing_constraints

    throughput = SimpleNamespace(
        successful=SimpleNamespace(mean=request_throughput_mean),
    )

    return SimpleNamespace(
        scheduler_state=state,
        request_throughput=throughput,
    )


def _make_failure_action(name: str = "test_constraint") -> SchedulerUpdateAction:
    """Create a SchedulerUpdateAction with stop_all (failure)."""
    return SchedulerUpdateAction(
        request_queuing="stop",
        request_processing="stop_all",
        metadata={f"{name}_triggered": True},
    )


def _make_normal_completion_action() -> SchedulerUpdateAction:
    """Create a SchedulerUpdateAction with stop_local (normal completion)."""
    return SchedulerUpdateAction(
        request_queuing="stop",
        request_processing="stop_local",
        metadata={"duration_exceeded": True},
    )


def _make_failure_benchmark(constraint_name: str = "over_saturation"):
    """Create a mock benchmark terminated by a stop_all constraint."""
    return _make_mock_benchmark(
        end_queuing_constraints={
            constraint_name: _make_failure_action(constraint_name),
        }
    )


def _make_normal_benchmark():
    """Create a mock benchmark that completed normally (stop_local)."""
    return _make_mock_benchmark(
        end_queuing_constraints={
            "max_duration": _make_normal_completion_action(),
        }
    )


def _advance_sweep_to_async_phase(profile: SweepProfile, sync_rate=2.0, tp_rate=10.0):
    """
    Advance a SweepProfile through sync and throughput phases, returning
    the first async strategy. Mutates profile.completed_strategies.

    Follows the same protocol as strategies_generator: each strategy is
    appended to completed_strategies before the next next_strategy call.
    """
    # Phase 1: synchronous
    sync_strat = SynchronousStrategy()
    sync_benchmark = _make_mock_benchmark(request_throughput_mean=sync_rate)
    throughput_strat = profile.next_strategy(sync_strat, sync_benchmark)
    profile.completed_strategies.append(sync_strat)

    # Phase 2: throughput
    throughput_benchmark = _make_mock_benchmark(request_throughput_mean=tp_rate)
    profile.completed_strategies.append(throughput_strat)
    first_async_strat = profile.next_strategy(throughput_strat, throughput_benchmark)

    return first_async_strat, throughput_strat


# ============================================================================
# Profile._should_stop_escalating tests
# ============================================================================


class TestShouldStopEscalating:
    """Tests for the shared Profile._should_stop_escalating static method."""

    def test_stop_all_triggers_escalation_stop(self):
        """A constraint with request_processing=stop_all should trigger stop."""
        benchmark = _make_failure_benchmark("over_saturation")
        assert Profile._should_stop_escalating(benchmark) is True

    def test_stop_local_does_not_trigger(self):
        """A constraint with request_processing=stop_local should not trigger."""
        benchmark = _make_normal_benchmark()
        assert Profile._should_stop_escalating(benchmark) is False

    def test_no_constraints(self):
        """No constraints triggered means no stop."""
        benchmark = _make_mock_benchmark(end_queuing_constraints={})
        assert Profile._should_stop_escalating(benchmark) is False

    def test_no_scheduler_state(self):
        """Benchmark without scheduler_state should not stop."""
        benchmark = SimpleNamespace()
        assert Profile._should_stop_escalating(benchmark) is False

    def test_mixed_constraints_with_one_stop_all(self):
        """If multiple constraints present and one is stop_all, should stop."""
        benchmark = _make_mock_benchmark(
            end_queuing_constraints={
                "max_duration": _make_normal_completion_action(),
                "over_saturation": _make_failure_action("over_saturation"),
            }
        )
        assert Profile._should_stop_escalating(benchmark) is True

    def test_multiple_stop_local_does_not_trigger(self):
        """Multiple stop_local constraints should not trigger stop."""
        benchmark = _make_mock_benchmark(
            end_queuing_constraints={
                "max_duration": _make_normal_completion_action(),
                "max_requests": _make_normal_completion_action(),
            }
        )
        assert Profile._should_stop_escalating(benchmark) is False


# ============================================================================
# Rate/stream sorting tests (parametrized across AsyncProfile & ConcurrentProfile)
# ============================================================================


@pytest.mark.parametrize(
    "profile_cls, rate_type, unsorted_input, sorted_output, output_key",
    [
        (AsyncProfile, "constant", [50.0, 10.0, 1.0, 25.0], [1.0, 10.0, 25.0, 50.0], "rate"),
        (AsyncProfile, "constant", [1.0, 5.0, 10.0], [1.0, 5.0, 10.0], "rate"),
        (AsyncProfile, "constant", [5.0], [5.0], "rate"),
        (ConcurrentProfile, "concurrent", [16.0, 4.0, 1.0, 8.0], [1, 4, 8, 16], "streams"),
        (ConcurrentProfile, "concurrent", [2.0, 4.0, 8.0], [2, 4, 8], "streams"),
        (ConcurrentProfile, "concurrent", [4.0], [4], "streams"),
    ],
    ids=[
        "async-unsorted", "async-sorted", "async-single",
        "concurrent-unsorted", "concurrent-sorted", "concurrent-single",
    ],
)
class TestRateSorting:
    """Rates and streams should be sorted ascending in resolve_args."""

    def test_sorting(self, profile_cls, rate_type, unsorted_input, sorted_output, output_key):
        resolved = profile_cls.resolve_args(
            rate_type=rate_type,
            rate=unsorted_input,
            random_seed=42,
        )
        assert resolved[output_key] == sorted_output


# ============================================================================
# Cross-rate early exit tests (parametrized across AsyncProfile & ConcurrentProfile)
# ============================================================================


class TestMultiRateProfileEarlyExit:
    """
    Tests for next_strategy() cross-rate early exit, parametrized across
    AsyncProfile and ConcurrentProfile which share the same behavior.
    """

    @staticmethod
    def _make_async_profile():
        profile = AsyncProfile(type_="constant", strategy_type="constant", rate=[1.0, 5.0, 10.0])
        first = AsyncConstantStrategy(rate=1.0)
        return profile, first

    @staticmethod
    def _make_concurrent_profile():
        profile = ConcurrentProfile(streams=[2, 4, 8])
        first = ConcurrentStrategy(streams=2)
        return profile, first

    @pytest.mark.parametrize("make_profile", ["_make_async_profile", "_make_concurrent_profile"])
    def test_normal_completion_continues(self, make_profile):
        """After normal completion (stop_local), should advance to next rate."""
        profile, first_strategy = getattr(self, make_profile)()
        profile.completed_strategies.append(first_strategy)

        next_strat = profile.next_strategy(first_strategy, _make_normal_benchmark())

        assert next_strat is not None

    @pytest.mark.parametrize("make_profile", ["_make_async_profile", "_make_concurrent_profile"])
    def test_failure_stops(self, make_profile):
        """After failure (stop_all), should return None."""
        profile, first_strategy = getattr(self, make_profile)()
        profile.completed_strategies.append(first_strategy)

        next_strat = profile.next_strategy(
            first_strategy, _make_failure_benchmark()
        )

        assert next_strat is None

    @pytest.mark.parametrize("make_profile", ["_make_async_profile", "_make_concurrent_profile"])
    def test_first_rate_always_runs(self, make_profile):
        """First rate should always run (no previous benchmark)."""
        profile, _ = getattr(self, make_profile)()

        next_strat = profile.next_strategy(None, None)

        assert next_strat is not None

    @pytest.mark.parametrize("make_profile", ["_make_async_profile", "_make_concurrent_profile"])
    def test_all_rates_completed_returns_none(self, make_profile):
        """When all rates are done, should return None regardless."""
        profile, first_strategy = getattr(self, make_profile)()
        for _ in range(len(getattr(profile, "rate", None) or profile.streams)):
            profile.completed_strategies.append(first_strategy)

        next_strat = profile.next_strategy(first_strategy, _make_mock_benchmark())

        assert next_strat is None

    def test_middle_rate_failure_skips_remaining(self):
        """Rate 1 succeeds, rate 2 succeeds, rate 3 fails, rate 4 is skipped.

        This is the core use case: the system handles low rates fine but
        fails at a mid-range rate, and higher rates are not attempted.
        """
        profile = AsyncProfile(
            type_="constant",
            strategy_type="constant",
            rate=[1.0, 5.0, 10.0, 50.0],
        )

        # Rate 1 (1.0 RPS): succeeds
        strat_1 = profile.next_strategy(None, None)
        assert strat_1 is not None
        assert strat_1.rate == 1.0
        profile.completed_strategies.append(strat_1)

        # Rate 2 (5.0 RPS): succeeds
        strat_2 = profile.next_strategy(strat_1, _make_normal_benchmark())
        assert strat_2 is not None
        assert strat_2.rate == 5.0
        profile.completed_strategies.append(strat_2)

        # Rate 3 (10.0 RPS): runs
        strat_3 = profile.next_strategy(strat_2, _make_normal_benchmark())
        assert strat_3 is not None
        assert strat_3.rate == 10.0
        profile.completed_strategies.append(strat_3)

        # Rate 4 (50.0 RPS): should be skipped because rate 3 failed
        strat_4 = profile.next_strategy(strat_3, _make_failure_benchmark())
        assert strat_4 is None

    def test_poisson_strategy_early_exit(self):
        """Poisson strategy type should work through the same early-exit path."""
        profile = AsyncProfile(
            type_="poisson",
            strategy_type="poisson",
            rate=[1.0, 5.0, 10.0],
            random_seed=42,
        )

        strat_1 = profile.next_strategy(None, None)
        assert strat_1 is not None
        assert isinstance(strat_1, AsyncPoissonStrategy)
        assert strat_1.rate == 1.0
        profile.completed_strategies.append(strat_1)

        # First rate fails -> second rate skipped
        strat_2 = profile.next_strategy(strat_1, _make_failure_benchmark())
        assert strat_2 is None


# ============================================================================
# SweepProfile cross-rate early exit tests
# ============================================================================


class TestSweepProfileEarlyExit:
    """Tests for SweepProfile.next_strategy() cross-rate early exit."""

    def _make_profile(self, sweep_size: int = 5) -> SweepProfile:
        return SweepProfile(sweep_size=sweep_size, strategy_type="constant")

    def test_sync_and_throughput_always_run(self):
        """Synchronous and throughput phases should always run."""
        profile = self._make_profile()

        strat = profile.next_strategy(None, None)
        assert strat is not None
        assert strat.type_ == "synchronous"

    def test_throughput_runs_after_sync_with_failure(self):
        """Throughput always runs after sync (sync returns early before check)."""
        profile = self._make_profile()

        sync_strategy = SynchronousStrategy()
        sync_benchmark = _make_mock_benchmark(
            request_throughput_mean=5.0,
            end_queuing_constraints={
                "over_saturation": _make_failure_action("over_saturation"),
            },
        )

        strat = profile.next_strategy(sync_strategy, sync_benchmark)
        assert strat is not None
        assert strat.type_ == "throughput"

    def test_throughput_failure_does_not_stop(self):
        """No failure during throughput should stop the sweep.

        Throughput pushes the system to its limit by design, so all failure
        checks are skipped after the throughput phase.
        """
        profile = self._make_profile(sweep_size=5)

        sync_strat = SynchronousStrategy()
        sync_benchmark = _make_mock_benchmark(request_throughput_mean=2.0)
        throughput_strat = profile.next_strategy(sync_strat, sync_benchmark)
        profile.completed_strategies.append(sync_strat)

        throughput_benchmark = _make_mock_benchmark(
            request_throughput_mean=10.0,
            end_queuing_constraints={
                "over_saturation": _make_failure_action("over_saturation"),
            },
        )
        profile.completed_strategies.append(throughput_strat)
        first_async_strat = profile.next_strategy(throughput_strat, throughput_benchmark)

        assert first_async_strat is not None

    def test_async_phase_stops_on_failure(self):
        """During async phase, stop_all constraint should stop remaining rates."""
        profile = self._make_profile(sweep_size=5)
        first_async_strat, _ = _advance_sweep_to_async_phase(profile)
        assert first_async_strat is not None

        profile.completed_strategies.append(first_async_strat)
        next_strat = profile.next_strategy(
            first_async_strat, _make_failure_benchmark()
        )

        assert next_strat is None

    def test_async_phase_continues_on_normal_completion(self):
        """During async phase, stop_local should advance to next rate."""
        profile = self._make_profile(sweep_size=5)
        first_async_strat, _ = _advance_sweep_to_async_phase(profile)
        assert first_async_strat is not None

        profile.completed_strategies.append(first_async_strat)
        next_strat = profile.next_strategy(
            first_async_strat, _make_normal_benchmark()
        )

        assert next_strat is not None

    def test_sweep_size_2_no_async_phase(self):
        """With sweep_size=2, only sync + throughput run; no async phase.

        Verifies the escalation check is never reached and the profile
        completes cleanly without generating any async rates.
        """
        profile = self._make_profile(sweep_size=2)

        # Phase 1: synchronous
        sync_strat = profile.next_strategy(None, None)
        assert sync_strat is not None
        assert sync_strat.type_ == "synchronous"
        profile.completed_strategies.append(sync_strat)

        # Phase 2: throughput
        sync_benchmark = _make_mock_benchmark(request_throughput_mean=5.0)
        throughput_strat = profile.next_strategy(sync_strat, sync_benchmark)
        assert throughput_strat is not None
        assert throughput_strat.type_ == "throughput"
        profile.completed_strategies.append(throughput_strat)

        # Phase 3: should be None — no async rates generated
        throughput_benchmark = _make_mock_benchmark(request_throughput_mean=10.0)
        next_strat = profile.next_strategy(throughput_strat, throughput_benchmark)

        assert profile.measured_rates == []
        assert next_strat is None
