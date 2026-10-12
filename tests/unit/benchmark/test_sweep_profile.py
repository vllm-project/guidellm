"""Unit tests for the sweep benchmark profile."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, SweepProfile
from guidellm.benchmark.schemas import GenerativeBenchmark
from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncRampStrategy,
    RampSaturationConstraint,
    SchedulerState,
    SchedulerUpdateAction,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.schemas.benchmark import SweepProfileArgs


class TestSweepProfileArgs:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"kind": "sweep", "sweep_size": 5}, 5),
            ({"kind": "sweep", "sweep_size": 8}, 8),
            ({"kind": "sweep", "sweep_size": 12}, 12),
            ({"kind": "sweep", "sweep_size": 10}, 10),
        ],
    )
    def test_sweep_size_validates(self, payload, expected):
        """
        Validate sweep_size from explicit sweep_size field.

        ## WRITTEN BY AI ##
        """
        args = SweepProfileArgs.model_validate(payload)
        assert args.sweep_size == expected

    @pytest.mark.smoke
    def test_profile_create_from_sweep_size(self):
        """
        Create sweep profile when sweep_size is provided explicitly.

        ## WRITTEN BY AI ##
        """
        profile = ProfileFactory.create(
            SweepProfileArgs.model_validate({"kind": "sweep", "sweep_size": 6}),
            42,
            {},
        )
        assert isinstance(profile, SweepProfile)
        assert profile.args.sweep_size == 6

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_resolve_profile_passes_sweep_size(self):
        """
        End-to-end resolve_profile passes sweep_size into the profile.

        ## WRITTEN BY AI ##
        """
        profile = await resolve_profile(
            profile=SweepProfileArgs.model_validate({"kind": "sweep", "sweep_size": 7}),
            constraints={},
        )
        assert isinstance(profile, SweepProfile)
        assert profile.args.sweep_size == 7

    @pytest.mark.smoke
    def test_sweep_size_enforces_minimum(self):
        """
        Reject sweep sizes below the profile minimum.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            SweepProfileArgs.model_validate({"kind": "sweep", "sweep_size": 1})


def _generative_benchmark(
    rate: float,
    latency: float = 0.5,
    constraint_actions: dict[str, SchedulerUpdateAction] | None = None,
):
    """A GenerativeBenchmark stand-in exposing only what the sweep reads."""
    benchmark = Mock(spec=GenerativeBenchmark)
    benchmark.request_throughput = SimpleNamespace(
        successful=SimpleNamespace(mean=rate)
    )
    benchmark.request_latency = SimpleNamespace(
        successful=SimpleNamespace(mean=latency)
    )
    state = SchedulerState()
    state.scheduler_constraints = constraint_actions or {}
    benchmark.scheduler_state = state
    return benchmark


def _ramp_profile(**overrides) -> SweepProfile:
    args = SweepProfileArgs.model_validate(
        {"kind": "sweep", "sweep_size": 4, "peak_detection": "ramp", **overrides}
    )
    return SweepProfile(args, random_seed=42, constraints=None)


def _run_to_ramp(profile: SweepProfile, sync_rate=2.0, latency=0.5):
    sync = SynchronousStrategy()
    profile.completed_strategies.append(sync)
    ramp = profile.next_strategy(sync, _generative_benchmark(sync_rate, latency))
    profile.completed_strategies.append(ramp)
    return ramp


class TestSweepRampPeakDetection:
    @pytest.mark.smoke
    def test_throughput_step_is_the_default(self):
        """
        Without peak_detection set, the sweep still runs the throughput step.

        ## WRITTEN BY AI ##
        """
        profile = SweepProfile(
            SweepProfileArgs.model_validate({"kind": "sweep", "sweep_size": 4}),
            random_seed=42,
            constraints=None,
        )

        assert profile.strategy_types[:2] == ["synchronous", "throughput"]
        assert isinstance(
            profile.next_strategy(SynchronousStrategy(), _generative_benchmark(2.0)),
            ThroughputStrategy,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(("latency", "interval"), [(0.5, 2.0), (1.5, 6.0)])
    def test_ramp_is_sized_from_the_synchronous_step(self, latency, interval):
        """
        The ramp starts at the synchronous rate and doubles every four
        synchronous latencies, with a two second floor.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile()

        ramp = _run_to_ramp(profile, sync_rate=2.0, latency=latency)

        assert profile.strategy_types[:2] == ["synchronous", "ramp"]
        assert isinstance(ramp, AsyncRampStrategy)
        assert ramp.start_rate == 2.0
        assert ramp.doubling_interval == pytest.approx(interval)

    @pytest.mark.sanity
    def test_ramp_gets_saturation_detection(self):
        """
        Only the ramp step gets the saturation constraint, sized to its interval.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile()
        ramp = _run_to_ramp(profile)

        ramp_constraints = profile.next_strategy_constraints(
            ramp, SynchronousStrategy(), None
        )
        constant_constraints = profile.next_strategy_constraints(
            AsyncConstantStrategy(rate=1.0), ramp, None
        )

        tracker = (ramp_constraints or {})["ramp_saturation"]
        assert isinstance(tracker, RampSaturationConstraint)
        assert tracker.window_seconds == ramp.doubling_interval
        assert "ramp_saturation" not in (constant_constraints or {})

    @pytest.mark.sanity
    def test_constant_steps_span_up_to_the_ramp_peak(self):
        """
        The peak the ramp measured sets the top of the range and the conclusion.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile()
        ramp = _run_to_ramp(profile, sync_rate=2.0)
        peak_action = SchedulerUpdateAction(
            metadata={"peak_rate": 20.0, "saturated_at": 12.0}
        )

        first = profile.next_strategy(
            ramp,
            _generative_benchmark(
                5.0, constraint_actions={"ramp_saturation": peak_action}
            ),
        )

        assert isinstance(first, AsyncConstantStrategy)
        assert profile.measured_rates == pytest.approx([11.0, 20.0])
        assert profile.conclusion == {
            "kind": "sweep_ramp",
            "synchronous_rate": 2.0,
            "peak_rate": 20.0,
            "peak_source": "saturated",
            "doubling_interval": ramp.doubling_interval,
        }

    @pytest.mark.sanity
    def test_falls_back_to_the_mean_rate_without_a_peak(self):
        """
        If the ramp ended before measuring a full window, the mean rate is used.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile()
        ramp = _run_to_ramp(profile, sync_rate=2.0)

        profile.next_strategy(ramp, _generative_benchmark(8.0))

        assert profile.throughput_rate == 8.0
        assert (profile.conclusion or {})["peak_source"] == "mean_rate"

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [({}, None), ({"max_concurrency": 64}, 64)],
    )
    def test_default_concurrency_cap_is_dropped(self, overrides, expected):
        """
        The default cap of 512 does not apply with a ramp, but an explicit one
        applies to the ramp and the constant steps.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile(**overrides)
        ramp = _run_to_ramp(profile)
        first = profile.next_strategy(ramp, _generative_benchmark(8.0))

        assert ramp.max_concurrency == expected
        assert first.max_concurrency == expected

    @pytest.mark.regression
    def test_ramp_stop_does_not_end_the_sweep(self):
        """
        Like the throughput step, a stop during the ramp does not end escalation.

        ## WRITTEN BY AI ##
        """
        profile = _ramp_profile()
        ramp = _run_to_ramp(profile)
        stopped = _generative_benchmark(8.0)
        stopped.scheduler_state.end_queuing_constraints = {
            "max_errors": SchedulerUpdateAction(
                request_queuing="stop", stopping_scope="all"
            )
        }

        assert isinstance(profile.next_strategy(ramp, stopped), AsyncConstantStrategy)
