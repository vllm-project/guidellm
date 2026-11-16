"""Unit tests for sweep benchmark profile argument validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, SweepProfile
from guidellm.scheduler import AsyncConstantStrategy, ThroughputStrategy
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


class TestSweepProfile:
    @pytest.mark.regression
    def test_constant_step_preserves_rampup_duration(self, mocker):
        """
        Preserve configured rampup duration in throughput and constant sweep steps.

        ## WRITTEN BY AI ##
        """
        args = SweepProfileArgs.model_validate(
            {
                "kind": "sweep",
                "sweep_size": 5,
                "strategy_type": "constant",
                "rampup_duration": 10.0,
            }
        )
        profile = SweepProfile(args, random_seed=42, constraints={})
        benchmark = mocker.Mock()
        # Rate, concurrency and latency agree, so the step rates do not depend
        # on how the sweep estimates the throughput rate.
        benchmark.request_latency.successful.mean = 1.0
        benchmark.request_throughput.successful.mean = 2.0
        benchmark.request_concurrency.total.mean = 2.0
        strategies = profile.strategies_generator()

        synchronous, _ = next(strategies)
        assert synchronous.type_ == "synchronous"

        throughput, _ = strategies.send(benchmark)
        assert isinstance(throughput, ThroughputStrategy)
        assert throughput.rampup_duration == args.rampup_duration

        benchmark.request_throughput.successful.mean = 10.0
        benchmark.request_concurrency.total.mean = 10.0
        constant, _ = strategies.send(benchmark)
        assert isinstance(constant, AsyncConstantStrategy)
        assert constant.rampup_duration == args.rampup_duration
