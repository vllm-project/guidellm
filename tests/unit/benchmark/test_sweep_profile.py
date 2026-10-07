"""Unit tests for sweep benchmark profile argument validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, SweepProfile
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


def _rate_benchmark(
    mean_rate: float, mean_concurrency: float, mean_latency: float
) -> SimpleNamespace:
    """Stand in for a compiled benchmark exposing only the sweep's rate inputs.

    ## WRITTEN BY AI ##
    """
    return SimpleNamespace(
        request_throughput=SimpleNamespace(successful=SimpleNamespace(mean=mean_rate)),
        request_concurrency=SimpleNamespace(
            total=SimpleNamespace(mean=mean_concurrency)
        ),
        request_latency=SimpleNamespace(successful=SimpleNamespace(mean=mean_latency)),
    )


def _first_async_rate(throughput_benchmark: SimpleNamespace) -> float:
    """Run a three-step sweep through its throughput step and return the next rate.

    With ``sweep_size=3`` the only interpolated step targets the throughput
    phase's rate, so it exposes the upper bound directly.

    ## WRITTEN BY AI ##
    """
    profile = ProfileFactory.create(
        SweepProfileArgs.model_validate({"kind": "sweep", "sweep_size": 3}), 42, {}
    )
    generator = profile.strategies_generator()
    synchronous, _ = next(generator)
    assert synchronous.type_ == "synchronous"
    throughput, _ = generator.send(
        _rate_benchmark(mean_rate=1.0, mean_concurrency=1.0, mean_latency=1.0)
    )
    assert throughput.type_ == "throughput"
    constant, _ = generator.send(throughput_benchmark)

    return constant.rate


class TestSweepThroughputRate:
    """
    Verify the upper bound the sweep takes from its throughput phase.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_upper_bound_applies_littles_law(self):
        """
        Set the top rate to mean concurrency over mean latency.

        32 requests held in flight that each take 10s complete 3.2 per second.
        A 15s run that waited 10s for the first completions reports a windowed
        mean of about 2.1, which must not become the upper bound.

        ## WRITTEN BY AI ##
        """
        rate = _first_async_rate(
            _rate_benchmark(mean_rate=2.13, mean_concurrency=32.0, mean_latency=10.0)
        )

        assert rate == pytest.approx(3.2)

    @pytest.mark.sanity
    def test_upper_bound_falls_back_without_latency(self):
        """
        Use the windowed mean when no successful request latency was measured.

        ## WRITTEN BY AI ##
        """
        rate = _first_async_rate(
            _rate_benchmark(mean_rate=2.0, mean_concurrency=0.0, mean_latency=0.0)
        )

        assert rate == pytest.approx(2.0)
