"""Unit tests for sweep benchmark profile argument validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_profile
from guidellm.benchmark.profiles import ProfileFactory, SweepProfile
from guidellm.benchmark.profiles.sweep import SweepProfileArgs


class TestSweepProfileArgs:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"kind": "sweep", "sweep_size": 5}, 5),
            ({"kind": "sweep", "rate": [8.0]}, 8),
            ({"kind": "sweep", "rate": [12]}, 12),
            ({"kind": "sweep", "rate": [10.0, 20.0]}, 10),
        ],
    )
    def test_sweep_size_accepts_scalar_or_rate_list(self, payload, expected):
        """
        Validate sweep_size from sweep_size or global rate list.

        ## WRITTEN BY AI ##
        """
        args = SweepProfileArgs.model_validate(payload)
        assert args.sweep_size == expected

    @pytest.mark.smoke
    def test_profile_create_from_rate_list(self):
        """
        Create sweep profile when resolve_profile passes rate as a list.

        ## WRITTEN BY AI ##
        """
        profile = ProfileFactory.create(
            SweepProfileArgs.model_validate({"kind": "sweep", "rate": [6.0]}),
            42,
            {},
        )
        assert isinstance(profile, SweepProfile)
        assert profile.args.sweep_size == 6

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_resolve_profile_coerces_rate_list(self):
        """
        End-to-end resolve_profile passes list rate into sweep_size.

        ## WRITTEN BY AI ##
        """
        profile = await resolve_profile(
            profile=SweepProfileArgs.model_validate(
                {"kind": "sweep", "rate": [7.0, 8.0]}
            ),
            constraints={},
            max_seconds=None,
            max_requests=None,
            max_errors=None,
            max_error_rate=None,
            max_global_error_rate=None,
            random_seed=42,
        )
        assert isinstance(profile, SweepProfile)
        assert profile.args.sweep_size == 7

    @pytest.mark.smoke
    def test_sweep_size_rejects_empty_rate_list(self):
        """
        Reject empty rate list for sweep profile.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            SweepProfileArgs.model_validate({"kind": "sweep", "rate": []})

    @pytest.mark.smoke
    def test_sweep_size_enforces_minimum(self):
        """
        Reject sweep sizes below the profile minimum.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            SweepProfileArgs.model_validate({"kind": "sweep", "rate": [1.0]})
