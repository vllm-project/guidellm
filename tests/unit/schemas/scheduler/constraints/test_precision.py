"""Unit tests for the target margin of error constraint argument schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.schemas.scheduler import ConstraintArgs, TargetMoeConstraintArgs


class TestTargetMoeConstraintArgs:
    """Test suite for TargetMoeConstraintArgs."""

    @pytest.mark.smoke
    def test_defaults(self):
        """
        Test that only the target margin of error is required.

        ## WRITTEN BY AI ##
        """
        args = TargetMoeConstraintArgs(moe=0.05)

        assert args.kind == "target_moe"
        assert args.constraint_key == "target_moe"
        assert args.metric == "time_to_first_token_ms"
        assert args.statistic == "mean"
        assert args.confidence == 0.95
        assert args.min_samples == 30
        assert args.check_interval == 10
        assert args.stopping_scope == "current"

    @pytest.mark.smoke
    def test_polymorphic_validation(self):
        """
        Test that a kind-discriminated dict resolves to the target_moe schema.

        ## WRITTEN BY AI ##
        """
        args = ConstraintArgs.model_validate(
            {
                "kind": "target_moe",
                "metric": "request_latency",
                "statistic": "p95",
                "moe": 0.1,
            }
        )

        assert isinstance(args, TargetMoeConstraintArgs)
        assert args.metric == "request_latency"
        assert args.statistic == "p95"
        assert args.moe == 0.1

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "overrides",
        [
            {"moe": 0.0},
            {"moe": 1.0},
            {"moe": -0.05},
            {"confidence": 0.4},
            {"confidence": 1.0},
            {"min_samples": 1},
            {"check_interval": 0},
            {"metric": "inter_token_latency_ms"},
            {"statistic": "p42"},
            {"unknown": 1},
        ],
    )
    def test_invalid_values(self, overrides):
        """
        Test that out of range values, unsupported metrics and extras are rejected.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            TargetMoeConstraintArgs(**{"moe": 0.05, **overrides})

    @pytest.mark.sanity
    def test_serialization_round_trip(self):
        """
        Test that serialized args validate back to an equal instance.

        ## WRITTEN BY AI ##
        """
        args = TargetMoeConstraintArgs(
            statistic="p99", moe=0.02, confidence=0.9, stopping_scope="all"
        )

        restored = ConstraintArgs.model_validate(args.model_dump())

        assert isinstance(restored, TargetMoeConstraintArgs)
        assert restored == args
