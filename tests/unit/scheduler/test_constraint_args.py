"""Unit tests for ConstraintArgs kind-based constraint configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.benchmark.entrypoints import resolve_constraints
from guidellm.benchmark.schemas import BenchmarkArgs
from guidellm.scheduler.constraints import (
    ConstraintArgs,
    ConstraintsInitializerFactory,
    MaxDurationConstraintArgs,
    MaxErrorRateConstraintArgs,
    MaxErrorsConstraintArgs,
    MaxGlobalErrorRateConstraintArgs,
    MaxRequestsConstraintArgs,
    OverSaturationConstraintArgs,
)
from guidellm.scheduler.constraints.error import (
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxGlobalErrorRateConstraint,
)
from guidellm.scheduler.constraints.request import (
    MaxDurationConstraint,
    MaxNumberConstraint,
)
from guidellm.scheduler.constraints.saturation import (
    OverSaturationConstraintInitializer,
)


def _minimal_args(**overrides) -> BenchmarkArgs:
    """
    Create a minimal valid BenchmarkArgs for testing.

    ## WRITTEN BY AI ##
    """
    base = {
        "data": [{"kind": "synthetic_text", "prompt_tokens": 10, "output_tokens": 10}],
        "backend": {"kind": "openai_http", "target": "http://localhost:8000"},
        "profile": {"kind": "sweep"},
        "tokenizer": {"kind": "huggingface_auto"},
        "data_column_mapper": {"kind": "generative_column_mapper"},
        "data_preprocessors": [{"kind": "encode_media"}],
        "data_finalizer": {"kind": "generative"},
        "data_loader": {"kind": "pytorch"},
    }
    base.update(overrides)
    return BenchmarkArgs.model_validate(base)


class TestConstraintArgsPolymorphicValidation:
    """Test polymorphic validation of ConstraintArgs via kind discriminator."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("payload", "expected_type"),
        [
            ({"kind": "max_duration", "max_duration": 60}, MaxDurationConstraintArgs),
            ({"kind": "max_requests", "max_num": 100}, MaxRequestsConstraintArgs),
            ({"kind": "max_errors", "max_errors": 5}, MaxErrorsConstraintArgs),
            (
                {"kind": "max_error_rate", "max_error_rate": 0.5},
                MaxErrorRateConstraintArgs,
            ),
            (
                {"kind": "max_global_error_rate", "max_error_rate": 0.3},
                MaxGlobalErrorRateConstraintArgs,
            ),
            (
                {"kind": "over_saturation", "mode": "active"},
                OverSaturationConstraintArgs,
            ),
        ],
    )
    def test_model_validate_dispatches_by_kind(self, payload, expected_type):
        """
        ConstraintArgs.model_validate produces the correct subclass based on kind.

        ## WRITTEN BY AI ##
        """
        result = ConstraintArgs.model_validate(payload)
        assert isinstance(result, expected_type)
        assert result.kind == payload["kind"]

    @pytest.mark.smoke
    def test_unknown_kind_raises_validation_error(self):
        """
        Unknown kind values are rejected during validation.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            ConstraintArgs.model_validate({"kind": "nonexistent_constraint"})

    @pytest.mark.smoke
    def test_missing_required_field_raises_validation_error(self):
        """
        Missing required fields on a known kind raise ValidationError.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            ConstraintArgs.model_validate({"kind": "max_duration"})

    @pytest.mark.smoke
    def test_extra_fields_forbidden(self):
        """
        Extra fields are rejected by the forbid config.

        ## WRITTEN BY AI ##
        """
        with pytest.raises(ValidationError):
            ConstraintArgs.model_validate(
                {"kind": "max_duration", "max_duration": 60, "bogus_field": True}
            )


class TestConstraintArgsToInitializerHelper:
    """Test ConstraintsInitializerFactory.create() helper for each subclass."""

    @pytest.mark.smoke
    def test_max_duration_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces MaxDurationConstraint.

        ## WRITTEN BY AI ##
        """
        args = MaxDurationConstraintArgs(max_duration=300)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxDurationConstraint)
        assert init.args.max_duration == 300

    @pytest.mark.smoke
    def test_max_duration_list_values(self):
        """
        MaxDurationConstraintArgs supports list of durations.

        ## WRITTEN BY AI ##
        """
        args = MaxDurationConstraintArgs(max_duration=[60, 120, 300])
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxDurationConstraint)
        assert init.args.max_duration == [60, 120, 300]

    @pytest.mark.smoke
    def test_max_requests_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces MaxNumberConstraint.

        ## WRITTEN BY AI ##
        """
        args = MaxRequestsConstraintArgs(max_num=1000)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxNumberConstraint)
        assert init.args.max_num == 1000

    @pytest.mark.smoke
    def test_max_errors_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces MaxErrorsConstraint.

        ## WRITTEN BY AI ##
        """
        args = MaxErrorsConstraintArgs(max_errors=10)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxErrorsConstraint)
        assert init.args.max_errors == 10

    @pytest.mark.smoke
    def test_max_error_rate_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces MaxErrorRateConstraint.

        ## WRITTEN BY AI ##
        """
        args = MaxErrorRateConstraintArgs(max_error_rate=0.5, window_size=50)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxErrorRateConstraint)
        assert init.args.max_error_rate == 0.5
        assert init.args.window_size == 50

    @pytest.mark.smoke
    def test_max_error_rate_default_window_size(self):
        """
        MaxErrorRateConstraintArgs uses settings default for window_size.

        ## WRITTEN BY AI ##
        """
        args = MaxErrorRateConstraintArgs(max_error_rate=0.3)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxErrorRateConstraint)
        assert init.args.window_size == 30  # settings default

    @pytest.mark.smoke
    def test_max_global_error_rate_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces MaxGlobalErrorRateConstraint.

        ## WRITTEN BY AI ##
        """
        args = MaxGlobalErrorRateConstraintArgs(max_error_rate=0.2, min_processed=50)
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, MaxGlobalErrorRateConstraint)
        assert init.args.max_error_rate == 0.2
        assert init.args.min_processed == 50

    @pytest.mark.smoke
    def test_over_saturation_to_initializer(self):
        """
        ConstraintsInitializerFactory.create produces
        OverSaturationConstraintInitializer.

        ## WRITTEN BY AI ##
        """
        args = OverSaturationConstraintArgs(
            mode="active", min_seconds=60, max_window_seconds=180
        )
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, OverSaturationConstraintInitializer)
        assert init.args.mode == "active"
        assert init.args.min_seconds == 60
        assert init.args.max_window_seconds == 180

    @pytest.mark.smoke
    def test_over_saturation_defaults(self):
        """
        OverSaturationConstraintArgs uses sensible defaults.

        ## WRITTEN BY AI ##
        """
        args = OverSaturationConstraintArgs()
        init = ConstraintsInitializerFactory.create(args)
        assert isinstance(init, OverSaturationConstraintInitializer)
        assert init.args.mode == "active"
        assert init.args.min_seconds == 30.0
        assert init.args.moe_threshold == 2.0


class TestConstraintArgsConstraintKey:
    """Test constraint_key property for correct dict key mapping."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("args_class", "kwargs", "expected_key"),
        [
            (MaxDurationConstraintArgs, {"max_duration": 60}, "max_duration"),
            (MaxRequestsConstraintArgs, {"max_num": 100}, "max_requests"),
            (MaxErrorsConstraintArgs, {"max_errors": 5}, "max_errors"),
            (MaxErrorRateConstraintArgs, {"max_error_rate": 0.5}, "max_error_rate"),
            (
                MaxGlobalErrorRateConstraintArgs,
                {"max_error_rate": 0.3},
                "max_global_error_rate",
            ),
            (OverSaturationConstraintArgs, {}, "over_saturation"),
        ],
    )
    def test_constraint_key_matches_expected(self, args_class, kwargs, expected_key):
        """
        Each ConstraintArgs subclass reports the correct constraint_key.

        ## WRITTEN BY AI ##
        """
        args = args_class(**kwargs)
        assert args.constraint_key == expected_key


class TestResolveConstraintsTranslation:
    """Test the resolve_constraints translation layer."""

    @pytest.mark.smoke
    def test_constraints_list_resolved(self):
        """
        Constraints list entries are resolved to constraint initializers.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(
            constraints=[
                {"kind": "max_duration", "max_duration": 120},
                {"kind": "max_requests", "max_num": 500},
            ]
        )
        resolved = resolve_constraints(args)

        assert "max_duration" in resolved
        assert "max_requests" in resolved
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert isinstance(resolved["max_requests"], MaxNumberConstraint)
        assert resolved["max_duration"].args.max_duration == 120
        assert resolved["max_requests"].args.max_num == 500

    @pytest.mark.smoke
    def test_no_constraints_returns_empty(self):
        """
        No constraints produces an empty resolved dict.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args()
        resolved = resolve_constraints(args)
        assert resolved == {}

    @pytest.mark.smoke
    def test_explicit_constraints_list(self):
        """
        Explicit constraints list from args is resolved.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(
            constraints=[
                {"kind": "max_duration", "max_duration": 60},
                {"kind": "max_errors", "max_errors": 3},
            ]
        )
        resolved = resolve_constraints(args)

        assert "max_duration" in resolved
        assert "max_errors" in resolved
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)

    @pytest.mark.smoke
    def test_extra_constraints_merged(self):
        """
        Programmatic extra_constraints are merged into the result.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(constraints=[{"kind": "max_duration", "max_duration": 60}])
        resolved = resolve_constraints(args, max_requests=200)
        assert "max_duration" in resolved
        assert "max_requests" in resolved

    @pytest.mark.smoke
    def test_over_saturation_via_constraints_list(self):
        """
        over_saturation passed via constraints list is correctly resolved.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(
            constraints=[
                {"kind": "over_saturation", "mode": "active", "min_seconds": 45}
            ]
        )
        resolved = resolve_constraints(args)

        assert "over_saturation" in resolved
        init = resolved["over_saturation"]
        assert isinstance(init, OverSaturationConstraintInitializer)
        assert init.args.mode == "active"
        assert init.args.min_seconds == 45

    @pytest.mark.smoke
    def test_all_constraint_types_together(self):
        """
        All constraint types are resolved simultaneously via constraints list.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(
            constraints=[
                {"kind": "max_duration", "max_duration": 120},
                {"kind": "max_requests", "max_num": 1000},
                {"kind": "max_errors", "max_errors": 10},
                {"kind": "max_error_rate", "max_error_rate": 0.5},
                {"kind": "max_global_error_rate", "max_error_rate": 0.3},
                {"kind": "over_saturation", "mode": "active"},
            ]
        )
        resolved = resolve_constraints(args)
        assert len(resolved) == 6
        assert set(resolved.keys()) == {
            "max_duration",
            "max_requests",
            "max_errors",
            "max_error_rate",
            "max_global_error_rate",
            "over_saturation",
        }

    @pytest.mark.smoke
    def test_single_constraint_in_list(self):
        """
        A single constraint dict in a list is correctly resolved.

        ## WRITTEN BY AI ##
        """
        args = _minimal_args(constraints=[{"kind": "max_requests", "max_num": 50}])
        assert len(args.constraints) == 1
        resolved = resolve_constraints(args)
        assert "max_requests" in resolved
        assert resolved["max_requests"].args.max_num == 50


class TestConstraintArgsSerialization:
    """Test serialization round-trips for ConstraintArgs."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "payload",
        [
            {"kind": "max_duration", "max_duration": 120},
            {"kind": "max_requests", "max_num": 500},
            {"kind": "max_errors", "max_errors": 10},
            {"kind": "max_error_rate", "max_error_rate": 0.5, "window_size": 20},
            {"kind": "max_global_error_rate", "max_error_rate": 0.3},
            {"kind": "over_saturation", "mode": "active", "min_seconds": 45},
        ],
    )
    def test_model_dump_round_trip(self, payload):
        """
        ConstraintArgs can be serialized and deserialized without data loss.

        ## WRITTEN BY AI ##
        """
        original = ConstraintArgs.model_validate(payload)
        dumped = original.model_dump()
        restored = ConstraintArgs.model_validate(dumped)
        assert type(restored) is type(original)
        assert restored.model_dump() == dumped
