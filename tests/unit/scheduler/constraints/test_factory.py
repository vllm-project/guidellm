import time

import pytest

from guidellm.scheduler import (
    Constraint,
    ConstraintsInitializerFactory,
    MaxDurationConstraint,
    MaxErrorRateConstraint,
    MaxErrorsConstraint,
    MaxNumberConstraint,
    SchedulerState,
    SchedulerUpdateAction,
)
from guidellm.schemas import RequestInfo


class TestConstraintsInitializerFactory:
    """Test the ConstraintsInitializerFactory implementation."""

    @pytest.mark.sanity
    def test_unregistered_key_fails(self):
        """Test that unregistered keys raise ValueError."""
        unregistered_key = "nonexistent_constraint"
        assert not ConstraintsInitializerFactory.is_registered(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create(unregistered_key)

        with pytest.raises(
            ValueError, match=f"Unknown constraint initializer key: {unregistered_key}"
        ):
            ConstraintsInitializerFactory.create_constraint(unregistered_key)

    @pytest.mark.smoke
    def test_resolve_mixed_types(self):
        """Test resolve method with mixed constraint types."""
        max_num_constraint = MaxNumberConstraint(max_num=25)
        max_duration_initializer = MaxDurationConstraint(max_duration=120.0)

        mixed_spec = {
            "max_number": max_num_constraint,
            "max_duration": max_duration_initializer,
            "max_errors": {"max_errors": 15},
            "max_error_rate": 0.08,
        }

        resolved = ConstraintsInitializerFactory.resolve(mixed_spec)

        assert len(resolved) == 4
        assert all(isinstance(c, Constraint) for c in resolved.values())
        assert resolved["max_number"] is max_num_constraint
        assert isinstance(resolved["max_duration"], MaxDurationConstraint)
        assert isinstance(resolved["max_errors"], MaxErrorsConstraint)
        assert isinstance(resolved["max_error_rate"], MaxErrorRateConstraint)
        assert resolved["max_error_rate"].max_error_rate == 0.08

    @pytest.mark.sanity
    def test_resolve_with_invalid_key(self):
        """Test that resolve raises ValueError for unregistered keys."""
        invalid_spec = {
            "max_number": {"max_num": 100},
            "invalid_constraint": {"some_param": 42},
        }

        with pytest.raises(
            ValueError, match="Unknown constraint initializer key: invalid_constraint"
        ):
            ConstraintsInitializerFactory.resolve(invalid_spec)

    @pytest.mark.smoke
    def test_functional_constraint_creation(self):
        """Test that created constraints are functionally correct."""
        constraint = ConstraintsInitializerFactory.create_constraint(
            "max_number", max_num=10
        )
        start_time = time.time()
        state = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=5,
            processed_requests=5,
        )
        request = RequestInfo(
            request_id="test-request",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=start_time,
        )

        action = constraint(state, request)
        assert isinstance(action, SchedulerUpdateAction)
        assert action.request_queuing == "continue"
        assert action.request_processing == "continue"

        state_exceeded = SchedulerState(
            node_id=0,
            num_processes=1,
            start_time=start_time,
            created_requests=15,
            processed_requests=15,
        )
        action_exceeded = constraint(state_exceeded, request)
        assert action_exceeded.request_queuing == "stop"
        assert action_exceeded.request_processing == "stop_local"
