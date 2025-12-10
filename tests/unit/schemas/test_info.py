"""
Unit tests for RequestInfo and RequestTimings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guidellm.schemas import (
    RequestInfo,
    RequestTimings,
    StandardBaseDict,
    StandardBaseModel,
)


class TestRequestTimings:
    """Test cases for RequestTimings model."""

    @pytest.fixture(
        params=[
            {},
            {"targeted_start": 1234567880.0},
            {"queued": 1234567885.0, "dequeued": 1234567886.0},
            {"scheduled_at": 1234567887.0, "resolve_start": 1234567888.0},
            {
                "request_start": 1234567890.0,
                "first_request_iteration": 1234567891.0,
                "first_token_iteration": 1234567892.0,
            },
            {
                "last_token_iteration": 1234567895.0,
                "last_request_iteration": 1234567896.0,
            },
            {
                "request_end": 1234567900.0,
                "resolve_end": 1234567905.0,
                "finalized": 1234567910.0,
            },
            {
                "request_iterations": 5,
                "token_iterations": 10,
            },
            {
                "queued": 1234567885.0,
                "scheduled_at": 1234567887.0,
                "request_start": 1234567890.0,
                "request_end": 1234567900.0,
                "request_iterations": 3,
                "token_iterations": 7,
            },
        ],
        ids=[
            "empty",
            "targeted_start",
            "queue_lifecycle",
            "scheduling",
            "request_start_tokens",
            "request_end_tokens",
            "completion_lifecycle",
            "iteration_counts",
            "full_lifecycle",
        ],
    )
    def valid_instances(self, request):
        """Fixture providing valid RequestTimings instances."""
        constructor_args = request.param
        instance = RequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test RequestTimings inheritance and type relationships."""
        assert issubclass(RequestTimings, StandardBaseDict)
        assert hasattr(RequestTimings, "model_dump")
        assert hasattr(RequestTimings, "model_validate")

        # Check fields are defined
        fields = RequestTimings.model_fields
        expected_fields = [
            "targeted_start",
            "queued",
            "dequeued",
            "scheduled_at",
            "resolve_start",
            "request_start",
            "first_request_iteration",
            "first_token_iteration",
            "last_token_iteration",
            "last_request_iteration",
            "request_iterations",
            "token_iterations",
            "request_end",
            "resolve_end",
            "finalized",
        ]
        for field in expected_fields:
            assert field in fields

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test RequestTimings initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, RequestTimings)

        # Check field values
        for key, expected_value in constructor_args.items():
            assert getattr(instance, key) == expected_value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("targeted_start", "not_float"),
            ("queued", "not_float"),
            ("dequeued", [123]),
            ("scheduled_at", {}),
            ("resolve_start", "invalid"),
            ("request_start", "invalid"),
            ("first_request_iteration", "not_float"),
            ("first_token_iteration", "invalid"),
            ("last_token_iteration", []),
            ("last_request_iteration", "not_float"),
            ("request_iterations", "not_int"),
            ("request_iterations", 3.14),
            ("token_iterations", "not_int"),
            ("token_iterations", 2.5),
            ("request_end", []),
            ("resolve_end", {}),
            ("finalized", "invalid"),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test RequestTimings with invalid field values."""
        with pytest.raises(ValidationError):
            RequestTimings(**{field: value})

    @pytest.mark.smoke
    def test_last_reported_property(self):
        """Test RequestTimings.last_reported property."""
        timings = RequestTimings()
        assert timings.last_reported is None

        # Set timing values and verify last_reported updates
        timings.queued = 1234567890.0
        assert timings.last_reported == 1234567890.0

        timings.dequeued = 1234567891.0
        assert timings.last_reported == 1234567891.0

        timings.scheduled_at = 1234567895.0
        assert timings.last_reported == 1234567895.0

        timings.resolve_start = 1234567896.0
        assert timings.last_reported == 1234567896.0

        timings.request_start = 1234567897.0
        assert timings.last_reported == 1234567897.0

        timings.request_end = 1234567900.0
        assert timings.last_reported == 1234567900.0

        timings.resolve_end = 1234567905.0
        assert timings.last_reported == 1234567905.0

        # Fields not included in last_reported should not affect it
        timings.targeted_start = 1234567999.0
        assert timings.last_reported == 1234567905.0

        timings.finalized = 1234568000.0
        assert timings.last_reported == 1234567905.0

        timings.first_request_iteration = 1234568001.0
        assert timings.last_reported == 1234567905.0

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test RequestTimings serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Verify all expected fields are in the dict
        expected_fields = [
            "targeted_start",
            "queued",
            "dequeued",
            "scheduled_at",
            "resolve_start",
            "request_start",
            "first_request_iteration",
            "first_token_iteration",
            "last_token_iteration",
            "last_request_iteration",
            "request_iterations",
            "token_iterations",
            "request_end",
            "resolve_end",
            "finalized",
        ]
        for field in expected_fields:
            assert field in data_dict

        # Test reconstruction
        reconstructed = RequestTimings.model_validate(data_dict)
        for key, expected_value in constructor_args.items():
            assert getattr(reconstructed, key) == expected_value

        # Verify fields not in constructor_args have correct defaults
        for field in expected_fields:
            if field not in constructor_args:
                value = getattr(reconstructed, field)
                if field in ["request_iterations", "token_iterations"]:
                    assert value == 0
                else:
                    assert value is None


class TestRequestInfo:
    """Test cases for RequestInfo model."""

    @pytest.fixture(
        params=[
            {},
            {"request_id": "test-123", "status": "queued"},
            {
                "request_id": "test-456",
                "status": "completed",
                "scheduler_node_id": 1,
            },
            {
                "status": "in_progress",
                "scheduler_node_id": 2,
                "scheduler_process_id": 3,
            },
            {
                "status": "errored",
                "error": "Test error message",
            },
            {
                "request_id": "test-789",
                "status": "pending",
                "scheduler_node_id": 0,
                "scheduler_process_id": 1,
                "scheduler_start_time": 1234567890.0,
            },
        ],
        ids=[
            "empty",
            "basic",
            "with_node_id",
            "with_process_ids",
            "with_error",
            "full_fields",
        ],
    )
    def valid_instances(self, request):
        """Fixture providing valid RequestInfo instances."""
        constructor_args = request.param
        instance = RequestInfo(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test RequestInfo inheritance and type relationships."""
        assert issubclass(RequestInfo, StandardBaseModel)
        assert hasattr(RequestInfo, "model_dump")
        assert hasattr(RequestInfo, "model_validate")
        assert hasattr(RequestInfo, "model_copy")

        # Check fields
        fields = RequestInfo.model_fields
        expected_fields = [
            "request_id",
            "status",
            "scheduler_node_id",
            "scheduler_process_id",
            "scheduler_start_time",
            "timings",
            "error",
        ]
        for field in expected_fields:
            assert field in fields

        # Check computed properties
        assert hasattr(RequestInfo, "started_at")
        assert hasattr(RequestInfo, "completed_at")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test RequestInfo initialization."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, RequestInfo)

        # Check field values
        for key, expected_value in constructor_args.items():
            assert getattr(instance, key) == expected_value

        # Check defaults
        if "request_id" not in constructor_args:
            assert isinstance(instance.request_id, str)
            assert len(instance.request_id) > 0
        if "status" not in constructor_args:
            assert instance.status == "queued"
        if "scheduler_node_id" not in constructor_args:
            assert instance.scheduler_node_id == -1
        if "scheduler_process_id" not in constructor_args:
            assert instance.scheduler_process_id == -1
        if "scheduler_start_time" not in constructor_args:
            assert instance.scheduler_start_time == -1
        if "timings" not in constructor_args:
            assert isinstance(instance.timings, RequestTimings)
        if "error" not in constructor_args:
            assert instance.error is None

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("status", "invalid_status"),
            ("status", 123),
            ("status", None),
            ("scheduler_node_id", "not_int"),
            ("scheduler_node_id", 3.14),
            ("scheduler_process_id", "not_int"),
            ("scheduler_process_id", []),
            ("scheduler_start_time", "not_float"),
            ("scheduler_start_time", {}),
            ("timings", "not_timings"),
            ("timings", 123),
            ("timings", []),
        ],
    )
    def test_invalid_initialization_values(self, field, value):
        """Test RequestInfo with invalid field values."""
        with pytest.raises(ValidationError):
            RequestInfo(**{field: value})

    @pytest.mark.smoke
    def test_auto_id_generation(self):
        """Test that request_id is auto-generated if not provided."""
        import uuid

        info1 = RequestInfo()
        info2 = RequestInfo()

        assert info1.request_id != info2.request_id
        assert len(info1.request_id) > 0
        assert len(info2.request_id) > 0

        # Should be valid UUIDs
        uuid.UUID(info1.request_id)
        uuid.UUID(info2.request_id)

    @pytest.mark.smoke
    def test_status_values(self):
        """Test RequestInfo with different status values."""
        valid_statuses = [
            "queued",
            "pending",
            "in_progress",
            "completed",
            "errored",
            "cancelled",
        ]

        for status in valid_statuses:
            info = RequestInfo(status=status)
            assert info.status == status

    @pytest.mark.smoke
    def test_started_at_property(self):
        """Test RequestInfo.started_at computed property."""
        info = RequestInfo()
        assert info.started_at is None

        # Set request_start
        info.timings.request_start = 1234567890.0
        assert info.started_at == 1234567890.0

        # Set resolve_start (should be used if request_start is None)
        info2 = RequestInfo()
        info2.timings.resolve_start = 1234567895.0
        assert info2.started_at == 1234567895.0

        # Both set - should prefer request_start
        info3 = RequestInfo()
        info3.timings.resolve_start = 1234567895.0
        info3.timings.request_start = 1234567900.0
        assert info3.started_at == 1234567900.0

    @pytest.mark.smoke
    def test_completed_at_property(self):
        """Test RequestInfo.completed_at computed property."""
        info = RequestInfo()
        assert info.completed_at is None

        # Set request_end
        info.timings.request_end = 1234567900.0
        assert info.completed_at == 1234567900.0

        # Set resolve_end (should be used if request_end is None)
        info2 = RequestInfo()
        info2.timings.resolve_end = 1234567905.0
        assert info2.completed_at == 1234567905.0

        # Both set - should prefer request_end
        info3 = RequestInfo()
        info3.timings.resolve_end = 1234567905.0
        info3.timings.request_end = 1234567910.0
        assert info3.completed_at == 1234567910.0

    @pytest.mark.smoke
    def test_model_copy(self):
        """Test RequestInfo.model_copy creates independent copies."""
        info = RequestInfo(request_id="test-123")
        info.timings.request_start = 1234567890.0

        # Create copy
        copied = info.model_copy()

        # Verify it's a different instance
        assert copied is not info
        assert copied.timings is not info.timings

        # Verify values are the same
        assert copied.request_id == info.request_id
        assert copied.timings.request_start == info.timings.request_start

        # Modify original and verify copy is independent
        info.timings.request_end = 1234567900.0
        assert copied.timings.request_end is None

    @pytest.mark.sanity
    def test_marshalling(self, valid_instances):
        """Test RequestInfo serialization and deserialization."""
        instance, constructor_args = valid_instances
        data_dict = instance.model_dump()
        assert isinstance(data_dict, dict)

        # Verify all expected fields are in the dict
        expected_fields = [
            "request_id",
            "status",
            "scheduler_node_id",
            "scheduler_process_id",
            "scheduler_start_time",
            "timings",
            "error",
        ]
        for field in expected_fields:
            assert field in data_dict

        # Verify timings is a dict
        assert isinstance(data_dict["timings"], dict)

        # Test reconstruction
        reconstructed = RequestInfo.model_validate(data_dict)
        assert isinstance(reconstructed, RequestInfo)
        assert isinstance(reconstructed.timings, RequestTimings)

        for key, expected_value in constructor_args.items():
            assert getattr(reconstructed, key) == expected_value
