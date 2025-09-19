from __future__ import annotations

from datetime import datetime

import pytest

from guidellm.utils.functions import (
    all_defined,
    safe_add,
    safe_divide,
    safe_format_timestamp,
    safe_getattr,
    safe_multiply,
)


class TestAllDefined:
    """Test suite for all_defined function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ((1, 2, 3), True),
            (("test", "hello"), True),
            ((0, False, ""), True),
            ((1, None, 3), False),
            ((None,), False),
            ((None, None), False),
            ((), True),
        ],
    )
    def test_invocation(self, values, expected):
        """Test all_defined with valid inputs."""
        result = all_defined(*values)
        assert result == expected

    @pytest.mark.sanity
    def test_mixed_types(self):
        """Test all_defined with mixed data types."""
        result = all_defined(1, "test", [], {}, 0.0, False)
        assert result is True

        result = all_defined(1, "test", None, {})
        assert result is False


class TestSafeGetattr:
    """Test suite for safe_getattr function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("obj", "attr", "default", "expected"),
        [
            (None, "any_attr", "default_val", "default_val"),
            (None, "any_attr", None, None),
            ("test_string", "nonexistent", "default_val", "default_val"),
        ],
    )
    def test_invocation(self, obj, attr, default, expected):
        """Test safe_getattr with valid inputs."""
        result = safe_getattr(obj, attr, default)
        assert result == expected

    @pytest.mark.smoke
    def test_with_object(self):
        """Test safe_getattr with actual object attributes."""

        class TestObj:
            test_attr = "test_value"

        obj = TestObj()
        result = safe_getattr(obj, "test_attr", "default")
        assert result == "test_value"

        result = safe_getattr(obj, "missing_attr", "default")
        assert result == "default"

        # Test with method attribute
        result = safe_getattr("test_string", "upper", None)
        assert callable(result)
        assert result() == "TEST_STRING"


class TestSafeDivide:
    """Test suite for safe_divide function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("numerator", "denominator", "num_default", "den_default", "expected"),
        [
            (10, 2, 0.0, 1.0, 5.0),
            (None, 2, 6.0, 1.0, 3.0),
            (10, None, 0.0, 5.0, 2.0),
            (None, None, 8.0, 4.0, 2.0),
            (10, 0, 0.0, 1.0, 10 / 1e-10),
        ],
    )
    def test_invocation(
        self, numerator, denominator, num_default, den_default, expected
    ):
        """Test safe_divide with valid inputs."""
        result = safe_divide(numerator, denominator, num_default, den_default)
        assert result == pytest.approx(expected, rel=1e-6)

    @pytest.mark.sanity
    def test_zero_division_protection(self):
        """Test safe_divide protection against zero division."""
        result = safe_divide(10, 0)
        assert result == 10 / 1e-10

        result = safe_divide(5, None, den_default=0)
        assert result == 5 / 1e-10


class TestSafeMultiply:
    """Test suite for safe_multiply function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("values", "default", "expected"),
        [
            ((2, 3, 4), 1.0, 24.0),
            ((2, None, 4), 1.0, 8.0),
            ((None, None), 5.0, 5.0),
            ((), 3.0, 3.0),
            ((2, 3, None, 5), 2.0, 60.0),
        ],
    )
    def test_invocation(self, values, default, expected):
        """Test safe_multiply with valid inputs."""
        result = safe_multiply(*values, default=default)
        assert result == expected

    @pytest.mark.sanity
    def test_with_zero(self):
        """Test safe_multiply with zero values."""
        result = safe_multiply(2, 0, 3, default=1.0)
        assert result == 0.0

        result = safe_multiply(None, 0, None, default=5.0)
        assert result == 0.0


class TestSafeAdd:
    """Test suite for safe_add function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("values", "signs", "default", "expected"),
        [
            ((1, 2, 3), None, 0.0, 6.0),
            ((1, None, 3), None, 5.0, 9.0),
            ((10, 5), [1, -1], 0.0, 5.0),
            ((None, None), [1, -1], 2.0, 0.0),
            ((), None, 3.0, 3.0),
            ((1, 2, 3), [1, 1, -1], 0.0, 0.0),
        ],
    )
    def test_invocation(self, values, signs, default, expected):
        """Test safe_add with valid inputs."""
        result = safe_add(*values, signs=signs, default=default)
        assert result == expected

    @pytest.mark.sanity
    def test_invalid_signs_length(self):
        """Test safe_add with invalid signs length."""
        with pytest.raises(
            ValueError, match="Length of signs must match length of values"
        ):
            safe_add(1, 2, 3, signs=[1, -1])

    @pytest.mark.sanity
    def test_single_value(self):
        """Test safe_add with single value."""
        result = safe_add(5, default=1.0)
        assert result == 5.0

        result = safe_add(None, default=3.0)
        assert result == 3.0


class TestSafeFormatTimestamp:
    """Test suite for safe_format_timestamp function."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("timestamp", "format_", "default", "expected"),
        [
            (1609459200.0, "%Y-%m-%d", "N/A", "2020-12-31"),
            (1609459200.0, "%H:%M:%S", "N/A", "19:00:00"),
            (None, "%H:%M:%S", "N/A", "N/A"),
            (-1, "%H:%M:%S", "N/A", "N/A"),
            (2**32, "%H:%M:%S", "N/A", "N/A"),
        ],
    )
    def test_invocation(self, timestamp, format_, default, expected):
        """Test safe_format_timestamp with valid inputs."""
        result = safe_format_timestamp(timestamp, format_, default)
        assert result == expected

    @pytest.mark.sanity
    def test_edge_cases(self):
        """Test safe_format_timestamp with edge case timestamps."""
        result = safe_format_timestamp(0.0, "%Y", "N/A")
        assert result == "1969"

        result = safe_format_timestamp(1.0, "%Y", "N/A")
        assert result == "1969"

        result = safe_format_timestamp(2**31 - 1, "%Y", "N/A")
        expected_year = datetime.fromtimestamp(2**31 - 1).strftime("%Y")
        assert result == expected_year

    @pytest.mark.sanity
    def test_invalid_timestamp_ranges(self):
        """Test safe_format_timestamp with invalid timestamp ranges."""
        result = safe_format_timestamp(2**31 + 1, "%Y", "ERROR")
        assert result == "ERROR"

        result = safe_format_timestamp(-1000, "%Y", "ERROR")
        assert result == "ERROR"
