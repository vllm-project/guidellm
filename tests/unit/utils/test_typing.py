"""
Test suite for the typing utilities module.
"""

from typing import Annotated, Literal, TypeAlias

import pytest

from guidellm.utils.typing import get_literal_vals

# Local type definitions to avoid imports from other modules
LocalProfileType = Literal["synchronous", "async", "concurrent", "throughput", "sweep"]
LocalStrategyType = Annotated[
    Literal["synchronous", "concurrent", "throughput", "constant", "poisson"],
    "Valid strategy type identifiers for scheduling request patterns",
]
StrategyProfileType: TypeAlias = LocalStrategyType | LocalProfileType


class TestGetLiteralVals:
    """Test cases for the get_literal_vals function."""

    @pytest.mark.sanity
    def test_profile_type(self):
        """
        Test extracting values from ProfileType.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(LocalProfileType)
        expected = frozenset(
            {"synchronous", "async", "concurrent", "throughput", "sweep"}
        )
        assert result == expected

    @pytest.mark.sanity
    def test_strategy_type(self):
        """
        Test extracting values from StrategyType.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(LocalStrategyType)
        expected = frozenset(
            {"synchronous", "concurrent", "throughput", "constant", "poisson"}
        )
        assert result == expected

    @pytest.mark.smoke
    def test_inline_union_type(self):
        """
        Test extracting values from inline union of ProfileType | StrategyType.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(LocalProfileType | LocalStrategyType)
        expected = frozenset(
            {
                "synchronous",
                "async",
                "concurrent",
                "throughput",
                "constant",
                "poisson",
                "sweep",
            }
        )
        assert result == expected

    @pytest.mark.smoke
    def test_type_alias(self):
        """
        Test extracting values from type alias union.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(StrategyProfileType)
        expected = frozenset(
            {
                "synchronous",
                "async",
                "concurrent",
                "throughput",
                "constant",
                "poisson",
                "sweep",
            }
        )
        assert result == expected

    @pytest.mark.sanity
    def test_single_literal(self):
        """
        Test extracting values from single Literal type.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(Literal["test"])
        expected = frozenset({"test"})
        assert result == expected

    @pytest.mark.sanity
    def test_multi_literal(self):
        """
        Test extracting values from multi-value Literal type.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(Literal["test", "test2"])
        expected = frozenset({"test", "test2"})
        assert result == expected

    @pytest.mark.smoke
    def test_literal_union(self):
        """
        Test extracting values from union of Literal types.

        ### WRITTEN BY AI ###
        """
        result = get_literal_vals(Literal["test", "test2"] | Literal["test3"])
        expected = frozenset({"test", "test2", "test3"})
        assert result == expected
