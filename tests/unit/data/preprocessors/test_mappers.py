"""
Unit tests for GenerativeColumnMapper column resolution.

### WRITTEN BY AI ###
"""

from __future__ import annotations

import pytest
from datasets import Dataset

from guidellm.data.preprocessors import GenerativeColumnMapper


def _dataset(*columns: str) -> Dataset:
    return Dataset.from_dict({column: ["x"] for column in columns})


class TestDatasetsMappings:
    """Test suite for GenerativeColumnMapper.datasets_mappings."""

    @pytest.mark.smoke
    def test_default_names_follow_priority_not_column_order(self):
        """
        An earlier default name wins even when a later one comes first in the
        dataset. Open-Platypus lists `input` (empty for most rows) before
        `instruction`, and `instruction` is the higher priority default.

        ### WRITTEN BY AI ###
        """
        dataset = _dataset("input", "output", "instruction", "data_source")

        mappings = GenerativeColumnMapper.datasets_mappings([dataset])

        assert mappings[("text_column", 0)] == [(0, "instruction")]

    @pytest.mark.sanity
    def test_explicit_names_follow_priority_not_column_order(self):
        """
        Candidate names passed in `input_mappings` are tried in the order given.

        ### WRITTEN BY AI ###
        """
        dataset = _dataset("alpha", "beta")

        mappings = GenerativeColumnMapper.datasets_mappings(
            [dataset], {"text_column": ["beta", "alpha"]}
        )

        assert mappings[("text_column", 0)] == [(0, "beta")]

    @pytest.mark.sanity
    def test_lower_priority_name_used_when_higher_is_absent(self):
        """
        With no higher-priority column present, the next candidate is used.

        ### WRITTEN BY AI ###
        """
        dataset = _dataset("input", "output")

        mappings = GenerativeColumnMapper.datasets_mappings([dataset])

        assert mappings[("text_column", 0)] == [(0, "input")]

    @pytest.mark.sanity
    def test_turn_suffixed_columns_still_resolve_in_priority_order(self):
        """
        Turn-suffixed and plural variants of the winning name are still found.

        ### WRITTEN BY AI ###
        """
        dataset = _dataset("input", "prompt-1", "prompt-0")

        mappings = GenerativeColumnMapper.datasets_mappings([dataset])

        assert mappings[("text_column", 0)] == [(0, "prompt-0")]
        assert mappings[("text_column", 1)] == [(0, "prompt-1")]

    @pytest.mark.sanity
    def test_no_matching_column_is_omitted(self):
        """
        A column type with no matching column is left out of the result.

        ### WRITTEN BY AI ###
        """
        dataset = _dataset("foo", "bar")

        assert GenerativeColumnMapper.datasets_mappings([dataset]) == {}
