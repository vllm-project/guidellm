"""
Unit tests for JSON unwrapping and DatasetDict handling in
guidellm.data.preprocessors.mappers.

## WRITTEN BY AI ##
"""

from __future__ import annotations

import json

import pytest
from datasets import Dataset, DatasetDict

from guidellm.data.preprocessors.mappers import (
    GenerativeColumnMapper,
    GenerativeColumnMapperArgs,
    _detect_json_wrapper,
    _extract_json_field,
    _resolve_virtual_columns,
    _unwrap_dataset_dict,
)


class TestDetectJsonWrapper:
    """Tests for _detect_json_wrapper helper.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_detects_single_json_string_column(self):
        """A dataset with one column containing JSON dicts is detected.

        ## WRITTEN BY AI ##
        """
        data = {"json": [json.dumps({"messages": [], "tools": []})]}
        ds = Dataset.from_dict(data)
        result = _detect_json_wrapper(ds, ["json"])
        assert result == "json"

    @pytest.mark.sanity
    def test_returns_none_for_multiple_columns(self):
        """Datasets with more than one column are not treated as wrapped.

        ## WRITTEN BY AI ##
        """
        data = {"col_a": ["hello"], "col_b": ["world"]}
        ds = Dataset.from_dict(data)
        result = _detect_json_wrapper(ds, ["col_a", "col_b"])
        assert result is None

    @pytest.mark.sanity
    def test_returns_none_for_non_json_string(self):
        """A single string column that isn't valid JSON returns None.

        ## WRITTEN BY AI ##
        """
        data = {"text": ["this is plain text"]}
        ds = Dataset.from_dict(data)
        result = _detect_json_wrapper(ds, ["text"])
        assert result is None

    @pytest.mark.sanity
    def test_returns_none_for_non_dict_json(self):
        """A JSON column containing a list (not dict) returns None.

        ## WRITTEN BY AI ##
        """
        data = {"json": [json.dumps([1, 2, 3])]}
        ds = Dataset.from_dict(data)
        result = _detect_json_wrapper(ds, ["json"])
        assert result is None


class TestResolveVirtualColumns:
    """Tests for _resolve_virtual_columns helper.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_returns_inner_keys(self):
        """Virtual columns are the keys of the parsed JSON dict.

        ## WRITTEN BY AI ##
        """
        data = {"json": [json.dumps({"messages": [], "tools": [], "metadata": {}})]}
        ds = Dataset.from_dict(data)
        result = _resolve_virtual_columns(ds, "json")
        assert set(result) == {"messages", "tools", "metadata"}


class TestExtractJsonField:
    """Tests for _extract_json_field helper.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_extracts_field_from_json_string(self):
        """Correctly parses JSON and returns the requested field.

        ## WRITTEN BY AI ##
        """
        row = {"json": json.dumps({"messages": [{"role": "user", "content": "hi"}]})}
        result = _extract_json_field(row, "json", "messages")
        assert result == [{"role": "user", "content": "hi"}]

    @pytest.mark.sanity
    def test_returns_none_for_missing_field(self):
        """Returns None when the requested field is not in the JSON.

        ## WRITTEN BY AI ##
        """
        row = {"json": json.dumps({"messages": []})}
        result = _extract_json_field(row, "json", "tools")
        assert result is None

    @pytest.mark.sanity
    def test_returns_none_for_invalid_json(self):
        """Returns None when the wrapper column contains invalid JSON.

        ## WRITTEN BY AI ##
        """
        row = {"json": "not valid json{{{"}
        result = _extract_json_field(row, "json", "messages")
        assert result is None


class TestUnwrapDatasetDict:
    """Tests for _unwrap_dataset_dict helper.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_returns_train_split_from_dict(self):
        """Prefers the 'train' split when available.

        ## WRITTEN BY AI ##
        """
        train_ds = Dataset.from_dict({"col": [1, 2, 3]})
        test_ds = Dataset.from_dict({"col": [4, 5]})
        dd = DatasetDict({"train": train_ds, "test": test_ds})
        result = _unwrap_dataset_dict(dd)
        assert len(result) == 3

    @pytest.mark.sanity
    def test_returns_first_split_when_no_train(self):
        """Falls back to first split if 'train' is not present.

        ## WRITTEN BY AI ##
        """
        val_ds = Dataset.from_dict({"col": [10, 20]})
        dd = DatasetDict({"validation": val_ds})
        result = _unwrap_dataset_dict(dd)
        assert len(result) == 2

    @pytest.mark.sanity
    def test_returns_dataset_unchanged(self):
        """A plain Dataset passes through without modification.

        ## WRITTEN BY AI ##
        """
        ds = Dataset.from_dict({"col": [1]})
        result = _unwrap_dataset_dict(ds)
        assert result is ds


class TestColumnMapperJsonUnwrapping:
    """Integration test for JSON unwrapping through the full column mapper flow.

    ## WRITTEN BY AI ##
    """

    @pytest.mark.smoke
    def test_json_wrapped_dataset_maps_virtual_columns(self):
        """Column mapper resolves virtual columns from a JSON-wrapped dataset.

        ## WRITTEN BY AI ##
        """
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test_fn"}}]
        row = json.dumps({"messages": messages, "tools": tools})
        ds = Dataset.from_dict({"json": [row, row]})

        config = GenerativeColumnMapperArgs(
            column_mappings={"text_column": "messages", "tools_column": "tools"}
        )
        mapper = GenerativeColumnMapper(config)
        mapper.setup_data(datasets=[ds])

        assert mapper._json_wrappers == {0: "json"}

        # Test __call__ extracts the right values
        items = [{"dataset": {"json": row}}]
        result = mapper(items)
        assert len(result) == 1
        assert result[0]["text_column"] == [messages]
        assert result[0]["tools_column"] == [tools]

    @pytest.mark.sanity
    def test_normal_dataset_no_json_wrapper(self):
        """Datasets with proper columns don't trigger JSON unwrapping.

        ## WRITTEN BY AI ##
        """
        ds = Dataset.from_dict(
            {
                "prompt": ["hello", "world"],
                "tools": [None, None],
            }
        )

        config = GenerativeColumnMapperArgs(
            column_mappings={"text_column": "prompt", "tools_column": "tools"}
        )
        mapper = GenerativeColumnMapper(config)
        mapper.setup_data(datasets=[ds])

        assert mapper._json_wrappers == {}

    @pytest.mark.sanity
    def test_dataset_dict_unwrapped_before_mapping(self):
        """A DatasetDict is unwrapped to a single split before column mapping.

        ## WRITTEN BY AI ##
        """
        ds = Dataset.from_dict({"prompt": ["hello"]})
        dd = DatasetDict({"train": ds})

        config = GenerativeColumnMapperArgs(column_mappings={"text_column": "prompt"})
        mapper = GenerativeColumnMapper(config)
        mapper.setup_data(datasets=[dd])

        assert mapper.datasets_column_mappings is not None
