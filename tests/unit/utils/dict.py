"""
Unit tests for dictionary utility functions.
"""

from __future__ import annotations

import pytest

from guidellm.utils.dict import deep_filter, deep_update, recursive_key_update


def update_str(string):
    return string + "_updated"


@pytest.mark.smoke
def test_recursive_key_update_updates_keys():
    my_dict = {
        "my_key": {
            "my_nested_key": {"my_double_nested_key": "someValue"},
            "my_other_nested_key": "someValue",
        },
        "my_other_key": "value",
    }
    my_updated_dict = {
        "my_key_updated": {
            "my_nested_key_updated": {"my_double_nested_key_updated": "someValue"},
            "my_other_nested_key_updated": "someValue",
        },
        "my_other_key_updated": "value",
    }
    recursive_key_update(my_dict, update_str)
    assert my_dict == my_updated_dict


def truncate_str_to_ten(string):
    return string[:10]


@pytest.mark.smoke
def test_recursive_key_update_leaves_unchanged_keys():
    my_dict = {
        "my_key": {
            "my_nested_key": {"my_double_nested_key": "someValue"},
            "my_other_nested_key": "someValue",
        },
        "my_other_key": "value",
    }
    my_updated_dict = {
        "my_key": {
            "my_nested_": {"my_double_": "someValue"},
            "my_other_n": "someValue",
        },
        "my_other_k": "value",
    }
    recursive_key_update(my_dict, truncate_str_to_ten)
    assert my_dict == my_updated_dict


@pytest.mark.smoke
def test_recursive_key_update_updates_dicts_in_list():
    my_dict = {
        "my_key": [
            {"my_list_item_key_1": "someValue"},
            {"my_list_item_key_2": "someValue"},
            {"my_list_item_key_3": "someValue"},
        ]
    }
    my_updated_dict = {
        "my_key_updated": [
            {"my_list_item_key_1_updated": "someValue"},
            {"my_list_item_key_2_updated": "someValue"},
            {"my_list_item_key_3_updated": "someValue"},
        ]
    }
    recursive_key_update(my_dict, update_str)
    assert my_dict == my_updated_dict


class TestDeepUpdate:
    """Test suite for deep_update function."""

    @pytest.mark.smoke
    def test_signature_validation(self):
        """
        Test deep_update function signature.

        ### WRITTEN BY AI ###
        """
        # Should accept two dict arguments
        dict1 = {}
        dict2 = {}
        deep_update(dict1, dict2)

    @pytest.mark.smoke
    def test_basic_shallow_merge(self):
        """
        Test basic shallow merge of top-level keys.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        deep_update(dict1, dict2)
        assert dict1 == {"a": 1, "b": 2, "c": 3, "d": 4}

    @pytest.mark.smoke
    def test_overwriting_non_dict_values(self):
        """
        Test that non-dict values are overwritten.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 99, "c": 3}
        deep_update(dict1, dict2)
        assert dict1 == {"a": 1, "b": 99, "c": 3}

    @pytest.mark.sanity
    def test_recursive_merge_nested_dicts(self):
        """
        Test recursive merge of nested dictionaries (KEY TEST).

        This is the critical test that verifies the shallow merge bug is fixed.

        ### WRITTEN BY AI ###
        """
        dict1 = {"outer": {"inner": {"a": 1, "b": 2}}}
        dict2 = {"outer": {"inner": {"b": 99, "c": 3}}}
        deep_update(dict1, dict2)
        # Should preserve "a", update "b", and add "c"
        assert dict1 == {"outer": {"inner": {"a": 1, "b": 99, "c": 3}}}

    @pytest.mark.sanity
    def test_in_place_modification(self):
        """
        Test that deep_update modifies dict1 in-place.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        original_id = id(dict1)
        deep_update(dict1, dict2)
        assert id(dict1) == original_id
        assert dict1 == {"a": 1, "b": 2}

    @pytest.mark.sanity
    def test_empty_dict2(self):
        """
        Test updating with an empty dictionary.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": 1, "b": 2}
        dict2 = {}
        deep_update(dict1, dict2)
        assert dict1 == {"a": 1, "b": 2}

    @pytest.mark.sanity
    def test_empty_dict1(self):
        """
        Test updating an empty dictionary.

        ### WRITTEN BY AI ###
        """
        dict1 = {}
        dict2 = {"a": 1, "b": 2}
        deep_update(dict1, dict2)
        assert dict1 == {"a": 1, "b": 2}

    @pytest.mark.sanity
    def test_multiple_nesting_levels(self):
        """
        Test deep merge with 3+ levels of nesting.

        ### WRITTEN BY AI ###
        """
        dict1 = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        dict2 = {"level1": {"level2": {"level3": {"b": 99, "c": 3}}}}
        deep_update(dict1, dict2)
        assert dict1 == {"level1": {"level2": {"level3": {"a": 1, "b": 99, "c": 3}}}}

    @pytest.mark.regression
    def test_mixed_dict_and_non_dict_replacement(self):
        """
        Test that dict value replaced by non-dict value.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": {"nested": "value"}}
        dict2 = {"a": "scalar"}
        deep_update(dict1, dict2)
        assert dict1 == {"a": "scalar"}

    @pytest.mark.regression
    def test_non_dict_replaced_by_dict(self):
        """
        Test that non-dict value replaced by dict value.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": "scalar"}
        dict2 = {"a": {"nested": "value"}}
        deep_update(dict1, dict2)
        assert dict1 == {"a": {"nested": "value"}}

    @pytest.mark.regression
    def test_none_value_handling(self):
        """
        Test that None values are handled correctly.

        ### WRITTEN BY AI ###
        """
        dict1 = {"a": 1, "b": None}
        dict2 = {"b": 2, "c": None}
        deep_update(dict1, dict2)
        assert dict1 == {"a": 1, "b": 2, "c": None}

    @pytest.mark.regression
    def test_complex_real_world_scenario(self):
        """
        Test complex nested structure matching real-world use case.

        This simulates the GenerationRequestArguments.model_combine use case.

        ### WRITTEN BY AI ###
        """
        dict1 = {
            "headers": {"Authorization": "Bearer token1", "User-Agent": "client/1.0"},
            "body": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.5, "top_p": 0.9},
                "options": {"timeout": 30},
            },
        }
        dict2 = {
            "headers": {"Content-Type": "application/json"},
            "body": {
                "parameters": {"temperature": 0.7, "max_tokens": 100},
                "options": {"retry": True},
            },
        }
        deep_update(dict1, dict2)

        expected = {
            "headers": {
                "Authorization": "Bearer token1",
                "User-Agent": "client/1.0",
                "Content-Type": "application/json",
            },
            "body": {
                "model": "gpt-4",
                "parameters": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100},
                "options": {"timeout": 30, "retry": True},
            },
        }
        assert dict1 == expected


class TestDeepFilter:
    """Test suite for deep_filter function."""

    @pytest.mark.smoke
    def test_signature_validation(self):
        """
        Test deep_filter function signature.

        ### WRITTEN BY AI ###
        """
        # Should accept a dict and a predicate function
        d = {}
        deep_filter(d, lambda _k, _v: True)

    @pytest.mark.smoke
    def test_basic_filtering_none(self):
        """
        Test basic filtering with None predicate.

        ### WRITTEN BY AI ###
        """
        d = {"a": 1, "b": None, "c": 3}
        deep_filter(d, lambda k, v: v is not None)
        assert d == {"a": 1, "c": 3}

    @pytest.mark.smoke
    def test_filter_by_value_type(self):
        """
        Test filtering by value type.

        ### WRITTEN BY AI ###
        """
        d = {"a": 1, "b": "string", "c": 2, "d": "text"}
        deep_filter(d, lambda _k, v: isinstance(v, int))
        assert d == {"a": 1, "c": 2}

    @pytest.mark.sanity
    def test_filter_by_key(self):
        """
        Test filtering by key name.

        ### WRITTEN BY AI ###
        """
        d = {"keep_a": 1, "remove_b": 2, "keep_c": 3}
        deep_filter(d, lambda k, _v: str(k).startswith("keep"))
        assert d == {"keep_a": 1, "keep_c": 3}

    @pytest.mark.sanity
    def test_recursive_filtering_nested_dicts(self):
        """
        Test recursive filtering in nested dictionaries (KEY TEST).

        ### WRITTEN BY AI ###
        """
        d = {
            "outer": {
                "keep": 1,
                "remove": None,
                "inner": {"a": 2, "b": None, "c": 3},
            },
            "top": None,
        }
        deep_filter(d, lambda k, v: v is not None)
        assert d == {"outer": {"keep": 1, "inner": {"a": 2, "c": 3}}}

    @pytest.mark.sanity
    def test_in_place_modification(self):
        """
        Test that deep_filter modifies dict in-place.

        ### WRITTEN BY AI ###
        """
        d = {"a": 1, "b": None}
        original_id = id(d)
        deep_filter(d, lambda _k, v: v is not None)
        assert id(d) == original_id
        assert d == {"a": 1}

    @pytest.mark.sanity
    def test_empty_dictionary(self):
        """
        Test filtering an empty dictionary.

        ### WRITTEN BY AI ###
        """
        d = {}
        deep_filter(d, lambda _k, v: v is not None)
        assert d == {}

    @pytest.mark.sanity
    def test_all_entries_filtered_out(self):
        """
        Test when all entries are filtered out.

        ### WRITTEN BY AI ###
        """
        d = {"a": None, "b": None, "c": None}
        deep_filter(d, lambda _k, v: v is not None)
        assert d == {}

    @pytest.mark.sanity
    def test_no_entries_filtered(self):
        """
        Test when no entries match the filter.

        ### WRITTEN BY AI ###
        """
        d = {"a": 1, "b": 2, "c": 3}
        deep_filter(d, lambda _k, v: v is not None)
        assert d == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.regression
    def test_nested_dict_becomes_empty(self):
        """
        Test that nested dict becomes empty after filtering.

        ### WRITTEN BY AI ###
        """
        d = {"outer": {"a": None, "b": None}, "keep": 1}
        deep_filter(d, lambda _k, v: v is not None)
        # Empty nested dicts are preserved (they're dicts, not filtered values)
        assert d == {"outer": {}, "keep": 1}

    @pytest.mark.regression
    def test_multiple_nesting_levels(self):
        """
        Test filtering with 3+ levels of nesting.

        ### WRITTEN BY AI ###
        """
        d = {
            "level1": {
                "level2": {"level3": {"a": 1, "b": None}, "c": None},
                "d": 2,
            },
            "e": None,
        }
        deep_filter(d, lambda _k, v: v is not None)
        assert d == {"level1": {"level2": {"level3": {"a": 1}}, "d": 2}}

    @pytest.mark.regression
    def test_complex_predicate_scenarios(self):
        """
        Test with complex predicates.

        ### WRITTEN BY AI ###
        """
        # Filter by both key and value
        d = {"a": 1, "b": 2, "c": 3, "d": 4}
        deep_filter(d, lambda k, v: k != "b" and v > 1)
        assert d == {"c": 3, "d": 4}

    @pytest.mark.regression
    def test_real_world_openai_request_body(self):
        """
        Test with real-world OpenAI request body scenario.

        This simulates the OpenAI HTTP backend use case.

        ### WRITTEN BY AI ###
        """
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": None,
            "stream": None,
            "frequency_penalty": 0.0,
        }
        deep_filter(body, lambda _k, v: v is not None)
        assert body == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "frequency_penalty": 0.0,
        }

    @pytest.mark.regression
    def test_lists_are_not_filtered(self):
        """
        Test that lists containing None are not filtered.

        Per docstring: "Does not filter lists."

        ### WRITTEN BY AI ###
        """
        d = {
            "array": [1, None, 3],
            "remove": None,
            "nested": {"list": [None, "keep"]},
        }
        deep_filter(d, lambda _k, v: v is not None)
        # Lists should remain unchanged
        assert d == {
            "array": [1, None, 3],
            "nested": {"list": [None, "keep"]},
        }

    @pytest.mark.regression
    def test_falsy_values_preserved(self):
        """
        Test that falsy values (except None) are preserved.

        ### WRITTEN BY AI ###
        """
        d = {
            "zero": 0,
            "false": False,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "none": None,
        }
        deep_filter(d, lambda _k, v: v is not None)
        assert d == {
            "zero": 0,
            "false": False,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
        }
