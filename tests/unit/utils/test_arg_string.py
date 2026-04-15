"""
Tests for arg string parsing utilities.
"""

import pytest

from guidellm.utils import arg_string


class TestArgStringParser:
    """Test cases for ArgStringParser class."""

    @pytest.mark.smoke
    def test_parser_decode_simple(self):
        """
        Test using ArgStringParser.decode() directly with simple input.

        ### WRITTEN BY AI ###
        """
        parser = arg_string.ArgStringParser()
        result = parser.decode("key=value,count=42")
        assert result == {"key": "value", "count": 42}

    @pytest.mark.sanity
    def test_parser_decode_nested(self):
        """
        Test using ArgStringParser.decode() with nested structures.

        ### WRITTEN BY AI ###
        """
        parser = arg_string.ArgStringParser()
        result = parser.decode("items[0].name=apple,items[0].count=5")
        assert result == {"items": [{"name": "apple", "count": 5}]}

    @pytest.mark.sanity
    def test_parser_custom_fill_value(self):
        """
        Test ArgStringParser with custom fill_value for sparse lists.

        ### WRITTEN BY AI ###
        """
        parser = arg_string.ArgStringParser(fill_value=lambda: 0)
        result = parser.decode("items[5]=value")
        assert result == {"items": [0, 0, 0, 0, 0, "value"]}


class TestArgStringLoads:
    """Test cases for arg_string.loads function."""

    @pytest.mark.smoke
    def test_loads_simple(self):
        """
        Test using arg_string.loads with simple input.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("key=value,count=42")
        assert result == {"key": "value", "count": 42}

    @pytest.mark.sanity
    def test_loads_nested(self):
        """
        Test using arg_string.loads with nested structures.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("items[0].name=apple,items[0].count=5")
        assert result == {"items": [{"name": "apple", "count": 5}]}

    @pytest.mark.sanity
    def test_loads_with_fill_value(self):
        """
        Test arg_string.loads with custom fill_value parameter.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("items[3]=value", fill_value=lambda: "empty")
        assert result == {"items": ["empty", "empty", "empty", "value"]}

    @pytest.mark.sanity
    def test_loads_with_custom_class(self):
        """
        Test arg_string.loads with custom decoder class.

        ### WRITTEN BY AI ###
        """

        class CustomParser(arg_string.ArgStringParser):
            def decode(self, s):
                result = super().decode(s)
                # Add a custom metadata field
                result["_custom"] = True
                return result

        result = arg_string.loads("a=1,b=2", cls=CustomParser)
        assert result == {"a": 1, "b": 2, "_custom": True}

    @pytest.mark.smoke
    def test_simple_key_value_pairs(self):
        """
        Test parsing simple key=value pairs with type conversion.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("prompt_tokens=10,output_tokens=30")
        assert result == {"prompt_tokens": 10, "output_tokens": 30}

    @pytest.mark.smoke
    def test_nested_dictionaries(self):
        """
        Test parsing nested dictionaries using dot notation.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("prompt.tokens=15,output.tokens=45")
        assert result == {"prompt": {"tokens": 15}, "output": {"tokens": 45}}

    @pytest.mark.smoke
    def test_list_indexing(self):
        """
        Test parsing list items with bracket notation.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("fruit[0]=apple,fruit[1]=orange")
        assert result == {"fruit": ["apple", "orange"]}

    @pytest.mark.smoke
    def test_sparse_list_with_none_filling(self):
        """
        Test parsing sparse lists with None values filling gaps.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("fruit[0]=apple,fruit[3]=orange")
        assert result == {"fruit": ["apple", None, None, "orange"]}

    @pytest.mark.smoke
    def test_combined_list_and_nested_dict(self):
        """
        Test parsing complex structures combining lists and nested dicts.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads(
            "prefix_buckets[0].bucket_weight=10,"
            "prefix_buckets[0].prefix_count=10,"
            "prefix_buckets[1].bucket_weight=20,"
            "prefix_buckets[1].prefix_count=4",
        )
        expected = {
            "prefix_buckets": [
                {"bucket_weight": 10, "prefix_count": 10},
                {"bucket_weight": 20, "prefix_count": 4},
            ]
        }
        assert result == expected

    @pytest.mark.sanity
    def test_type_conversion_int(self):
        """
        Test that integer strings are converted to int type.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("count=42")
        assert result == {"count": 42}
        assert isinstance(result["count"], int)

    @pytest.mark.sanity
    def test_type_conversion_float(self):
        """
        Test that float strings are converted to float type.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("temperature=0.7")
        assert result == {"temperature": 0.7}
        assert isinstance(result["temperature"], float)

    @pytest.mark.sanity
    def test_type_conversion_bool_true(self):
        """
        Test that 'true' is converted to bool True.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("enabled=true")
        assert result == {"enabled": True}
        assert isinstance(result["enabled"], bool)

    @pytest.mark.sanity
    def test_type_conversion_bool_false(self):
        """
        Test that 'false' is converted to bool False.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("enabled=false")
        assert result == {"enabled": False}
        assert isinstance(result["enabled"], bool)

    @pytest.mark.sanity
    def test_type_conversion_bool_case_insensitive(self):
        """
        Test that boolean conversion is case-insensitive.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("a=True,b=FALSE,c=true")
        assert result == {"a": True, "b": False, "c": True}

    @pytest.mark.sanity
    def test_type_conversion_string(self):
        """
        Test that non-numeric strings remain as strings.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("name=hello")
        assert result == {"name": "hello"}
        assert isinstance(result["name"], str)

    @pytest.mark.sanity
    def test_empty_string(self):
        """
        Test parsing an empty string returns empty dict.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("")
        assert result == {}

    @pytest.mark.sanity
    def test_whitespace_only(self):
        """
        Test parsing whitespace-only string returns empty dict.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("   ")
        assert result == {}

    @pytest.mark.sanity
    def test_whitespace_handling(self):
        """
        Test that whitespace around keys and values is stripped.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads(" key1 = value1 , key2 = value2 ")
        assert result == {"key1": "value1", "key2": "value2"}

    @pytest.mark.sanity
    def test_deep_nesting(self):
        """
        Test parsing deeply nested dictionary structures.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("a.b.c.d=value")
        assert result == {"a": {"b": {"c": {"d": "value"}}}}

    @pytest.mark.sanity
    def test_multiple_lists(self):
        """
        Test parsing multiple separate lists.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("colors[0]=red,colors[1]=blue,sizes[0]=small")
        assert result == {
            "colors": ["red", "blue"],
            "sizes": ["small"],
        }

    @pytest.mark.sanity
    def test_overwrite_error_dict_key(self):
        """
        Test that overwriting a dict key raises an error by default.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at key 'key'",
        ):
            arg_string.loads("key=first,key=second")

    @pytest.mark.sanity
    def test_overwrite_error_nested_dict_key(self):
        """
        Test that overwriting a nested dict key raises an error by default.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at key 'b'",
        ):
            arg_string.loads("a.b=1,a.b=2")

    @pytest.mark.sanity
    def test_overwrite_error_list_value(self):
        """
        Test that overwriting a list value raises an error by default.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at 'items\\[0\\]'",
        ):
            arg_string.loads("items[0]=first,items[0]=second")

    @pytest.mark.regression
    def test_mixed_value_types(self):
        """
        Test parsing with mixed value types in a single string.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("count=42,temperature=0.7,enabled=true,name=test")
        assert result == {
            "count": 42,
            "temperature": 0.7,
            "enabled": True,
            "name": "test",
        }

    @pytest.mark.regression
    def test_complex_nested_structure(self):
        """
        Test parsing complex nested structure with multiple levels.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads(
            "config.server[0].host=localhost,"
            "config.server[0].port=8080,"
            "config.server[1].host=remote,"
            "config.server[1].port=9090"
        )
        expected = {
            "config": {
                "server": [
                    {"host": "localhost", "port": 8080},
                    {"host": "remote", "port": 9090},
                ]
            }
        }
        assert result == expected

    @pytest.mark.regression
    def test_value_with_equals_sign(self):
        """
        Test parsing values that contain equals signs.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("equation=a=b+c")
        assert result == {"equation": "a=b+c"}

    @pytest.mark.regression
    def test_numeric_string_key(self):
        """
        Test that numeric keys remain as strings in dict keys.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("123=value")
        assert result == {"123": "value"}

    @pytest.mark.regression
    def test_underscore_in_keys(self):
        """
        Test that underscores in keys are preserved.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("my_key_name=value")
        assert result == {"my_key_name": "value"}

    @pytest.mark.regression
    def test_list_with_missing_first_element(self):
        """
        Test creating a list starting at index > 0.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("items[2]=third")
        assert result == {"items": [None, None, "third"]}

    @pytest.mark.regression
    def test_nested_dict_in_list_element(self):
        """
        Test nested dictionaries within list elements.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("items[0].config.name=first,items[0].config.value=10")
        assert result == {"items": [{"config": {"name": "first", "value": 10}}]}

    @pytest.mark.regression
    def test_invalid_pair_no_equals(self):
        """
        Test that pairs without '=' raise an error.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Invalid key=value pair",
        ):
            arg_string.loads("invalidpair,key=value2")

    @pytest.mark.regression
    def test_invalid_key_pair_skip(self):
        """
        Test that pairs with invalid keys are skipped
        when skip_invalid is True.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads(
            "valid_key=value,invalid-pair,key2=value2",
            skip_invalid=True,
        )
        assert result == {"valid_key": "value", "key2": "value2"}

    @pytest.mark.regression
    def test_empty_key(self):
        """
        Test handling of empty keys

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Empty key in pair",
        ):
            arg_string.loads("=value,key=value2")

    @pytest.mark.regression
    def test_empty_key_skip(self):
        """
        Test that empty keys are skipped when skip_invalid is True.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("=value,key=value2", skip_invalid=True)
        assert result == {"key": "value2"}

    @pytest.mark.regression
    def test_empty_value(self):
        """
        Test handling of empty values.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("key=")
        assert result == {"key": None}

    @pytest.mark.regression
    def test_special_characters_in_string_value(self):
        """
        Test that special characters in string values are preserved.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("path=/usr/local/bin,url=http://example.com")
        assert result == {
            "path": "/usr/local/bin",
            "url": "http://example.com",
        }

    @pytest.mark.regression
    def test_overwrite_error_replace_value_with_dict(self):
        """
        Test that replacing a raw value with a dict raises an error.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-dict value at 'key' with dict",
        ):
            arg_string.loads("key=value,key.subkey=other")

    @pytest.mark.regression
    def test_overwrite_error_replace_value_with_list(self):
        """
        Test that replacing a raw value with a list raises an error.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-list value at 'key' with list",
        ):
            arg_string.loads("key=value,key[0]=item")

    @pytest.mark.regression
    def test_overwrite_error_replace_list_value_with_dict(self):
        """
        Test that replacing a list value with a dict raises an error.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-dict value at 'items\\[0\\]' with dict",
        ):
            arg_string.loads("items[0]=value,items[0].field=other")

    @pytest.mark.regression
    def test_no_overwrite_error_for_fill_values(self):
        """
        Test that setting values in sparse lists doesn't raise overwrite errors.

        ### WRITTEN BY AI ###
        """
        # This should NOT raise an error because items[0-4] are fill values
        result = arg_string.loads("items[5]=value")
        assert result == {"items": [None, None, None, None, None, "value"]}

    @pytest.mark.regression
    def test_no_overwrite_error_for_fill_values_with_nested(self):
        """
        Test that navigating through fill values doesn't raise overwrite errors.

        ### WRITTEN BY AI ###
        """
        # This should NOT raise an error because items[0] is a fill value
        result = arg_string.loads("items[2].field=value")
        assert result == {"items": [None, None, {"field": "value"}]}

    @pytest.mark.regression
    def test_allow_overwrite_false_dict_key(self):
        """
        Test that allow_overwrite=False raises error for dict key overwrites.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at key 'key'",
        ):
            arg_string.loads("key=first,key=second", allow_overwrite=False)

    @pytest.mark.regression
    def test_allow_overwrite_false_nested_dict_key(self):
        """
        Test that allow_overwrite=False raises error for nested dict overwrites.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at key 'b'",
        ):
            arg_string.loads("a.b=1,a.b=2", allow_overwrite=False)

    @pytest.mark.regression
    def test_allow_overwrite_false_list_value(self):
        """
        Test that allow_overwrite=False raises error for list value overwrites.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite existing value at 'items\\[0\\]'",
        ):
            arg_string.loads("items[0]=first,items[0]=second", allow_overwrite=False)

    @pytest.mark.regression
    def test_allow_overwrite_false_value_to_dict(self):
        """
        Test that allow_overwrite=False raises error for value to dict conversion.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-dict value at 'key' with dict",
        ):
            arg_string.loads("key=value,key.subkey=other", allow_overwrite=False)

    @pytest.mark.regression
    def test_allow_overwrite_false_value_to_list(self):
        """
        Test that allow_overwrite=False raises error for value to list conversion.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-list value at 'key' with list",
        ):
            arg_string.loads("key=value,key[0]=item", allow_overwrite=False)

    @pytest.mark.regression
    def test_allow_overwrite_false_list_value_to_dict(self):
        """
        Test that allow_overwrite=False raises error for list value to dict.

        ### WRITTEN BY AI ###
        """
        with pytest.raises(
            arg_string.ArgStringParseError,
            match="Cannot overwrite non-dict value at 'items\\[0\\]' with dict",
        ):
            arg_string.loads(
                "items[0]=value,items[0].field=other", allow_overwrite=False
            )

    @pytest.mark.regression
    def test_overwrite_value(self):
        """
        Test that later values overwrite earlier ones for the same key.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("key=first,key=second", allow_overwrite=True)
        assert result == {"key": "second"}

    @pytest.mark.regression
    def test_overwrite_in_nested_structure(self):
        """
        Test overwriting values in nested structures.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads("a.b=1,a.b=2", allow_overwrite=True)
        assert result == {"a": {"b": 2}}

    @pytest.mark.regression
    def test_overwrite_in_list(self):
        """
        Test overwriting values in list elements.

        ### WRITTEN BY AI ###
        """
        result = arg_string.loads(
            "items[0]=first,items[0]=second", allow_overwrite=True
        )
        assert result == {"items": ["second"]}
