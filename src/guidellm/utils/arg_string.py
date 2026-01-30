"""
Utilities for parsing argument strings into Python dictionaries.

Provides functions to parse comma-separated key=value strings with support for
simple key-value pairs, nested dictionaries using dot notation, list indexing
with bracket notation, and combined syntax for complex nested structures.
Automatically converts values to appropriate Python types (int, float, bool, str).

Usage:
::
    from guidellm.utils import arg_string
    result = arg_string.loads("prompt_tokens=10,output_tokens=30")
    # {'prompt_tokens': 10, 'output_tokens': 30}
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import yaml

__all__ = [
    "ArgStringParseError",
    "ArgStringParser",
    "loads",
]


class ArgStringParseError(Exception):
    """Exception raised for errors during argument string parsing."""


class ArgStringParser:
    """
    Decoder for converting argument strings to Python dictionaries.

    Parses comma-separated key=value strings with support for nested dictionaries
    (dot notation), list indexing (bracket notation), and automatic type conversion
    using YAML safe_load. Similar to json.JSONDecoder.

    Example:
    ::
        parser = ArgStringParser()
        result = parser.decode("prompt_tokens=10,output_tokens=30")
        # {'prompt_tokens': 10, 'output_tokens': 30}

        # Custom fill value for sparse lists
        parser = ArgStringParser(fill_value=lambda: 0)
        result = parser.decode("items[5]=value")
        # {'items': [0, 0, 0, 0, 0, 'value']}
    """

    def __init__(
        self,
        skip_invalid: bool = False,
        fill_value: Callable[[], Any] | None = None,
        allow_overwrite: bool = False,
    ):
        """
        Initialize the argument string parser.

        :param skip_invalid: If True, skips invalid key=value pairs rather than raising
        :param fill_value: Callable that returns fill value for sparse lists.
            Defaults to lambda: None.
        :param allow_overwrite: If True, allows overwriting existing values.
            Defaults to False, which raises an error on overwrites.
        """
        self.skip_invalid = skip_invalid
        self.fill_value = fill_value() if fill_value is not None else None
        self.allow_overwrite = allow_overwrite

    def decode(self, s: str) -> dict[str, Any]:
        """
        Decode an arg string into a Python dictionary with automatic type conversion.

        Converts comma-separated key=value pairs into nested dictionary structures.
        Supports dot notation for nested dictionaries (parent.child=value), bracket
        notation for lists (items[0]=value), and combined syntax.
        Values are parsed using YAML safe_load for automatic type conversion.

        :param s: Comma-separated list of key=value pairs supporting nested
            structures via dot and bracket notation
        :return: Parsed dictionary with appropriate types and nesting
        :raises ArgStringParseError: If the string format is invalid

        Example:
        ::
            parser = ArgStringParser()

            parser.decode("prompt_tokens=10,output_tokens=30")
            # {'prompt_tokens': 10, 'output_tokens': 30}

            parser.decode("prompt.tokens=15,output.tokens=45")
            # {'prompt': {'tokens': 15}, 'output': {'tokens': 45}}

            parser.decode("fruit[0]=apple,fruit[1]=orange")
            # {'fruit': ['apple', 'orange']}

            parser.decode("fruit[0]=apple,fruit[3]=orange")
            # {'fruit': ['apple', None, None, 'orange']}

            parser.decode("items[0].weight=10,items[0].count=5")
            # {'items': [{'weight': 10, 'count': 5}]}
        """
        result: dict[str, Any] = {}

        if not s or not s.strip():
            return result

        # Split by commas
        pairs = s.split(",")

        for pair_raw in pairs:
            pair_stripped = pair_raw.strip()
            # Handle invalid pairs
            if not pair_stripped or "=" not in pair_stripped:
                if self.skip_invalid:
                    continue
                raise ArgStringParseError(f"Invalid key=value pair: '{pair_raw}'")

            key, value = pair_stripped.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Handle empty keys
            if not key:
                if self.skip_invalid:
                    continue
                raise ArgStringParseError(f"Empty key in pair: '{pair_raw}'")

            # Parse the key into segments
            segments = self._parse_key(key)

            # Convert value to appropriate type
            parsed_value = yaml.safe_load(value)

            # Set the value in the nested structure
            self._value_set_nested(result, segments, parsed_value)

        return result

    def _parse_key(self, key: str) -> list[tuple[str, int | None]]:
        """
        Parse a key string into segments with optional array indices.

        Splits a key by dots and identifies array access patterns using
        bracket notation. Returns a list of (name, index) tuples where index
        is None for dictionary keys and an integer for list indices.

        :param key: Key string to parse, may contain dots and brackets
        :return: List of (name, index) tuples representing the parsed key segments
        """
        segments: list[tuple[str, int | None]] = []

        # Split by dots first
        parts = key.split(".")

        for part in parts:
            # Check if part has array notation: name[index]
            match = re.match(r"^([^\[]+)\[(\d+)\]$", part)
            if match:
                name = match.group(1)
                index = int(match.group(2))
                segments.append((name, index))
            else:
                segments.append((part, None))

        return segments

    def _list_ensure_size(self, lst: list, index: int) -> None:
        """
        Ensure a list is large enough to access the given index.

        Extends the list with fill values until it can safely accommodate
        the specified index position.

        :param lst: The list to resize if necessary
        :param index: The minimum index that must be accessible
        """
        while len(lst) <= index:
            lst.append(self.fill_value)

    def _list_set_value(
        self,
        current: dict[str, Any],
        name: str,
        index: int,
        value: Any,
    ) -> None:
        """
        Set a value in a list at the given index within a dictionary.

        Creates the list if it doesn't exist and ensures it's large enough
        to accommodate the specified index before setting the value.

        :param current: The current dictionary context containing the list
        :param name: The key name for the list in the dictionary
        :param index: The list index where the value should be set
        :param value: The value to set at the specified index
        :raises ArgStringParseError: If overwriting and allow_overwrite is False
        """
        if name not in current:
            current[name] = []
        elif not isinstance(current[name], list):
            # Trying to replace non-list with list
            if not self.allow_overwrite:
                raise ArgStringParseError(
                    f"Cannot overwrite non-list value at '{name}' with list"
                )
            current[name] = []

        self._list_ensure_size(current[name], index)

        # Check for overwrite of non-fill values
        existing_value = current[name][index]
        if not self.allow_overwrite and existing_value != self.fill_value:
            raise ArgStringParseError(
                f"Cannot overwrite existing value at '{name}[{index}]'"
            )

        current[name][index] = value

    def _list_navigate_to_element(
        self,
        current: dict[str, Any],
        name: str,
        index: int,
    ) -> Any:
        """
        Navigate to or create a list element for intermediate path segments.

        Creates the list if it doesn't exist, ensures it's large enough for the index,
        and initializes the element as an empty dictionary if needed. Returns the
        element for further navigation.

        :param current: The current dictionary context containing the list
        :param name: The key name for the list in the dictionary
        :param index: The list index to navigate to
        :return: The element at the given index (creating empty dict if needed)
        :raises ArgStringParseError: If overwriting and allow_overwrite is False
        """
        if name not in current:
            current[name] = []
        elif not isinstance(current[name], list):
            # Trying to replace non-list with list
            if not self.allow_overwrite:
                raise ArgStringParseError(
                    f"Cannot overwrite non-list value at '{name}' with list"
                )
            current[name] = []

        self._list_ensure_size(current[name], index)

        # Create nested structure if needed (only for fill values)
        existing_value = current[name][index]
        if existing_value == self.fill_value:
            current[name][index] = {}
        elif not isinstance(existing_value, dict):
            # Trying to navigate into non-dict value
            if not self.allow_overwrite:
                raise ArgStringParseError(
                    f"Cannot overwrite non-dict value at '{name}[{index}]' with dict"
                )
            current[name][index] = {}

        return current[name][index]

    def _dict_navigate_to_key(
        self,
        current: dict[str, Any],
        name: str,
    ) -> dict[str, Any]:
        """
        Navigate to or create a dictionary key for intermediate path segments.

        Creates the dictionary key if it doesn't exist and returns the nested
        dictionary for further navigation.

        :param current: The current dictionary context
        :param name: The key name to navigate to
        :return: The nested dictionary at the given key
        :raises ArgStringParseError: If overwriting and allow_overwrite is False
        """
        if name not in current:
            current[name] = {}
        elif not isinstance(current[name], dict):
            # Trying to navigate into non-dict value
            if not self.allow_overwrite:
                raise ArgStringParseError(
                    f"Cannot overwrite non-dict value at '{name}' with dict"
                )
            current[name] = {}

        return current[name]

    def _value_set_nested(
        self,
        result: dict[str, Any],
        segments: list[tuple[str, int | None]],
        value: Any,
    ) -> None:
        """
        Set a value in a nested dictionary and list structure.

        Navigates through the segment path, creating intermediate dictionaries and lists
        as needed. For sparse lists, fills gaps with fill values. Supports mixed nesting
        of dictionaries and lists.

        :param result: The root dictionary to modify in place
        :param segments: List of (name, index) tuples representing the navigation path
        :param value: The value to set at the final destination
        :raises ArgStringParseError: If overwriting and allow_overwrite is False
        """
        current: Any = result

        for i, (name, index) in enumerate(segments):
            is_last = i == len(segments) - 1

            if is_last:
                # If we are at the last segment, set the value
                if index is not None:
                    self._list_set_value(current, name, index, value)
                else:
                    if name in current and not self.allow_overwrite:
                        raise ArgStringParseError(
                            f"Cannot overwrite existing value at key '{name}'"
                        )
                    current[name] = value
            # If not last, navigate/create intermediate structures
            elif index is not None:
                # Intermediate segment with array access
                current = self._list_navigate_to_element(current, name, index)
            else:
                # Intermediate segment with dict access
                current = self._dict_navigate_to_key(current, name)


def loads(
    s: str,
    *,
    cls: type[ArgStringParser] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Parse an arg string into a Python dictionary with automatic type conversion.

    This function follows the json.loads pattern, allowing customization through
    a custom decoder class. By default, uses ArgStringParser.

    :param s: Comma-separated list of key=value pairs supporting nested
        structures via dot and bracket notation
    :param cls: Custom decoder class to use (must have a decode method).
        Defaults to ArgStringParser.
    :param kwargs: Additional keyword arguments passed to the decoder class
        constructor (e.g., fill_value)
    :return: Parsed dictionary with appropriate types and nesting
    :raises ArgStringParseError: If the string format is invalid

    Example:
    ::
        from guidellm.utils import arg_string

        arg_string.loads("prompt_tokens=10,output_tokens=30")
        # {'prompt_tokens': 10, 'output_tokens': 30}

        # With custom fill value
        arg_string.loads("items[5]=value", fill_value=lambda: 0)
        # {'items': [0, 0, 0, 0, 0, 'value']}

        # With custom decoder class
        class CustomParser(arg_string.ArgStringParser):
            def decode(self, s):
                result = super().decode(s)
                # Custom post-processing
                return result

        arg_string.loads("a=1,b=2", cls=CustomParser)
    """
    if cls is None:
        cls = ArgStringParser

    return cls(**kwargs).decode(s)
