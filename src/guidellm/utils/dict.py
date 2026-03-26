"""Utility functions for working with dictionaries."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any


def recursive_key_update(d, key_update_func):
    if not isinstance(d, dict) and not isinstance(d, list):
        return d

    if isinstance(d, list):
        for item in d:
            recursive_key_update(item, key_update_func)
        return d

    updated_key_pairs = []
    for key, _ in d.items():
        updated_key = key_update_func(key)
        if key != updated_key:
            updated_key_pairs.append((key, updated_key))

    for key_pair in updated_key_pairs:
        old_key, updated_key = key_pair
        d[updated_key] = d[old_key]
        del d[old_key]

    for _, value in d.items():
        recursive_key_update(value, key_update_func)
    return d


def deep_update(dict1: dict, dict2: dict) -> None:
    """
    Update dict1 with values from dict2 recursively.

    Modifies dict1 in-place. Does not handle circular references.
    Does not copy values. Does not merge lists.
    """
    for key, val in dict2.items():
        if isinstance(val, dict) and key in dict1 and isinstance(dict1[key], dict):
            deep_update(dict1[key], val)
        else:
            dict1[key] = val


def deep_filter(d: dict, predicate: Callable[[Hashable, Any], bool]) -> None:
    """
    Recursively filters a dictionary based on a predicate function.

    Modifies the input dictionary in-place. Does not handle circular references.
    Does not copy values. Does not filter lists.
    """
    for key, value in list(d.items()):
        if isinstance(value, dict):
            deep_filter(value, predicate)
        elif not predicate(key, value):
            d.pop(key)
