from __future__ import annotations

from collections.abc import Iterator
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

# Backwards compatibility for Python <3.12
try:
    from typing import TypeAliasType  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAliasType


__all__ = ["get_literal_vals"]


def get_literal_vals(alias) -> frozenset[str]:
    """Extract all literal values from a (possibly nested) type alias."""

    def resolve(alias) -> Iterator[str]:
        origin = get_origin(alias)

        # Base case: Literal types
        if origin is Literal:
            for literal_val in get_args(alias):
                yield str(literal_val)
        # Unwrap Annotated type
        elif origin is Annotated:
            yield from resolve(get_args(alias)[0])
        # Unwrap TypeAliasTypes
        elif isinstance(alias, TypeAliasType):
            yield from resolve(alias.__value__)
        # Iterate over unions
        elif origin in (Union, UnionType):
            for arg in get_args(alias):
                yield from resolve(arg)
        # Fallback
        else:
            yield str(alias)

    return frozenset(resolve(alias))
