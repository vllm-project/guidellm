from __future__ import annotations

from datetime import datetime
from typing import Any, Final

__all__ = [
    "UNSET",
    "UnsetType",
    "all_defined",
    "safe_add",
    "safe_divide",
    "safe_format_timestamp",
    "safe_getattr",
    "safe_multiply",
    "safe_subtract",
]


class UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = UnsetType()


def safe_getattr(obj: Any | None, attr: str, default: Any = None) -> Any:
    """
    Safely get an attribute from an object or return a default value.

    :param obj: The object to get the attribute from.
    :param attr: The name of the attribute to get.
    :param default: The default value to return if the attribute is not found.
    :return: The value of the attribute or the default value.
    """
    if obj is None:
        return default

    return getattr(obj, attr, default)


def all_defined(*values: Any | None) -> bool:
    """
    Check if all values are defined (not None).

    :param values: The values to check.
    :return: True if all values are defined, False otherwise.
    """
    return all(value is not None for value in values)


def safe_divide(
    numerator: int | float | None,
    denominator: int | float | None,
    num_default: float = 0.0,
    den_default: float = 1.0,
) -> float:
    numerator = numerator if numerator is not None else num_default
    denominator = denominator if denominator is not None else den_default

    return numerator / (denominator or 1e-10)


def safe_multiply(*values: int | float | None, default: float = 1.0) -> float:
    result = default
    for val in values:
        result *= val if val is not None else 1.0
    return result


def safe_add(*values: int | float | None, default: float = 0.0) -> float:
    result = default
    for val in values:
        result += val if val is not None else 0.0
    return result


def safe_subtract(*values: int | float | None, default: float = 0.0) -> float:
    result = default
    for val in values:
        if val is not None:
            result -= val

    return result


def safe_format_timestamp(
    timestamp: float | None, format_: str = "%H:%M:%S", default: str = "N/A"
) -> str:
    if timestamp is not None and timestamp >= 0 and timestamp <= 2**31:
        try:
            return datetime.fromtimestamp(timestamp).strftime(format_)
        except (ValueError, OverflowError, OSError):
            return default

    return default
