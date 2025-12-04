import contextlib
import json
import os
from typing import Any

import click

__all__ = [
    "Union",
    "format_list_arg",
    "list_set_env",
    "parse_json",
    "parse_list",
    "parse_list_floats",
    "set_if_not_default",
]


def list_set_env(prefix: str = "GUIDELLM_") -> list[str]:
    """
    List all set environment variables prefixed with the given prefix.

    :param prefix: The prefix to filter environment variables.
    :return: List of environment variable names that are set with the given prefix.
    """
    return [key for key in os.environ if key.startswith(prefix)]


def parse_list(ctx, param, value) -> list[str] | None:
    """
    Click callback to parse the input value into a list of strings.
    Supports single strings, comma-separated strings,
    and lists/tuples of any of these formats (when multiple=True is used).

    :param ctx: Click context
    :param param: Click parameter
    :param value: The input value to parse
    :return: List of parsed strings
    """
    if value is None or value == [None]:
        # Handle null values directly or nested null (when multiple=True)
        return None

    if isinstance(value, list | tuple):
        # Handle multiple=True case by recursively parsing each item and combining
        parsed = []
        for val in value:
            if (items := parse_list(ctx, param, val)) is not None:
                parsed.extend(items)
        return parsed

    if isinstance(value, str) and "," in value:
        # Handle comma-separated strings
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, str):
        # Handle single string
        return [value.strip()]

    # Fall back to returning as a single-item list
    return [value]


def parse_list_floats(ctx, param, value):
    str_list = parse_list(ctx, param, value)
    if str_list is None:
        return None

    item = None  # For error reporting
    try:
        return [float(item) for item in str_list]
    except ValueError as err:
        # Raise a Click error if any part isn't a valid float
        raise click.BadParameter(
            f"Input '{value}' is not a valid comma-separated list "
            f"of floats/ints. Failed on {item} Error: {err}"
        ) from err


def parse_json(ctx, param, value):  # noqa: ARG001, C901, PLR0911
    if isinstance(value, dict):
        return value

    if value is None or value == [None]:
        return None

    if isinstance(value, str) and not value.strip():
        return None

    if isinstance(value, list | tuple):
        return [parse_json(ctx, param, val) for val in value]

    if "{" not in value and "}" not in value and "=" in value:
        # Treat it as a key=value pair if it doesn't look like JSON.
        result = {}
        for pair in value.split(","):
            if "=" not in pair:
                raise click.BadParameter(
                    f"{param.name} must be a valid JSON string or key=value pairs."
                )
            key, val = pair.split("=", 1)
            result[key.strip()] = val.strip()
        return result

    if "{" not in value and "}" not in value:
        # Treat it as a primitive if it doesn't look like JSON.
        try:
            value = int(value)
        except ValueError:
            with contextlib.suppress(ValueError):
                value = float(value)
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError as err:
        raise click.BadParameter(f"{param.name} must be a valid JSON string.") from err


def set_if_not_default(ctx: click.Context, **kwargs) -> dict[str, Any]:
    """
    Set the value of a click option if it is not the default value.
    This is useful for setting options that are not None by default.
    """
    values = {}
    for k, v in kwargs.items():
        if ctx.get_parameter_source(k) != click.core.ParameterSource.DEFAULT:  # type: ignore[attr-defined]
            values[k] = v

    return values


def format_list_arg(
    value: Any, default: Any = None, simplify_single: bool = False
) -> list[Any] | Any:
    """
    Format a multi-argument value for display.

    :param value: The value to format, which can be a single value or a list/tuple.
    :param default: The default value to set if the value is non truthy.
    :param simplify_single: If True and the value is a single-item list/tuple,
        return the single item instead of a list.
    :return: Formatted list of values, or single value if simplify_single and applicable
    """
    if not value:
        return default

    if isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        value = [value]

    return value if not simplify_single or len(value) != 1 else value[0]


class Union(click.ParamType):
    """
    A custom click parameter type that allows for multiple types to be accepted.
    """

    def __init__(self, *types: click.ParamType):
        self.types = types
        self.name = "".join(t.name for t in types)

    def convert(self, value, param, ctx):
        fails = []
        for t in self.types:
            try:
                return t.convert(value, param, ctx)
            except click.BadParameter as e:
                fails.append(str(e))
                continue

        self.fail("; ".join(fails) or f"Invalid value: {value}")  # noqa: RET503

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str:
        def get_choices(t: click.ParamType) -> str:
            meta = t.get_metavar(param, ctx)
            return meta if meta is not None else t.name

        # Get the choices for each type in the union.
        choices_str = "|".join(map(get_choices, self.types))

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"
