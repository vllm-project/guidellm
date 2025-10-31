import json
from typing import Any

import click

__all__ = [
    "Union",
    "format_list_arg",
    "parse_json",
    "parse_list_floats",
    "set_if_not_default",
]


def parse_list_floats(ctx, param, value):  # noqa: ARG001
    """
    Callback to parse a comma-separated string into a list of floats.
    """
    # This callback only runs if the --rate option is provided by the user.
    # If it's not, 'value' will be None, and Click will use the 'default'.
    if value is None:
        return None  # Keep the default

    try:
        # Split by comma, strip any whitespace, and convert to float
        return [float(item.strip()) for item in value.split(",")]
    except ValueError as e:
        # Raise a Click error if any part isn't a valid float
        raise click.BadParameter(
            f"Value '{value}' is not a valid comma-separated list "
            f"of floats/ints. Error: {e}"
        ) from e

def parse_json(ctx, param, value):  # noqa: ARG001
    if value is None or value == [None]:
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
        # Treat it as a plain string if it doesn't look like JSON.
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

    def get_metavar(self, param: click.Parameter) -> str:
        def get_choices(t: click.ParamType) -> str:
            meta = t.get_metavar(param)
            return meta if meta is not None else t.name

        # Get the choices for each type in the union.
        choices_str = "|".join(map(get_choices, self.types))

        # Use curly braces to indicate a required argument.
        if param.required and param.param_type_name == "argument":
            return f"{{{choices_str}}}"

        # Use square braces to indicate an option or optional argument.
        return f"[{choices_str}]"
