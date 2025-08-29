import json
from typing import Any

import click


def parse_json(ctx, param, value):  # noqa: ARG001
    if value is None:
        return None
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


class Union(click.ParamType):
    """
    A custom click parameter type that allows for multiple types to be accepted.
    """

    def __init__(self, *types: click.ParamType):
        self.types = types
        self.name = "".join(t.name for t in types)

    def convert(self, value, param, ctx):  # noqa: RET503
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
