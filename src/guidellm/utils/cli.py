import codecs
from typing import Any

import click
import yaml

# NOTE: Sentinel is sentinel in newer (unreleased) version of typing_extensions
# which matches the accepted version of PEP 661 in Python 3.15+
# NOTE: Not sure why but mypy doesn't recognize Sentinel as a type
from typing_extensions import Sentinel  # type: ignore[attr-defined]

from guidellm.utils import arg_string

__all__ = [
    "BLANK",
    "Union",
    "decode_escaped_str",
    "overrides_to_benchmarks",
    "parse_arguments",
    "parse_list",
    "parse_overrides",
    "set_if_not_default",
]

BLANK = Sentinel("BLANK")


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
        result = []
        for item in value.split(","):
            stripped = item.strip()
            if stripped:
                result.append(stripped)
            else:
                result.append(BLANK)
        return result

    if isinstance(value, str):
        # Handle single string
        return [value.strip()]

    # Fall back to returning as a single-item list
    return [value]


def parse_arguments(
    ctx: click.Context, param: click.Parameter, value: list[str] | tuple[str, ...] | str
) -> Any:
    """
    Parse a string value into a Python object using YAML, JSON, or key=value pairs.

    This functions uses a combination of YAML and arg_string parsers to handle any input
    format since PyYAML will parse JSON and raw types.

    :param ctx: Click context
    :param param: Click parameter
    :param value: The input value to parse, string or list/tuple of strings
    """
    if isinstance(value, list | tuple):
        return [parse_arguments(ctx, param, val) for val in value]
    if isinstance(value, str):
        yaml_parsed = False
        try:
            value_parsed = yaml.safe_load(value)
            yaml_parsed = True
        except yaml.YAMLError:
            value_parsed = value
        # If no change from YAML parsing, try arg_string parsing
        if value_parsed == value:
            try:
                value_parsed = arg_string.loads(value)
            # If arg_string parsing fails, attempt to parse the original string
            except arg_string.ArgStringParseError as e:
                if not yaml_parsed:
                    raise click.BadParameter(
                        f"{param.name} must be a valid YAML, JSON, or key=value string."
                    ) from e
        return value_parsed

    return value


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


def decode_escaped_str(_ctx, _param, value):
    """
    Decode escape sequences in Click option values.

    Click automatically escapes characters converting sequences like "\\n" to
    "\\\\n". This function decodes these sequences to their intended characters.

    :param _ctx: Click context (unused)
    :param _param: Click parameter (unused)
    :param value: String value to decode
    :return: Decoded string with proper escape sequences, or None if input is None
    :raises click.BadParameter: When escape sequence decoding fails
    """
    if value is None:
        return None
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception as e:
        raise click.BadParameter(f"Could not decode escape sequences: {e}") from e


def overrides_to_benchmarks(*overrides: tuple[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of (benchmark_name, override_list) tuples into a list of benchmarks

    Resulting benchmark list is as long as the longest override list.
    None values are omitted.

    Example::
        >>> overrides_to_benchmarks(
        ...     ("profile.streams", [1,2,3,4]),
        ...     ("constraint[0].seconds", [10,20,<BLANK>,30])
        ... )
        [
            {"profile.streams": 1, "constraint[0].seconds": 10}
            {"profile.streams": 2, "constraint[0].seconds": 20}
            {"profile.streams": 3}
            {"profile.streams": 4, "constraint[0].seconds": 30}
        ]
    """
    benchmarks: list[dict[str, Any]] = []
    for name, values in overrides:
        for i, value in enumerate(values):
            if len(benchmarks) <= i:
                benchmarks.append({})
            if value is not BLANK:
                benchmarks[i][name] = value
    return benchmarks


def parse_overrides(ctx, param, value):
    """
    Click callback to parse override arguments into a list of benchmark dicts.

    Expects input as multiple occurrences of `--override <key name> <override values>`.
    """
    if not value or not isinstance(value, list | tuple):
        return []

    overrides: list[tuple[str, list[Any]]] = []
    for k, v in value:
        values_list = parse_list(ctx, param, v)
        if values_list is None:
            continue

        values_parsed = parse_arguments(ctx, param, values_list)
        overrides.append((k, values_parsed))

    return overrides_to_benchmarks(*overrides)


def parse_kv_str(
    ctx: click.Context, param: click.Parameter, value: str | tuple[str, ...] | list[str]
):
    """
    Parse a key=value string into a dictionary.

    :param ctx: Click context
    :param param: Click parameter
    :param value: The input string to parse
    :return: Dictionary with the parsed key-value pair, or None if input is None
    :raises click.BadParameter: When parsing fails or the format is invalid
    """
    if isinstance(value, list | tuple):
        return [parse_kv_str(ctx, param, v) for v in value]

    try:
        key, val = value.split("=", 1)
        return key, val
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid key=value format for '{value}'. Expected format is 'key=value'."
        ) from e
