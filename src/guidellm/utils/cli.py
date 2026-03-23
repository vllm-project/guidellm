import codecs
import contextlib
import json
import os
from typing import Any, cast, get_args, get_origin

import click
from pydantic import BaseModel

from guidellm.settings import settings

__all__ = [
    "EnvVarValidator",
    "Union",
    "decode_escaped_str",
    "format_list_arg",
    "list_set_env",
    "parse_json",
    "parse_list",
    "parse_list_floats",
    "set_if_not_default",
]


class EnvVarValidator:
    """
    Static utility class for validating environment variables.

    Provides methods to extract and validate environment variable names
    from Click contexts and Pydantic Settings.
    """

    @staticmethod
    def get_valid_env_vars(
        ctx: click.Context | None = None,
        include_settings: bool = True,
    ) -> set[str]:
        """
        Get all valid environment variable names from Click context and Settings.

        Collects environment variable names that are recognized by the application.

        :param ctx: Click context for extracting Click option env vars.
        :param include_settings: If True, include Settings env vars (default: True).
        :return: Set of valid environment variable names.
        """
        valid_envs = set()

        if ctx is not None:
            valid_envs.update(EnvVarValidator._extract_click_env_vars(ctx))

        if include_settings:
            valid_envs.update(EnvVarValidator._extract_settings_env_vars())

        return valid_envs

    @staticmethod
    def _extract_click_env_vars(ctx: click.Context) -> set[str]:
        """
        Extract all environment variable names from Click context hierarchy.

        :param ctx: Click context to extract from.
        :return: Set of environment variable names.
        """
        env_vars = set()
        current_ctx: click.Context | None = ctx

        # Traverse context hierarchy (including parent contexts)
        while current_ctx is not None:
            # Check if context has auto_envvar_prefix
            auto_prefix = getattr(current_ctx, "auto_envvar_prefix", None)

            if auto_prefix and current_ctx.command:
                # Extract env vars from command parameters
                for param in current_ctx.command.params:
                    # Check for explicit envvar attribute
                    if hasattr(param, "envvar") and param.envvar:
                        if isinstance(param.envvar, str):
                            env_vars.add(param.envvar)
                        elif isinstance(param.envvar, list | tuple):
                            env_vars.update(param.envvar)

                    # Generate auto env var name if no explicit envvar
                    elif hasattr(param, "name") and param.name:
                        env_var_name = f"{auto_prefix}_{param.name.upper()}"
                        env_vars.add(env_var_name)

            # Move to parent context
            current_ctx = current_ctx.parent

        return env_vars

    @staticmethod
    def _extract_settings_env_vars() -> set[str]:
        """
        Extract all environment variable names from Pydantic Settings.

        Uses the global settings instance from guidellm.settings.

        :return: Set of environment variable names.
        """
        config = settings.model_config
        env_prefix = str(config.get("env_prefix", ""))
        env_delimiter = str(config.get("env_nested_delimiter", "__"))

        # Walk the settings model fields recursively
        return EnvVarValidator._walk_settings_fields(
            type(settings),
            env_prefix,
            env_delimiter,
        )

    @staticmethod
    def _walk_settings_fields(
        model_class: type[BaseModel],
        prefix: str,
        delimiter: str,
    ) -> set[str]:
        """
        Recursively walk Pydantic model fields to build env var names.

        :param model_class: Pydantic model class to walk.
        :param prefix: Current environment variable prefix.
        :param delimiter: Delimiter for nested fields.
        :return: Set of environment variable names.
        """
        env_vars = set()

        for field_name, field_info in model_class.model_fields.items():
            # Build environment variable name for this field
            field_env_name = f"{prefix}{field_name.upper()}"

            # Get the field type annotation
            field_type = field_info.annotation

            # Unwrap Optional/Union types to find the actual type
            if get_origin(field_type) is type(None) or get_origin(field_type) is type(
                None
            ):
                # Handle Optional[T] which is Union[T, None]
                args = get_args(field_type)
                if args:
                    # Get the non-None type
                    field_type = next(
                        (arg for arg in args if arg is not type(None)), args[0]
                    )

            # Check if the field type is a nested BaseModel
            is_base_model = False
            with contextlib.suppress(TypeError):
                # Attempt to check if field_type is a BaseModel subclass
                is_base_model = isinstance(field_type, type) and issubclass(
                    field_type, BaseModel
                )

            if is_base_model:
                # Recursively walk nested model
                nested_prefix = f"{field_env_name}{delimiter}"
                env_vars.update(
                    EnvVarValidator._walk_settings_fields(
                        cast("type[BaseModel]", field_type),
                        nested_prefix,
                        delimiter,
                    )
                )
            else:
                # Regular field - add to set
                env_vars.add(field_env_name)

        return env_vars


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

    try:
        return json.loads(value)
    except json.JSONDecodeError as err:
        # If json parsing fails, check if it looks like a plain string
        if "{" not in value and "}" not in value:
            return value

        raise click.BadParameter(f"{param.name} must be a valid JSON string.") from err


def parse_json_list(ctx, param, value):
    list_value = parse_list(ctx, param, value)

    if list_value is None:
        return None

    return [parse_json(ctx, param, item) for item in list_value]


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
