import contextlib
import os
from typing import cast, get_args, get_origin

import click
from more_itertools import partition
from pydantic import BaseModel

from guidellm.settings import settings

__all_ = [
    "validate_env_vars",
    "list_set_env",
    "get_valid_env_vars",
]


def validate_env_vars(ctx: click.Context | None = None) -> tuple[set[str], set[str]]:
    """
    Validate if the given environment variable name is recognized.

    Checks if the environment variable is in the set of valid environment variables
    extracted from the Click context and Pydantic Settings.

    :param env_var: The environment variable name to validate.
    :return: True if the environment variable is valid, False otherwise.
    """
    set_vars = list_set_env()
    valid_vars = get_valid_env_vars(ctx)

    invaild_set_vars, valid_set_vars = partition(
        lambda e: e in valid_vars,
        set_vars,
    )
    return set(invaild_set_vars), set(valid_set_vars)


def list_set_env(prefix: str = "GUIDELLM_") -> set[str]:
    """
    List all set environment variables prefixed with the given prefix.

    :param prefix: The prefix to filter environment variables.
    :return: List of environment variable names that are set with the given prefix.
    """
    return {key for key in os.environ if key.startswith(prefix)}


def get_valid_env_vars(
    ctx: click.Context | None = None, include_settings: bool = True
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
        valid_envs.update(_extract_click_env_vars(ctx))

    if include_settings:
        valid_envs.update(_extract_settings_env_vars())

    return valid_envs


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
    return _walk_settings_fields(
        type(settings),
        env_prefix,
        env_delimiter,
    )


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
        if get_origin(field_type) is type(None) or get_origin(field_type) is type(None):
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
                _walk_settings_fields(
                    cast("type[BaseModel]", field_type),
                    nested_prefix,
                    delimiter,
                )
            )
        else:
            # Regular field - add to set
            env_vars.add(field_env_name)

    return env_vars
