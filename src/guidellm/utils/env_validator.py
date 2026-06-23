import contextlib
import os
from typing import Any, get_args

import click
from more_itertools import partition
from pydantic import BaseModel

__all_ = [
    "validate_env_vars",
    "list_set_env",
    "get_valid_env_vars",
]


def validate_env_vars(
    *models: type[BaseModel],
    ctx: click.Context | None = None,
) -> tuple[set[str], set[str]]:
    """
    Validate if the given environment variable name is recognized.

    Checks if the environment variable is in the set of valid environment variables
    extracted from the Click context and Pydantic Settings.

    :param models: Pydantic model classes to extract valid env var names from.
    :param ctx: Click context for extracting Click option env vars.
    :return: Tuple of (invalid env vars, valid env vars) that are currently set.
    """
    set_vars = list_set_env()
    valid_vars, valid_prefixes = get_valid_env_vars(*models, ctx=ctx)

    def _is_valid(env_var: str) -> bool:
        return env_var in valid_vars or any(
            env_var.startswith(p) for p in valid_prefixes
        )

    invalid_set_vars, valid_set_vars = partition(_is_valid, set_vars)
    return set(invalid_set_vars), set(valid_set_vars)


def list_set_env(prefix: str = "GUIDELLM_") -> set[str]:
    """
    List all set environment variables prefixed with the given prefix.

    :param prefix: The prefix to filter environment variables.
    :return: List of environment variable names that are set with the given prefix.
    """
    return {key for key in os.environ if key.startswith(prefix)}


def get_valid_env_vars(
    *models: type[BaseModel],
    ctx: click.Context | None = None,
) -> tuple[set[str], set[str]]:
    """
    Get all valid environment variable names from Click context and Settings.

    Collects environment variable names that are recognized by the application.

    :param models: Pydantic model classes to extract valid env var names from.
    :param ctx: Click context for extracting Click option env vars.
    :return: Tuple of (exact env var names, valid env var prefixes).
    """
    valid_envs: set[str] = set()
    valid_prefixes: set[str] = set()

    if ctx is not None:
        valid_envs.update(_extract_click_env_vars(ctx))

    for model in models:
        envs, prefixes = _extract_model_env_vars(model)
        valid_envs.update(envs)
        valid_prefixes.update(prefixes)

    return valid_envs, valid_prefixes


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


def _extract_model_env_vars(
    model_class: type[BaseModel],
) -> tuple[set[str], set[str]]:
    """
    Extract all environment variable names from a Pydantic model.

    :param model_class: Pydantic model class to extract env var names from.
    :return: Tuple of (exact env var names, valid env var prefixes).
    """
    config = model_class.model_config
    env_prefix = str(config.get("env_prefix", ""))
    env_delimiter = str(config.get("env_nested_delimiter", "__"))

    return _walk_model_fields(model_class, env_prefix, env_delimiter)


def _resolve_model_type(annotation: Any) -> type[BaseModel] | None:
    """
    Recursively unwrap a type annotation to find a BaseModel subclass.

    Handles Optional, Union, list, and other generic types.

    :param annotation: The type annotation to resolve.
    :return: The BaseModel subclass if found, otherwise None.
    """
    if annotation is None:
        return None
    with contextlib.suppress(TypeError):
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    for arg in get_args(annotation) or ():
        if arg is type(None):
            continue
        result = _resolve_model_type(arg)
        if result is not None:
            return result
    return None


def _walk_model_fields(
    model_class: type[BaseModel],
    prefix: str,
    delimiter: str,
) -> tuple[set[str], set[str]]:
    """
    Recursively walk Pydantic model fields to build env var names and prefixes.

    For leaf fields, the exact env var name is added. For nested BaseModel
    fields, the prefix is added to allow any env var under that prefix,
    consistent with pydantic-settings' lenient ``explode_env_vars`` behavior.

    :param model_class: Pydantic model class to walk.
    :param prefix: Current environment variable prefix.
    :param delimiter: Delimiter for nested fields.
    :return: Tuple of (exact env var names, valid env var prefixes).
    """
    env_vars: set[str] = set()
    prefixes: set[str] = set()

    for field_name, field_info in model_class.model_fields.items():
        field_env_name = f"{prefix}{field_name.upper()}"
        model_type = _resolve_model_type(field_info.annotation)

        if model_type is not None:
            nested_prefix = f"{field_env_name}{delimiter}"
            prefixes.add(nested_prefix)
            sub_vars, sub_prefixes = _walk_model_fields(
                model_type, nested_prefix, delimiter
            )
            env_vars.update(sub_vars)
            prefixes.update(sub_prefixes)
        else:
            env_vars.add(field_env_name)

    return env_vars, prefixes
