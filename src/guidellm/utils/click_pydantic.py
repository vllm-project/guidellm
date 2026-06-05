from __future__ import annotations

import functools
from typing import Any, get_args, get_origin

import click
from pydantic import BaseModel

from guidellm.schemas import PydanticClassRegistryMixin
from guidellm.utils.cli import parse_arguments

__all__ = [
    "registry_option",
    "registry_options_from_model",
]


class _RegistryChoiceConfig(click.ParamType):
    """
    Custom two-value Click parameter type for ``<discriminator> <config>`` pairs.

    The first value is validated against the registered names of a
    ``PydanticClassRegistryMixin`` subclass. The second value is a free-form
    config string (JSON, YAML, or key=value) displayed as ``CONFIG``.

    :param registry_type: The ``PydanticClassRegistryMixin`` subclass to
        pull valid discriminator names from
    """

    name = "CHOICE CONFIG"

    def __init__(self, registry_type: type[PydanticClassRegistryMixin]) -> None:  # type: ignore[type-arg]
        self._registry_type = registry_type
        if registry_type.registry is not None:
            self.registry_choices: click.Choice | None = click.Choice(
                list(registry_type.registered_names())
            )
        else:
            self.registry_choices = None

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str:
        """
        Return a metavar string like ``{openai_http|vllm_python} CONFIG``.

        Falls back to ``KIND CONFIG`` when no registry entries exist yet.

        :param param: The Click parameter
        :param ctx: The Click context
        :return: Formatted metavar string
        """
        if self.registry_choices is not None:
            choice_meta = self.registry_choices.get_metavar(param, ctx)
        else:
            choice_meta = "KIND"
        return f"{choice_meta} CONFIG"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> Any:
        """
        Pass through raw values; discriminator validation happens in the callback.

        :param value: The raw value from Click
        :param param: The Click parameter
        :param ctx: The Click context
        :return: The value unchanged
        """
        return value


def _make_common_field_callback(
    discriminator: str,
    is_list: bool,
    choice_type: _RegistryChoiceConfig,
):
    """
    Create a Click callback that parses ``<discriminator_value> <config>`` pairs.

    The discriminator value is validated against the registry's
    ``click.Choice`` type when available.

    :param discriminator: The discriminator field name (e.g. ``"kind"``)
    :param is_list: Whether the field is a list type (``multiple=True``)
    :param choice_type: The ``_RegistryChoiceConfig`` instance for discriminator
        validation
    :return: A Click callback function
    """

    def _parse_pair(
        ctx: click.Context, param: click.Parameter, disc_value: str, config_str: str
    ) -> dict[str, Any]:
        if choice_type.registry_choices is not None:
            disc_value = choice_type.registry_choices.convert(disc_value, param, ctx)
        parsed = parse_arguments(ctx, param, config_str)
        if isinstance(parsed, dict):
            return {discriminator: disc_value, **parsed}
        return {discriminator: disc_value}

    def callback(
        ctx: click.Context,
        param: click.Parameter,
        value: Any,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if is_list:
            if not value:
                return None
            return [
                _parse_pair(ctx, param, disc_value, config_str)
                for disc_value, config_str in value
            ]

        if value is None:
            return None
        disc_value, config_str = value
        return _parse_pair(ctx, param, disc_value, config_str)

    return callback


def _introspect_registry_fields(model: type[BaseModel]) -> list[dict[str, Any]]:
    """
    Extract field specs from a Pydantic model whose fields are
    ``PydanticClassRegistryMixin`` subclasses or lists of them.

    :param model: A Pydantic model class to introspect
    :return: List of field spec dicts with name, discriminator, is_list, description,
        and field_type
    """
    field_specs: list[dict[str, Any]] = []

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        is_list = get_origin(annotation) is list

        if is_list:
            args = get_args(annotation)
            field_type = args[0] if args else None
        else:
            field_type = annotation

        if field_type is None or not issubclass(field_type, PydanticClassRegistryMixin):
            continue

        extra = field_info.json_schema_extra
        argument_alias = (
            extra.get("argument_alias") if isinstance(extra, dict) else None
        )

        field_specs.append(
            {
                "name": field_name,
                "discriminator": field_type.schema_discriminator,
                "is_list": is_list,
                "description": field_info.description or "",
                "field_type": field_type,
                "argument_alias": argument_alias,
            }
        )

    return field_specs


def registry_option(
    *param_decls: str,
    registry: type[PydanticClassRegistryMixin],  # type: ignore[type-arg]
    multiple: bool = False,
    **extra: Any,
):
    """
    Click option decorator for ``PydanticClassRegistryMixin``-backed fields.

    Produces a ``--name <discriminator_value> <config>`` option whose value is
    parsed into ``{schema_discriminator: disc_value, **parsed_config}``.

    :param param_decls: Click-style parameter declarations (e.g. ``"--output"``,
        ``"--output", "outputs"``)
    :param registry: A ``PydanticClassRegistryMixin`` subclass whose registered
        names populate a ``click.Choice`` for the first argument
    :param multiple: When ``True``, the option may be repeated and values are
        collected into a list of dicts
    :param help: Help text shown in ``--help``
    :param extra: Additional keyword arguments forwarded to ``click.option``
    :return: A decorator that adds the option to a Click command
    """
    choice_type = _RegistryChoiceConfig(registry)
    kwargs: dict[str, Any] = {
        "nargs": 2,
        "type": choice_type,
        "multiple": multiple,
        "callback": _make_common_field_callback(
            registry.schema_discriminator,
            multiple,
            choice_type,
        ),
        "expose_value": True,
        "is_eager": False,
        **extra,
    }
    if not multiple:
        kwargs.setdefault("default", None)

    return click.option(*param_decls, **kwargs)


def registry_options_from_model(model: type[BaseModel], group_key: str | None = None):
    """
    Decorator factory that introspects a Pydantic model and generates
    ``click.option`` decorators for each registry-backed field.

    Each field in the model that has a ``schema_discriminator`` attribute gets a
    CLI option in the form ``--field <discriminator_value> <config>``. All parsed
    values are aggregated into a ``common`` kwarg passed to the decorated function.

    The discriminator value is validated against the registry's known names via
    ``click.Choice``. Non-list fields use Click's default last-wins behavior
    when specified more than once.

    :param model: A Pydantic model class whose fields are
        ``PydanticClassRegistryMixin`` subclasses or lists of them
    :return: A decorator that adds click options and wraps the function
    """
    field_specs = _introspect_registry_fields(model)

    def decorator(func):
        wrapped = func

        for spec in reversed(field_specs):
            cli_name = spec["argument_alias"] or spec["name"]
            option_name = f"--{cli_name.replace('_', '-')}"
            wrapped = registry_option(
                option_name,
                spec["name"],
                registry=spec["field_type"],
                multiple=spec["is_list"],
                help=spec["description"],
            )(wrapped)

        @functools.wraps(wrapped)
        def wrapper(**kw):
            ctx = click.get_current_context()
            group: dict[str, Any] = {}

            for spec in field_specs:
                value = kw.pop(spec["name"], None)
                source = ctx.get_parameter_source(spec["name"])
                if source is not None and source != click.core.ParameterSource.DEFAULT:
                    group[spec["name"]] = value

            if group:
                if group_key is not None:
                    kw[group_key] = group
                else:
                    kw.update(group)

            return func(**kw)

        if hasattr(wrapped, "__click_params__"):
            wrapper.__click_params__ = wrapped.__click_params__  # type: ignore[attr-defined]

        return wrapper

    return decorator
