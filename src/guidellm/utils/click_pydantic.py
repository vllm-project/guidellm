from __future__ import annotations

import contextlib
import functools
from typing import Any, get_args, get_origin

import click
from pydantic import BaseModel, ValidationError

from guidellm.schemas import PydanticClassRegistryMixin
from guidellm.utils.cli import parse_arguments

__all__ = [
    "format_validation_errors",
    "registry_option",
    "registry_options_from_model",
]


class _RegistryParamType(click.ParamType):
    """
    Click parameter type for registry-backed config strings.

    Accepts a single comma-separated key=value string such as
    ``kind=constant,rate=10``.  The metavar lists the discriminator field
    and its registered choices so ``--help`` is informative.

    :param registry_type: The ``PydanticClassRegistryMixin`` subclass whose
        registered names populate the metavar
    :param discriminator: The discriminator field name (e.g. ``"kind"``)
    """

    name = "CONFIG"

    def __init__(
        self,
        registry_type: type[PydanticClassRegistryMixin],  # type: ignore[type-arg]
        discriminator: str,
    ) -> None:
        self._registry_type = registry_type
        self._discriminator = discriminator

    def get_metavar(self, param: click.Parameter, ctx: click.Context) -> str:  # noqa: ARG002
        """
        Return a metavar like ``kind=[constant|sweep|...],...``.

        :param param: The Click parameter
        :param ctx: The Click context
        :return: Formatted metavar string
        """
        registry = self._registry_type.registry
        if registry:
            names = "|".join(self._registry_type.registered_names())
            return f"{self._discriminator}=[{names}],..."
        return f"{self._discriminator}=KIND,..."

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> Any:
        """
        Pass through raw values; validation is handled by Pydantic.

        :param value: The raw value from Click
        :param param: The Click parameter
        :param ctx: The Click context
        :return: The value unchanged
        """
        return value


def _make_common_field_callback(
    is_list: bool,
    discriminator: str,
    registry_type: type[PydanticClassRegistryMixin],  # type: ignore[type-arg]
):
    """
    Create a Click callback that parses a config string into a dict.

    The input string is a comma-separated key=value string such as
    ``kind=constant,rate=10``, parsed by :func:`parse_arguments`.

    :param is_list: Whether the field is a list type (``multiple=True``)
    :param discriminator: The discriminator field name (e.g. ``"kind"``)
    :param registry_type: The registry subclass for error messages
    :return: A Click callback function
    """

    def _parse_and_check(
        ctx: click.Context, param: click.Parameter, raw: str
    ) -> dict[str, Any]:
        parsed = parse_arguments(ctx, param, raw)
        if isinstance(parsed, dict) and discriminator not in parsed:
            names = registry_type.registered_names()
            choices = ", ".join(names) if names else "(none registered)"
            raise click.BadParameter(
                f"missing required key '{discriminator}'. Valid options: {choices}",
                ctx=ctx,
                param=param,
            )
        return parsed

    def callback(
        ctx: click.Context,
        param: click.Parameter,
        value: Any,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if is_list:
            if not value:
                return None
            return [_parse_and_check(ctx, param, v) for v in value]

        if value is None:
            return None
        return _parse_and_check(ctx, param, value)

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

    Produces a ``--name <config>`` option where ``<config>`` is a single
    comma-separated key=value string (e.g. ``kind=constant,rate=10``).

    :param param_decls: Click-style parameter declarations (e.g. ``"--output"``,
        ``"--output", "outputs"``)
    :param registry: A ``PydanticClassRegistryMixin`` subclass whose registered
        names populate the metavar
    :param multiple: When ``True``, the option may be repeated and values are
        collected into a list of dicts
    :param extra: Additional keyword arguments forwarded to ``click.option``
    :return: A decorator that adds the option to a Click command
    """
    param_type = _RegistryParamType(registry, registry.schema_discriminator)
    if multiple and "help" in extra:
        extra["help"] = f"{extra['help']}  [repeatable]"
    kwargs: dict[str, Any] = {
        "type": param_type,
        "multiple": multiple,
        "callback": _make_common_field_callback(
            multiple, registry.schema_discriminator, registry
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
    CLI option accepting a single comma-separated key=value string (e.g.
    ``--profile kind=constant,rate=10``).  All parsed values are aggregated
    into a group kwarg passed to the decorated function.

    :param model: A Pydantic model class whose fields are
        ``PydanticClassRegistryMixin`` subclasses or lists of them
    :param group_key: Optional key under which to group all registry values
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


def _error_to_message(loc: tuple, msg: str) -> str:
    """
    Format a single pydantic validation error into a human-readable message.

    Includes the full location path, such as
    ``data[0].synthetic_text.output_tokens``, so callers can identify which
    nested subfield failed rather than only the top-level CLI option.

    :param err: A pydantic error dict as returned by ``ValidationError.errors()``
    :return: Formatted error string including the failing field path
    """
    if not loc:
        return msg

    path = str(loc[0])
    for component in loc[1:]:
        if isinstance(component, int):
            path += f"[{component}]"
        else:
            path += f".{component}"

    return f"{msg} (at '{path}')"


def _resolve_param_name(
    loc: tuple[str | int, ...],
    base_class: type[BaseModel],
) -> tuple[tuple[str | int, ...], str]:
    current_model = base_class

    for i, component in enumerate(loc):
        if isinstance(component, int):
            continue

        fields = current_model.model_fields
        if component not in fields:
            raise KeyError("Key does not exist")

        field_info = fields[component]
        extra = field_info.json_schema_extra
        if isinstance(extra, dict):
            alias = extra.get("argument_alias")
            if alias is not None:
                return loc[i:], str(alias)

        annotation = field_info.annotation
        if get_origin(annotation) is list:
            args = get_args(annotation)
            annotation = args[0] if args else None

        if annotation is None:
            raise KeyError("Key does not exist")

        try:
            is_model = issubclass(annotation, BaseModel)
        except TypeError as e:
            raise KeyError("Key is not a model") from e

        if not is_model:
            raise KeyError("Key does not exist")

        current_model = annotation

    raise KeyError("Key does not exist")


def format_validation_errors(
    ctx: click.Context,
    err: ValidationError,
    base_class: type[BaseModel] | None = None,
) -> click.BadParameter:
    """
    Translate a pydantic ``ValidationError`` into a ``click.BadParameter`` error.

    Includes the top-level field names and full location paths, such as
    ``data[0].synthetic_text.output_tokens``, so callers can identify which
    nested subfield failed rather than only the top-level CLI option.

    :param ctx: The active Click context
    :param err: The pydantic validation error to translate
    :param base_class: Optional root model class for resolving argument aliases
    :return: A ``click.BadParameter`` with a formatted message and param hint
    """
    errs = err.errors(include_url=False, include_context=True, include_input=True)

    param_names = set()
    msgs = []
    for e in errs:
        loc = e.get("loc", ())
        msg = e.get("msg", "validation error")

        top_field: str | None = None
        if len(loc) > 0:
            if base_class is not None:
                with contextlib.suppress(KeyError):
                    loc, top_field = _resolve_param_name(loc, base_class)
            top_field = top_field or str(loc[0]) or None

            if top_field:
                param_names.add("--" + top_field.replace("_", "-"))

        msgs.append(_error_to_message(loc, msg))

    error_msg = "".join(f"\n  - {msg}" for msg in msgs)
    return click.BadParameter(error_msg, ctx=ctx, param_hint=list(param_names))
