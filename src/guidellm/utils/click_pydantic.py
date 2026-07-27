from __future__ import annotations

import contextlib
import functools
from typing import Any, get_args, get_origin

import click
from pydantic import BaseModel, ValidationError

from guidellm.schemas import PydanticClassRegistryMixin
from guidellm.utils.cli import parse_arguments

__all__ = [
    "RegistryAwareCommand",
    "format_kind_config_usage",
    "format_validation_errors",
    "registry_option",
    "registry_options_from_model",
]

# Pydantic error types where the stock message does not list valid kinds, so we
# append format_kind_config_usage. ``union_tag_invalid`` already lists tags and
# would be redundant if we appended again.
_REGISTRY_SHAPE_ERROR_TYPES = frozenset({"model_attributes_type"})


def format_kind_config_usage(
    registry_type: type[PydanticClassRegistryMixin],  # type: ignore[type-arg]
) -> str:
    """
    Build a self-documenting usage hint for a registry-backed config option.

    :param registry_type: The ``PydanticClassRegistryMixin`` subclass whose
        registered names are listed
    :return: Multi-line string describing expected format and valid kinds
    """
    disc = registry_type.schema_discriminator
    names = registry_type.registered_names()
    choices = ", ".join(names) if names else "(none registered)"
    return (
        f"Expected format: {disc}=<type>,key=value,... "
        f"or JSON/YAML object with a '{disc}' field\n"
        f"  Valid {disc}s: {choices}"
    )


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


class RegistryConfigOption(click.Option):
    """
    Click option subclass that carries a reference to its registry type.

    Used by :class:`RegistryAwareCommand` to enrich missing-argument errors
    with the expected format and valid kinds.
    """

    def __init__(
        self,
        *args: Any,
        registry_type: type[PydanticClassRegistryMixin] | None = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        self.registry_type = registry_type
        super().__init__(*args, **kwargs)


class RegistryAwareCommand(click.Command):
    """
    Command subclass that enriches missing-argument errors for registry options.

    When Click's parser raises ``BadOptionUsage`` for a registry-backed option
    (e.g. bare ``--constraint`` with no value), this override catches it and
    re-raises a ``BadParameter`` with the expected format and valid kinds.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """
        Override to intercept missing-value errors for registry options.

        :param ctx: Click context
        :param args: Command-line arguments to parse
        :return: Remaining arguments after parsing
        """
        try:
            return super().parse_args(ctx, args)
        except click.BadOptionUsage as e:
            for param in self.params:
                if (
                    isinstance(param, RegistryConfigOption)
                    and param.registry_type is not None
                    and hasattr(e, "option_name")
                    and e.option_name in param.opts
                ):
                    usage = format_kind_config_usage(param.registry_type)
                    raise click.BadParameter(
                        f"option requires a value.\n{usage}",
                        ctx=ctx,
                        param=param,
                    ) from e
            raise


def _parse_and_check_kind_config(
    ctx: click.Context,
    param: click.Parameter,
    raw: str,
    discriminator: str,
    registry_type: type[PydanticClassRegistryMixin],  # type: ignore[type-arg]
    usage: str,
) -> dict[str, Any]:
    """
    Parse a registry config string and require a valid discriminator value.

    :param ctx: Click context
    :param param: Click parameter
    :param raw: Raw CLI string value
    :param discriminator: Discriminator field name (e.g. ``"kind"``)
    :param registry_type: Registry used to validate the discriminator value
    :param usage: Preformatted usage hint to append on errors
    :return: Parsed config dictionary
    :raises click.BadParameter: When parsing fails, the value is not a dict
        with the required discriminator key, or the kind is unregistered
    """
    try:
        parsed = parse_arguments(ctx, param, raw)
    except click.BadParameter as e:
        raise click.BadParameter(
            f"{e.message}\n{usage}",
            ctx=ctx,
            param=param,
        ) from e

    if not isinstance(parsed, dict):
        raise click.BadParameter(
            f"invalid config value (expected key=value pairs or "
            f"JSON/YAML object).\n{usage}",
            ctx=ctx,
            param=param,
        )

    if discriminator not in parsed:
        raise click.BadParameter(
            f"missing required key '{discriminator}'.\n{usage}",
            ctx=ctx,
            param=param,
        )

    # Reject unknown kinds early so the error is a single clean usage block.
    kind_value = parsed[discriminator]
    valid_kinds = registry_type.registered_names()
    if kind_value not in valid_kinds:
        raise click.BadParameter(
            f"invalid {discriminator} {kind_value!r}.\n{usage}",
            ctx=ctx,
            param=param,
        )
    return parsed


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
    usage = format_kind_config_usage(registry_type)

    def callback(
        ctx: click.Context,
        param: click.Parameter,
        value: Any,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if is_list:
            if not value:
                return None
            return [
                _parse_and_check_kind_config(
                    ctx, param, v, discriminator, registry_type, usage
                )
                for v in value
            ]

        if value is None:
            return None
        return _parse_and_check_kind_config(
            ctx, param, value, discriminator, registry_type, usage
        )

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
        "cls": RegistryConfigOption,
        "registry_type": registry,
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


def _resolve_registry_type(
    loc: tuple[str | int, ...],
    base_class: type[BaseModel],
) -> type[PydanticClassRegistryMixin] | None:  # type: ignore[type-arg]
    """
    Walk a pydantic error location to the nearest registry-backed field type.

    :param loc: Validation error location tuple
    :param base_class: Root model to start walking from
    :return: The registry mixin class if found, otherwise ``None``
    """
    current_model: type[BaseModel] = base_class

    for component in loc:
        if isinstance(component, int):
            continue

        fields = current_model.model_fields
        if component not in fields:
            return None

        annotation = fields[component].annotation
        if get_origin(annotation) is list:
            args = get_args(annotation)
            annotation = args[0] if args else None

        if annotation is None:
            return None

        try:
            if issubclass(annotation, PydanticClassRegistryMixin):
                return annotation
            if issubclass(annotation, BaseModel):
                current_model = annotation
                continue
        except TypeError:
            return None

        return None

    return None


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

    For registry shape failures where the stock message does not list kinds
    (non-object input), appends a usage hint with expected format and valid kinds.
    Invalid ``kind`` tags are left as-is because pydantic already lists them.

    :param ctx: The active Click context
    :param err: The pydantic validation error to translate
    :param base_class: Optional root model class for resolving argument aliases
    :return: A ``click.BadParameter`` with a formatted message and param hint
    """
    errs = err.errors(include_url=False, include_context=True, include_input=True)

    param_names = set()
    msgs = []
    for e in errs:
        original_loc = e.get("loc", ())
        loc = original_loc
        msg = e.get("msg", "validation error")
        err_type = e.get("type")

        top_field: str | None = None
        if len(loc) > 0:
            if base_class is not None:
                with contextlib.suppress(KeyError):
                    loc, top_field = _resolve_param_name(loc, base_class)
            top_field = top_field or str(loc[0]) or None

            if top_field:
                param_names.add("--" + top_field.replace("_", "-"))

        formatted = _error_to_message(loc, msg)

        if (
            base_class is not None
            and err_type in _REGISTRY_SHAPE_ERROR_TYPES
            and (registry := _resolve_registry_type(original_loc, base_class))
            is not None
        ):
            formatted = f"{formatted}\n{format_kind_config_usage(registry)}"

        msgs.append(formatted)

    error_msg = "".join(f"\n  - {msg}" for msg in msgs)
    return click.BadParameter(error_msg, ctx=ctx, param_hint=list(param_names))
