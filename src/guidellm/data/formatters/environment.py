from __future__ import annotations

from typing import Any, ClassVar

from jinja2 import Template
from jinja2.nativetypes import NativeEnvironment, NativeTemplate

from guidellm.data.formatters.filters import JinjaFiltersRegistry
from guidellm.data.formatters.globals import JinjaGlobalsRegistry
from guidellm.data.formatters.templates import JinjaTemplatesRegistry

__all__ = ["JinjaEnvironmentMixin"]


class JinjaEnvironmentMixin:
    jinja_environment: ClassVar[NativeEnvironment | None] = None

    @classmethod
    def create_environment(cls, **env_kwargs: Any) -> NativeEnvironment:
        if "autoescape" not in env_kwargs:
            env_kwargs["autoescape"] = False

        extensions = env_kwargs.pop("extensions", [])
        extensions = set(extensions) | {"jinja2.ext.do"}

        env = NativeEnvironment(extensions=list(extensions), **env_kwargs)  # noqa: S701

        # Attach registered filters
        filters_registry = JinjaFiltersRegistry.registry  # type: ignore[misc]
        if filters_registry:
            for name, func in filters_registry.items():
                env.filters[name] = func

        # Attach registered globals
        globals_registry = JinjaGlobalsRegistry.registry  # type: ignore[misc]
        if globals_registry:
            for name, value in globals_registry.items():
                env.globals[name] = value

        cls.jinja_environment = env
        return env

    @classmethod
    def get_environment(cls) -> NativeEnvironment:
        if cls.jinja_environment is None:
            raise ValueError(
                "Jinja environment is not initialized. Call create_environment first."
            )
        return cls.jinja_environment

    @classmethod
    def template_from_source(cls, source: str | Template) -> NativeTemplate:
        if isinstance(source, Template):
            return source
        env = cls.get_environment()
        return env.from_string(source)

    @classmethod
    def template_from_registry(cls, name: str) -> NativeTemplate:
        template = JinjaTemplatesRegistry.get_registered_object(name)
        if template is None:
            raise ValueError(f"Template '{name}' not found in registry.")
        return cls.template_from_source(template)
