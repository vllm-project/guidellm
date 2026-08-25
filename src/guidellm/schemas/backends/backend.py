"""
Base backend Args schema for polymorphic backend configuration.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from pydantic import Field

from guidellm.schemas import PydanticClassRegistryMixin, standard_model_config

__all__ = ["BackendArgs"]


class BackendArgs(PydanticClassRegistryMixin["BackendArgs"], ABC):
    """
    Base class for backend creation arguments.

    This class serves as a base for defining argument models used in the creation
    of backend instances. It inherits from PydanticClassRegistryMixin to enable
    automatic registration of subclasses, allowing for flexible and extensible
    backend configurations.

    :cvar schema_discriminator: Field name for polymorphic deserialization
    """

    model_config = standard_model_config()

    schema_discriminator: ClassVar[str] = "kind"

    @classmethod
    def __pydantic_schema_base_type__(cls) -> type[BackendArgs]:
        """
        Return base type for polymorphic validation hierarchy.

        :return: Base BackendArgs class for schema validation
        """
        if cls.__name__ == "BackendArgs":
            return cls

        return BackendArgs

    kind: str = Field(
        description="Identify the desired backend implementation.",
        examples=["openai_http"],
    )
